"""
Snake.io Aggressive Bot — gl.readPixels Edition
================================================
Captures frames by calling gl.readPixels() via JS, which reads directly
from the GPU framebuffer. Returns raw RGBA bytes as a JS array, which we
decode into a numpy array in Python.

Note: still requires the game to be mid-frame when we call it, so we hook
requestAnimationFrame to capture at the right moment.

Requirements:
    pip install playwright opencv-python numpy
    playwright install chromium
    playwright install-deps    # Linux only
"""


import pytesseract
import base64
from PIL import Image
import asyncio
import math
import numpy as np
import cv2
import time
from playwright.async_api import async_playwright
from argparse import ArgumentParser
from geometry import line_intersects_contour, detect_game_objects
from objects import GameObject

arg_parser = ArgumentParser()
arg_parser.add_argument("--debug", "-d", action="store_true")
arg_parser.add_argument("--time", "-t", action="store_true")
args = arg_parser.parse_args()


# ─── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/home/mac/projects/snake/out"
GAME_URL       = "https://snake.io"


# ─── Frame Info ──────────────────────────────────────────────────────────
BOTTOM_CROP = 40

# ─── JS: capture WebGL frame as JPEG base64 via a 2D proxy canvas ─────────────
# Hooking rAF lets us read pixels right after the game draws (before buffer clear).
# We encode as JPEG in JS — far faster to transfer than a raw Array.from(pixels).
HOOK_JS = """
() => {
    if (window.__botHooked) return 'already_hooked';

    const c = document.querySelector('canvas');
    if (!c) return 'no_canvas';

    const gl = c.getContext('webgl') || c.getContext('webgl2');
    if (!gl) return 'no_gl';

    const proxy = document.createElement('canvas');
    proxy.width  = c.width;
    proxy.height = c.height;
    const ctx = proxy.getContext('2d');

    window.__botFrame  = null;
    window.__botHooked = true;

    const origRAF = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = function(cb) {
        return origRAF(function(ts) {
            cb(ts);
            try {
                const w = c.width, h = c.height;
                const buf = new Uint8Array(w * h * 4);
                gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                // Flip vertically — WebGL origin is bottom-left
                const flipped = new Uint8Array(w * h * 4);
                for (let y = 0; y < h; y++) {
                    const src = (h - 1 - y) * w * 4;
                    flipped.set(buf.subarray(src, src + w * 4), y * w * 4);
                }
                ctx.putImageData(new ImageData(new Uint8ClampedArray(flipped), w, h), 0, 0);
                window.__botFrame = proxy.toDataURL('image/jpeg', 0.6);
            } catch(e) {
                window.__botFrame = null;
            }
        });
    };

    return 'hooked';
}
"""

READ_JS = "() => window.__botFrame"


CURSOR_JS = """
(() => {
  if (window.top !== window) return;
  // Grab the real rAF now, before the bot's HOOK_JS replaces it.
  // Otherwise every cursor redraw would trigger an extra readPixels.
  const raf = window.requestAnimationFrame.bind(window);
  let dot = null, x = 0, y = 0, queued = false;

  const draw = () => {
    queued = false;
    if (dot) dot.style.transform = `translate3d(${x}px,${y}px,0)`;
  };

  window.addEventListener('mousemove', e => {
    x = e.clientX; y = e.clientY;
    if (!queued) { queued = true; raf(draw); }
  }, { passive: true });

  const install = () => {
    dot = document.createElement('div');
    dot.style.cssText = 'position:fixed;left:0;top:0;width:16px;height:16px;' +
      'margin:-8px 0 0 -8px;border:2px solid red;border-radius:50%;' +
      'box-sizing:border-box;pointer-events:none;z-index:2147483647;' +
      'will-change:transform;contain:strict;';
    document.documentElement.appendChild(dot);
  };
  if (document.documentElement) install();
  else document.addEventListener('DOMContentLoaded', install, { once: true });
})();
"""

# ─── Frame decode ──────────────────────────────────────────────────────────────
def b64_to_cv2(data_url: str) -> np.ndarray:
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─── BOTS ─────────────────────────────────────────────────────────────────
class AggressiveBot:  # OUTDATED, needs refactoring for GameObject type
    def __init__(self):
        self.boost_active = False
        self.drift_angle = 0.0
        self.head_x = 1265 / 2
        self.head_y = 565 / 2
        self.circle_input = (
            self.head_x + math.cos(self.drift_angle) * 15,
            self.head_y + math.sin(self.drift_angle) * 15,
            False,
        )

    def decide(self, frame: np.ndarray, tick: int):
        enemies, food = detect_game_objects(frame, args.debug)

        best_enemy, best_dist = None, float("inf")
        for ex, ey, area in enemies:
            d = math.hypot(ex - self.head_x, ey - self.head_y)
            if d < best_dist and d < CHASE_RADIUS:
                best_dist, best_enemy = d, (ex, ey)

        if best_enemy:
            tx, ty = best_enemy
            angle = math.atan2(ty - self.head_y, tx - self.head_x)
            self.drift_angle = angle
            return (
                tx + math.cos(angle) * 40,
                ty + math.sin(angle) * 40,
                BOOST_MIN_DIST < best_dist < BOOST_MAX_DIST,
            )

        if food:
            fx, fy = min(
                food, key=lambda p: math.hypot(p[0] - self.head_x, p[1] - self.head_y)
            )
            self.drift_angle = math.atan2(fy - self.head_y, fx - self.head_x)
            return fx, fy, False

        self.drift_angle += 0.3


class PassiveBot:
    def __init__(self, frame_width, frame_height):
        self.boost_active = False
        self.drift_angle = 0.0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width / 2
        self.center_y = frame_height / 2
        self.heading = (0, 0)

    def circle(self):
        self.drift_angle += 1
        return (
            (self.center_x + 15, self.center_y, False)
            if self.drift_angle % 2 == 0
            else (self.center_x - 15, self.center_y, False)
        )

    def decide(self, frame: np.ndarray, tick: int):
        enemies, foods, us, debug_frame = detect_game_objects(frame, tick)

        if not foods:
            return self.circle()
        else:
            for f in foods:
                f.dist_moment_to_center = math.hypot(f.moment[0] - self.center_x, f.moment[1] - self.center_y)
            foods.sort(key=lambda f: f.dist_moment_to_center)

            nearest_food_index = 0

            while nearest_food_index != len(foods):
                nearest_food = foods[nearest_food_index]
                f_x, f_y = nearest_food.moment
                f_x = (f_x - self.center_x) * 1.3 + f_x
                f_y = (f_y - self.center_y) * 1.3 + f_y
                if enemies:
                    for enemy in enemies:
                        if line_intersects_contour(
                            (self.frame_width,self.frame_height),
                            enemy.contour,
                            (round(self.center_x), round(self.center_y)),
                            (round(f_x),round(f_y))
                        ):
                            nearest_food_index += 1
                            if args.debug:
                                debug_frame = cv2.line(
                                    debug_frame,
                                    (round(self.center_x), round(self.center_y * 3)),
                                    (
                                        round(f_x),
                                        round(f_y + self.center_y * 2),
                                    ),
                                    (0, 0, 255),
                                )
                            break
                        self.heading_x = f_x
                        self.heading_y = f_y
                        if args.debug:
                            debug_frame = cv2.line(
                                debug_frame,
                                (round(self.center_x), round(self.center_y * 3)),
                                (round(f_x), round(f_y + self.center_y * 2)),
                                (255, 0, 0),
                            )
                            cv2.imwrite(f"{OUTPUT_DIR}/frame{tick}.jpg", debug_frame)
                        return (f_x, f_y, False)
                else: 
                    self.heading_x = f_x
                    self.heading_y = f_y
                    if args.debug:
                        debug_frame = cv2.line(
                            debug_frame,
                            (round(self.center_x), round(self.center_y * 3)),
                            (round(f_x), round(f_y + self.center_y * 2)),
                            (255, 0, 0),
                        )
                        cv2.imwrite(f"{OUTPUT_DIR}/frame{tick}.jpg", debug_frame)
                    return (f_x, f_y, False)


            if args.debug:
                cv2.imwrite(f"{OUTPUT_DIR}/frame{tick}.jpg", debug_frame)
            return self.circle()


# ─── Main loop ─────────────────────────────────────────────────────────────────
async def run_bot():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False, args=["--window-size=1280,720", "--mute-audio"]
        )
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        await context.add_init_script(CURSOR_JS)
        page = await context.new_page()

        print("[BOT] Opening Snake.io ...")
        await page.goto(GAME_URL, wait_until="domcontentloaded")
        await asyncio.sleep(6)

        # Install the rAF hook
        result = await page.evaluate(HOOK_JS)
        print(f"[BOT] Hook result: {result}")
        if result not in ("hooked", "already_hooked"):
            print("[BOT] WARNING: hook failed, frame capture may not work")

        # Get canvas size for coordinate mapping
        cb = await page.evaluate("""
            () => {
                const c = document.querySelector('canvas');
                if (!c) return {left: 0, top: 0, w: 1280, h: 720};
                const r = c.getBoundingClientRect();
                return {left: Math.round(r.left), top: Math.round(r.top),
                        w: c.width, h: c.height};
            }
        """)
        cl, ct, cw, ch = cb["left"], cb["top"], cb["w"], cb["h"]
        bot = PassiveBot(frame_width=cw, frame_height=ch-BOTTOM_CROP)
        print(f"[BOT] Canvas: {cw}x{ch} at page ({cl},{ct})")

        game_loaded = False
        with open("tesseract_output", "w") as f:
            f.write("")
        while not game_loaded:
            start = time.time()
            result = await page.evaluate(READ_JS)
            if result:
                frame = b64_to_cv2(result)
                image = Image.frombytes("RGB", (frame.shape[1], frame.shape[0]), frame)
                text = pytesseract.image_to_string(image).lower()
                if "download" not in text and (
                    "player" in text
                    or "input" in text
                    or "lag" in text
                    or "ping" in text
                ):
                    game_loaded = True
                    print("Game loaded, starting bot")
            end = time.time()
            if args.time:
                print(f"[DEBUG] tesseract loop took {end-start:.3f} secs")
            await asyncio.sleep(0.5)

        tick = 0
        while True:
            try:
                start = time.time()
                result = await page.evaluate(READ_JS)

                if not result:
                    if tick % 20 == 0:
                        print("[BOT] Waiting for first frame...")
                    await asyncio.sleep(0.1)
                    continue

                frame = b64_to_cv2(result)
                frame = frame[:-BOTTOM_CROP, :, :] # remove text from bottom
                end2 = time.time()
                if args.time:
                    print(f"[DEBUG] js eval  took {end2-start:.3f} secs")

                start2 = time.time()
                tx, ty, boost = bot.decide(frame, tick)
                end2 = time.time()
                if args.time:
                    print(f"[DEBUG] decision took {end2-start2:.3f} secs")

                start2 = time.time()
                await page.mouse.move(cl + tx, ct + ty)
                end2 = time.time()
                if args.time:
                    print(f"[DEBUG] input    took {end2-start2:.3f} secs")

                if boost and not bot.boost_active:
                    await page.mouse.down()
                    bot.boost_active = True
                elif not boost and bot.boost_active:
                    await page.mouse.up()
                    bot.boost_active = False

                end2 = time.time()
                if args.time:
                    print(f"[DEBUG] tick {tick} took {end2-start:.3f} secs")
                tick += 1

            except KeyboardInterrupt:
                print(f"[INFO] avg tick time: {sum(times)/len(times)}")
                print("\n[BOT] Stopped.")
                break


if __name__ == "__main__":
    asyncio.run(run_bot())
