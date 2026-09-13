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
from PIL import Image
import asyncio
import math
import numpy as np
import cv2
import time
from playwright.async_api import async_playwright
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--debug", "-d", action="store_true")
args = arg_parser.parse_args()


# ─── Config ────────────────────────────────────────────────────────────────────
OUTPUT="/home/mac/projects/snake/out"
GAME_URL       = "https://snake.io"
FRAME_RATE     = 0.05
CHASE_RADIUS   = 450
BOOST_MIN_DIST = 80
BOOST_MAX_DIST = 380

# ─── HSV Color Ranges ──────────────────────────────────────────────────────────
ENEMY_LOWER     = np.array([0,   120, 120])
ENEMY_UPPER     = np.array([179, 255, 255])
BG_LOWER        = np.array([82,  50,  130])
BG_UPPER        = np.array([99, 145, 225])

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


# ─── Frame decode ──────────────────────────────────────────────────────────────

import base64

def b64_to_cv2(data_url: str) -> np.ndarray:
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─── Vision ────────────────────────────────────────────────────────────────────

def find_blobs(mask: np.ndarray, min_area: int):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if area > min_area:
            if M["m00"] > 0:
                blobs.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), area))
    return blobs


def detect_game_objects(frame: np.ndarray,tick: int):
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #enemy_mask = cv2.inRange(hsv, ENEMY_LOWER, ENEMY_UPPER)
    bg_mask    = cv2.inRange(hsv, BG_LOWER, BG_UPPER)
    bg_mask = cv2.bitwise_not(bg_mask)
    bg_mask[50:135,550:725] = 0
    bg_mask[:275,968:] = 0
    bg_mask = cv2.erode(bg_mask, np.ones((3,3), np.uint8), iterations=5)

    #enemy_mask = cv2.bitwise_and(enemy_mask, cv2.bitwise_not(bg_mask))
    #enemy_mask = cv2.dilate(enemy_mask, np.ones((7, 7), np.uint8), iterations=2)

    min_area = 40
    blobs = find_blobs(bg_mask, min_area)
    
    if args.debug:
        bg_mask = cv2.cvtColor(bg_mask,cv2.COLOR_GRAY2BGR)
        for blob in blobs:
            bg_mask = cv2.circle(bg_mask, (blob[0],blob[1]), 10, (0,0,255))
        cv2.imwrite(f"{OUTPUT}/frame{tick}.jpg", np.vstack((frame,bg_mask)))
        blobs.sort(key=lambda x: x[1])
        with open("out/blobs.txt", "a") as f:
            f.write(f"tick {tick:<3} |  ")
            for i,blob in enumerate(blobs):
                f.write(f"{i}:{blob[2]}  ")
            f.write("\n")

    blobs.sort(key=lambda x: x[2])
    area_cutoff = 1000
    index_cutoff = len(blobs)
    for i,blob in enumerate(blobs):
        if blob[1] > area_cutoff:
            index_cutoff = i - 1

    food = [(b[0],b[1]) for b in blobs[:index_cutoff]]
    enemies = blobs[index_cutoff:]
            
    return enemies, food


# ─── Bot brain ─────────────────────────────────────────────────────────────────

class AggressiveBot:
    def __init__(self):
        self.boost_active = False
        self.drift_angle  = 0.0

    def decide(self, frame: np.ndarray,tick: int):
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        enemies, food = detect_game_objects(frame,tick)
        head_x = cx
        head_y = cy

        best_enemy, best_dist = None, float("inf")
        for ex, ey, area in enemies:
            d = math.hypot(ex - head_x, ey - head_y)
            if d < best_dist and d < CHASE_RADIUS:
                best_dist, best_enemy = d, (ex, ey)

        if best_enemy:
            tx, ty = best_enemy
            angle = math.atan2(ty - head_y, tx - head_x)
            self.drift_angle = angle
            return (tx + math.cos(angle) * 40,
                    ty + math.sin(angle) * 40,
                    BOOST_MIN_DIST < best_dist < BOOST_MAX_DIST)

        if food:
            fx, fy = min(food, key=lambda p: math.hypot(p[0] - head_x, p[1] - head_y))
            self.drift_angle = math.atan2(fy - head_y, fx - head_x)
            return fx, fy, False

        self.drift_angle += 0.3
        return (head_x + math.cos(self.drift_angle) * 15,
                head_y + math.sin(self.drift_angle) * 15,
                False)

class PassiveBot:
    def __init__(self):
        self.boost_active = False
        self.drift_angle  = 0.0

    def decide(self, frame: np.ndarray,tick: int):
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        enemies, food = detect_game_objects(frame,tick)
        head_x = cx
        head_y = cy


 


# ─── Main loop ─────────────────────────────────────────────────────────────────

async def run_bot():
    bot = AggressiveBot()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--window-size=1280,720", "--mute-audio"])
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
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
        print(f"[BOT] Canvas: {cw}x{ch} at page ({cl},{ct})")
        print("[BOT] Running — Ctrl+C to stop\n")

        game_loaded = False
        with open("tesseract_output", "w") as f:
            f.write("")
        while not game_loaded:
            start = time.time()
            result = await page.evaluate(READ_JS)
            if result:
                frame = b64_to_cv2(result) 
                image = Image.frombytes("RGB",(frame.shape[1],frame.shape[0]),frame)
                text = pytesseract.image_to_string(image).lower()
                if "download" not in text and ("player" in text or "input" in text or "lag" in text or "ping" in text):
                    game_loaded=True
                    print("Game loaded, starting bot")
            end = time.time()
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
                frame = frame[:-40,:,:]
                end2 = time.time()
                print(f"[DEBUG] js eval  took {end2-start:.3f} secs")

                start2 = time.time()
                tx, ty, boost = bot.decide(frame,tick)
                end2 = time.time()
                print(f"[DEBUG] decision took {end2-start2:.3f} secs")

                start2 = time.time()
                #await page.mouse.move(cl + tx, ct + ty)
                end2 = time.time()
                print(f"[DEBUG] input    took {end2-start2:.3f} secs")

                if boost and not bot.boost_active:
                    await page.mouse.down()
                    bot.boost_active = True
                elif not boost and bot.boost_active:
                    await page.mouse.up()
                    bot.boost_active = False

                end2 = time.time()
                print(f"[DEBUG] tick {tick} took {end2-start:.3f} secs")
                tick += 1

            except KeyboardInterrupt:
                print("\n[BOT] Stopped.")
                break

        await page.mouse.up()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_bot())