"""
Snake.io Aggressive Bot — Tuned Vision Edition
===============================================
Uses Playwright's page.screenshot() cropped to the canvas rect instead of
canvas.toDataURL(), which returns black due to WebGL cross-origin tainting.

Requirements:
    pip install playwright opencv-python numpy
    playwright install chromium
    playwright install-deps    # Linux only
"""

import asyncio
import math
import numpy as np
import cv2
from playwright.async_api import async_playwright


# ─── Config ────────────────────────────────────────────────────────────────────
GAME_URL       = "https://snake.io"
FRAME_RATE     = 0.05     # seconds per tick (~20 fps)
CHASE_RADIUS   = 450      # px — max distance to chase an enemy
BOOST_MIN_DIST = 80       # px — don't boost when this close (cut-off mode)
BOOST_MAX_DIST = 380      # px — boost when enemy is within this range

# ─── HSV Color Ranges (calibrated from real screenshot) ────────────────────────
OUR_SNAKE_LOWER = np.array([10,  180, 180])   # orange snake
OUR_SNAKE_UPPER = np.array([30,  255, 255])

ENEMY_LOWER     = np.array([0,   120, 120])   # any saturated color
ENEMY_UPPER     = np.array([179, 255, 255])

FOOD_LOWER      = np.array([0,   150, 200])   # bright small dots
FOOD_UPPER      = np.array([179, 255, 255])

BG_LOWER        = np.array([82,  50,  130])   # muted teal background
BG_UPPER        = np.array([103, 120, 225])


# ─── Vision ────────────────────────────────────────────────────────────────────

def png_bytes_to_cv2(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_blobs(mask: np.ndarray, min_area: int, max_area: int):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                blobs.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), area))
    return blobs


def detect_game_objects(frame: np.ndarray):
    h, w = frame.shape[:2]
    canvas_cx, canvas_cy = w / 2, h / 2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Our snake (orange)
    our_mask = cv2.inRange(hsv, OUR_SNAKE_LOWER, OUR_SNAKE_UPPER)
    our_blobs = find_blobs(our_mask, 50, 100_000)
    our_center = None
    if our_blobs:
        best = min(our_blobs, key=lambda b: math.hypot(b[0] - canvas_cx, b[1] - canvas_cy))
        our_center = (best[0], best[1])

    # Enemy snakes (any saturated non-orange, non-background blob)
    enemy_mask = cv2.inRange(hsv, ENEMY_LOWER, ENEMY_UPPER)
    bg_mask    = cv2.inRange(hsv, BG_LOWER, BG_UPPER)
    enemy_mask = cv2.bitwise_and(enemy_mask, cv2.bitwise_not(bg_mask))
    enemy_mask = cv2.bitwise_and(enemy_mask, cv2.bitwise_not(our_mask))
    ex, ey = (int(our_center[0]), int(our_center[1])) if our_center else (int(canvas_cx), int(canvas_cy))
    cv2.circle(enemy_mask, (ex, ey), 80, 0, -1)
    kernel = np.ones((7, 7), np.uint8)
    enemy_mask = cv2.dilate(enemy_mask, kernel, iterations=2)
    enemies = find_blobs(enemy_mask, 300, 500_000)

    # Food orbs (small bright dots)
    food_mask = cv2.inRange(hsv, FOOD_LOWER, FOOD_UPPER)
    food_mask = cv2.bitwise_and(food_mask, cv2.bitwise_not(our_mask))
    food_mask = cv2.bitwise_and(food_mask, cv2.bitwise_not(enemy_mask))
    food = [(b[0], b[1]) for b in find_blobs(food_mask, 4, 120)]

    return our_center, enemies, food


# ─── Bot brain ─────────────────────────────────────────────────────────────────

class AggressiveBot:
    def __init__(self):
        self.boost_active = False
        self.drift_angle  = 0.0

    def decide(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        our_center, enemies, food = detect_game_objects(frame)
        head_x = our_center[0] if our_center else cx
        head_y = our_center[1] if our_center else cy

        # 1. Hunt nearest enemy
        best_enemy, best_dist = None, float("inf")
        for ex, ey, area in enemies:
            d = math.hypot(ex - head_x, ey - head_y)
            if d < best_dist and d < CHASE_RADIUS:
                best_dist, best_enemy = d, (ex, ey)

        if best_enemy:
            tx, ty = best_enemy
            angle = math.atan2(ty - head_y, tx - head_x)
            lead_x = tx + math.cos(angle) * 40
            lead_y = ty + math.sin(angle) * 40
            boost = BOOST_MIN_DIST < best_dist < BOOST_MAX_DIST
            self.drift_angle = angle
            return lead_x, lead_y, boost

        # 2. Eat nearest food
        if food:
            fx, fy = min(food, key=lambda p: math.hypot(p[0] - head_x, p[1] - head_y))
            self.drift_angle = math.atan2(fy - head_y, fx - head_x)
            return fx, fy, False

        # 3. Tight circle in place
        self.drift_angle += 0.3
        return (head_x + math.cos(self.drift_angle) * 15,
                head_y + math.sin(self.drift_angle) * 15,
                False)


# ─── Main loop ─────────────────────────────────────────────────────────────────

async def run_bot():
    bot = AggressiveBot()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--window-size=1280,720"])
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        print("[BOT] Opening Snake.io ...")
        await page.goto(GAME_URL, wait_until="domcontentloaded")
        await asyncio.sleep(8)

        for selector in ["text=Play", ".btn-play", "#play", "button"]:
            try:
                await page.locator(selector).first.click(timeout=2000)
                print(f"[BOT] Clicked play: {selector}")
                await asyncio.sleep(0.5)
                break
            except Exception:
                pass

        await asyncio.sleep(3)

        # Get canvas bounding box (used for cropping screenshots and mapping coords)
        cb = await page.evaluate("""
            () => {
                const c = document.querySelector('canvas');
                if (!c) return {left: 0, top: 0, w: 1280, h: 720};
                const r = c.getBoundingClientRect();
                return {left: Math.round(r.left), top: Math.round(r.top),
                        w: Math.round(r.width),   h: Math.round(r.height)};
            }
        """)
        cl, ct, cw, ch = cb["left"], cb["top"], cb["w"], cb["h"]
        print(f"[BOT] Canvas: {cw}x{ch} at ({cl},{ct})")
        print("[BOT] Running — Ctrl+C to stop\n")

        tick = 0
        while True:
            try:
                # Capture full page then crop in numpy — avoids the flash/flicker
                # that occurs when using Playwright's clip parameter
                png = await page.screenshot()
                frame = png_bytes_to_cv2(png)[ct:ct+ch, cl:cl+cw]

                tx, ty, boost = bot.decide(frame)

                await page.mouse.move(cl + tx, ct + ty)

                if boost and not bot.boost_active:
                    await page.mouse.down()
                    bot.boost_active = True
                elif not boost and bot.boost_active:
                    await page.mouse.up()
                    bot.boost_active = False


                await asyncio.sleep(FRAME_RATE)
                tick += 1

            except KeyboardInterrupt:
                print("\n[BOT] Stopped.")
                break
            except Exception as e:
                print(f"[BOT] Error tick {tick}: {e}")
                await asyncio.sleep(0.3)

        await page.mouse.up()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_bot())