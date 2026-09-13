
from playwright.async_api import async_playwright
import asyncio

GAME_URL       = "https://snake.io"
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--window-size=1280,720"])
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        await page.goto(GAME_URL, wait_until="domcontentloaded")
        while True:
            await asyncio.sleep(0.016)

if __name__ == "__main__":
    asyncio.run(main())