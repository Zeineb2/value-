from playwright.sync_api import sync_playwright
import os
import json
import re
import time
from scrapers.flaresolverr_scraper import fetch_with_flaresolverr

def sanitize_filename(url):
    filename = url.strip().rstrip("/").split("/")[-1]
    if not filename or "." in filename:
        filename = re.sub(r"[^\w]", "_", url.strip().split("/")[-2]) if len(url.strip().split("/")) > 1 else "page"
    return filename + ".html"

def fetch_pages_with_playwright_from_json(json_path, save_dir="data/html"):
    with open(json_path, "r", encoding="utf-8") as f:
        urls = json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
            timezone_id="Europe/Paris",
            locale="en-US"
        )

        for url in urls:
            page = context.new_page()
            try:
                print(f"🔍 Visiting: {url}")
                page.goto(url, timeout=60000)
                time.sleep(10)
                content = page.content()

                # Fallback to FlareSolverr if blocked
                if any(kw in content.lower() for kw in ["verifying you are human", "cloudflare", "access denied"]):
                 print(f"⚠️ Blocked: {url}. Retrying with FlareSolverr...")
                 fetch_with_flaresolverr(url, save_dir)
                continue
            
                filename = sanitize_filename(url)
                path = os.path.join(save_dir, filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Saved {url} → {path}")
            except Exception as e:
                print(f"❌ Failed to fetch {url}: {e}")
            finally:
                page.close()

        browser.close()
