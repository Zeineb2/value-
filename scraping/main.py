import asyncio
from services.serper import fetch_links
from scrapers.scrape_and_download import scrape_and_download
from core.parse_html import extract_text_from_html
from core.extractor import extract_structured_indicators
from services.crawler4Ai import run_crawler
from scrapers.scrape_and_download import scrape_and_download

def main():
    print("🚀 STEP 1: Fetching from Serper...")
    fetch_links()

    print("🌐 STEP 2: Scraping and downloading...")
    asyncio.run(scrape_and_download("serper_links.xlsx"))

    print("🧠 STEP 2.5: Scraping HTML pages with Playwright...")
    asyncio.run(scrape_and_download("serper_links.xlsx"))

    print("🧼 STEP 3: Parsing HTML...")
    extract_text_from_html()

    print("📊 STEP 4: Extracting indicators...")
    extract_structured_indicators()

    print("🕷️ STEP 5: Crawling for more...")
    asyncio.run(run_crawler())

    print("✅ Pipeline complete.")

if __name__ == "__main__":
    main()