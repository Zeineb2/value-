from crawl4ai import AsyncWebCrawler
import asyncio
import json
import os

async def run_crawler(seed_file="serper_links.json", output_file="crawled_links.json"):
    print("🕷️ Running crawler...")

    if not os.path.exists(seed_file):
        print(f"❌ Seed file '{seed_file}' not found.")
        return

    with open(seed_file, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    crawled = []

    async with AsyncWebCrawler() as crawler:
        for idx, url in enumerate(seeds[:10], 1):  # Adjust limit if needed
            try:
                print(f"🔗 Crawling {idx}: {url}")
                result = await crawler.arun(url=url)
                if hasattr(result, "links"):
                    links = [
                        link.url if hasattr(link, "url") else link
                        for link in result.links
                    ]
                    crawled.extend(links)
            except Exception as e:
                print(f"❌ Error crawling {url}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(set(crawled)), f, indent=2)

    print(f"✅ Crawling completed. {len(crawled)} links saved to {output_file}")
