from duckduckgo_search import DDGS
import pandas as pd

query = "Tunisia economic indicators 2018..2025 site:ins.tn OR site:worldbank.org OR site:imf.org"

results = []
with DDGS(useragent="Mozilla/5.0") as ddgs:  # Add useragent here
    for r in ddgs.text(query, max_results=10):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", "")
        })

df = pd.DataFrame(results)
df.to_excel("tunisia_economic_links_2018_2025.xlsx", index=False)
