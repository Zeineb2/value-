import json
import unicodedata
import os

def normalize(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8").lower().strip()

def title_case_indicator(name):
    return " ".join(word.capitalize() for word in name.strip().split())

# === Define keyword → category mapping rules
CATEGORY_RULES = [
    # National Accounts
    ("gdp", "National Accounts"),
    ("gross domestic product", "National Accounts"),
    ("gni", "National Accounts"),
    ("national income", "National Accounts"),
    ("economic output", "National Accounts"),

    # Households
    ("disposable income", "Households"),
    ("household", "Households"),
    ("ménage", "Households"),

    # Demographics
    ("population", "Demographics"),
    ("birth rate", "Demographics"),
    ("death rate", "Demographics"),
    ("life expectancy", "Demographics"),
    ("age dependency", "Demographics"),

    # Prices
    ("inflation", "Prices"),
    ("cpi", "Prices"),
    ("consumer price index", "Prices"),
    ("price", "Prices"),
    ("deflator", "Prices"),

    # Employment
    ("unemployment", "Employment"),
    ("labor force", "Employment"),
    ("employment rate", "Employment"),
    ("jobless", "Employment"),

    # Savings
    ("savings", "Savings"),
    ("épargne", "Savings"),

    # Investment
    ("capital", "Investment"),
    ("formation", "Investment"),
    ("gross fixed capital", "Investment"),
    ("investment", "Investment"),

    # Trade
    ("imports", "Trade"),
    ("exports", "Trade"),
    ("current account", "Trade"),
    ("balance of trade", "Trade"),
    ("net exports", "Trade"),
    ("trade balance", "Trade"),

    # Finance
    ("debt", "Finance"),
    ("budget", "Finance"),
    ("deficit", "Finance"),
    ("public", "Finance"),
    ("fiscal", "Finance"),
    ("account balance", "Finance"),
    ("government expenditure", "Finance"),

    # Expenditure
    ("consumption", "Expenditure"),
    ("household consumption", "Expenditure"),

    # Other (generic catch)
    ("ratio", "Other"),
    ("rate", "Other"),
    ("index", "Other")
]


def assign_category(canonical_name):
    name = normalize(canonical_name)
    for keyword, category in CATEGORY_RULES:
        if keyword in name:
            return category
    return "Other"

# === Load indicators
with open("economic_indicator.json", "r", encoding="utf-8") as f:
    indicators = json.load(f)

canonical_map = {}

# === Build canonical alias map with categories
for item in indicators:
    canonical_raw = item.get("Canonical Name", "").strip()
    if not canonical_raw:
        continue

    canonical = title_case_indicator(canonical_raw)
    category = assign_category(canonical)
    aliases = item.get("Aliases", [])

    for alias in aliases + [canonical_raw]:
        key = normalize(alias)
        if key not in canonical_map:
            canonical_map[key] = {
                "canonical": canonical,
                "category": category
            }

# === Save result
os.makedirs("utils", exist_ok=True)
output_path = "utils/canonical_indicators.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dict(sorted(canonical_map.items())), f, indent=2, ensure_ascii=False)

print(f"✅ Generated {output_path} with {len(canonical_map)} entries and auto-categorized them.")
