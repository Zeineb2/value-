import pandas as pd

VALID_KEYWORDS = [
    "/publication", "/statistiques", "/data", "/indicator", "/dataportal",
    "/documents", "/enquetes", "/economic", "/indicators", "/document"
]

def pre_filter_links(input_excel="serper_links.xlsx", output_excel="filtered_links.xlsx"):
    df = pd.read_excel(input_excel)
    df["URL"] = df["URL"].astype(str).str.lower().str.strip()
    df_filtered = df[df["URL"].apply(lambda u: any(k in u for k in VALID_KEYWORDS))]
    df_filtered.to_excel(output_excel, index=False)
    print(f"✅ Filtered links saved to {output_excel} ({len(df_filtered)} kept)")

if __name__ == "__main__":
    pre_filter_links()
