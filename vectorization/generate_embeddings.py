# generate_embeddings.py  — drop-in fix
import os, json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


INPUT_JSON = "../scraping/output/improved_structured_indicators.json"
FAISS_DIR = "faiss_index"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# 1) Load
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    entries = json.load(f)

texts, metadatas = [], []

def to_text(e):
    year = e.get("Year")
    ind  = (e.get("CanonicalIndicator") or e.get("Indicator") or "").strip()
    val  = e.get("DisplayValue") or e.get("Value")
    unit = e.get("Unit") or ""
    src  = e.get("Source") or "unknown"
    raw  = (e.get("RawText") or "")[:800]
    # BGE likes passage prefix
    return f"passage: Tunisia | {ind} | year={year} | value={val} {unit} | source={src}. Context: {raw}"

for i, e in enumerate(entries):
    texts.append(to_text(e))
    # keep rich meta for UI / later use
    metadatas.append({
        "id": i,
        "year": e.get("Year"),
        "indicator": e.get("CanonicalIndicator") or e.get("Indicator"),
        "value": e.get("DisplayValue") or e.get("Value"),
        "unit": e.get("Unit"),
        "source": e.get("Source"),
        "category": e.get("Category"),
        "confidence": e.get("Confidence"),
    })

# 2) Embed model (normalize=True for cosine via IP)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda"}, 
    encode_kwargs={"normalize_embeddings": True}
)

# 3) Build & save
os.makedirs(FAISS_DIR, exist_ok=True)
vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
vectorstore.save_local(FAISS_DIR)
print(f"✅ Saved {len(texts)} docs to {FAISS_DIR} with {EMBED_MODEL}")
