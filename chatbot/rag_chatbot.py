# rag_chatbot.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import subprocess
import sys
import os

# === CONFIG ===
FAISS_INDEX_PATH = "../vectorization/faiss_index"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
OLLAMA_MODEL = "nous-hermes2-mixtral"

# === Load Vector Store ===
print("📦 Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# === Ask user ===
while True:
    query = input("\n💬 Ask your question (or type 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting chatbot.")
        break

    print("🔍 Retrieving relevant documents...")
    docs = vectorstore.similarity_search(query, k=4)

    context = "\n---\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful economic assistant. Use only the context below to answer the question.
If the context does not contain enough info, say you don't know.

Context:
{context}

Question: {query}
Answer:
"""

    print("🤖 Sending to Ollama model...")

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print("\n📤 Response:\n")
        print(result.stdout.decode("utf-8"))
    else:
        print("❌ Error talking to Ollama:")
        print(result.stderr.decode("utf-8"))
