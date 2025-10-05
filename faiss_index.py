import os
import json
import faiss
import openai
from tqdm import tqdm
from typing import List
from openai import OpenAI
from pathlib import Path

# ✅ Hardcoded OpenAI API key (your request)
client = OpenAI(api_key="OPENAI_API_KEY")

# Directory containing category folders of JSONL product data
SCRAPED_RESULTS_DIR = "scraped_results"

# Output files
FAISS_INDEX_FILE = "product_faiss.index"
METADATA_FILE = "product_metadata.json"

def load_all_jsonl_files(base_dir: str) -> List[dict]:
    """Recursively loads all .jsonl product entries from the base directory"""
    all_entries = []
    base_path = Path(base_dir)

    for file_path in base_path.rglob("*.jsonl"):
        category = file_path.parent.name
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("full_text") and entry.get("title"):
                        entry["category"] = category
                        all_entries.append(entry)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON in {file_path}")
    return all_entries

def get_embedding(text: str) -> List[float]:
    """Calls OpenAI API to get embedding"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Failed to embed: {e}")
        return None

def build_faiss_index():
    print("🔍 Loading product entries...")
    products = load_all_jsonl_files(SCRAPED_RESULTS_DIR)
    print(f"✅ Loaded {len(products)} products")

    if not products:
        print("⚠️ No valid product data found.")
        return

    embeddings = []
    metadata = []

    print("🧠 Generating embeddings in batches...")

    BATCH_SIZE = 100  # tweak to 50–200 depending on your rate limit
    embeddings = []
    metadata = []

    batch_texts = []
    batch_meta = []

    for i, product in enumerate(tqdm(products, desc="Batching")):
        text = f"{product['title']}\n{product.get('summary', '')}\n{product.get('full_text', '')}"
        batch_texts.append(text)
        batch_meta.append({
            "title": product["title"],
            "url": product["url"],
            "price": product.get("price"),
            "rating": product.get("rating"),
            "category": product.get("category")
        })

        # when batch is full or last product reached
        if len(batch_texts) >= BATCH_SIZE or i == len(products) - 1:
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts
                )
                # store all embeddings + their metadata
                for j, item in enumerate(response.data):
                    embeddings.append(item.embedding)
                    metadata.append(batch_meta[j])

                print(f"✅ Embedded batch {len(embeddings)} total so far")
            except Exception as e:
                print(f"❌ Batch failed: {e}")

            # reset for next batch
            batch_texts.clear()
            batch_meta.clear()


    if not embeddings:
        print("❌ No embeddings generated.")
        return

    print("📦 Building FAISS index...")
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Save
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Index saved to {FAISS_INDEX_FILE}")
    print(f"✅ Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    import numpy as np
    build_faiss_index()
