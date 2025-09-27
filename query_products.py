import faiss
import json
import openai
import numpy as np
from tqdm import tqdm

# Set your OpenAI API key directly here
openai.api_key = "OPENAI_API_KEY"

# Load FAISS index and metadata
index = faiss.read_index("product_faiss.index")
with open("product_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def embed_query(query):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype="float32")

def search_products(user_query, top_k=5):
    query_vector = embed_query(user_query)
    D, I = index.search(np.array([query_vector]), top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

def generate_response_with_rag(user_query, results):
    context_blocks = []
    for item in results:
        context_blocks.append(
            f"Title: {item.get('title', '')}\nPrice: {item.get('price', '')}\nRating: {item.get('rating', '')}\nSummary: {item.get('summary', '')}\nURL: {item.get('url', '')}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a smart shopping assistant. A user asked: "{user_query}"

Based on the following product data, recommend the most relevant product(s), explain why, and present it in a helpful tone.

PRODUCT DATA:
{context}

Answer:
"""

    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ RAG generation failed: {e}")
        return None

if __name__ == "__main__":
    query = input("Enter your product query: ")
    results = search_products(query, top_k=5)

    print("\n🔎 Top Matches:\n")
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['title']}  —  {item['price']}\n   {item['url']}\n   Rating: {item['rating']}\n")

    print("\n🧠 Smart Recommendation:\n")
    rag_response = generate_response_with_rag(query, results)
    if rag_response:
        print(rag_response)
