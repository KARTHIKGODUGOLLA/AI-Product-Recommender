# app.py

import streamlit as st
import faiss
import json
import openai
import numpy as np
import os

# Load FAISS index and metadata
index = faiss.read_index("product_faiss.index")
with open("product_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not api_key:
    st.error("No OpenAI API key found. Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")
else:
    openai.api_key = api_key

# Embedding function
def embed_query(query):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype="float32")

# Search top-k products
def search_products(query, top_k=5):
    query_vector = embed_query(query)
    D, I = index.search(np.array([query_vector]), top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

# Generate response from top documents
def generate_response_with_rag(query, docs):
    context = "\n\n".join(
        f"Title: {item['title']}\nPrice: {item.get('price', '')}\nRating: {item.get('rating', '')}\nSummary: {item.get('summary', '')}\nURL: {item['url']}"
        for item in docs
    )

    prompt = f"""Given the following product documents, answer this query:

{context}

Query: {query}

Return a helpful, concise answer for the user.
"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Streamlit UI
st.title("🔍 Smart Product Search & Recommendation")
query = st.text_input("Enter your product search:", "mirrorless camera with 4K and flip screen")

if st.button("Search") and query:
    with st.spinner("Searching and thinking..."):
        top_results = search_products(query)

        st.subheader("Top Results")
        for item in top_results:
            st.markdown(f"**{item['title']}**")
            st.markdown(f"- Price: {item.get('price', 'N/A')}")
            st.markdown(f"- Rating: {item.get('rating', 'N/A')}")
            st.markdown(f"- [Link]({item['url']})")
            st.markdown("---")

        rag_response = generate_response_with_rag(query, top_results)
        st.subheader("🧠 Smart Recommendation")
        st.write(rag_response)
