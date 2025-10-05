import streamlit as st
import faiss
import json
import openai
import numpy as np
import os

# ──────────────────────────────────────────────
# 🔧 Page setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ShopLyst AI",
    page_icon="🛒",
    layout="wide"
)
st.title("ShopLyst — Smart Product Finder")
st.markdown("Find products based on **meaning**, not just keywords.")

# ──────────────────────────────────────────────
# 🔐 API & Data Setup
# ──────────────────────────────────────────────
def load_api_key():
    """Fetch OpenAI API key from environment or Streamlit secrets."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("No OpenAI API key found. Please set OPENAI_API_KEY in environment variables.")
        st.stop()
    openai.api_key = api_key


def load_data():
    """Load FAISS index and product metadata."""
    try:
        index = faiss.read_index("product_faiss.index")
        with open("product_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index or metadata: {e}")
        st.stop()


# ──────────────────────────────────────────────
# 🧠 Core Functions
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def embed_query(query: str):
    """Generate embedding for a given query."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype="float32")

def search_products(query: str, index, metadata, top_k):
    """Search top-k similar products using FAISS."""
    query_vector = embed_query(query)

    # Ensure the query embedding and FAISS index match in dimension
    if query_vector.shape[0] != index.d:
        st.error(f"Embedding dimension mismatch: query({query_vector.shape[0]}) vs index({index.d})")
        st.stop()

    # Perform FAISS similarity search
    D, I = index.search(np.array([query_vector]), top_k)

    # Collect matching product metadata
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return results



def generate_response_with_rag(query: str, docs: list):
    """Generate a single overall recommendation (not per item)."""
    context = "\n\n".join(
        f"Title: {item['title']}\n"
        f"Price: {item.get('price', 'Not available')}\n"
        f"Rating: {item.get('rating', 'Not available')}\n"
        f"Summary: {item.get('summary', 'No summary provided')}\n"
        f"URL: {item['url']}"
        for item in docs
    )

    prompt = f"""
You are ShopLyst, an AI-powered shopping assistant.

The user searched for: '{query}'.

Below are some related product listings. Your job:
- Analyze them collectively.
- Identify **the single most suitable option or combination** for the user.
- Give a clear, concise recommendation — 3 to 5 sentences max.
- Maintain a confident, helpful tone (like a top-tier shopping guide).
- If any product lacks price or rating, naturally phrase around it (e.g., "well-reviewed" or "affordable option").
- End your message with a small call-to-action like “You might want to start here.”

PRODUCT DATA:
{context}

Return only the final recommendation text (no bullets, no headings).
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI reasoning failed: {e}"



# ──────────────────────────────────────────────
# 🎨 UI Functions
# ──────────────────────────────────────────────
def render_sidebar(metadata, index):
    """Render sidebar filters and search history."""
    st.sidebar.header("⚙️ Options")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    top_k = st.sidebar.slider("Number of results", 3, 10, 5)

    # Category filter
    if metadata and "category" in metadata[0]:
        categories = sorted({item["category"] for item in metadata})
        selected_categories = st.sidebar.multiselect("Filter by category", categories)
    else:
        selected_categories = []

    # Reload index
    if st.sidebar.button("Reload Index", use_container_width=True):
        index, metadata = load_data()
        st.sidebar.success("Index reloaded!")

    # Recent searches
    st.sidebar.subheader("🕒 Recent Searches")
    for q in st.session_state["history"][-5:][::-1]:
        st.sidebar.markdown(f"🔹 {q}")

    return top_k, selected_categories


def render_search_ui(index, metadata, top_k, selected_categories):
    """Render main search and results layout."""
    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("🔍 Search")
        query = st.text_input("Describe what you want:", "mirrorless camera with 4K and flip screen")

        if st.button("🔍 Search", use_container_width=True):
            st.session_state["history"].append(query)

            with st.spinner("✨ Searching for the best matches..."):
                top_results = search_products(query, index, metadata, top_k=top_k)

                # Apply category filter
                if selected_categories:
                    top_results = [r for r in top_results if r.get("category") in selected_categories]

                if not top_results:
                    st.warning("No results found.")
                    return None, query
                else:
                    st.success(f"Found {len(top_results)} similar products!")
                    return top_results, query

    return None, None

def render_results(top_results, query):
    """Display product results and ShopLyst's overall AI suggestion."""
    if not top_results:
        return

    st.header("🛍️ Results")

    # Product cards
    for item in top_results:
        price = item.get("price", "Not available")
        rating = item.get("rating", "Not available")

        st.markdown(f"""
        <div style='padding:15px; border-radius:10px; background:#f8f9fa; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.1);'>
            <h4 style='margin-bottom:4px;'>{item['title']}</h4>
            <p style='margin:0;'><b>Price:</b> {price}</p>
            <p style='margin:0;'><b>Rating:</b> {rating}</p>
            <a href='{item['url']}' target='_blank'>🔗 View Product</a>
        </div>
        """, unsafe_allow_html=True)

    # 🧠 Highlighted AI Suggestion Section
    st.markdown("### 💡 ShopLyst Smart Suggestion")

    with st.spinner("🧠 Analyzing results to find the best overall match..."):
        rag_response = generate_response_with_rag(query, top_results)

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #fffbe6, #fff8d6);
        border-left: 5px solid #FFD700;
        border-radius: 10px;
        padding: 18px;
        margin-top: 20px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        font-size: 1.05em;
        line-height: 1.6;
    '>
    <b>🧠 ShopLyst Recommends:</b><br>
    {rag_response}
    </div>
    """, unsafe_allow_html=True)



# ──────────────────────────────────────────────
# 🚀 Main App Logic
# ──────────────────────────────────────────────
def main():
    load_api_key()
    index, metadata = load_data()
    top_k, selected_categories = render_sidebar(metadata, index)
    top_results, query = render_search_ui(index, metadata, top_k, selected_categories)
    if top_results:
        render_results(top_results, query)


if __name__ == "__main__":
    main()
