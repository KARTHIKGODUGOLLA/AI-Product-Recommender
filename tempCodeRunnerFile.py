import os
import json
from serpapi import GoogleSearch

SERPAPI_KEY = "9ff6cf3d4e10e8998d98d10d87a1af0d127806a32ea38130928aa699c1319bed"


def get_amazon_products(search_query, max_results=5):
    params = {
        "engine": "amazon",
        "amazon_domain": "amazon.com",
        "type": "search",
        "search_term": search_query,
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    products = results.get("organic_results", [])[:max_results]

    cleaned_products = []

    for product in products:
        cleaned = {
            "title": product.get("title"),
            "price": product.get("price"),
            "rating": product.get("rating"),
            "review_count": product.get("reviews"),
            "url": product.get("link"),
            "thumbnail": product.get("thumbnail")
        }
        cleaned_products.append(cleaned)

    return cleaned_products


if __name__ == "__main__":
    query = input("Enter your product search query: ")
    products = get_amazon_products(query, max_results=5)
    if not products:
        print("No results found. Try rephrasing the query with simpler keywords.")

    print(json.dumps(products, indent=2))
