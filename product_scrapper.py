from serpapi import GoogleSearch
import json

def search_google_products(query, max_results=5):
    params = {
        "engine": "google",
        "q": query,
        "api_key": "ebbfebbe7150ef11199379894908e983c3160046e4177c30ee1b6fc8c871b61d",
        "num": max_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic_results = results.get("organic_results", [])
    cleaned = []

    for r in organic_results:
        cleaned.append({
            "title": r.get("title"),
            "snippet": r.get("snippet"),
            "link": r.get("link")
        })

    return cleaned

