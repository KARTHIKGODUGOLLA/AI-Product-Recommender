import os
import requests
from bs4 import BeautifulSoup
import tldextract
import json
from tqdm import tqdm
import re
from readability import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

from product_scrapper import search_google_products  # Google scraping logic
from prompt_feeder import get_prompts_by_category, get_timestamp  # Prompt and timestamp utilities

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def try_fetch(url, retries=3):
    for i in range(retries):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = res.apparent_encoding
            return res
        except Exception as e:
            print(f"Retry {i+1} for {url} failed: {e}")
            sleep(2)
    return None

def extract_product_info(url):
    try:
        res = try_fetch(url)
        if not res:
            return None

        doc = Document(res.text)
        clean_html = doc.summary()
        soup_clean = BeautifulSoup(clean_html, "lxml")
        visible_text = soup_clean.get_text(separator=" ", strip=True)
        soup = BeautifulSoup(res.text, "lxml")

        title = soup.title.string.strip() if soup.title else ""
        if title.lower().startswith("sorry!") or "something went wrong" in title.lower():
            print(f"⚠️ Skipping broken or error page: {url}")
            return None

        meta_desc = soup.find("meta", attrs={"name": "description"})
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        description = (meta_desc.get("content") if meta_desc else "") or (og_desc.get("content") if og_desc else "")

        source = tldextract.extract(url).registered_domain

        price_matches = re.findall(r"\$\d{1,5}(?:\.\d{2})?", soup.text)
        price_text = price_matches[0] if price_matches else ""

        match = re.search(r"(\d\.\d)\s*out of\s*5", soup.text, re.IGNORECASE)
        if match:
            rating_text = match.group(1)
        else:
            star_match = re.search(r"(★{1,5})", soup.text)
            rating_text = str(len(star_match.group(1))) if star_match else ""

        return {
            "title": title,
            "summary": description,
            "full_text": visible_text[:5000],
            "url": url,
            "source": source,
            "price": price_text,
            "rating": rating_text
        }

    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return None

def run_scraper_for_prompt(prompt, category):
    print(f"\n🔍 Searching for: {prompt}")
    links = search_google_products(prompt)
    final_data = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_product_info, item["link"]): item["link"] for item in links}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scraping: {prompt}"):
            data = future.result()
            if data:
                final_data.append(data)

    if final_data:
        timestamp = get_timestamp()
        out_dir = os.path.join("scraped_results", category)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{prompt.replace(' ', '_')}_{timestamp}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for p in final_data:
                f.write(json.dumps(p) + "\n")
        print(f"✅ Saved {len(final_data)} items to {output_path}")
    else:
        print("⚠️ No valid results found for:", prompt)

if __name__ == "__main__":
    prompt_map = get_prompts_by_category()
    for category, prompts in prompt_map.items():
        for prompt in prompts:
            run_scraper_for_prompt(prompt, category)
