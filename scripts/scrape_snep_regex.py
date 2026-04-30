"""
Scrape SNEP certifications - Optimisé avec regex pur pour extraction rapide
Stratégie : regex pure pour extraire champs clés par certification div
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://snepmusique.com/les-certifications/page/{page}/"
TOTAL_PAGES = 378
TIMEOUT = 25
MAX_WORKERS = 6
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "snep_certifications.csv"
PIVOT_PAGE = 189

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip",
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def extract_text(html, cls):
    """Extract text from <div class="cls">...</div> using regex."""
    pattern = rf'<div class="{cls}"[^>]*>(.*?)</div>'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        # Clean HTML tags
        text = match.group(1)
        text = re.sub(r'<[^>]+>', '', text).strip()
        return text
    return ""


def extract_dates(html):
    """Extract all dates from <div class="date"> blocks."""
    dates = {}
    pattern = r'<div class="date"[^>]*>(.*?)</div>'
    for match in re.finditer(pattern, html, re.DOTALL):
        block = match.group(1)
        # Extract label from <span>...</span>
        label_match = re.search(r'<span[^>]*>(.*?)</span>', block)
        label = label_match.group(1).strip() if label_match else ""
        # Extract value (everything after span)
        value = re.sub(r'<span[^>]*>.*?</span>', '', block).strip()
        value = re.sub(r'<[^>]+>', '', value).strip()
        
        if "sortie" in label.lower():
            dates["Date de sortie"] = value
        elif "constat" in label.lower():
            dates["Date de constat"] = value
        elif "dur" in label.lower():
            dates["Duree obtention"] = value
    
    return dates


def parse_certification_div(div_html):
    """Parse a single <div class="certification">...</div> block."""
    try:
        record = {
            "Interprete": extract_text(div_html, "artiste"),
            "Titre": extract_text(div_html, "titre"),
            "Editeur / Distributeur": extract_text(div_html, "editeur"),
            "Categorie": extract_text(div_html, "categorie"),
            "Certification": extract_text(div_html, "certif"),
            "Date de sortie": "",
            "Date de constat": "",
            "Duree obtention": "",
        }
        
        # Merge dates
        dates = extract_dates(div_html)
        record.update(dates)
        
        return record
    except Exception:
        return None


def _parse_html(raw, page):
    """Extract certifications from page HTML."""
    # Extract ALL <div class="certification">...</div> blocks
    pattern = r'<div class="certification"[^>]*>.*?(?=<div class="certification"|$)'
    divs = re.findall(pattern, raw, re.DOTALL)
    
    if not divs:
        return []
    
    # Select which divs to keep based on page range
    if page <= PIVOT_PAGE:
        # Pages 1-189: only last 30
        selected_divs = divs[-30:]
    else:
        # Pages 190-378: ALL (different time window)
        selected_divs = divs
    
    # Parse each div
    records = []
    for div_html in selected_divs:
        rec = parse_certification_div(div_html)
        if rec and rec["Interprete"]:  # Only if we got at least the artist name
            records.append(rec)
    
    return records


def fetch_and_parse(page):
    session = requests.Session()
    url = BASE_URL.format(page=page)
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return page, _parse_html(r.text, page)
        except requests.RequestException as e:
            if attempt == 2:
                print(f"  [!] Erreur page {page} : {e}")
                return page, []
            time.sleep(3)
    return page, []


def make_key(r):
    return (r["Interprete"], r["Titre"], r["Categorie"], r["Certification"])


def main():
    all_records = []
    seen_keys = set()
    results = {}

    print(f"==> Scraping de {TOTAL_PAGES} pages ({MAX_WORKERS} workers parallèles)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_and_parse, p): p for p in range(1, TOTAL_PAGES + 1)}
        done = 0
        for future in as_completed(futures):
            page, records = future.result()
            results[page] = records
            done += 1
            if done % 50 == 0 or done == TOTAL_PAGES:
                elapsed = time.time() - t0
                print(f"  ... {done}/{TOTAL_PAGES} pages téléchargées ({elapsed:.0f}s écoulées)")

    print("==> Déduplication et assemblage...")
    for page in range(1, TOTAL_PAGES + 1):
        for rec in results.get(page, []):
            key = make_key(rec)
            if key not in seen_keys:
                seen_keys.add(key)
                all_records.append(rec)

    if not all_records:
        print("[!] Aucune donnée récupérée.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_FILE, sep=";", encoding="utf-8-sig", index=False)

    elapsed = time.time() - t0
    print(f"\nFichier exporté : {OUTPUT_FILE}")
    print(f"  Lignes totales : {len(df)}")
    print(f"  Colonnes       : {list(df.columns)}")
    print(f"  Durée totale   : {elapsed:.0f}s")


if __name__ == "__main__":
    main()
