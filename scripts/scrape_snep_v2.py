"""
Scrape les certifications SNEP : https://snepmusique.com/les-certifications/
Stratégie optimisée :
- Itère sur les 378 pages en parallèle (6 workers)
- Extrait TOUS les divs.certification par regex
- Parse chaque div individuellement avec BS4 (plus rapide que parsing tout d'un coup)
- Deux fenêtres cumulatives : pages 1-189 (récente, prend last 30), pages 190-378 (ancienne, prend ALL)
- Déduplication globale par clé (Interprete, Titre, Categorie, Certification)
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ────────────────────────────────────────────────────────────
BASE_URL = "https://snepmusique.com/les-certifications/page/{page}/"
TOTAL_PAGES = 378
TIMEOUT = 25
MAX_WORKERS = 6
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "snep_certifications.csv"
PIVOT_PAGE = 189  # Page de transition entre deux fenêtres cumulatives

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip",
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def _extract_divs(raw):
    """Extract ALL certification div blocks from raw HTML using regex."""
    pattern = re.compile(r'<div class="certification">.*?</div>\s*</div>', re.DOTALL)
    return pattern.findall(raw)


def _parse_div(div_html):
    """Parse a single certification div into a record."""
    soup = BeautifulSoup(div_html, "html.parser")
    
    def get_text(cls):
        el = soup.find("div", class_=cls)
        return el.get_text(strip=True) if el else ""
    
    certif_div = soup.find("div", class_="certif")
    date_sortie = date_constat = duree = ""
    
    for date_div in soup.find_all("div", class_="date"):
        label_el = date_div.find("span")
        label = label_el.get_text(strip=True) if label_el else ""
        value = date_div.get_text(strip=True).replace(label, "").strip()
        if "sortie" in label.lower():
            date_sortie = value
        elif "constat" in label.lower():
            date_constat = value
        elif "dur" in label.lower():
            duree = value
    
    return {
        "Interprete": get_text("artiste"),
        "Titre": get_text("titre"),
        "Editeur / Distributeur": get_text("editeur"),
        "Categorie": get_text("categorie"),
        "Certification": certif_div.get_text(strip=True) if certif_div else "",
        "Date de sortie": date_sortie,
        "Date de constat": date_constat,
        "Duree obtention": duree,
    }


def _parse_html(raw, page):
    """Extract records from a page, selecting appropriately based on page range."""
    divs = _extract_divs(raw)
    if not divs:
        return []
    
    # Select which divs to keep based on pagination structure
    if page <= PIVOT_PAGE:
        # Pages 1-189: cumulative increasing, new items at the END (last 30)
        selected_divs = divs[-30:]
    else:
        # Pages 190-378: cumulative decreasing, ALL items are from a different window
        selected_divs = divs
    
    records = []
    for div_html in selected_divs:
        try:
            rec = _parse_div(div_html)
            records.append(rec)
        except Exception:
            pass  # Skip malformed divs
    
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
                print(f"  [!] Erreur page {page} après 3 tentatives : {e}")
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
