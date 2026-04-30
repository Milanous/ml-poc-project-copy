"""
Scrape les certifications SNEP : https://snepmusique.com/les-certifications/
Stratégie finale :
- Pages 1-189 : prendre 30 derniers items chaque page (nouvelles certifications ajoutées)
- Pages 190-378 : prendre TOUS les items (fenêtre de temps différente, pas de doublons attendus avec 1-189)
- Déduplication globale: clé = (Interprete, Titre, Categorie, Certification)
- Export CSV avec BOM UTF-8
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
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


def _parse_html(raw, page):
    """Extract certifications from HTML page."""
    pattern = re.compile(r'<div class="certification"')
    positions = [m.start() for m in pattern.finditer(raw)]
    if not positions:
        return []
    
    # Pages 1-189: take only last 30 (new items added)
    # Pages 190-378: take ALL (different time window, no overlap with 1-189)
    if page <= PIVOT_PAGE:
        selected = positions[-30:]
    else:
        selected = positions
    
    # Extract chunks
    chunks = []
    for i, pos in enumerate(selected):
        end = selected[i + 1] if i + 1 < len(selected) else len(raw)
        chunks.append(raw[pos:end])
    
    # Parse
    mini_html = "<html><body>" + "".join(chunks) + "</body></html>"
    soup = BeautifulSoup(mini_html, "html.parser")
    
    records = []
    for div in soup.find_all("div", class_="certification"):
        try:
            def get_text(cls, _div=div):
                el = _div.find("div", class_=cls)
                return el.get_text(strip=True) if el else ""
            
            certif_div = div.find("div", class_="certif")
            date_sortie = date_constat = duree = ""
            
            for date_div in div.find_all("div", class_="date"):
                label_el = date_div.find("span")
                label = label_el.get_text(strip=True) if label_el else ""
                value = date_div.get_text(strip=True).replace(label, "").strip()
                if "sortie" in label.lower():
                    date_sortie = value
                elif "constat" in label.lower():
                    date_constat = value
                elif "dur" in label.lower():
                    duree = value
            
            records.append({
                "Interprete": get_text("artiste"),
                "Titre": get_text("titre"),
                "Editeur / Distributeur": get_text("editeur"),
                "Categorie": get_text("categorie"),
                "Certification": certif_div.get_text(strip=True) if certif_div else "",
                "Date de sortie": date_sortie,
                "Date de constat": date_constat,
                "Duree obtention": duree,
            })
        except Exception:
            pass
    
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
