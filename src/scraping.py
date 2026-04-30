"""
src/scraping.py
---------------
Fonctions de scraping des certifications SNEP depuis snepmusique.com/les-certifications/

Stratégie :
- 378 pages paginées sur le site SNEP
- Pages 1–189 : pagination cumulative (nouvelles certifs ajoutées en haut) → prendre les 30 derniers éléments
- Pages 190–378 : fenêtre temporelle fixe → prendre TOUS les éléments
- Déduplication par clé (Interprete, Titre, Categorie, Certification)
- Scraping parallèle avec ThreadPoolExecutor
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Constantes ────────────────────────────────────────────────────────────────

BASE_URL = "https://snepmusique.com/les-certifications/page/{page}/"
TOTAL_PAGES = 378
PIVOT_PAGE = 189       # Seuil entre les deux fenêtres cumulatives
TIMEOUT = 25           # Timeout HTTP en secondes
MAX_WORKERS = 6        # Threads parallèles
MAX_RETRIES = 3        # Tentatives par page en cas d'erreur réseau

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip",
}

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "snep_certifications.csv"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_html_page(raw_html: str, page: int, pivot_page: int = PIVOT_PAGE) -> list[dict]:
    """
    Extrait les certifications depuis le HTML brut d'une page SNEP.

    Paramètres
    ----------
    raw_html : str
        Contenu HTML brut de la page.
    page : int
        Numéro de la page (1-indexed), utilisé pour appliquer la stratégie
        de sélection (pages ≤ pivot_page → 30 derniers ; pages > pivot_page → tous).
    pivot_page : int
        Numéro de page séparant les deux fenêtres cumulatives.

    Retourne
    --------
    list[dict]
        Liste de dictionnaires, chacun représentant une certification.
    """
    pattern = re.compile(r'<div class="certification"')
    positions = [m.start() for m in pattern.finditer(raw_html)]
    if not positions:
        return []

    selected = positions[-30:] if page <= pivot_page else positions

    chunks = []
    for i, pos in enumerate(selected):
        end = selected[i + 1] if i + 1 < len(selected) else len(raw_html)
        chunks.append(raw_html[pos:end])

    mini_html = "<html><body>" + "".join(chunks) + "</body></html>"
    soup = BeautifulSoup(mini_html, "html.parser")

    records = []
    for div in soup.find_all("div", class_="certification"):
        record = _parse_certification_div(div)
        if record:
            records.append(record)

    return records


def _parse_certification_div(div) -> dict | None:
    """
    Parse un élément <div class="certification"> BeautifulSoup en dictionnaire.

    Retourne None en cas d'échec.
    """
    try:
        def get_text(cls):
            el = div.find("div", class_=cls)
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
    except Exception:
        return None


# ── Récupération HTTP ─────────────────────────────────────────────────────────

def fetch_page(page: int, session: requests.Session | None = None,
               headers: dict | None = None) -> tuple[int, list[dict]]:
    """
    Télécharge et parse une page de certifications SNEP.

    Paramètres
    ----------
    page : int
        Numéro de page à récupérer.
    session : requests.Session, optionnel
        Session HTTP réutilisable (recommandé pour les appels en série).
    headers : dict, optionnel
        En-têtes HTTP. Utilise DEFAULT_HEADERS si non fourni.

    Retourne
    --------
    tuple[int, list[dict]]
        (numéro_de_page, liste_de_certifications)
    """
    _headers = headers or DEFAULT_HEADERS
    _session = session or requests.Session()
    url = BASE_URL.format(page=page)

    for attempt in range(MAX_RETRIES):
        try:
            response = _session.get(url, headers=_headers, timeout=TIMEOUT)
            response.raise_for_status()
            return page, parse_html_page(response.text, page)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  [!] Erreur page {page} après {MAX_RETRIES} tentatives : {exc}")
                return page, []
            time.sleep(3)

    return page, []


# ── Déduplication ─────────────────────────────────────────────────────────────

def make_dedup_key(record: dict) -> tuple:
    """Retourne la clé de déduplication d'un enregistrement."""
    return (
        record.get("Interprete", ""),
        record.get("Titre", ""),
        record.get("Categorie", ""),
        record.get("Certification", ""),
    )


def deduplicate(records: list[dict]) -> list[dict]:
    """Supprime les doublons d'une liste d'enregistrements en conservant le premier."""
    seen = set()
    unique = []
    for rec in records:
        key = make_dedup_key(rec)
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


# ── Scraping complet ──────────────────────────────────────────────────────────

def scrape_snep(
    total_pages: int = TOTAL_PAGES,
    max_workers: int = MAX_WORKERS,
    output_file: Path | str | None = DEFAULT_OUTPUT,
    headers: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scrape l'intégralité des certifications SNEP sur toutes les pages.

    Paramètres
    ----------
    total_pages : int
        Nombre total de pages à scraper (défaut : 378).
    max_workers : int
        Nombre de threads parallèles (défaut : 6).
    output_file : Path | str | None
        Chemin de sortie CSV. Si None, aucun fichier n'est écrit.
    headers : dict | None
        En-têtes HTTP personnalisés.
    verbose : bool
        Affiche la progression si True.

    Retourne
    --------
    pd.DataFrame
        DataFrame avec toutes les certifications dédupliquées.
    """
    all_results: dict[int, list[dict]] = {}
    t0 = time.time()

    if verbose:
        print(f"==> Scraping de {total_pages} pages ({max_workers} workers parallèles)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_page, p, None, headers): p
            for p in range(1, total_pages + 1)
        }
        done = 0
        for future in as_completed(futures):
            page, records = future.result()
            all_results[page] = records
            done += 1
            if verbose and (done % 50 == 0 or done == total_pages):
                print(f"  ... {done}/{total_pages} pages ({time.time() - t0:.0f}s)")

    if verbose:
        print("==> Déduplication et assemblage...")

    flat = [rec for p in range(1, total_pages + 1) for rec in all_results.get(p, [])]
    unique_records = deduplicate(flat)

    df = pd.DataFrame(unique_records)

    if output_file is not None:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep=";", encoding="utf-8-sig", index=False)
        if verbose:
            print(f"\nFichier exporté : {out}")

    if verbose:
        elapsed = time.time() - t0
        print(f"  Lignes   : {len(df)}")
        print(f"  Colonnes : {list(df.columns)}")
        print(f"  Durée    : {elapsed:.0f}s")

    return df


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_snep()
