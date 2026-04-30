#!/usr/bin/env python3
"""
SNEP Certifications Scraper - All Pages with Deduplication
Scrapes all 378 pages and deduplicates globally by artist+title+category+certification
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://snepmusique.com/les-certifications/page/{page}/"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "snep_certifications.csv"
TOTAL_PAGES = 378
MAX_WORKERS = 6
TIMEOUT = 25

def extract_certification_data(cert_div):
    """Extract all fields from a single certification div"""
    try:
        # Extract artist
        artist_elem = cert_div.find('div', class_='artiste')
        artist = artist_elem.get_text(strip=True) if artist_elem else ""
        
        # Extract title
        title_elem = cert_div.find('div', class_='titre')
        title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Extract publisher
        pub_elem = cert_div.find('div', class_='editeur')
        publisher = pub_elem.get_text(strip=True) if pub_elem else ""
        
        # Extract category
        cat_elem = cert_div.find('div', class_='categorie')
        category = cat_elem.get_text(strip=True) if cat_elem else ""
        
        # Extract certification level
        certif_elem = cert_div.find('div', class_='certif')
        certification = certif_elem.get_text(strip=True) if certif_elem else ""
        
        # Extract dates from block_dates
        dates_block = cert_div.find('div', class_='block_dates')
        release_date = ""
        cert_date = ""
        duration = ""
        
        if dates_block:
            date_divs = dates_block.find_all('div', class_='date')
            for date_div in date_divs:
                span = date_div.find('span')
                if span:
                    label = span.get_text(strip=True)
                    # Get the text after the span
                    value = date_div.get_text(strip=True).replace(label, '').strip()
                    
                    if 'sortie' in label:
                        release_date = value
                    elif 'constat' in label:
                        cert_date = value
                    elif 'Durée' in label or 'obtention' in label:
                        duration = value
        
        return {
            'Interprete': artist,
            'Titre': title,
            'Editeur / Distributeur': publisher,
            'Categorie': category,
            'Certification': certification,
            'Date de sortie': release_date,
            'Date de constat': cert_date,
            'Duree obtention': duration
        }
    except Exception as e:
        return None

def scrape_page(page_num):
    """Scrape a single page and return extracted data"""
    url = BASE_URL.format(page=page_num)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all certification divs
            cert_divs = soup.find_all('div', class_='certification')
            
            data = []
            for div in cert_divs:
                cert_data = extract_certification_data(div)
                if cert_data and cert_data['Interprete']:
                    data.append(cert_data)
            
            return page_num, len(cert_divs), data
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                continue
            print(f"  ⚠ Page {page_num}: Failed after retries ({e})")
            return page_num, 0, []
    
    return page_num, 0, []

def main():
    """Main execution - scrape all pages with parallel downloading"""
    print("SNEP Certifications Scraper - All Pages")
    print("=" * 60)
    
    start_time = time.time()
    all_data = []
    
    print(f"Downloading {TOTAL_PAGES} pages with {MAX_WORKERS} workers...")
    print()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_page, page): page for page in range(1, TOTAL_PAGES + 1)}
        completed = 0
        
        for future in as_completed(futures):
            page_num, item_count, data = future.result()
            all_data.extend(data)
            completed += 1
            
            if completed % 50 == 0 or completed == TOTAL_PAGES:
                elapsed = time.time() - start_time
                print(f"  ... {completed}/{TOTAL_PAGES} pages downloaded ({elapsed:.0f}s elapsed)")
    
    print(f"\nTotal items across all pages: {len(all_data)}")
    
    # Global deduplication by composite key
    seen = set()
    unique_data = []
    duplicates = 0
    
    for row in all_data:
        key = (row['Interprete'], row['Titre'], row['Categorie'], row['Certification'])
        if key not in seen:
            seen.add(key)
            unique_data.append(row)
        else:
            duplicates += 1
    
    print(f"After deduplication: {len(unique_data)} unique rows ({duplicates} duplicates removed)")
    
    # Create DataFrame and export
    df = pd.DataFrame(unique_data)
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV
    df.to_csv(OUTPUT_FILE, sep=';', index=False, encoding='utf-8-sig')
    
    elapsed = time.time() - start_time
    print(f"\n✓ Exported to {OUTPUT_FILE}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Time: {elapsed:.0f}s")
    
    # Certification distribution
    print(f"\nCertification distribution:")
    cert_counts = df['Certification'].value_counts()
    for cert, count in cert_counts.items():
        print(f"  {cert}: {count}")
    
    print(f"\nCategory distribution:")
    cat_counts = df['Categorie'].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}")

if __name__ == '__main__':
    main()
