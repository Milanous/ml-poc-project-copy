#!/usr/bin/env python3
"""
SNEP Certifications Scraper - Cumulative Approach
Strategy: Extract from last page only (all cumulative results)
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

BASE_URL = "https://snepmusique.com/les-certifications/page/{page}/"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "snep_certifications.csv"

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
        
        # Extract certification level - contains icon class too
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
        print(f"Error parsing certification: {e}")
        return None

def scrape_page(page_num):
    """Scrape a single page"""
    url = BASE_URL.format(page=page_num)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all certification divs
        cert_divs = soup.find_all('div', class_='certification')
        print(f"Page {page_num}: Found {len(cert_divs)} cumulative items")
        
        data = []
        for div in cert_divs:
            cert_data = extract_certification_data(div)
            if cert_data and cert_data['Interprete']:  # Only add if we have an artist
                data.append(cert_data)
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page_num}: {e}")
        return []

def main():
    """Main execution - scrape last page which contains all cumulative data"""
    print("SNEP Certifications Scraper - Cumulative Strategy")
    print("=" * 60)
    print("Scraping page 378 (contains all cumulative certifications)...")
    
    start_time = time.time()
    
    # Get all data from last page (cumulative)
    all_data = scrape_page(378)
    
    print(f"\nTotal certifications found: {len(all_data)}")
    
    # Remove duplicates by creating a composite key
    seen = set()
    unique_data = []
    
    for row in all_data:
        key = (row['Interprete'], row['Titre'], row['Categorie'], row['Certification'])
        if key not in seen:
            seen.add(key)
            unique_data.append(row)
    
    print(f"After deduplication: {len(unique_data)} unique rows")
    
    # Create DataFrame and export
    df = pd.DataFrame(unique_data)
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV
    df.to_csv(OUTPUT_FILE, sep=';', index=False, encoding='utf-8-sig')
    
    elapsed = time.time() - start_time
    print(f"\n✓ Exported to {OUTPUT_FILE}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Time: {elapsed:.1f}s")
    
    # Certification distribution
    print(f"\nCertification distribution:")
    print(df['Certification'].value_counts())

if __name__ == '__main__':
    main()
