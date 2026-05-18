"""Thin wrapper around src.scraping.scrape_snep().

Run with:
    python scripts/scrape_snep.py
"""

import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scraping import scrape_snep  # noqa: E402


if __name__ == "__main__":
    scrape_snep()
