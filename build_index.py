"""
build_index.py — One-time script to build the FAISS vector index from catalog.json.
Run this after scraper.py has generated catalog.json.

Usage:
    python build_index.py
"""

import json
import sys
import os

CATALOG_PATH = "catalog.json"


def main():
    if not os.path.exists(CATALOG_PATH):
        print(f"ERROR: {CATALOG_PATH} not found.")
        print("Run `python scraper.py` first to generate the catalog.")
        sys.exit(1)

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not catalog:
        print("ERROR: catalog.json is empty. Re-run scraper.py.")
        sys.exit(1)

    print(f"Loaded {len(catalog)} items from {CATALOG_PATH}")

    # Import here so module-level singletons are fresh
    from retriever import build_and_save_index

    build_and_save_index(catalog)
    print("\n✓ Index build complete. You can now start the API server.")


if __name__ == "__main__":
    main()
