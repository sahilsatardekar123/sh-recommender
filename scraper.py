"""
SHL Product Catalog Scraper
Scrapes Individual Test Solutions from https://www.shl.com/solutions/products/productcatalog/
"""

import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/productcatalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def get_page(url: str, params: dict = None) -> BeautifulSoup:
    """Fetch a page and return a BeautifulSoup object."""
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_test_type(row) -> str:
    """Extract test type letter codes from a catalog row."""
    type_codes = []
    # SHL uses small colored badges / spans for test type codes
    badges = row.select("span.product-catalogue__key, .catalogue-type, [class*='type']")
    for badge in badges:
        text = badge.get_text(strip=True)
        if text and len(text) <= 3 and text.isalpha():
            type_codes.append(text)

    # Fallback: look for single-letter cells or title attributes
    if not type_codes:
        cells = row.find_all("td")
        for cell in cells:
            txt = cell.get_text(strip=True)
            if txt in ("K", "P", "A", "B", "C", "S", "E", "M"):
                type_codes.append(txt)

    return ", ".join(type_codes) if type_codes else "N/A"


def scrape_assessment_detail(url: str) -> str:
    """Visit an individual assessment page and extract its description."""
    try:
        soup = get_page(url)
        # Try common description selectors on SHL product pages
        for selector in [
            ".product-detail__description",
            ".product-description",
            "section.overview p",
            "div.overview p",
            ".content-area p",
            "main p",
        ]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 40:
                    return text[:600]

        # Fallback: grab first substantial paragraph in main content
        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 80:
                return text[:600]

        return ""
    except Exception as e:
        print(f"  Warning: could not fetch detail for {url}: {e}")
        return ""


def parse_catalog_page(soup: BeautifulSoup) -> list[dict]:
    """Parse one page of the catalog and return list of assessment dicts."""
    assessments = []

    # SHL catalog renders as a table; each row is one product
    # Rows are under: div.product-catalogue or table with product rows
    rows = soup.select("tr.product-catalogue__row, tr[data-course-id], .js-filter-item")

    if not rows:
        # Try generic table rows inside the catalog container
        catalog_section = soup.find("div", class_=lambda c: c and "catalogue" in c.lower())
        if catalog_section:
            rows = catalog_section.find_all("tr")[1:]  # skip header
        else:
            rows = soup.select("table tr")[1:]  # skip header row

    for row in rows:
        try:
            # Skip pre-packaged job solutions — they appear in a separate section
            # marked with "Job Solution" or a different table heading
            row_text = row.get_text(" ", strip=True).lower()
            if "job solution" in row_text or "pre-packaged" in row_text:
                continue

            # Extract name and URL
            link = row.find("a", href=True)
            if not link:
                continue

            name = link.get_text(strip=True)
            if not name:
                continue

            href = link["href"]
            url = href if href.startswith("http") else urljoin(BASE_URL, href)

            # Must be an shl.com URL
            if "shl.com" not in url:
                continue

            test_type = extract_test_type(row)

            assessments.append(
                {
                    "name": name,
                    "url": url,
                    "test_type": test_type,
                    "description": "",  # filled in next pass
                }
            )
        except Exception as e:
            print(f"  Warning: error parsing row: {e}")
            continue

    return assessments


def find_next_page(soup: BeautifulSoup, current_url: str) -> str | None:
    """Return the URL of the next pagination page, or None if last page."""
    # SHL uses ?start=N or ?page=N pagination
    next_btn = soup.select_one("a[rel='next'], .pagination__next a, li.next a")
    if next_btn and next_btn.get("href"):
        href = next_btn["href"]
        return href if href.startswith("http") else urljoin(BASE_URL, href)

    # Check for numbered pagination links
    pager = soup.select(".pagination a, .pager a")
    for link in pager:
        if "next" in link.get_text(strip=True).lower():
            href = link["href"]
            return href if href.startswith("http") else urljoin(BASE_URL, href)

    return None


def scrape_catalog() -> list[dict]:
    """
    Main scrape function.
    Returns list of assessment dicts with name, url, test_type, description.
    """
    all_assessments = []
    visited_urls = set()

    url = CATALOG_URL
    page_num = 1

    print(f"Scraping SHL Individual Test Solutions catalog...")

    while url:
        if url in visited_urls:
            break
        visited_urls.add(url)

        print(f"  Page {page_num}: {url}")
        try:
            # SHL catalog may use query params for pagination: ?start=0, ?start=12, etc.
            soup = get_page(url)
        except Exception as e:
            print(f"  Error fetching page {page_num}: {e}")
            break

        page_assessments = parse_catalog_page(soup)
        print(f"    Found {len(page_assessments)} items on this page")

        # De-duplicate by URL
        for item in page_assessments:
            if item["url"] not in {a["url"] for a in all_assessments}:
                all_assessments.append(item)

        next_url = find_next_page(soup, url)
        if not next_url or next_url == url:
            # Try offset-based pagination manually (?start=N)
            # SHL uses increments of 12
            if "start=" not in url and page_num == 1:
                next_url = url + "?start=12" if "?" not in url else url + "&start=12"
                # Only follow if we got items; otherwise assume single page
                if len(page_assessments) == 0:
                    next_url = None
            else:
                next_url = None

        url = next_url
        page_num += 1

        if page_num > 20:  # Safety cap
            break

        time.sleep(1)  # Be polite

    if not all_assessments:
        print("\nWARNING: Primary scraper found 0 items. Trying alternate selector strategy...")
        all_assessments = fallback_scrape()

    # Enrich with descriptions from detail pages
    print(f"\nEnriching {len(all_assessments)} items with descriptions...")
    for i, item in enumerate(all_assessments):
        print(f"  [{i+1}/{len(all_assessments)}] {item['name']}")
        if not item["description"]:
            item["description"] = scrape_assessment_detail(item["url"])
        time.sleep(0.5)

    return all_assessments


def fallback_scrape() -> list[dict]:
    """
    Alternate scraping strategy using the catalog's API/filter endpoint
    that SHL may expose for their AJAX-loaded table.
    """
    assessments = []
    # SHL catalog sometimes loads via an API endpoint
    api_candidates = [
        "https://www.shl.com/solutions/products/productcatalog/?type=A",
        "https://www.shl.com/solutions/products/productcatalog/?type=individual",
    ]
    for api_url in api_candidates:
        try:
            soup = get_page(api_url)
            items = parse_catalog_page(soup)
            if items:
                assessments.extend(items)
                break
        except Exception:
            continue

    # If still nothing, use the known static structure
    if not assessments:
        try:
            soup = get_page(CATALOG_URL)
            # Try every anchor that points to /solutions/products/
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "/solutions/products/" in href and href != CATALOG_URL:
                    url = href if href.startswith("http") else urljoin(BASE_URL, href)
                    name = link.get_text(strip=True)
                    if name and len(name) > 3 and "shl.com" in url:
                        assessments.append(
                            {
                                "name": name,
                                "url": url,
                                "test_type": "N/A",
                                "description": "",
                            }
                        )
        except Exception as e:
            print(f"Fallback scrape failed: {e}")

    # Deduplicate
    seen = set()
    unique = []
    for a in assessments:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)
    return unique


def main():
    assessments = scrape_catalog()

    if not assessments:
        print("\nERROR: Could not scrape any assessments. Check network access and selectors.")
        return

    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Scraped {len(assessments)} Individual Test Solutions")
    print("✓ Saved to catalog.json")


if __name__ == "__main__":
    main()
