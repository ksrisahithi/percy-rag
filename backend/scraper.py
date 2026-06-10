"""
Fetches character and lore pages from the Riordan Wiki via the MediaWiki API.

Why action=parse instead of prop=extracts:
  - riordan.fandom.com blocks all direct HTML requests at the Cloudflare/IP level
    (HTTP 403, CF-RAY header on every response, no selectors matched).
  - The MediaWiki API endpoint (api.php) returns HTTP 200 and bypasses this block.
  - However, Fandom wikis do NOT have the TextExtracts extension, so
    prop=extracts returns "Unrecognized value for parameter prop: extracts".
  - action=parse&prop=text returns rendered HTML inside JSON via the same
    unblocked api.php path; BeautifulSoup can then extract clean text from it.

Debug mode: run with --debug to print the raw API JSON for the first page.
"""

import os
import sys
import time
import requests
from bs4 import BeautifulSoup

API_URL = "https://riordan.fandom.com/api.php"
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

SEED_TITLES = [
    "Percy_Jackson",
    "Annabeth_Chase",
    "Grover_Underwood",
    "Nico_di_Angelo",
    "Thalia_Grace",
    "Luke_Castellan",
    "Clarisse_La_Rue",
    "Tyson",
    "Camp_Half-Blood",
    "Olympus",
    "Poseidon",
    "Athena",
    "Zeus",
    "Hades",
    "Kronos",
    "The_Lightning_Thief",
    "The_Sea_of_Monsters",
    "The_Titan's_Curse",
    "The_Battle_of_the_Labyrinth",
    "The_Last_Olympian",
]

HEADERS = {
    "User-Agent": "percy-rag/1.0 (educational RAG project; python-requests)",
    "Accept": "application/json",
}


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup

    for tag in content(["script", "style", "table", "sup"]):
        tag.decompose()
    for tag in content.find_all("span", class_="reference"):
        tag.decompose()

    lines = []
    for tag in content.find_all(["p", "h2", "h3"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            lines.append(text)

    return "\n\n".join(lines)


def fetch_page_text(session: requests.Session, title: str, debug: bool = False) -> str | None:
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",           # rendered HTML returned inside JSON — bypasses Cloudflare
        "disablelimitreport": "1",
        "format": "json",
        "formatversion": "2",
    }

    try:
        response = session.get(API_URL, params=params, timeout=15)
        print(f"  [http {response.status_code}] {response.url[:100]}")
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [error] {title}: {e}")
        return None

    data = response.json()

    if debug:
        import json
        # Print the structure without the full HTML blob
        preview = {k: v for k, v in data.get("parse", {}).items() if k != "text"}
        print(f"\n  --- API response structure (text blob omitted) ---")
        print(json.dumps({"parse": preview}, indent=2))
        if "text" in data.get("parse", {}):
            html_preview = data["parse"]["text"][:500]
            print(f"\n  --- Rendered HTML (first 500 chars) ---\n{html_preview}\n  ---\n")

    if "error" in data:
        code = data["error"].get("code", "")
        info = data["error"].get("info", "")
        print(f"  [api error] {code}: {info}")
        return None

    html = data.get("parse", {}).get("text", "")
    if not html:
        print(f"  [warn] Empty HTML in parse response for: {title}")
        return None

    return extract_text_from_html(html) or None


def slugify(title: str) -> str:
    return title.replace(" ", "_").replace("'", "")


def run(titles: list[str] = SEED_TITLES, delay: float = 1.0, debug: bool = False):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Saving raw text to: {os.path.abspath(RAW_DATA_DIR)}\n")

    session = requests.Session()
    session.headers.update(HEADERS)
    first = True

    for title in titles:
        name = slugify(title)
        out_path = os.path.join(RAW_DATA_DIR, f"{name}.txt")

        if os.path.exists(out_path):
            print(f"  [skip] {name}.txt already exists")
            continue

        print(f"  [fetch] {title}")
        text = fetch_page_text(session, title, debug=(debug and first))
        first = False

        if text:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  [saved] {name}.txt ({len(text):,} chars)")
        else:
            print(f"  [empty] {name}.txt — nothing extracted")

        time.sleep(delay)

    print("\nScraping complete.")


if __name__ == "__main__":
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        print("[debug mode] Raw API response will be printed for the first page.\n")
    run(debug=debug_mode)
