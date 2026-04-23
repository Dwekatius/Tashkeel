"""Scrape fully-vocalized Arabic articles from easyarabicnews.com.

The site (WordPress) has 3 level-tagged categories:
  - مبتدئ  (beginner)
  - متوسط  (intermediate)
  - تقرير خاص (special report)

For each category we walk /page/1/, /page/2/, ... collecting article URLs
until we hit a page that has no new article links. For each article we
extract the main body text (heading + paragraphs) and emit it in the
same format as datasets/arabic_tashkeel/raw/long_passages_raw.txt:

    ## <article title>
    <paragraph 1>

    <paragraph 2>
    ...

So the existing augment_long_passages.py pipeline can consume the output
without changes.

Filters applied during extraction:
  - drop paragraphs with tashkeel-density < 0.50 (site has some unvocalized
    boilerplate like video captions / related-article links)
  - drop paragraphs shorter than 4 Arabic words
  - deduplicate across articles by dedup_key (tashkeel-stripped, spaces removed)

Per-category output files plus a combined file land in:
    datasets/arabic_tashkeel/raw/easyarabicnews_<slug>.txt
    datasets/arabic_tashkeel/raw/easyarabicnews_all.txt

Usage:
    python scripts/data_prep/scrape_easyarabicnews.py
    python scripts/data_prep/scrape_easyarabicnews.py --max-pages 3  (quick test)
    python scripts/data_prep/scrape_easyarabicnews.py --category intermediate
"""

import argparse
import os
import re
import sys
import time
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing deps. Install with:")
    print("  python -m pip install requests beautifulsoup4 lxml")
    sys.exit(1)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "datasets", "arabic_tashkeel", "raw")

BASE = "https://easyarabicnews.com"
CATEGORIES = {
    "beginner":        "/category/%d9%85%d8%a8%d8%aa%d8%af%d8%a6/",
    "intermediate":    "/category/%d9%85%d8%aa%d9%88%d8%b3%d8%b7/",
    "special_report":  "/category/%d8%aa%d9%82%d8%b1%d9%8a%d8%b1-%d8%ae%d8%a7%d8%b5/",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQ_DELAY = 0.8         # seconds between requests — be polite
PAGE_TIMEOUT = 25

TASHKEEL_RE = re.compile(r"[ً-ْٰ]")
ARABIC_LETTER_RE = re.compile(r"[ء-ي]")


def tashkeel_density(s: str) -> float:
    letters = len(ARABIC_LETTER_RE.findall(s))
    if not letters:
        return 0.0
    return len(TASHKEEL_RE.findall(s)) / letters


def arabic_word_count(s: str) -> int:
    return len(re.sub(TASHKEEL_RE, "", s).split())


def dedup_key(s: str) -> str:
    return TASHKEEL_RE.sub("", s).replace(" ", "").replace("\n", "").strip()


def get(url: str, session: requests.Session) -> BeautifulSoup | None:
    try:
        r = session.get(url, timeout=PAGE_TIMEOUT)
    except requests.RequestException as e:
        print(f"  ! request failed: {e}")
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        print(f"  ! status {r.status_code} on {url}")
        return None
    # site serves UTF-8 but some servers mislabel; force it
    r.encoding = "utf-8"
    return BeautifulSoup(r.text, "lxml")


def find_article_links(soup: BeautifulSoup) -> list[str]:
    """Return unique article URLs on a category listing page.
    WordPress category pages put article links inside <h2>/<h3> headings or
    <a class="... more ..."> read-more links. We take the union of
    <article> blocks' first <a href>.
    """
    links = []
    seen = set()
    # WordPress puts each post in <article> or in an <h2 class="entry-title"><a>
    for a in soup.select("article a[href], h2.entry-title a[href], h3.entry-title a[href]"):
        href = a.get("href", "")
        if not href or "/category/" in href or href.endswith("/feed/"):
            continue
        # keep only same-site absolute URLs
        if not href.startswith(BASE):
            continue
        # drop anchors and query strings
        href = href.split("#", 1)[0].split("?", 1)[0]
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
    return links


def extract_article(soup: BeautifulSoup) -> tuple[str, list[str]] | None:
    """Return (title, paragraphs) or None if body isn't found/vocalized."""
    # Title: <h1 class="entry-title"> or <h1>
    title_el = (
        soup.select_one("h1.entry-title")
        or soup.select_one("article h1")
        or soup.select_one("h1")
    )
    title = title_el.get_text(strip=True) if title_el else ""

    # Body: try common WordPress containers in order of specificity
    body = (
        soup.select_one("div.entry-content")
        or soup.select_one("article .entry-content")
        or soup.select_one("article")
        or soup.select_one("main")
    )
    if body is None:
        return None

    # Remove obvious non-article blocks
    for tag in body.select(
        "script, style, iframe, noscript, .sharedaddy, .jp-relatedposts, "
        ".related, nav, footer, header, .wp-block-embed, .wp-block-buttons, "
        "form, .comments-area"
    ):
        tag.decompose()

    paragraphs = []
    for p in body.find_all(["p", "h2", "h3", "li"]):
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        # Only keep if it's mostly Arabic AND has real tashkeel
        if not ARABIC_LETTER_RE.search(txt):
            continue
        if tashkeel_density(txt) < 0.50:
            continue
        if arabic_word_count(txt) < 4:
            continue
        paragraphs.append(txt)

    if not paragraphs:
        return None
    return title, paragraphs


def scrape_category(name: str, path: str, session: requests.Session, max_pages: int, max_articles: int) -> list[tuple[str, list[str]]]:
    print(f"\n=== Category: {name} ===")
    articles = []
    seen_article_urls = set()
    page = 1
    while page <= max_pages:
        if page == 1:
            url = urljoin(BASE, path)
        else:
            url = urljoin(BASE, path.rstrip("/") + f"/page/{page}/")
        print(f"  page {page}: {url}")
        soup = get(url, session)
        time.sleep(REQ_DELAY)
        if soup is None:
            print(f"  -> no page {page}, stopping.")
            break
        links = find_article_links(soup)
        new_links = [l for l in links if l not in seen_article_urls]
        if not new_links:
            print(f"  -> no new articles on page {page}, stopping.")
            break
        print(f"  -> found {len(new_links)} new articles")
        for url in new_links:
            if len(articles) >= max_articles:
                break
            seen_article_urls.add(url)
            art_soup = get(url, session)
            time.sleep(REQ_DELAY)
            if art_soup is None:
                continue
            res = extract_article(art_soup)
            if res is None:
                print(f"    skip (no vocalized body): {url[-60:]}")
                continue
            title, paras = res
            total_words = sum(arabic_word_count(p) for p in paras)
            print(f"    kept: {title[:50]}  ({len(paras)} paras, {total_words} words)")
            articles.append((title, paras))
        if len(articles) >= max_articles:
            print(f"  -> hit max-articles cap ({max_articles}), stopping.")
            break
        page += 1
    return articles


def write_articles(path: str, articles: list[tuple[str, list[str]]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen_keys = set()
    kept = 0
    dropped_dupes = 0
    with open(path, "w", encoding="utf-8") as f:
        for title, paras in articles:
            unique_paras = []
            for p in paras:
                k = dedup_key(p)
                if k in seen_keys:
                    dropped_dupes += 1
                    continue
                seen_keys.add(k)
                unique_paras.append(p)
            if not unique_paras:
                continue
            f.write(f"## {title}\n\n")
            f.write("\n\n".join(unique_paras))
            f.write("\n\n")
            kept += 1
    return kept, dropped_dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=list(CATEGORIES.keys()) + ["all"], default="all")
    ap.add_argument("--max-pages", type=int, default=20, help="max listing pages per category")
    ap.add_argument("--max-articles", type=int, default=500, help="per-category cap")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ar,en;q=0.8"})

    cats = CATEGORIES.items() if args.category == "all" else [(args.category, CATEGORIES[args.category])]

    combined: list[tuple[str, list[str]]] = []
    for name, path in cats:
        arts = scrape_category(name, path, session, args.max_pages, args.max_articles)
        out_path = os.path.join(OUT_DIR, f"easyarabicnews_{name}.txt")
        kept, dup = write_articles(out_path, arts)
        print(f"  wrote {kept} articles to {out_path}  (deduped {dup} paragraphs)")
        combined.extend(arts)

    if args.category == "all":
        combined_path = os.path.join(OUT_DIR, "easyarabicnews_all.txt")
        kept, dup = write_articles(combined_path, combined)
        print(f"\nCombined: {kept} articles -> {combined_path}  (deduped {dup} paragraphs)")


if __name__ == "__main__":
    main()
