import re
import requests
from bs4 import BeautifulSoup


def find_urls_in_text(text: str):
    """Return all http/https URLs in a string."""
    return re.findall(r"https?://\S+", text or "")


def scrape_url(url: str, max_chars: int = 6000) -> str:
    """Fetch a URL and return concatenated <p> text."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[scraper] Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n".join(paragraphs)[:max_chars]
