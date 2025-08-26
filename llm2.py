#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm2.py - Enhanced LLM decision engine with two-phase analysis:
1. Initial analysis: Find up to 10 related stocks with buy/sell confidence
2. Enrichment: For high-confidence stocks, fetch additional data and refine
"""

import argparse
import json
import csv
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import feedparser
import lmstudio as lms

# Directories
DATA_DIR = Path("./data")
NEWS_DIR = DATA_DIR / "news"
RESULT_DIR = DATA_DIR / "result"
LISTING_DIR = DATA_DIR / "us-stock-listing"

# User Agent for HTTP requests
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}
HTTP_TIMEOUT = (8, 15)

# Simple rate limiter for external APIs
class SimpleRateLimiter:
    """Simple rate limiter without backoff."""
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.last_call = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

# Global rate limiter for external APIs
api_rate_limiter = SimpleRateLimiter(min_interval=1.0)

# ----------------------
# Utilities
# ----------------------

def log(msg: str, verbose: bool = True):
    """Log message with timestamp."""
    if verbose:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}", file=sys.stderr)

def ensure_dirs():
    """Create necessary directories."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_server_host(host: str) -> str:
    """Extract host:port from various URL formats."""
    if not host:
        return "localhost:1234"
    # Remove protocol and path if present
    host = re.sub(r"^https?://", "", host)
    host = re.sub(r"/.*$", "", host)
    return host

# ----------------------
# Text cleaning
# ----------------------

INVISIBLE_CATEGORIES = {"Cf", "Cc", "Cs"}

def clean_text(s: Any) -> Any:
    """Clean text by normalizing unicode and removing invisible characters."""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in INVISIBLE_CATEGORIES)
    return s.strip()

def truncate_with_ellipsis(text: str, max_length: int = 240) -> str:
    """Truncate text to max length with ellipsis."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

# ----------------------
# Stock Listings
# ----------------------

@dataclass
class ListingItem:
    ticker: str
    name: Optional[str]
    exchange: Optional[str]
    type: Optional[str]
    is_etf: bool
    is_adr: bool
    cik: Optional[str]

def load_listings() -> Dict[str, ListingItem]:
    """Load US stock listings from JSON."""
    path = LISTING_DIR / "us-listings-latest.json"
    if not path.exists():
        # Try dated files
        dated = sorted(LISTING_DIR.glob("us-listings-*.json"))
        if not dated:
            raise FileNotFoundError(f"No listings found in {LISTING_DIR}")
        path = dated[-1]
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping = {}
    for rec in data:
        ticker = (rec.get("ticker") or "").upper()
        if not ticker:
            continue
        name = rec.get("name") or rec.get("company_name_sec") or rec.get("security_name")
        mapping[ticker] = ListingItem(
            ticker=ticker,
            name=name,
            exchange=rec.get("exchange"),
            type=rec.get("type") or rec.get("security_type"),
            is_etf=bool(rec.get("is_etf")),
            is_adr=bool(rec.get("is_adr")),
            cik=rec.get("cik"),
        )
    return mapping

# ----------------------
# News handling
# ----------------------

def find_latest_news_file() -> Optional[Path]:
    """Find the most recent news file (YYMMDDHHMM.json)."""
    files = []
    for p in NEWS_DIR.glob("*.json"):
        stem = p.stem
        if len(stem) == 10 and stem.isdigit():
            files.append(p)
    if not files:
        return None
    files.sort()
    return files[-1]

def load_news(path: Path) -> List[Dict[str, Any]]:
    """Load news items from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data

# ----------------------
# Google News RSS
# ----------------------

def fetch_google_news(company: str, ticker: str, days: int = 30, max_items: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent news from Google News RSS for a company/ticker."""
    q = f'"{ticker}" OR "{company}" when:{days}d'
    url = (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    )
    
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        fp = feedparser.parse(resp.content)
        
        items = []
        for entry in fp.entries[:max_items * 2]:  # Oversample for deduplication
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            published = entry.get("published", "") or entry.get("updated", "")
            source = ""
            if "source" in entry and isinstance(entry.source, dict):
                source = entry.source.get("title", "")
            
            if title and link:
                items.append({
                    "title": clean_text(title),
                    "link": clean_text(link),
                    "published": clean_text(published),
                    "source": clean_text(source),
                })
        
        # Deduplicate by link
        seen = set()
        deduped = []
        for item in items:
            if item["link"] not in seen:
                seen.add(item["link"])
                deduped.append(item)
        
        return deduped[:max_items]
    except Exception as e:
        log(f"Error fetching Google News for {ticker}: {e}")
        return []


# ----------------------
# News providers (non-Yahoo) with unified fallback
# ----------------------

def _dedup_by_link(items: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        lk = it.get("link", "")
        if lk and lk not in seen:
            seen.add(lk)
            out.append(it)
        if len(out) >= max_items:
            break
    return out


def fetch_finnhub_company_news(ticker: str, days: int = 30, max_items: int = 20) -> List[Dict[str, Any]]:
    """Finnhub company news (requires FINNHUB_API_KEY). Docs: https://finnhub.io/docs/api/company-news"""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []
    symbol = ticker.replace(".", "-")
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=days)
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": str(from_date), "to": str(to_date), "token": api_key}
    try:
        api_rate_limiter.wait_if_needed()
        resp = requests.get(url, headers=HTTP_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or []
        items: List[Dict[str, Any]] = []
        for e in data:
            title = (e.get("headline") or "").strip()
            link = (e.get("url") or "").strip()
            ts = e.get("datetime")  # epoch seconds
            published = ""
            if isinstance(ts, (int, float)):
                try:
                    published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                except Exception:
                    published = ""
            if title and link:
                items.append({
                    "title": clean_text(title),
                    "link": clean_text(link),
                    "published": clean_text(published),
                    "source": clean_text(e.get("source") or "Finnhub"),
                })
        return _dedup_by_link(items, max_items)
    except Exception as e:
        log(f"Error fetching Finnhub news for {ticker}: {e}")
        return []


def fetch_polygon_news(ticker: str, max_items: int = 20) -> List[Dict[str, Any]]:
    """Polygon.io ticker news (requires POLYGON_API_KEY). Docs: https://polygon.io/docs/rest/stocks/news"""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return []
    symbol = ticker.replace(".", "-")
    url = "https://api.polygon.io/v2/reference/news"
    params = {"ticker": symbol, "limit": str(max_items), "apiKey": api_key}
    try:
        api_rate_limiter.wait_if_needed()
        resp = requests.get(url, headers=HTTP_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results") or []
        items: List[Dict[str, Any]] = []
        for e in results:
            title = (e.get("title") or "").strip()
            link = (e.get("article_url") or e.get("url") or "").strip()
            published = (e.get("published_utc") or "").strip()
            publisher = e.get("publisher") or {}
            source = publisher.get("name") if isinstance(publisher, dict) else publisher or "Polygon News"
            if title and link:
                items.append({
                    "title": clean_text(title),
                    "link": clean_text(link),
                    "published": clean_text(published),
                    "source": clean_text(source),
                })
        return _dedup_by_link(items, max_items)
    except Exception as e:
        log(f"Error fetching Polygon news for {ticker}: {e}")
        return []


def fetch_alpha_vantage_news(ticker: str, max_items: int = 20) -> List[Dict[str, Any]]:
    """Alpha Vantage news & sentiment (requires ALPHAVANTAGE_API_KEY). Docs: https://www.alphavantage.co/documentation/"""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return []
    symbol = ticker.replace(".", "-")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "sort": "LATEST",
        "limit": str(max_items),
        "apikey": api_key,
    }
    try:
        api_rate_limiter.wait_if_needed()
        resp = requests.get(url, headers=HTTP_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        feed = data.get("feed") or []
        items: List[Dict[str, Any]] = []
        for e in feed:
            title = (e.get("title") or "").strip()
            link = (e.get("url") or "").strip()
            tp = (e.get("time_published") or "").strip()  # e.g., 20250119T141000
            published = ""
            if len(tp) >= 8:
                try:
                    # Parse as UTC (AlphaVantage timestamps are UTC without TZ)
                    dt = datetime.strptime(tp[:15], "%Y%m%dT%H%M%S") if len(tp) >= 15 else datetime.strptime(tp[:8], "%Y%m%d")
                    published = dt.replace(tzinfo=timezone.utc).isoformat()
                except Exception:
                    published = tp
            source = e.get("source") or (e.get("source_domain") or "Alpha Vantage")
            if title and link:
                items.append({
                    "title": clean_text(title),
                    "link": clean_text(link),
                    "published": clean_text(published),
                    "source": clean_text(source),
                })
        return _dedup_by_link(items, max_items)
    except Exception as e:
        log(f"Error fetching AlphaVantage news for {ticker}: {e}")
        return []


def fetch_gdelt_news(query_text: str, days: int = 30, max_items: int = 20) -> List[Dict[str, Any]]:
    """GDELT 2.0 Doc API (no key). Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/"""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": query_text, "mode": "ArtList", "maxrecords": str(max_items), "format": "json"}
    try:
        api_rate_limiter.wait_if_needed()
        resp = requests.get(url, headers=HTTP_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        # The JSON can expose different field names across modes; handle robustly
        raw = data.get("articles") or data.get("documents") or data.get("result") or []
        items: List[Dict[str, Any]] = []
        for e in raw:
            title = (e.get("title") or e.get("DocumentTitle") or "").strip()
            link = (e.get("url") or e.get("DocumentIdentifier") or "").strip()
            published = (e.get("seendate") or e.get("publishedDate") or e.get("Date") or "").strip()
            if title and link:
                items.append({
                    "title": clean_text(title),
                    "link": clean_text(link),
                    "published": clean_text(published),
                    "source": "GDELT",
                })
        return _dedup_by_link(items, max_items)
    except Exception as e:
        log(f"Error fetching GDELT news: {e}")
        return []


def fetch_company_news(ticker: str, company: Optional[str] = None, days: int = 30, max_items: int = 20) -> List[Dict[str, Any]]:
    """Unified news fetch with provider fallback (Polygon -> Finnhub -> AlphaVantage -> GDELT)."""
    # Try Polygon
    items = fetch_polygon_news(ticker, max_items)
    if items:
        return items
    # Try Finnhub
    items = fetch_finnhub_company_news(ticker, days, max_items)
    if items:
        return items
    # Try Alpha Vantage
    items = fetch_alpha_vantage_news(ticker, max_items)
    if items:
        return items
    # Fallback to GDELT using a broader company/ticker query
    q = f'"{ticker}"' + (f' OR "{company}"' if company else "")
    return fetch_gdelt_news(q, days, max_items)

# ----------------------
# Price providers (non-Yahoo) with unified fallback
# ----------------------

def _price_points_stats(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not points:
        return {}
    closes_valid = [p["close"] for p in points if p.get("close") is not None]
    highs_valid = [p["high"] for p in points if p.get("high") is not None]
    lows_valid = [p["low"] for p in points if p.get("low") is not None]
    stats = {
        "current": closes_valid[-1] if closes_valid else None,
        "previous_close": closes_valid[-2] if len(closes_valid) >= 2 else None,
        "period_high": max(highs_valid) if highs_valid else None,
        "period_low": min(lows_valid) if lows_valid else None,
        "period_first": closes_valid[0] if closes_valid else None,
        "period_last": closes_valid[-1] if closes_valid else None,
    }
    if stats["period_first"] is not None and stats["period_last"] is not None:
        change = stats["period_last"] - stats["period_first"]
        pct_change = (change / stats["period_first"]) * 100 if stats["period_first"] else None
        stats["change"] = round(change, 4)
        stats["pct_change"] = round(pct_change, 2) if pct_change is not None else None
    return stats


def fetch_polygon_prices(ticker: str, days: int = 7, interval: str = "5m") -> Dict[str, Any]:
    """Polygon aggregates (requires POLYGON_API_KEY). Docs: https://polygon.io/docs/rest/stocks/aggregates"""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return {}
    symbol = ticker.replace(".", "-")
    multiplier = 5 if interval == "5m" else 60
    timespan = "minute" if interval == "5m" else "minute"
    to_dt = datetime.utcnow()
    from_dt = to_dt - timedelta(days=days)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
        f"{from_dt.strftime('%Y-%m-%d')}/{to_dt.strftime('%Y-%m-%d')}"
    )
    params = {"adjusted": "true", "limit": "50000", "apiKey": api_key, "sort": "asc"}
    try:
        api_rate_limiter.wait_if_needed()
        resp = requests.get(url, headers=HTTP_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results") or []
        points: List[Dict[str, Any]] = []
        for r in results:
            ts_ms = r.get("t")
            if ts_ms is None:
                continue
            ts = int(ts_ms) / 1000.0
            points.append({
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "close": r.get("c"),
                "volume": r.get("v"),
            })
        stats = _price_points_stats(points)
        return {"symbol": symbol, "interval": interval, "stats": stats, "points": points[-100:]}
    except Exception as e:
        log(f"Error fetching Polygon prices for {ticker}: {e}")
        return {}


def fetch_finnhub_prices(ticker: str, days: int = 7, interval: str = "5m") -> Dict[str, Any]:
    """Finnhub stock candles (requires FINNHUB_API_KEY). Docs: https://finnhub.io/docs/api/stock-candles"""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return {}
    symbol = ticker.replace(".", "-")
    resolution = "5" if interval == "5m" else "60"
    to_ts = int(datetime.utcnow().timestamp())
    from_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {"symbol": symbol, "resolution": resolution, "from": from_ts, "to": to_ts, "token": api_key}
    try:
        api_rate_limiter.wait_if_needed()
        resp = requests.get(url, headers=HTTP_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        if data.get("s") != "ok":
            return {}
        t = data.get("t") or []
        o = data.get("o") or []
        h = data.get("h") or []
        l = data.get("l") or []
        c = data.get("c") or []
        v = data.get("v") or []
        points: List[Dict[str, Any]] = []
        for i in range(min(len(t), len(c))):
            ts = t[i]
            points.append({
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "open": o[i] if i < len(o) else None,
                "high": h[i] if i < len(h) else None,
                "low": l[i] if i < len(l) else None,
                "close": c[i] if i < len(c) else None,
                "volume": v[i] if i < len(v) else None,
            })
        stats = _price_points_stats(points)
        return {"symbol": symbol, "interval": interval, "stats": stats, "points": points[-100:]}
    except Exception as e:
        log(f"Error fetching Finnhub prices for {ticker}: {e}")
        return {}


def fetch_alpha_vantage_intraday(ticker: str, interval: str = "5min") -> Dict[str, Any]:
    """Alpha Vantage intraday (requires ALPHAVANTAGE_API_KEY). Docs: https://www.alphavantage.co/documentation/"""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {}
    symbol = ticker.replace(".", "-")
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": interval, "outputsize": "compact", "apikey": api_key}
    try:
        api_rate_limiter.wait_if_needed()
        resp = requests.get(url, headers=HTTP_HEADERS, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or {}
        key = f"Time Series ({interval})"
        series = data.get(key) or {}
        if not series:
            return {}
        # AlphaVantage timestamps are in local US/Eastern without TZ; treat as UTC for simplicity
        rows = []
        for ts_str, row in series.items():
            try:
                # Attempt parse as naive and set UTC
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            rows.append((dt, row))
        rows.sort(key=lambda x: x[0])
        points = []
        for dt, row in rows:
            points.append({
                "timestamp": dt.isoformat(),
                "open": float(row.get("1. open")) if row.get("1. open") is not None else None,
                "high": float(row.get("2. high")) if row.get("2. high") is not None else None,
                "low": float(row.get("3. low")) if row.get("3. low") is not None else None,
                "close": float(row.get("4. close")) if row.get("4. close") is not None else None,
                "volume": float(row.get("5. volume")) if row.get("5. volume") is not None else None,
            })
        stats = _price_points_stats(points)
        return {"symbol": symbol, "interval": interval, "stats": stats, "points": points[-100:]}
    except Exception as e:
        log(f"Error fetching AlphaVantage prices for {ticker}: {e}")
        return {}


def fetch_price_data(ticker: str, days: int = 7, interval: str = "5m") -> Dict[str, Any]:
    """Unified price fetch with provider fallback (Polygon -> Finnhub -> AlphaVantage)."""
    # Try Polygon intraday
    data = fetch_polygon_prices(ticker, days, interval)
    if data:
        return data
    # Try Finnhub intraday
    data = fetch_finnhub_prices(ticker, days, interval)
    if data:
        return data
    # Try AlphaVantage intraday
    data = fetch_alpha_vantage_intraday(ticker, interval="5min")
    if data:
        return data
    return {}

# ----------------------
# LM Studio Client
# ----------------------

class LMStudioClient:
    """Client for LM Studio server using lmstudio SDK."""
    
    def __init__(self, server_host: str, verbose: bool = False, timeout: float = 60.0):
        host = normalize_server_host(server_host)
        lms.configure_default_client(host)
        self.model = lms.llm()
        self.host = host
        self.verbose = verbose
        self.timeout = timeout
    
    def get_model_info(self) -> Optional[str]:
        """Get current model identifier."""
        try:
            info = self.model.get_info()
            return info.get("identifier") or info.get("modelKey") or info.get("displayName")
        except Exception:
            return None
    
    def call_structured(self, messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Call model with structured output schema."""
        import threading
        result_holder = {}
        error_holder = []
        done = threading.Event()
        
        def _worker():
            try:
                chat = {"messages": messages}
                config = {"temperature": 0.3, "max_new_tokens": 512}
                res = self.model.respond(chat, response_format=schema, config=config)
                parsed = getattr(res, "parsed", {})
                if isinstance(parsed, dict):
                    result_holder.update(parsed)
            except Exception as e:
                error_holder.append(e)
            finally:
                done.set()
        
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        done.wait(self.timeout)
        
        if not done.is_set():
            log(f"Model call timed out after {self.timeout}s")
            return {}
        
        if error_holder:
            raise error_holder[0]
        
        return result_holder

# ----------------------
# Phase 1: Initial Analysis
# ----------------------

INITIAL_SYSTEM_PROMPT = """You are a financial analyst. Given a news article and the US stock listings universe, identify up to 10 US-listed stocks that are most directly affected by this news.

For each stock, provide:
- Ticker symbol (must be from the provided listings)
- Action: BUY or SELL based on the news sentiment and implications
- Confidence: 0-100 (100 = extremely confident, act immediately; 70+ = strong signal; 50-69 = moderate; below 50 = weak)
- Reason: Clear explanation (less than 200 words) of why this news affects the stock

Key guidelines:
1. If an article explicitly recommends buying a stock, give it BUY with HIGH confidence (70-90)
2. If an article highlights positive developments, growth, or opportunities, lean toward BUY
3. If an article mentions problems, risks, or negative developments, lean toward SELL
4. Companies directly mentioned should have the highest confidence
5. Competitors and related companies should have moderate confidence

Focus on:
- Companies directly mentioned in the article (highest priority)
- Direct competitors or suppliers
- Sector/industry peers that would be affected
- Related ETFs if applicable

For example, if an article says "2 Growth Stocks to Buy Right Now" and mentions Unity Software (U) and CoreWeave (CRWV), 
you should return BUY recommendations for both with confidence 70+ since they are explicitly recommended.

Return only valid JSON matching the schema."""

def build_initial_schema() -> Dict[str, Any]:
    """Build JSON schema for initial analysis."""
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "pattern": r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$"},
                        "action": {"type": "string", "enum": ["BUY", "SELL"]},
                        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reason": {"type": "string", "minLength": 3, "maxLength": 600}
                    },
                    "required": ["ticker", "action", "confidence", "reason"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["decisions"],
        "additionalProperties": False
    }

def analyze_news_initial(client: LMStudioClient, news_item: Dict[str, Any], listings: Dict[str, ListingItem]) -> List[Dict[str, Any]]:
    """Perform initial analysis on a news item."""
    # Extract mentioned tickers from article
    article_body = news_item.get("article-body", "")
    title = news_item.get("title", "")
    
    # Look for explicitly mentioned tickers in the article
    mentioned_tickers = set()
    text_to_search = title + " " + article_body
    
    # Look for patterns like "NYSE: U" or "NASDAQ: CRWV" or "(NASDAQ: CRWV)"
    exchange_pattern = re.compile(r'\(?\b(?:NYSE|NASDAQ|NYSEARCA):\s*([A-Z]{1,5})\b\)?')
    for match in exchange_pattern.finditer(text_to_search):
        ticker = match.group(1)
        if ticker in listings:
            mentioned_tickers.add(ticker)
            if client.verbose:
                log(f"      Found exchange mention: {ticker}")
    
    # Look for specific company names mentioned
    companies_to_check = {
        "Unity Software": "U",
        "CoreWeave": "CRWV",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Nvidia": "NVDA",
        "Tesla": "TSLA",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Meta": "META",
        "Netflix": "NFLX",
    }
    
    for company_name, ticker in companies_to_check.items():
        if company_name.lower() in text_to_search.lower() and ticker in listings:
            mentioned_tickers.add(ticker)
            if client.verbose:
                log(f"      Found company name: {company_name} -> {ticker}")
    
    # Also check for exact full company names from listings
    for ticker, listing in listings.items():
        if listing.name and len(listing.name) > 5:  # Skip very short names
            # Check for exact company name matches (case insensitive)
            if listing.name.lower() in text_to_search.lower():
                mentioned_tickers.add(ticker)
                if client.verbose:
                    log(f"      Found full company name: {listing.name} -> {ticker}")
    
    # Build focused ticker list for context (mentioned + sample of others)
    ticker_context = list(mentioned_tickers) + list(listings.keys())[:200]
    
    user_content = {
        "news": {
            "title": title,
            "link": news_item.get("link", ""),
            "body": article_body[:4000],  # Increase body size
            "published": news_item.get("published_at_utc", ""),
            "source": news_item.get("source_title", ""),
        },
        "mentioned_tickers": list(mentioned_tickers),
        "sample_tickers": ticker_context[:500],
        "instructions": (
            "Analyze this news and identify affected stocks with trading recommendations. "
            f"The article explicitly mentions these tickers: {list(mentioned_tickers)}. "
            "Focus on companies directly mentioned, their competitors, and related ETFs."
        )
    }
    
    messages = [
        {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
    ]
    
    schema = build_initial_schema()
    
    # Add debug logging
    if client.verbose:
        log(f"    Mentioned tickers found in article: {mentioned_tickers}")
    
    result = client.call_structured(messages, schema)
    
    # Debug log the raw result
    if client.verbose and result:
        log(f"    Raw LLM response: {json.dumps(result, indent=2)[:500]}...")
    
    # Validate and clean results
    decisions = result.get("decisions", [])
    if not decisions and client.verbose:
        log("    WARNING: LLM returned no decisions")
    
    valid_decisions = []
    
    for decision in decisions:
        ticker = decision.get("ticker", "").upper()
        # Only include tickers that exist in our listings
        if ticker in listings:
            # Get the raw values first for debugging
            raw_confidence = decision.get("confidence")
            raw_reason = decision.get("reason", "")
            
            if client.verbose:
                log(f"    Processing decision for {ticker}: confidence={raw_confidence}, reason_len={len(raw_reason)}")
            
            # Clean and validate
            confidence = max(0, min(100, int(raw_confidence) if raw_confidence is not None else 0))
            reason = clean_text(raw_reason) if raw_reason else "No reason provided"
            
            valid_decisions.append({
                "ticker": ticker,
                "action": decision.get("action", "BUY"),
                "confidence": confidence,
                "reason": truncate_with_ellipsis(reason, 200),
                "company_name": listings[ticker].name,
            })
        elif client.verbose:
            log(f"    Skipping unknown ticker: {ticker}")
    
    return valid_decisions

# ----------------------
# Phase 2: Enrichment and Refinement
# ----------------------

REFINE_SYSTEM_PROMPT = """You are a senior financial analyst performing a detailed second review of a trading recommendation.

Given:
1. The original news article and initial recommendation
2. Recent news from Google News and Yahoo Finance (last 30 days)
3. Detailed price data with high precision (5-minute intervals)

Your task:
- Provide a REVISED confidence level (0-100) for the same action
- Specify an expected target price based on your analysis
- Give a detailed reason (up to 200 words) for your revised assessment

Consider:
- Has the stock already moved significantly in response to similar news?
- Are there contradicting signals in recent news?
- What does the price action suggest about market sentiment?
- Is the opportunity still actionable or has it passed?

Be conservative - if the stock has already moved substantially, reduce confidence accordingly.
Return only valid JSON matching the schema."""

def build_refine_schema() -> Dict[str, Any]:
    """Build JSON schema for refinement analysis."""
    return {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "action": {"type": "string", "enum": ["BUY", "SELL"]},
            "original_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "revised_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "expected_high_price": {"type": "number", "minimum": 0},
            "horizon_hours": {"type": "integer", "minimum": 1, "maximum": 168},  # Up to 1 week
            "reasoning": {"type": "string", "minLength": 10, "maxLength": 600}
        },
        "required": ["ticker", "action", "original_confidence", "revised_confidence", 
                   "expected_high_price", "horizon_hours", "reasoning"],
        "additionalProperties": False
    }

def refine_decision(client: LMStudioClient, news_item: Dict[str, Any], decision: Dict[str, Any],
                    google_news: List[Dict[str, Any]], yahoo_news: List[Dict[str, Any]], 
                    price_data: Dict[str, Any]) -> Dict[str, Any]:
    """Refine a high-confidence decision with additional data."""
    
    # Combine news sources - prioritize Google News if Yahoo fails
    all_recent_news = []
    
    # If Yahoo failed but Google succeeded, use more Google news
    if not yahoo_news and google_news:
        for item in google_news[:10]:  # Use more Google news when Yahoo fails
            all_recent_news.append({
                "title": item["title"],
                "source": item.get("source", "Google News"),
                "published": item.get("published", ""),
            })
    else:
        # Normal case: mix both sources
        for item in google_news[:5]:
            all_recent_news.append({
                "title": item["title"],
                "source": item.get("source", "Google News"),
                "published": item.get("published", ""),
            })
        for item in yahoo_news[:5]:
            all_recent_news.append({
                "title": item["title"],
                "source": "Yahoo Finance",
                "published": item.get("published", ""),
            })
    
    # Prepare price summary
    price_summary = {}
    if price_data and "stats" in price_data:
        stats = price_data["stats"]
        price_summary = {
            "current_price": stats.get("current"),
            "period_high": stats.get("period_high"),
            "period_low": stats.get("period_low"),
            "change_pct": stats.get("pct_change"),
            "recent_points": price_data.get("points", [])[-20:],  # Last 20 data points
        }
    
    user_content = {
        "original_news": {
            "title": news_item.get("title", ""),
            "body_excerpt": news_item.get("article-body", "")[:1000],
            "published": news_item.get("published_at_utc", ""),
        },
        "initial_decision": {
            "ticker": decision["ticker"],
            "action": decision["action"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
        },
        "recent_news": all_recent_news,
        "price_data": price_summary,
        "instructions": (
            "Review the initial decision with this additional context. "
            "If the stock has already moved significantly based on similar news, reduce confidence. "
            "Provide realistic target price and timeframe."
        )
    }
    
    messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
    ]
    
    schema = build_refine_schema()
    result = client.call_structured(messages, schema)
    
    # Merge with original decision
    refined = decision.copy()
    if result:
        refined["revised_confidence"] = max(0, min(100, result.get("revised_confidence", decision["confidence"])))
        refined["expected_high_price"] = result.get("expected_high_price")
        refined["horizon_hours"] = result.get("horizon_hours", 24)
        refined["refined_reason"] = truncate_with_ellipsis(clean_text(result.get("reasoning", "")), 200)
        refined["refinement_timestamp"] = datetime.now(timezone.utc).isoformat()
    
    return refined

# ----------------------
# Main Processing
# ----------------------

def process_news_item(client: LMStudioClient, news_item: Dict[str, Any], listings: Dict[str, ListingItem],
                      confidence_threshold: int = 70, news_days: int = 30, price_days: int = 7,
                      verbose: bool = False) -> Dict[str, Any]:
    """Process a single news item through both phases."""
    
    news_id = news_item.get("id", "unknown")
    log(f"Processing news: {news_item.get('title', 'Unknown')[:80]}...", verbose)
    
    # Phase 1: Initial analysis
    log("  Phase 1: Initial analysis...", verbose)
    initial_decisions = analyze_news_initial(client, news_item, listings)
    log(f"  Found {len(initial_decisions)} affected stocks", verbose)
    
    # Phase 2: Enrichment for high-confidence decisions
    refined_decisions = []
    for decision in initial_decisions:
        ticker = decision["ticker"]
        confidence = decision["confidence"]
        
        if confidence >= confidence_threshold:
            log(f"  Phase 2: Enriching {ticker} (confidence: {confidence}%)...", verbose)
            
            # Fetch additional data
            company_name = decision.get("company_name", ticker)
            google_news = fetch_google_news(company_name, ticker, news_days)
            alt_news = fetch_company_news(ticker, company_name, news_days)
            price_data = fetch_price_data(ticker, price_days, interval="5m")
            
            log(f"    Fetched {len(google_news)} Google news, {len(alt_news)} alt news items", verbose)
            
            # Refine decision
            refined = refine_decision(client, news_item, decision, google_news, alt_news, price_data)
            
            # Add enrichment data to result
            refined["google_news_count"] = len(google_news)
            refined["yahoo_news_count"] = len(alt_news)
            refined["has_price_data"] = bool(price_data)
            
            if verbose and "revised_confidence" in refined:
                log(f"    Revised confidence: {refined['revised_confidence']}% (was {confidence}%)", verbose)
            
            refined_decisions.append(refined)
        else:
            refined_decisions.append(decision)
    
    # Build result
    result = {
        "news_id": news_id,
        "news_title": news_item.get("title", ""),
        "news_link": news_item.get("link", ""),
        "published_at": news_item.get("published_at_utc", ""),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "model": client.get_model_info(),
        "decisions": refined_decisions,
        "summary": {
            "total_decisions": len(refined_decisions),
            "high_confidence_count": sum(1 for d in refined_decisions 
                                        if d.get("revised_confidence", d.get("confidence", 0)) >= confidence_threshold),
            "enriched_count": sum(1 for d in refined_decisions if "revised_confidence" in d),
        }
    }
    
    return result

def save_results(results: List[Dict[str, Any]], minute_key: str):
    """Save processing results to file."""
    ensure_dirs()
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{minute_key}_{timestamp}.json"
    filepath = RESULT_DIR / filename
    
    # Also save as latest
    latest_path = RESULT_DIR / "latest_results.json"
    
    # Save to both files
    for path in [filepath, latest_path]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    log(f"Saved results to {filepath}")
    return filepath

# ----------------------
# Main Entry Point
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Enhanced LLM decision engine (llm2.py)")
    parser.add_argument("--server-host", default="localhost:1234", 
                       help="LM Studio server host:port")
    parser.add_argument("--confidence-threshold", type=int, default=70,
                       help="Minimum confidence for enrichment (0-100)")
    parser.add_argument("--news-days", type=int, default=30,
                       help="Days of news history to fetch")
    parser.add_argument("--price-days", type=int, default=7,
                       help="Days of price history to fetch")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--test", action="store_true",
                       help="Run test with data/news/2508242333.json")
    
    args = parser.parse_args()
    
    # Initialize
    ensure_dirs()
    log("Loading stock listings...", args.verbose)
    listings = load_listings()
    log(f"Loaded {len(listings)} stocks", args.verbose)
    
    # Connect to LM Studio
    log(f"Connecting to LM Studio at {args.server_host}...", args.verbose)
    client = LMStudioClient(args.server_host, verbose=args.verbose)
    model_info = client.get_model_info()
    log(f"Connected. Model: {model_info}", args.verbose)
    
    # Determine which news file to use
    if args.test:
        news_file = Path("data/news/2508242333.json")
        if not news_file.exists():
            print(f"Test file not found: {news_file}")
            sys.exit(1)
        log(f"Using test file: {news_file}", args.verbose)
    else:
        news_file = find_latest_news_file()
        if not news_file:
            print("No news files found in data/news/")
            sys.exit(1)
        log(f"Using latest news: {news_file}", args.verbose)
    
    # Load and process news
    news_items = load_news(news_file)
    log(f"Found {len(news_items)} news items", args.verbose)
    
    if not news_items:
        print("No news items to process")
        sys.exit(0)
    
    # Process each news item
    results = []
    for i, news_item in enumerate(news_items, 1):
        log(f"\n[{i}/{len(news_items)}] Processing news item...", args.verbose)
        
        try:
            result = process_news_item(
                client, news_item, listings,
                confidence_threshold=args.confidence_threshold,
                news_days=args.news_days,
                price_days=args.price_days,
                verbose=args.verbose
            )
            results.append(result)
            
            # Show summary
            if result["decisions"]:
                log(f"  Decisions made: {len(result['decisions'])}", True)
                for dec in result["decisions"]:
                    conf = dec.get("revised_confidence", dec.get("confidence", 0))
                    log(f"    {dec['ticker']}: {dec['action']} @ {conf}%", True)
        except Exception as e:
            log(f"  Error processing: {e}", True)
            results.append({
                "news_id": news_item.get("id", "unknown"),
                "news_title": news_item.get("title", ""),
                "error": str(e),
                "decisions": []
            })
    
    # Save results
    minute_key = news_file.stem  # YYMMDDHHMM
    output_path = save_results(results, minute_key)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    total_decisions = sum(len(r.get("decisions", [])) for r in results)
    high_conf = sum(
        1 for r in results 
        for d in r.get("decisions", [])
        if d.get("revised_confidence", d.get("confidence", 0)) >= args.confidence_threshold
    )
    print(f"News items processed: {len(results)}")
    print(f"Total decisions: {total_decisions}")
    print(f"High confidence (>={args.confidence_threshold}%): {high_conf}")
    print(f"Results saved to: {output_path}")
    
    # If test mode, show specific results for U and CRWV
    if args.test:
        print("\n" + "="*60)
        print("TEST RESULTS FOR U AND CRWV:")
        print("="*60)
        for result in results:
            for decision in result.get("decisions", []):
                if decision["ticker"] in ["U", "CRWV"]:
                    print(f"\n{decision['ticker']}:")
                    print(f"  Action: {decision['action']}")
                    print(f"  Initial confidence: {decision['confidence']}%")
                    if "revised_confidence" in decision:
                        print(f"  Revised confidence: {decision['revised_confidence']}%")
                        print(f"  Expected price: ${decision.get('expected_high_price', 'N/A')}")
                        print(f"  Reason: {decision.get('refined_reason', decision['reason'])}")
                    else:
                        print(f"  Reason: {decision['reason']}")

if __name__ == "__main__":
    main()