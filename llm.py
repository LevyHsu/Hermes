#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm.py — Decision engine that reads the latest minute-bucketed news JSON,
consults the latest US stock listings, and asks a local LLM (LM Studio)
for up to 0–10 related tickers with BUY/SELL decisions + confidence.

Requirements / assumptions
- LM Studio server API is running (e.g., localhost:1234 or 192.168.0.198:1234) and a model
  is loaded/selected in the LM Studio UI.
- The news harvester stores files at ./data/news/YYMMDDHHMM.json (array of items, with
  mandatory fields including "id", "title", "link", and "article-body").
- Listings are in ./data/us-stock-listing/us-listings-latest.json (or the dated one).
- This script runs continuously until interrupted (Ctrl+C / SIGTERM):
    1) Locate the most recent news minute file
    2) For each unseen news item, query the model
    3) Save per-item decisions to ./data/llm/<minute>/<news_id>.json
    4) Save per-minute aggregate to ./data/llm/<minute>/<minute>.json

CLI options:
  --server-host host:port (default: localhost:1234)
  --poll-interval seconds (default 10)
  --verbose

Notes on MCP:
  LM Studio can host MCP servers (e.g., a RAG server) in the app itself; when the
  server is connected, tool use happens inside LM Studio.
"""

import argparse
import json
import os
import re
import signal
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import feedparser

# lmstudio-python SDK (native)
# Docs: https://lmstudio.ai/docs/python/getting-started/project-setup
#       https://lmstudio.ai/docs/python/llm-prediction/structured-response
import lmstudio as lms

DATA_DIR = Path("./data")
NEWS_DIR = DATA_DIR / "news"
LLM_OUT_DIR = DATA_DIR / "llm"
LISTING_DIR = DATA_DIR / "us-stock-listing"
DECISIONS_DB = LLM_OUT_DIR / ".decisions.db"
# Store per-news processed JSON under data/llm/<YYMMDDHHMM>/<news_id>.json
PROCESSED_DIR = LLM_OUT_DIR

DEFAULT_SERVER_HOST = os.environ.get("LMSTUDIO_SERVER_HOST", "192.168.0.198:1234")

# ----------------------
# Server host normalization
# ----------------------
_HOST_RE = re.compile(r"^(?:https?://)?([^/]+?)(?:/.*)?$")
def normalize_server_host(host: str) -> str:
    if not host:
        return "localhost:1234"
    m = _HOST_RE.match(host.strip())
    if not m:
        return host
    return m.group(1)

# ----------------------
# Small utilities
# ----------------------

def log(msg: str, verbose: bool):
    if verbose:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}", file=sys.stderr)

def ensure_dirs():
    LLM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# SQLite for dedupe
# ----------------------

def decisions_db_init() -> sqlite3.Connection:
    ensure_dirs()
    con = sqlite3.connect(DECISIONS_DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS processed (news_id TEXT PRIMARY KEY, minute TEXT NOT NULL, created_utc TEXT NOT NULL)"
    )
    con.commit()
    return con

def already_processed(con: sqlite3.Connection, news_id: str) -> bool:
    cur = con.execute("SELECT 1 FROM processed WHERE news_id=?", (news_id,))
    return cur.fetchone() is not None

def mark_processed(con: sqlite3.Connection, news_id: str, minute: str):
    with con:
        con.execute(
            "INSERT OR IGNORE INTO processed(news_id, minute, created_utc) VALUES (?, ?, ?)",
            (news_id, minute, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )

# ----------------------
# Listings loader + tools
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
    """Load latest listings (prefer us-listings-short-latest.json if present)."""
    short_latest = LISTING_DIR / "us-listings-short-latest.json"
    full_latest = LISTING_DIR / "us-listings-latest.json"
    path = short_latest if short_latest.exists() else full_latest
    if not path.exists():
        # Fall back to latest dated file
        dated = sorted(LISTING_DIR.glob("us-listings-*.json"))
        if not dated:
            raise FileNotFoundError("No listings JSON found in ./data/us-stock-listing")
        path = dated[-1]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, ListingItem] = {}
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

def find_by_name_partial(listings: Dict[str, ListingItem], query: str, limit: int = 10) -> List[Dict[str, Any]]:
    q = query.lower()
    out: List[Tuple[str, float]] = []
    for tkr, item in listings.items():
        nm = (item.name or "").lower()
        if not nm:
            continue
        if q in nm:
            score = 1.0 - (nm.find(q) / max(1, len(nm)))
            out.append((tkr, score))
    out.sort(key=lambda x: x[1], reverse=True)
    results: List[Dict[str, Any]] = []
    for tkr, _s in out[:limit]:
        it = listings[tkr]
        results.append({
            "ticker": it.ticker,
            "name": it.name,
            "exchange": it.exchange,
            "type": it.type,
            "is_etf": it.is_etf,
            "is_adr": it.is_adr,
        })
    return results

# ----------------------
# News loader
# ----------------------

def latest_news_file() -> Optional[Path]:
    # Accept files named YYMMDDHHMM.json (10 digits in the stem)
    files: List[Path] = []
    for p in NEWS_DIR.glob("*.json"):
        stem = p.stem
        if len(stem) == 10 and stem.isdigit():
            files.append(p)
    if not files:
        return None
    files.sort()
    return files[-1]

def load_news(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data

# ----------------------
# Prompt + schema
# ----------------------

SYSTEM_TEMPLATE = (
    "You are a trading assistant. Given one news article about US markets and the US "
    "listings universe, select up to 10 US-listed tickers most directly affected. "
    "For each, output an action and a confidence index (0-100).\n\n"
    "Rules:\n"
    "- Consider first-order impact (issuer mentioned, direct competitors, suppliers, ETFs).\n"
    "- If impact is ambiguous, generic personal finance content, opinion/column with no issuer impact, or otherwise immaterial, return ZERO decisions.\n"
    "- Only include tickers that are listed in the provided US listings (you can call tools to check).\n"
    "- BUY means open/accumulate long; SELL means reduce/close long (or short).\n"
    "- Confidence 100 = act now; 0 = not sure.\n"
    "- Each decision MUST include a short non-empty reason (<= 240 chars).\n"
    "- Output JSON ONLY matching the response schema. No commentary."
)

RESPONSE_SCHEMA_EXAMPLE = {
    "news_id": "<string>",
    "news_title": "<string>",
    "link": "<url>",
    "relevance": "none",  # one of: none, low, medium, high
    "relevance_reason": "<short reason>",
    "decisions": [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 87,
            "reason": "iPhone demand surprise; revenue guide beat; peers lift."
        }
    ]
}

def build_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "news_id": {"type": "string"},
            "news_title": {"type": "string", "minLength": 1},
            "link": {"type": "string", "minLength": 1, "pattern": r"^https?://.*$"},
            "relevance": {"type": "string", "enum": ["none", "low", "medium", "high"]},
            "relevance_reason": {"type": "string"},
            "decisions": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "pattern": r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$"},
                        "action": {"type": "string", "enum": ["BUY", "SELL"]},
                        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reason": {"type": "string", "minLength": 3}
                    },
                    "required": ["ticker", "action", "confidence"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["news_id", "news_title", "link", "relevance", "relevance_reason", "decisions"],
        "additionalProperties": False
    }

# ----------------------
# Text normalization helpers
# ----------------------
INVISIBLE_CATEGORIES = {"Cf", "Cc", "Cs"}
def clean_text(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in INVISIBLE_CATEGORIES)
    return s.strip()

# ----------------------
# Core processing helpers
# ----------------------

def normalize_model_output(nid: str, title: str, link: str, parsed: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        out.update(parsed)

    # Always trust our source metadata over model-provided copies
    out["news_id"] = nid
    out["news_title"] = clean_text(title)
    out["link"] = clean_text(link)

    decisions = out.get("decisions")
    if not isinstance(decisions, list):
        decisions = []
    norm_decisions = []
    for d in decisions[:10]:
        if not isinstance(d, dict):
            continue
        tk = clean_text((d.get("ticker") or "").upper())
        if not tk:
            continue
        act = clean_text((d.get("action") or "").upper())
        if act not in {"BUY", "SELL"}:
            continue
        # Drop unknown/unlisted tickers
        try:
            if GLOBAL_LISTINGS and tk not in GLOBAL_LISTINGS:
                continue
        except NameError:
            pass
        try:
            conf = int(d.get("confidence", 0))
        except Exception:
            conf = 0
        conf = max(0, min(100, conf))
        raw_reason = d.get("reason") or ""
        reason = clean_text(raw_reason) if raw_reason else ""
        if not reason:
            reason = "Model provided no rationale; included due to article context."
        norm_decisions.append({
            "ticker": tk,
            "action": act,
            "confidence": conf,
            "reason": reason[:240],
        })
    out["decisions"] = norm_decisions

    rel = out.get("relevance")
    if rel not in {"none", "low", "medium", "high"}:
        rel = "none" if not norm_decisions else "low"
    out["relevance"] = rel

    rr = out.get("relevance_reason") or ""
    rr = clean_text(rr)
    if not rr:
        if rel == "none" and not norm_decisions:
            rr = "No directly affected US-listed issuer; generic/irrelevant to equities."
        else:
            rr = "Model-provided decisions imply some impact."
    out["relevance_reason"] = rr[:280]
    return out

def write_processed_record(minute_key: str, news_id: str, record: Dict[str, Any]):
    # Create ./data/llm/<YYMMDDHHMM>/ and write <news_id>.json inside it
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    minute_dir = PROCESSED_DIR / minute_key
    minute_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{news_id}.json" if news_id else "unknown.json"
    out_path = minute_dir / fname
    tmp = out_path.with_suffix(".tmp")

    rec = {
        **record,
        "news_id": clean_text(record.get("news_id")),
        "minute": clean_text(record.get("minute")),
        "model": clean_text(record.get("model")),
    }
    if isinstance(record.get("input"), dict):
        rec["input"] = {
            "title": clean_text(record["input"].get("title")),
            "link": clean_text(record["input"].get("link")),
        }
    if isinstance(record.get("output"), dict):
        outp = dict(record["output"])
        outp["news_id"] = clean_text(outp.get("news_id"))
        outp["news_title"] = clean_text(outp.get("news_title"))
        outp["link"] = clean_text(outp.get("link"))
        outp["relevance"] = clean_text(outp.get("relevance"))
        outp["relevance_reason"] = clean_text(outp.get("relevance_reason"))
        decs = []
        for d in outp.get("decisions", []) or []:
            if not isinstance(d, dict):
                continue
            entry = {
                "ticker": clean_text(d.get("ticker")),
                "action": clean_text(d.get("action")),
                "confidence": int(d.get("confidence", 0)),
                "reason": clean_text(d.get("reason"))[:240],
            }
            # Optional refined fields
            if d.get("refined_confidence") is not None:
                try:
                    entry["refined_confidence"] = int(d.get("refined_confidence"))
                except Exception:
                    pass
            if d.get("refined_reason"):
                entry["refined_reason"] = clean_text(d.get("refined_reason"))[:240]
            if d.get("refined_at_utc"):
                entry["refined_at_utc"] = clean_text(d.get("refined_at_utc"))
            decs.append(entry)
        outp["decisions"] = decs
        rec["output"] = outp

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)

TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b")

def extract_candidate_tickers(text: str, listings: Dict[str, ListingItem]) -> List[str]:
    if not text:
        return []
    found = set()
    for m in TICKER_RE.finditer(text.upper()):
        t = m.group(0)
        if t in listings:
            found.add(t)
    return sorted(found)

# ----------------------
# Enrichment: Google News RSS + Yahoo Finance chart API
# ----------------------

HTTP_TIMEOUT = (8, 15)  # (connect, read) seconds
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}

def google_news_rss(company: str, ticker: str, days: int, max_items: int = 10) -> List[Dict[str, Any]]:
    """Query Google News RSS for the last N days using a robust query of name OR ticker."""
    # Example: https://news.google.com/rss/search?q=AAPL%20OR%20%22Apple%20Inc%22%20when%3A30d&hl=en-US&gl=US&ceid=US:en
    q = f"{ticker} OR \"{company}\" when:{days}d"
    url = (
        "https://news.google.com/rss/search?" 
        + f"q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        fp = feedparser.parse(resp.content)
        items = []
        for e in fp.entries[: max_items * 2]:  # oversample then trim uniques
            title = e.get("title", "").strip()
            link = (e.get("link") or "").strip()
            published = e.get("published", "") or e.get("updated", "")
            source = ""
            if "source" in e and isinstance(e.source, dict):
                source = e.source.get("title") or ""
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
        for it in items:
            if it["link"] in seen:
                continue
            seen.add(it["link"])
            deduped.append(it)
        return deduped[:max_items]
    except Exception:
        return []

def yahoo_symbol_from(ticker: str) -> str:
    # Yahoo uses '-' instead of '.' for class shares (e.g., BRK-B)
    return ticker.replace(".", "-")

def yahoo_prices(ticker: str, days: int) -> Dict[str, Any]:
    """Fetch recent price series from Yahoo chart endpoint.
    Returns {meta: {...}, points: [{t, c, v}], interval: str} or {} on failure.
    """
    sym = yahoo_symbol_from(ticker)
    interval = "5m" if days <= 7 else "1h"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?"
        f"range={days}d&interval={interval}&includePrePost=true"
    )
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("chart", {}).get("result") or [None])[0]
        if not result:
            return {}
        meta = result.get("meta", {})
        tzoff = int(meta.get("gmtoffset", 0))
        ts = result.get("timestamp") or []
        quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
        closes = quotes.get("close") or []
        vols = quotes.get("volume") or []
        points = []
        for i, t in enumerate(ts):
            c = closes[i] if i < len(closes) else None
            v = vols[i] if i < len(vols) else None
            if c is None:
                continue
            # Convert epoch (sec) + tz offset to ISO UTC string for consistency
            iso = datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat(timespec="seconds")
            points.append({"t": iso, "c": float(c), "v": int(v) if v is not None else None})
        return {
            "meta": {"symbol": sym, "interval": interval, "timezone_offset": tzoff},
            "points": points,
            "interval": interval,
        }
    except Exception:
        return {}

def summarize_prices(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not points:
        return {}
    first = points[0]["c"]
    last = points[-1]["c"]
    chg = (last - first)
    pct = (chg / first * 100.0) if first else 0.0
    return {"first": first, "last": last, "abs_change": chg, "pct_change": pct}

REFINE_SYSTEM = (
    "You are a second-pass verifier for trading signals. Given the original decision "
    "and supplemental evidence (recent headlines and recent price action), output a revised "
    "confidence for the SAME action on the SAME ticker. Additionally:\n"
    "1) Estimate a *near-term horizon* in HOURS (positive integer) that is appropriate for this setup/impact/liquidity.\n"
    "2) Estimate the *expected high price* within that horizon and optionally provide up to two intermediate anchors "
    "(price and confidence) between the current price and the expected high.\n\n"
    "Rules:\n"
    "- Keep the action unchanged.\n"
    "- 'horizon_hours' must be a positive integer (e.g., 1–72 hours is typical; longer only if clearly justified by the article).\n"
    "- 'expected_high_price' must be a positive float in the same currency as the input prices.\n"
    "- Confidence is 0–100. Assume confidence decays to **0 at the expected_high_price** if action=BUY; "
    "if action=SELL, assume confidence decays to 0 at the expected_low_price (still use the field name 'expected_high_price' but it may be below current).\n"
    "- Anchors, if provided, must be strictly between the current price and the expected_high_price and have confidence between 0 and the revised confidence.\n"
    "Return JSON only."
)

def build_refine_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "action": {"type": "string", "enum": ["BUY", "SELL"]},
            "previous_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "revised_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "reasoning": {"type": "string", "minLength": 5},
            "expected_high_price": {"type": "number", "minimum": 0},
            "anchors": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "price": {"type": "number"},
                        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                        "label": {"type": "string"}
                    },
                    "required": ["price", "confidence"],
                    "additionalProperties": False
                }
            },
            "horizon_hours": {"type": "integer", "minimum": 1, "maximum": 240}
        },
        "required": ["ticker", "action", "previous_confidence", "revised_confidence", "reasoning", "horizon_hours"],
        "additionalProperties": False
    }

def build_refine_messages(news_item: Dict[str, Any], decision: Dict[str, Any], gnews: List[Dict[str, Any]], yprices: Dict[str, Any]) -> List[Dict[str, Any]]:
    price_summary = summarize_prices(yprices.get("points", []))
    payload = {
        "original_news": {
            "title": news_item.get("title"),
            "link": news_item.get("link"),
            "published_at_utc": news_item.get("published_at_utc"),
        },
        "decision": decision,
        "google_news": gnews,
        "price_summary": price_summary,
        "price_points_tail": yprices.get("points", [])[-50:],
        "current_price": (price_summary.get("last") if isinstance(price_summary, dict) else None),
        "instructions": (
            "Select a plausible near-term 'horizon_hours' based on article severity and recent price path. "
            "Estimate expected_high_price within that horizon. Confidence must be 0 at the expected high price (for BUY). "
            "Optionally provide up to 2 anchors between current price and expected high."
        )
    }
    return [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
def build_confidence_curve(current_price: Optional[float], current_conf: int, expected_high: Optional[float], anchors: Optional[List[Dict[str, Any]]], steps: int, action: str) -> Dict[str, Any]:
    """
    Build a monotonic curve from current price to expected_high where confidence decays to 0 at the endpoint.
    For SELL actions, the expected_high may be below current; we still treat it as the terminal price where confidence=0.
    Returns {"expected_high": float, "curve": [{"price":p,"confidence":c}, ...]}
    """
    try:
        steps = max(2, int(steps))
    except Exception:
        steps = 5
    if current_price is None or not isinstance(current_price, (int, float)) or current_price <= 0:
        return {"expected_high": None, "curve": []}
    # Fallback expected_high if missing or invalid: extrapolate using recent pct change or +1%
    if expected_high is None or not isinstance(expected_high, (int, float)) or expected_high <= 0 or expected_high == current_price:
        # simple fallback: 1% away from current in the direction of action
        bump = 0.01 * current_price
        expected_high = (current_price + bump) if (str(action).upper() == "BUY") else (current_price - bump)
    # Normalize anchors
    norm_anchors: List[Tuple[float, int]] = []
    if anchors:
        for a in anchors:
            try:
                ap = float(a.get("price"))
                ac = int(a.get("confidence"))
                norm_anchors.append((ap, max(0, min(100, ac))))
            except Exception:
                continue
    # Build base grid
    prices: List[float] = []
    if steps <= 2:
        prices = [current_price, expected_high]
    else:
        for i in range(steps):
            t = i / (steps - 1)
            p = current_price + (expected_high - current_price) * t
            prices.append(p)
    # Seed confidences with linear decay as default
    confs = [max(0, min(100, int(round(current_conf * (1 - (i / (len(prices) - 1))))))) for i in range(len(prices))]
    # Blend in anchors: snap confidence at nearest price index to anchor confidence, then ensure monotonic non-increasing toward endpoint
    for ap, ac in norm_anchors:
        # find nearest index
        idx = min(range(len(prices)), key=lambda k: abs(prices[k] - ap))
        confs[idx] = max(0, min(100, ac))
    # Enforce monotone decay to 0 at endpoint
    confs[0] = max(0, min(100, int(current_conf)))
    confs[-1] = 0
    for i in range(1, len(confs)):
        if confs[i] > confs[i-1]:
            confs[i] = confs[i-1]
    return {"expected_high": float(expected_high), "curve": [{"price": float(prices[i]), "confidence": int(confs[i])} for i in range(len(prices))]}

# ----------------------
# LLM client (lmstudio-python SDK)
# ----------------------

class LMStudioClient:
    def __init__(self, server_host: str, verbose: bool = False):
        # Must be the first call before using convenience API. See docs.
        # https://lmstudio.ai/docs/python/getting-started/project-setup
        host = normalize_server_host(server_host)
        lms.configure_default_client(host)
        # Use the currently selected model in LM Studio (no explicit identifier here)
        self.model = lms.llm()
        self.host = host
        self.verbose = verbose


    def get_current_model_identifier(self) -> Optional[str]:
        try:
            info = self.model.get_info()
            return info.get("identifier") or info.get("modelKey") or info.get("displayName")
        except Exception:
            return None

    def respond_structured(self, messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        chat = {"messages": messages}
        cfg = {"temperature": 0.2}  # effort/limits are configured in LM Studio UI
        result = self.model.respond(chat, response_format=schema, config=cfg)
        return getattr(result, "parsed", {}) or {}

# ----------------------
# Core processing
# ----------------------

def build_messages(listings: Dict[str, ListingItem], news: Dict[str, Any]) -> List[Dict[str, Any]]:
    news_id = news.get("id") or ""
    title = news.get("title") or ""
    link = news.get("link") or ""
    body = news.get("article-body") or ""

    candidates = extract_candidate_tickers(" ".join([title, body]), listings)

    system_msg = {"role": "system", "content": SYSTEM_TEMPLATE}

    user_payload = {
        "news_id": news_id,
        "news_title": title,
        "link": link,
        "published_at_utc": news.get("published_at_utc"),
        "source": news.get("source_title") or news.get("source_domain"),
        "candidate_tickers_in_text": candidates,
        "response_schema": RESPONSE_SCHEMA_EXAMPLE,
        "instructions": (
            "Return only valid JSON. Include 0–10 items in 'decisions'. "
            "If the article is generic personal finance/advice or otherwise irrelevant to listed issuers, set 'relevance'='none' and return an empty 'decisions' array. "
            "Use tools to validate tickers against the listings if unsure."
        ),
        "article_body": body[:8000],
    }

    return [system_msg, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}]

def write_decisions(minute_key: str, results: List[Dict[str, Any]]):
    # Aggregate file resides at ./data/llm/<YYMMDDHHMM>/<YYMMDDHHMM>.json
    base_dir = PROCESSED_DIR / minute_key
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"{minute_key}.json"

    # Merge with existing if present
    existing: List[Dict[str, Any]] = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as rf:
                prev = json.load(rf)
                if isinstance(prev, list):
                    existing = prev
        except Exception:
            existing = []
    existing_ids = {str(r.get("news_id")) for r in existing}
    new_unique = [r for r in results if str(r.get("news_id")) not in existing_ids]
    combined = existing + new_unique

    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)

def minute_from_filename(path: Path) -> str:
    return path.stem  # 'YYMMDDHHMM'

# Global listings cache
GLOBAL_LISTINGS: Dict[str, ListingItem] = {}

def main():
    parser = argparse.ArgumentParser(description="LLM decision engine using LM Studio (native SDK)")
    parser.add_argument(
        "--server-host",
        default=DEFAULT_SERVER_HOST,
        help=(
            "LM Studio server host:port (default: %s). 'http://' prefix is accepted and will be normalized."
            % DEFAULT_SERVER_HOST
        ),
    )
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between checks for new news")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--conf-threshold", type=int, default=60, help="Confidence threshold (inclusive) to trigger enrichment & refinement")
    parser.add_argument("--news-window-days", type=int, default=30, help="Days of Google News headlines to fetch per ticker")
    parser.add_argument("--price-window-days", type=int, default=7, help="Days of recent price history to fetch per ticker")
    parser.add_argument("--max-news", type=int, default=10, help="Max Google News items to include in refinement")
    # REMOVED: parser.add_argument("--horizon-hours", type=int, default=6, help="Near-term horizon for expected-high estimate (hours)")
    parser.add_argument("--confidence-curve-steps", type=int, default=5, help="Number of points (including endpoints) for the price→confidence curve")
    args = parser.parse_args()

    ensure_dirs()
    con = decisions_db_init()

    # Load listings (global)
    global GLOBAL_LISTINGS
    GLOBAL_LISTINGS = load_listings()
    if args.verbose:
        log(f"Loaded {len(GLOBAL_LISTINGS)} listings.", True)

    # Init client and sanity-check loaded models
    server_host_norm = normalize_server_host(args.server_host)
    if args.verbose:
        log(f"Connecting to LM Studio at {server_host_norm}", True)
    client = LMStudioClient(server_host=server_host_norm, verbose=args.verbose)
    if args.verbose:
        log(f"LM Studio client initialized for {server_host_norm}. Inference will confirm connectivity/model selection.", True)

    # Main loop: always use the latest news file available
    try:
        while True:
            nf = latest_news_file()
            if nf is None:
                log("No news files found yet…", args.verbose)
                time.sleep(args.poll_interval)
                continue

            minute_key = minute_from_filename(nf)
            log(f"Using latest news file {nf.name} (minute {minute_key})", args.verbose)

            news_items = load_news(nf)
            total_items = len(news_items)
            unseen = 0
            decisions: List[Dict[str, Any]] = []

            for item in news_items:
                nid = item.get("id") or ""
                if not nid:
                    continue
                if already_processed(con, nid):
                    continue
                unseen += 1

                messages = build_messages(GLOBAL_LISTINGS, item)
                schema = build_response_schema()
                if args.verbose:
                    log("LLM round 1…", True)
                try:
                    parsed = client.respond_structured(messages, schema)
                except Exception as e:
                    parsed = {"error": repr(e), "decisions": [], "relevance": "none", "relevance_reason": "Model call failed."}

                normalized = normalize_model_output(nid, item.get("title") or "", item.get("link") or "", parsed)

                # --- Enrichment + refinement for high-confidence decisions ---
                try:
                    threshold = max(0, min(100, int(args.conf_threshold)))
                except Exception:
                    threshold = 60
                refine_schema = build_refine_schema()

                # Update normalized decisions in-place with refined results
                for idx, dec in enumerate(list(normalized.get("decisions", []))):
                    try:
                        if int(dec.get("confidence", 0)) < threshold:
                            continue
                    except Exception:
                        continue
                    ticker = dec.get("ticker")
                    if not ticker:
                        continue
                    listing = GLOBAL_LISTINGS.get(ticker)
                    company = listing.name if listing and listing.name else ticker

                    # Fetch enrichment data
                    gnews = google_news_rss(company, ticker, int(args.news_window_days), max_items=int(args.max_news))
                    yprices = yahoo_prices(ticker, int(args.price_window_days))

                    # Ask the model to revise confidence
                    refine_msgs = build_refine_messages(item, dec, gnews, yprices)
                    try:
                        refined = client.respond_structured(refine_msgs, refine_schema)
                    except Exception as e:
                        refined = {
                            "ticker": ticker,
                            "action": dec.get("action"),
                            "previous_confidence": int(dec.get("confidence", 0)),
                            "revised_confidence": int(dec.get("confidence", 0)),
                            "reasoning": f"Refinement failed: {e!r}",
                        }

                    # Persist refined confidence & reason into the decision
                    try:
                        prev_c = int(refined.get("previous_confidence", dec.get("confidence", 0)))
                        new_c = int(refined.get("revised_confidence", prev_c))
                    except Exception:
                        prev_c = int(dec.get("confidence", 0))
                        new_c = prev_c
                    reason_text = refined.get("reasoning", "")

                    # Update the normalized decision entry
                    normalized["decisions"][idx]["refined_confidence"] = new_c
                    if reason_text:
                        normalized["decisions"][idx]["refined_reason"] = clean_text(str(reason_text))[:240]
                    normalized["decisions"][idx]["refined_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

                    # Build and attach a price→confidence curve
                    current_price = (summarize_prices(yprices.get("points", [])).get("last") if isinstance(yprices, dict) else None)
                    expected_high = refined.get("expected_high_price")
                    anchors = refined.get("anchors")
                    act = (refined.get("action") or dec.get("action") or "").upper()
                    curve = build_confidence_curve(current_price, new_c, expected_high, anchors, int(args.confidence_curve_steps), act)
                    chosen_h = refined.get("horizon_hours")
                    try:
                        chosen_h = int(chosen_h) if chosen_h is not None else 6
                    except Exception:
                        chosen_h = 6
                    normalized["decisions"][idx]["price_path"] = {
                        "horizon_hours": chosen_h,
                        "expected_high": curve.get("expected_high"),
                        "curve": curve.get("curve", [])
                    }

                    # Print on screen
                    try:
                        log(f"Refined {ticker} {act}: {prev_c}% -> {new_c}% — {str(reason_text)[:160]}", True)
                    except Exception:
                        log(f"Refined {ticker}: (unable to print refined output)", True)

                # Build record AFTER refinement so JSON includes refined fields
                record = {
                    "news_id": nid,
                    "minute": minute_key,
                    "model": client.get_current_model_identifier(),
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "input": {
                        "title": item.get("title"),
                        "link": item.get("link"),
                    },
                    "output": normalized,
                }

                # Write per-news processed JSON (now contains refined fields)
                write_processed_record(minute_key, nid, record)

                decisions.append(record)
                mark_processed(con, nid, minute_key)

            log(f"Latest file has {total_items} item(s); unseen this pass: {unseen}.", args.verbose)

            if decisions:
                write_decisions(minute_key, decisions)
                log(f"Wrote {len(decisions)} decision(s) to {minute_key}/{minute_key}.json", args.verbose)
            else:
                log("No new news to process in the latest file.", args.verbose)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        log("Interrupted — shutting down.", True)
    except SystemExit:
        pass

if __name__ == "__main__":
    main()