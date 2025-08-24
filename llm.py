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
            decs.append({
                "ticker": clean_text(d.get("ticker")),
                "action": clean_text(d.get("action")),
                "confidence": int(d.get("confidence", 0)),
                "reason": clean_text(d.get("reason"))[:240],
            })
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

    def list_models_loaded(self) -> List[str]:
        try:
            handles = lms.list_loaded_models("llm")  # list only LLMs
            out: List[str] = []
            for h in handles:
                try:
                    info = h.get_info()
                    ident = info.get("identifier") or info.get("modelKey") or info.get("displayName")
                    if ident:
                        out.append(str(ident))
                except Exception:
                    pass
            # Fallback: query the currently selected handle's info
            if not out and self.model is not None:
                try:
                    info = self.model.get_info()
                    ident = info.get("identifier") or info.get("modelKey") or info.get("displayName")
                    if ident:
                        out.append(str(ident))
                except Exception:
                    pass
            return out
        except Exception:
            return []

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
    current_model = client.get_current_model_identifier()
    loaded = client.list_models_loaded()
    if args.verbose:
        if current_model:
            log(f"Active LM Studio model: {current_model} (server {server_host_norm})", True)
        else:
            log(f"LM Studio ready at {server_host_norm}; proceeding without model listing (will confirm via inference).", True)
        if loaded:
            log(f"Loaded LLMs reported: {len(loaded)} (first 5): {loaded[:5]}", True)

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

                # Write per-news processed JSON immediately
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