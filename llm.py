#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm.py — Decision engine that reads the latest minute-bucketed news JSON,
consults the latest US stock listings, and asks a local LLM (LM Studio)
for up to 0–10 related tickers with BUY/SELL decisions + confidence.

Requirements / assumptions
- LM Studio is running its OpenAI-compatible server (default http://localhost:1234/v1)
  and a model is loaded (default: openai/gpt-oss-120b).  
  Docs: LM Studio exposes an OpenAI-compatible /v1/chat/completions endpoint, so the
  official openai SDK can be used by pointing base_url to localhost.  
  (Ref: LM Studio docs + OpenAI cookbook)  
- The news harvester stores files at ./data/news/YYMMDDHHMM.json (array of items, with
  mandatory fields including "id", "title", "link", and "article-body").
- Listings are in ./data/us-stock-listing/us-listings-latest.json (or the dated one).
- This script runs continuously until interrupted (Ctrl+C / SIGTERM):
    1) Locate the most recent news minute file
    2) For each unseen news item, start a chat
    3) Provide tools so the model can validate tickers against the listings
    4) Save decisions to ./data/llm/<minute>.json

CLI options:
  --model {openai/gpt-oss-120b,openai/gpt-oss-20b} (default: 120b)
  --effort {low,medium,high}  (reasoning effort, default: medium)
  --base-url http://localhost:1234/v1
  --poll-interval seconds (default 10)
  --verbose

Notes on MCP:
  LM Studio can host MCP servers (e.g., a RAG server) in the app itself; when the
  server is connected, tool use happens inside LM Studio. From this client, we expose
  simple function tools (OpenAI tool calling) for ticker validation and lookup.

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

# lmstudio-python SDK (native) — use LM Studio's convenience API
# Docs: https://lmstudio.ai/docs/python/getting-started/project-setup
# and Structured Response: https://lmstudio.ai/docs/python/llm-prediction/structured-response
import lmstudio as lms

DATA_DIR = Path("./data")
NEWS_DIR = DATA_DIR / "news"
LLM_OUT_DIR = DATA_DIR / "llm"
LISTING_DIR = DATA_DIR / "us-stock-listing"
DECISIONS_DB = LLM_OUT_DIR / ".decisions.db"
# Store per-news processed JSON directly under data/llm/<YYMMDDHHMM>/<news_id>.json
PROCESSED_DIR = LLM_OUT_DIR

DEFAULT_MODEL = "openai/gpt-oss-120b"
ALT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SERVER_HOST = os.environ.get("LMSTUDIO_SERVER_HOST", "localhost:1234")

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
        # Prefer the compact schema names if present
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
            # simple scoring by position length
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
# Prompt + tools for the model
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
        # 0 to 10 items
        {
            "ticker": "AAPL",
            "action": "BUY",  # or "SELL"
            "confidence": 87,   # integer 0-100
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
    """Normalize and remove invisible/control characters from strings.
    Returns input unchanged if it is not a string.
    """
    if not isinstance(s, str):
        return s
    # Normalize width/compatibility first
    s = unicodedata.normalize("NFKC", s)
    # Strip characters in Unicode categories: Format (Cf), Control (Cc), Surrogate (Cs)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in INVISIBLE_CATEGORIES)
    # Trim redundant whitespace
    return s.strip()

# ----------------------
# Core processing
# ----------------------

# Normalizer for model output: ensures required fields, coerces structure, sets relevance to "none" if no decisions.
def normalize_model_output(nid: str, title: str, link: str, parsed: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        out.update(parsed)

    # Always trust our source metadata over model-provided copies
    out["news_id"] = nid
    out["news_title"] = clean_text(title)
    out["link"] = clean_text(link)

    # decisions
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
            # Fallback if model leaves it blank
            reason = "Model provided no rationale; included due to article context."
        norm_decisions.append({
            "ticker": tk,
            "action": act,
            "confidence": conf,
            "reason": reason[:240],
        })
    out["decisions"] = norm_decisions

    # relevance
    rel = out.get("relevance")
    if rel not in {"none", "low", "medium", "high"}:
        rel = "none" if not norm_decisions else "low"
    out["relevance"] = rel

    # relevance reason
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
    # use the raw id (hex) as filename; fallback to safe name
    fname = f"{news_id}.json" if news_id else "unknown.json"
    out_path = minute_dir / fname
    tmp = out_path.with_suffix(".tmp")

    # Deep sanitize selected string fields to eliminate invisible chars
    rec = {
        **record,
        "news_id": clean_text(record.get("news_id")),
        "minute": clean_text(record.get("minute")),
        "model": clean_text(record.get("model")),
        "effort": clean_text(record.get("effort")),
    }
    if isinstance(record.get("input"), dict):
        rec["input"] = {
            "title": clean_text(record["input"].get("title")),
            "link": clean_text(record["input"].get("link")),
        }
    if isinstance(record.get("output"), dict):
        outp = dict(record["output"])  # shallow copy
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


# ---- OpenAI tool definitions (executed locally) ----

def tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "is_listed",
                "description": "Check if a ticker exists in the US listing universe.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_company_brief",
                "description": "Return brief info about a ticker (name, exchange, type, is_etf, is_adr).",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_by_name",
                "description": "Search for companies by partial name and return up to 10 results.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]


def execute_tool(name: str, args: Dict[str, Any], listings: Dict[str, ListingItem]) -> str:
    try:
        if name == "is_listed":
            t = (args.get("ticker") or "").upper()
            exists = t in listings
            return json.dumps({"ticker": t, "listed": bool(exists)})
        elif name == "get_company_brief":
            t = (args.get("ticker") or "").upper()
            it = listings.get(t)
            if not it:
                return json.dumps({"error": "not_found", "ticker": t})
            return json.dumps({
                "ticker": it.ticker,
                "name": it.name,
                "exchange": it.exchange,
                "type": it.type,
                "is_etf": it.is_etf,
                "is_adr": it.is_adr,
            })
        elif name == "find_by_name":
            q = args.get("query") or ""
            return json.dumps(find_by_name_partial(listings, q, limit=10))
    except Exception as e:
        return json.dumps({"error": repr(e)})
    return json.dumps({"error": "unknown_tool"})


# ----------------------
# LLM client (lmstudio-python SDK)
# ----------------------

class LMStudioClient:
    def __init__(self, server_host: str, model: str, effort: str = "medium", verbose: bool = False):
        # Must be the first call before using convenience API. See docs.
        # https://lmstudio.ai/docs/python/getting-started/project-setup
        host = normalize_server_host(server_host)
        lms.configure_default_client(host)
        self.model = lms.llm(model) if model else lms.llm()
        self.host = host
        if effort not in {"low", "medium", "high"}:
            effort = "medium"
        self.effort = effort
        self.verbose = verbose

    def list_models_loaded(self) -> List[str]:
        try:
            handles = lms.list_loaded_models()  # all types
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

    def _effort_to_max_tokens(self) -> int:
        return {"low": 400, "medium": 800, "high": 1200}.get(self.effort, 800)

    def respond_structured(self, messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        chat = {"messages": messages}
        cfg = {"temperature": 0.2, "maxTokens": self._effort_to_max_tokens()}
        # Structured output: result.parsed is a dict conforming to the schema.
        # https://lmstudio.ai/docs/python/llm-prediction/structured-response
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
        "candidate_tickers_in_text": candidates,  # hints
        "response_schema": RESPONSE_SCHEMA_EXAMPLE,
        "instructions": (
            "Return only valid JSON. Include 0–10 items in 'decisions'. "
            "If the article is generic personal finance/advice or otherwise irrelevant to listed issuers, set 'relevance'='none' and return an empty 'decisions' array. "
            "Use tools to validate tickers against the listings if unsure."
        ),
        # Keep the body last
        "article_body": body[:8000],  # keep context bounded
    }

    return [system_msg, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}]


def write_decisions(minute_key: str, results: List[Dict[str, Any]]):
    out_dir = LLM_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{minute_key}.json"

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

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def minute_from_filename(path: Path) -> str:
    return path.stem  # 'YYMMDDHHMM'


# Global listings cache (used inside tool functions)
GLOBAL_LISTINGS: Dict[str, ListingItem] = {}


def main():
    parser = argparse.ArgumentParser(description="LLM decision engine using LM Studio (native SDK)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL, ALT_MODEL], help="Model id to use")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="Reasoning effort")
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST, help="LM Studio server host:port (e.g., localhost:1234 or 192.168.0.198:1234). 'http://' prefix is accepted and will be normalized.")
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
    client = LMStudioClient(server_host=server_host_norm, model=args.model, effort=args.effort, verbose=args.verbose)
    loaded = client.list_models_loaded()
    if args.verbose:
        if loaded:
            log(f"LM Studio loaded models on {server_host_norm}: {len(loaded)} (showing first 5): {loaded[:5]}", True)
        else:
            log(f"No loaded models reported from {server_host_norm} — ensure the server is reachable, 'Serve on Local Network' is enabled, and a model is loaded.", True)

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

                # Build messages and ask the model
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
                    "model": args.model,
                    "effort": args.effort,
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
                log(f"Wrote {len(decisions)} decision(s) to {minute_key}.json", args.verbose)
            else:
                log("No new news to process in the latest file.", args.verbose)

            # Sleep until the next poll; we will still re-check the same latest file
            # and only act on previously unseen items
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        log("Interrupted — shutting down.", True)
    except SystemExit:
        pass


if __name__ == "__main__":
    main()