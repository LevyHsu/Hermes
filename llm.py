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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# OpenAI SDK works with LM Studio's OpenAI-compatible endpoint
# See: https://lmstudio.ai/docs/app/api/endpoints/openai and
# https://cookbook.openai.com/articles/gpt-oss/run-locally-lmstudio
from openai import OpenAI

DATA_DIR = Path("./data")
NEWS_DIR = DATA_DIR / "news"
LLM_OUT_DIR = DATA_DIR / "llm"
LISTING_DIR = DATA_DIR / "us-stock-listing"
DECISIONS_DB = LLM_OUT_DIR / ".decisions.db"

DEFAULT_MODEL = "openai/gpt-oss-120b"
ALT_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "not-needed")  # LM Studio ignores API keys

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
    files = [p for p in NEWS_DIR.glob("*.json") if p.name.isdigit() and p.suffix == ".json"]
    if not files:
        return None
    # sort by filename (YYMMDDHHMM.json lexicographically equals chronological)
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
    "- If impact is ambiguous or minor, either omit or assign low confidence.\n"
    "- Only include tickers that are listed in the provided US listings (you can call tools to check).\n"
    "- BUY means open/accumulate long; SELL means reduce/close long (or short).\n"
    "- Confidence 100 = act now; 0 = not sure.\n"
    "- Keep rationales concise (<= 240 chars each).\n"
    "- Output JSON ONLY matching the response schema. No commentary."
)

RESPONSE_SCHEMA_EXAMPLE = {
    "news_id": "<string>",
    "news_title": "<string>",
    "link": "<url>",
    "decisions": [
        {
            "ticker": "AAPL",
            "action": "BUY",  # or "SELL"
            "confidence": 87,   # integer 0-100
            "reason": "iPhone demand surprise; revenue guide beat; peers lift."
        }
    ]
}


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
# LLM client
# ----------------------

class LMStudioClient:
    def __init__(self, base_url: str, api_key: str, model: str, effort: str = "medium", verbose: bool = False):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        if effort not in {"low", "medium", "high"}:
            effort = "medium"
        self.effort = effort
        self.verbose = verbose

    def list_models(self) -> List[str]:
        try:
            res = self.client.models.list()
            return [m.id for m in getattr(res, "data", [])]
        except Exception:
            return []

    def chat_json(self, messages: List[Dict[str, Any]], tools=None, max_rounds: int = 4) -> str:
        """Run a tool-call loop until we get a final JSON response string."""
        tools = tools or []
        round_idx = 0
        chat = list(messages)
        while round_idx < max_rounds:
            round_idx += 1
            if self.verbose:
                log(f"LLM round {round_idx}…", True)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=0.2,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                    reasoning={"effort": self.effort},  # gpt-oss accepts the field; ignored if unsupported
                )
            except Exception as e:
                raise RuntimeError(f"Chat call failed: {e}")

            choice = resp.choices[0]
            msg = choice.message

            # Handle tool calls if any
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                chat.append({"role": "assistant", "tool_calls": [tc.model_dump() for tc in tool_calls]})
                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    tool_output = execute_tool(name, args, GLOBAL_LISTINGS)
                    chat.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": tool_output,
                    })
                # Continue loop
                continue

            # No tool calls: expect JSON content
            content = (msg.content or "").strip()
            if content:
                return content

            # Fallback: stop
            if choice.finish_reason == "stop":
                return "{}"
        # Max rounds reached
        return "{}"


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
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def minute_from_filename(path: Path) -> str:
    return path.stem  # 'YYMMDDHHMM'


# Global listings cache (used inside tool functions)
GLOBAL_LISTINGS: Dict[str, ListingItem] = {}


def main():
    parser = argparse.ArgumentParser(description="LLM decision engine using LM Studio (OpenAI-compatible API)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL, ALT_MODEL], help="Model id to use")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="Reasoning effort")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LM Studio OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key (ignored by LM Studio)")
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

    # Init client and sanity-check models list
    client = LMStudioClient(base_url=args.base_url, api_key=args.api_key, model=args.model, effort=args.effort, verbose=args.verbose)
    models = client.list_models()
    if args.verbose:
        if models:
            log(f"LM Studio models visible: {len(models)} (showing first 5): {models[:5]}", True)
        else:
            log("Could not list models from LM Studio — ensure the API server is running.", True)

    last_seen_news_file: Optional[Path] = None

    # Main loop
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
            decisions: List[Dict[str, Any]] = []

            for item in news_items:
                nid = item.get("id") or ""
                if not nid:
                    continue
                if already_processed(con, nid):
                    continue

                # Build messages and ask the model
                messages = build_messages(GLOBAL_LISTINGS, item)
                try:
                    model_json = client.chat_json(messages, tools=tool_schemas(), max_rounds=4)
                    parsed = json.loads(model_json)
                except Exception as e:
                    parsed = {"error": repr(e), "news_id": nid}

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
                    "output": parsed,
                }
                decisions.append(record)
                mark_processed(con, nid, minute_key)

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