#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
news_feed.py — Continuous, minute-bucketed RSS → JSONL harvester.

Changes per spec:
• First minutes after start: grab any items we've never seen before (first-seen).
• Thereafter: every minute, save any new (first-seen) items into the previous-minute bucket.
• Default 4 threads (ThreadPoolExecutor).
• Use a common Chrome User-Agent for HTTP requests.
• 20s time budget per minute (configurable).
• Never stops unless a signal is caught.

Output (local time):
  ./data/news/YYMMDDHHMM.json          # JSON array per minute

Item schema (per line):
{
  "id": "<stable hash from guid/id/link/title>",
  "title": "...",
  "link": "https://...",
  "published_at_local": "YYYY-MM-DDTHH:MM:SS±HH:MM" or null,
  "published_at_utc": "YYYY-MM-DDTHH:MM:SSZ" or null,
  "fetched_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "first_seen_local": "YYYY-MM-DDTHH:MM:SS±HH:MM",
  "first_seen_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "source_title": "...",
  "source_domain": "example.com",
  "rss_url": "https://...rss",
  "authors": ["..."],
  "categories": ["..."],
  "summary": "...",
  "content_html": "<p>...</p>",
  "guid": "..."   # when provided by the feed
}

Completeness rule (must have): title and link.
"""

import argparse
import calendar
import hashlib
import json
import os
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import feedparser   # pip install feedparser
import requests     # pip install requests
from pathlib import Path
import sqlite3

# ---------- Reliable default feeds (official) ----------
# (Verified official pages below in citations)
DEFAULT_FEEDS = [
    # Bloomberg
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.bloomberg.com/economics/news.rss",

    # MarketWatch (Dow Jones)
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://feeds.content.dowjones.io/public/rss/mw_bulletins",

    # Wall Street Journal (directory provides many sections; this one is common)
    "https://feeds.wsjonline.com/wsj/economics/feed",

    # Financial Times (sitewide)
    "https://www.ft.com/news-feed?format=rss",

    # Washington Post – Business
    "https://feeds.washingtonpost.com/rss/business/",

    # NPR – Business
    "https://feeds.npr.org/1006/rss.xml",

    # SEC – Latest EDGAR filings (Atom)
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom",

    # Federal Reserve – All press releases
    "https://www.federalreserve.gov/feeds/press_all.xml",

    # Tech (useful for market-moving tech headlines)
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/quickposts",  # public “quick posts” feed
    # Finance aggregators
    "https://finance.yahoo.com/news/rss",
    "https://www.investing.com/rss/news.rss",
    "https://feeds.bloomberg.com/technology/news.rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",            # WSJ Markets
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",          # WSJ U.S. Business
    "https://feeds.a.dj.com/rss/RSSEconomy.xml",                # WSJ Economy
    "https://www.ft.com/markets?format=rss",                    # FT Markets
    "https://www.ft.com/companies?format=rss",                  # FT Companies
    "https://www.ft.com/global-economy?format=rss",             # FT Global Economy
    "https://www.ft.com/technology?format=rss",                 # FT Technology
    "https://www.ft.com/world/us?format=rss",                   # FT U.S.
    "http://feeds.bbci.co.uk/news/business/rss.xml",            # BBC Business
    "https://www.theguardian.com/business/rss",                 # Guardian Business
    "https://www.theguardian.com/business/economics/rss",       # Guardian Economics
    "https://www.theguardian.com/us-news/rss",                  # Guardian U.S. news
    "https://www.marketwatch.com/feeds/topstories",             # MarketWatch Top
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",   # MarketWatch MarketPulse
    "https://www.investors.com/feed/",                          # Investor’s Business Daily
    "https://www.barrons.com/magazine/rss",                     # Barron's (magazine)

    # --- Tech (often market-moving) ---
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.wired.com/feed/rss",

    # --- U.S. regulators / government (market-moving) ---
    "https://www.federalreserve.gov/feeds/press_monetary.xml",  # Fed – monetary policy
    "https://apps.bea.gov/rss/rss.xml",                         # BEA news releases
    "https://www.sec.gov/news/pressreleases.rss",               # SEC press releases
    "https://www.cftc.gov/RSS/RSSGP/rssgp.xml",                 # CFTC press
    "https://www.cftc.gov/RSS/RSSENF/rssenf.xml",               # CFTC enforcement
    "https://www.cftc.gov/RSS/RSSST/rssst.xml",                 # CFTC speeches/testimony

    # --- Corporate press wires (official releases; high volume) ---
    "https://www.prnewswire.com/rss/news-releases-list.rss",    # PR Newswire (all releases)
    "https://rss.globenewswire.com/rssfeed/",                   # GlobeNewswire (general feed)
]

# Common (modern) Chrome UA; adjust if you prefer.
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

STOP = False

SEEN_DB_PATH = Path("./data/news/.seen.db")

def seen_db_init():
    SEEN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SEEN_DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, first_seen_utc TEXT NOT NULL)")
    con.commit()
    return con
def compute_entry_id(entry: dict) -> str:
    src = entry.get("id") or entry.get("guid") or entry.get("link") or (entry.get("title") or "")
    if isinstance(src, dict):  # some feeds nest guid as {"#text": "..."}
        src = src.get("#text") or json.dumps(src, sort_keys=True)
    return hashlib.sha256(str(src).encode("utf-8")).hexdigest()


def handle_signal(signum, frame):
    global STOP
    STOP = True
    print(f"[signal] Caught {signum}. Shutting down gracefully...", file=sys.stderr)


for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, handle_signal)
    except Exception:
        pass


def log(msg: str, verbose: bool):
    if verbose:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}", file=sys.stderr)


def safe_get(url: str, timeout: float = 10.0) -> Optional[bytes]:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass
    return None


def struct_time_to_aware_utc(st) -> Optional[datetime]:
    if not st:
        return None
    try:
        ts = calendar.timegm(st)  # treat struct_time as UTC
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def get_best_published(entry: dict) -> Optional[datetime]:
    dt = struct_time_to_aware_utc(entry.get("published_parsed"))
    if dt:
        return dt
    return struct_time_to_aware_utc(entry.get("updated_parsed"))


def entry_to_record(entry: dict, feed_url: str, feed_title: Optional[str]) -> Optional[Dict[str, Any]]:
    title = (entry.get("title") or "").strip()
    link = (entry.get("link") or "").strip()

    published_utc = get_best_published(entry)
    if not (title and link):
        return None  # incomplete → drop

    published_local = published_utc.astimezone() if published_utc else None

    authors = []
    if entry.get("authors"):
        for a in entry["authors"]:
            name = a.get("name") or a.get("email") or ""
            if name:
                authors.append(name)

    categories = []
    if entry.get("tags"):
        for t in entry["tags"]:
            term = t.get("term")
            if term:
                categories.append(term)

    summary = entry.get("summary")
    content_html = None
    if entry.get("content"):
        try:
            content_html = entry["content"][0].get("value")
        except Exception:
            content_html = None

    try:
        src_domain = urlparse(link).netloc
    except Exception:
        src_domain = ""

    rec_id = compute_entry_id(entry)

    return {
        "id": rec_id,
        "title": title,
        "link": link,
        "published_at_local": published_local.isoformat(timespec="seconds") if published_local else None,
        "published_at_utc": published_utc.isoformat(timespec="seconds").replace("+00:00", "Z") if published_utc else None,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "first_seen_local": None,
        "first_seen_utc": None,
        "source_title": (feed_title or src_domain or None),
        "source_domain": src_domain or None,
        "rss_url": feed_url,
        "authors": authors or None,
        "categories": categories or None,
        "summary": summary or None,
        "content_html": content_html or None,
        "guid": entry.get("id") or entry.get("guid"),
    }


def fetch_and_parse(feed_url: str, request_timeout: float = 10.0) -> List[Dict[str, Any]]:
    data = safe_get(feed_url, timeout=request_timeout)
    if not data:
        return []
    parsed = feedparser.parse(data)
    feed_title = (parsed.get("feed") or {}).get("title")
    out: List[Dict[str, Any]] = []
    for e in parsed.get("entries", []):
        rec = entry_to_record(e, feed_url, feed_title)
        if rec:
            out.append(rec)
    return out


def minute_bounds(dt_local: datetime):
    start = dt_local.replace(second=0, microsecond=0)
    end = start + timedelta(minutes=1)
    return start, end




def minute_key(dt_local: datetime) -> str:
    # YYMMDDHHMM in local time
    return dt_local.strftime("%y%m%d%H%M")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_extra_feeds(path: str, verbose: bool) -> List[str]:
    feeds: List[str] = []
    if not path:
        return feeds
    try:
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    feeds = [str(x).strip() for x in obj if str(x).strip()]
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        feeds.append(s)
    except Exception as e:
        print(f"[warn] Failed to load feeds file {path}: {e}", file=sys.stderr)
    if verbose and feeds:
        log(f"Loaded {len(feeds)} extra feeds from {path}", True)
    return feeds


def sleep_until_next_minute():
    now = datetime.now()
    next_min = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
    secs = (next_min - now).total_seconds()
    if secs > 0:
        time.sleep(secs)


# --- Backfill logic: collect history during first 1-2 minutes, update seen DB only ---
def backfill_history(feeds: List[str], threads: int, request_timeout: float, deadline: datetime,
                     con: sqlite3.Connection, inmem_seen: set, verbose: bool):
    """Continuously scan feeds and mark items as seen until either:
    - we complete at least one full pass with 0 new items (early complete), or
    - we hit the provided deadline (end of the second minute).
    No files are written during backfill. Only the seen DB is updated.
    """
    pass_count = 0
    while (not STOP) and (datetime.now() < deadline):
        pass_count += 1
        new_pairs = []  # [(id, first_seen_utc)] to persist
        try:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                futures = {ex.submit(fetch_and_parse, url, request_timeout): url for url in feeds}
                while futures and (datetime.now() < deadline) and (not STOP):
                    time_left = (deadline - datetime.now()).total_seconds()
                    if time_left <= 0:
                        break
                    # bound the wait by the smaller of remaining time or request timeout
                    wait_timeout = min(request_timeout, max(0.0, time_left))
                    done, _pending = wait(list(futures.keys()), timeout=wait_timeout, return_when=FIRST_COMPLETED)
                    for fut in done:
                        _url = futures.pop(fut, None)
                        try:
                            results = fut.result() or []
                            for r in results:
                                rec_id = r.get("id") or compute_entry_id({
                                    "id": None,
                                    "guid": None,
                                    "link": r.get("link"),
                                    "title": r.get("title"),
                                })
                                r["id"] = rec_id
                                if rec_id in inmem_seen:
                                    continue
                                cur = con.execute("SELECT 1 FROM seen WHERE id=?", (rec_id,))
                                if cur.fetchone():
                                    continue
                                now_utc = datetime.now(timezone.utc)
                                # stamp first-seen timestamps (not used for files in backfill)
                                r["first_seen_utc"] = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
                                r["first_seen_local"] = now_utc.astimezone().isoformat(timespec="seconds")
                                inmem_seen.add(rec_id)
                                new_pairs.append((rec_id, r["first_seen_utc"]))
                        except Exception:
                            pass
        except Exception:
            pass

        if new_pairs:
            try:
                with con:
                    con.executemany(
                        "INSERT OR IGNORE INTO seen(id, first_seen_utc) VALUES(?, ?)",
                        new_pairs,
                    )
            except Exception:
                pass

        if verbose:
            now_str = datetime.now().strftime("%H:%M:%S")
            dl_str = deadline.strftime("%H:%M:%S")
            log(f"Backfill pass {pass_count}: newly seen {len(new_pairs)} items (now {now_str}, deadline {dl_str}).", True)

        # Early completion heuristic: one full pass with zero new items
        if len(new_pairs) == 0 and pass_count >= 1:
            if verbose:
                log("Backfill appears complete early; switching to minute mode.", True)
            break


def main():
    parser = argparse.ArgumentParser(description="Continuous RSS harvester (minute-bucketed JSONL, current minute only).")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of worker threads (default: 4)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--feeds-file", type=str, default=None, help="Path to extra feeds file (json or newline list)")
    parser.add_argument("--out-dir", type=str, default="./data/news", help="Output base directory (default: ./data/news)")
    parser.add_argument("--request-timeout", type=float, default=10.0, help="Per-request timeout in seconds (default: 10)")
    parser.add_argument("--cycle-budget", type=float, default=20.0, help="Time budget (s) per minute cycle (default: 20)")
    args = parser.parse_args()

    feeds = list(dict.fromkeys(DEFAULT_FEEDS + load_extra_feeds(args.feeds_file, args.verbose)))  # de-dup preserve order
    if not feeds:
        print("No feeds to process.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        log(f"Starting with {len(feeds)} feeds, {args.threads} threads, {args.cycle_budget:.0f}s budget per minute.", True)

    # Initialize seen-store and in-memory cache
    con = seen_db_init()
    inmem_seen = set()

    # Bootstrap/backfill phase: first minute collects history, second minute continues if needed.
    start_ts = datetime.now()
    current_floor = start_ts.replace(second=0, microsecond=0)
    first_minute_end = current_floor + timedelta(minutes=1)
    second_minute_end = current_floor + timedelta(minutes=2)
    if args.verbose:
        log(f"Backfill starting; allowed until end of second minute ({second_minute_end.strftime('%H:%M:%S')}).", True)
    backfill_history(
        feeds=feeds,
        threads=args.threads,
        request_timeout=args.request_timeout,
        deadline=second_minute_end,
        con=con,
        inmem_seen=inmem_seen,
        verbose=args.verbose,
    )

    # Main loop: minute-by-minute mode
    while not STOP:
        cycle_start_local = datetime.now()
        # Process the PREVIOUS local minute to avoid double-catching the current minute
        now_floor = cycle_start_local.replace(second=0, microsecond=0)
        minute_start = now_floor - timedelta(minutes=1)
        minute_end = now_floor
        bucket_name = minute_key(minute_start)
        if args.verbose:
            log(f"[minute-mode] Processing previous minute bucket {bucket_name} (local) — budget {args.cycle_budget:.0f}s…", True)

        deadline = cycle_start_local + timedelta(seconds=args.cycle_budget)
        records_for_minute: List[Dict[str, Any]] = []
        fetched_items = 0

        try:
            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                futures = {ex.submit(fetch_and_parse, url, args.request_timeout): url for url in feeds}

                while futures and datetime.now() < deadline and not STOP:
                    time_left = (deadline - datetime.now()).total_seconds()
                    if time_left <= 0:
                        break
                    done, _pending = wait(list(futures.keys()), timeout=time_left, return_when=FIRST_COMPLETED)
                    for fut in done:
                        url = futures.pop(fut, None)
                        try:
                            results = fut.result() or []
                            fetched_items += len(results)
                            for r in results:
                                rec_id = r.get("id") or compute_entry_id({"id": None, "guid": None, "link": r.get("link"), "title": r.get("title")})
                                r["id"] = rec_id
                                if rec_id in inmem_seen:
                                    continue
                                cur = con.execute("SELECT 1 FROM seen WHERE id=?", (rec_id,))
                                if cur.fetchone():
                                    continue
                                now_utc = datetime.now(timezone.utc)
                                r["first_seen_utc"] = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
                                r["first_seen_local"] = now_utc.astimezone().isoformat(timespec="seconds")
                                inmem_seen.add(rec_id)
                                records_for_minute.append(r)
                        except Exception:
                            pass
                # Executor exits here; remaining futures will be cleaned up.
        except Exception:
            pass

        # Write ONLY previous minute’s items as a single JSON array file:
        # ./data/news/YYMMDDHHMM.json
        if records_for_minute:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"{bucket_name}.json")
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(records_for_minute, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, out_path)
        if records_for_minute:
            with con:
                con.executemany(
                    "INSERT OR IGNORE INTO seen(id, first_seen_utc) VALUES (?, ?)",
                    [(r["id"], r["first_seen_utc"]) for r in records_for_minute]
                )

        if args.verbose:
            took = (datetime.now() - cycle_start_local).total_seconds()
            log(f"Fetched ~{fetched_items} items; saved {len(records_for_minute)} to {bucket_name}.json in {took:.2f}s.", True)

        if STOP:
            break
        sleep_until_next_minute()

    if args.verbose:
        log("Stopped.", True)


if __name__ == "__main__":
    try:
        import multiprocessing as _mp  # noqa
        _mp.freeze_support()
    except Exception:
        pass
    main()