#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
news_harvester.py - Single-cycle news harvester for main.py integration

This is a modified version of news_feed.py that:
1. Runs for exactly one minute cycle
2. Returns immediately after saving the news file
3. Can be called repeatedly by main.py
"""

import argparse
import calendar
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from pathlib import Path
import sqlite3

import feedparser
import requests
from readability import Document
from bs4 import BeautifulSoup

# Import feed list from news_feed
from news_feed import (
    DEFAULT_FEEDS,
    UA,
    SEEN_DB_PATH,
    seen_db_init,
    compute_entry_id,
    safe_get,
    struct_time_to_aware_utc,
    get_best_published,
    entry_to_record,
    fetch_and_parse,
    minute_key,
    html_to_text,
    fetch_article_html,
    readability_extract,
    enrich_records_with_article_body,
    load_extra_feeds
)

def log(msg: str, verbose: bool):
    """Log message with timestamp."""
    if verbose:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}", file=sys.stderr)

def harvest_single_cycle(
    feeds: List[str],
    threads: int = 4,
    request_timeout: float = 10.0,
    cycle_budget: float = 20.0,
    out_dir: str = "./data/news",
    verbose: bool = False
) -> Optional[Path]:
    """
    Harvest news for a single minute cycle and return the output file path.
    
    Returns:
        Path to the created JSON file, or None if no news was found
    """
    cycle_start = datetime.now()
    
    # Use previous minute as the bucket
    now_floor = cycle_start.replace(second=0, microsecond=0)
    minute_start = now_floor - timedelta(minutes=1)
    bucket_name = minute_key(minute_start)
    
    if verbose:
        log(f"Harvesting news for bucket {bucket_name}", True)
    
    deadline = cycle_start + timedelta(seconds=cycle_budget)
    records_for_minute: List[Dict[str, Any]] = []
    fetched_items = 0
    
    # Initialize seen database
    con = seen_db_init()
    
    # Load in-memory seen cache
    inmem_seen = set()
    try:
        cursor = con.execute("SELECT id FROM seen")
        for row in cursor:
            inmem_seen.add(row[0])
    except Exception:
        pass
    
    # Fetch from all feeds concurrently
    try:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = {ex.submit(fetch_and_parse, url, request_timeout): url for url in feeds}
            
            while futures and datetime.now() < deadline:
                time_left = (deadline - datetime.now()).total_seconds()
                if time_left <= 0:
                    break
                    
                done, _pending = wait(list(futures.keys()), timeout=min(1.0, time_left), return_when=FIRST_COMPLETED)
                
                for fut in done:
                    url = futures.pop(fut, None)
                    try:
                        results = fut.result() or []
                        fetched_items += len(results)
                        
                        for r in results:
                            rec_id = r.get("id") or compute_entry_id({
                                "id": None,
                                "guid": None,
                                "link": r.get("link"),
                                "title": r.get("title")
                            })
                            r["id"] = rec_id
                            
                            # Skip if already seen
                            if rec_id in inmem_seen:
                                continue
                            
                            # Check database
                            cur = con.execute("SELECT 1 FROM seen WHERE id=?", (rec_id,))
                            if cur.fetchone():
                                continue
                            
                            # Mark as seen
                            now_utc = datetime.now(timezone.utc)
                            r["first_seen_utc"] = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
                            r["first_seen_local"] = now_utc.astimezone().isoformat(timespec="seconds")
                            inmem_seen.add(rec_id)
                            records_for_minute.append(r)
                            
                    except Exception as e:
                        if verbose:
                            log(f"Error processing {url}: {e}", True)
    except Exception as e:
        if verbose:
            log(f"Error in thread pool: {e}", True)
    
    # Enrich with article bodies
    records_for_minute = enrich_records_with_article_body(
        records_for_minute, deadline, threads, request_timeout, verbose
    )
    
    # Save to file if we have records
    out_path = None
    if records_for_minute:
        os.makedirs(out_dir, exist_ok=True)
        out_path = Path(out_dir) / f"{bucket_name}.json"
        tmp_path = str(out_path) + ".tmp"
        
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records_for_minute, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
        
        # Update seen database
        with con:
            con.executemany(
                "INSERT OR IGNORE INTO seen(id, first_seen_utc) VALUES (?, ?)",
                [(r["id"], r["first_seen_utc"]) for r in records_for_minute]
            )
        
        if verbose:
            took = (datetime.now() - cycle_start).total_seconds()
            log(f"Saved {len(records_for_minute)} items (from {fetched_items} fetched) to {out_path.name} in {took:.1f}s", True)
    else:
        if verbose:
            log(f"No new items found (fetched {fetched_items} total)", True)
    
    con.close()
    return out_path

def main():
    """Main entry point for standalone execution"""
    parser = argparse.ArgumentParser(description="Single-cycle RSS news harvester")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of worker threads (default: 4)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--feeds-file", type=str, default=None, help="Path to extra feeds file")
    parser.add_argument("--out-dir", type=str, default="./data/news", help="Output directory")
    parser.add_argument("--request-timeout", type=float, default=10.0, help="Request timeout in seconds")
    parser.add_argument("--cycle-budget", type=float, default=20.0, help="Time budget in seconds")
    
    args = parser.parse_args()
    
    # Load feeds
    feeds = list(dict.fromkeys(DEFAULT_FEEDS + load_extra_feeds(args.feeds_file, args.verbose)))
    
    if not feeds:
        print("No feeds to process", file=sys.stderr)
        return 1
    
    if args.verbose:
        log(f"Processing {len(feeds)} feeds with {args.threads} threads", True)
    
    # Run single harvest cycle
    output_file = harvest_single_cycle(
        feeds=feeds,
        threads=args.threads,
        request_timeout=args.request_timeout,
        cycle_budget=args.cycle_budget,
        out_dir=args.out_dir,
        verbose=args.verbose
    )
    
    if output_file:
        print(output_file)  # Print path for main.py to capture
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())