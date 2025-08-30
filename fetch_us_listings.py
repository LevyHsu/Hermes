#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch all listings for NASDAQ / NYSE / NYSE American from free public sources
(Nasdaq Trader + SEC), enrich with basic tags and CIK, and write to JSON.

Outputs:
  data/us-listings-YYYY-MM-DD.json
  data/us-listings-latest.json

Python 3.9+ required (uses zoneinfo).
"""

import csv
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

# Import centralized configuration
from args import LISTING_DIR as DATA_DIR, SEC_USER_AGENT

# Ensure directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Config
# --------------------
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Timezone (US/Eastern) to decide "today"
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET_TZ = ZoneInfo("America/New_York")

HTTP_TIMEOUT = 30  # seconds

# Mapping for "Exchange" column in otherlisted.txt
EXCH_CODE_MAP = {
    "A": "NYSE American",  # AMEX/NYSE MKT
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

TARGET_EXCHANGES = {"NASDAQ", "NYSE", "NYSE American"}

# Lightweight name-based tags (extend as needed)
TYPE_PATTERNS = [
    ("ADR", r"\bADR\b|\bADS\b|American Depositary", "is_adr"),
    ("ETF", r"\bETF\b", "is_etf_name"),
    ("ETN", r"\bETN\b", "is_etn"),
    ("Unit", r"\bUnit[s]?\b", "is_unit"),
    ("Warrant", r"\bWarrant[s]?\b", "is_warrant"),
    ("Preferred", r"\bPreferred\b|\bPfd\b", "is_preferred"),
]

COMMON_STOCK_HINTS = [
    "Common Stock", "Ordinary Shares", "Class A", "Class B", "Class C", "Shares"
]


# --------------------
# Helpers
# --------------------
def now_et():
    return datetime.now(tz=ET_TZ)


def today_str_et():
    return now_et().strftime("%Y-%m-%d")


def http_get(url, headers=None):
    h = {
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }
    if "sec.gov" in url:
        h.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
    if headers:
        h.update(headers)
    resp = requests.get(url, headers=h, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp


def parse_pipe_table(text):
    """
    Parse Nasdaq Trader pipe-delimited text into (rows, file_creation_time).
    The last line often starts with "File Creation Time".
    """
    rows = []
    fc_time = None
    reader = csv.reader(io.StringIO(text), delimiter="|")
    headers = next(reader)
    headers = [h.strip() for h in headers]
    for parts in reader:
        if not parts or all(p == "" for p in parts):
            continue
        if parts[0].startswith("File Creation Time"):
            # Example: "File Creation Time: 0817202517:03"
            try:
                ts_raw = parts[0].split(":", 1)[1].strip()
                fc_time = ts_raw
            except Exception:
                fc_time = parts[0]
            break
        row = {headers[i]: (parts[i].strip() if i < len(parts) else "") for i in range(len(headers))}
        rows.append(row)
    return rows, fc_time


def load_nasdaq_listed():
    r = http_get(NASDAQ_LISTED_URL)
    rows, fc = parse_pipe_table(r.text)
    out = []
    for row in rows:
        symbol = row.get("Symbol") or row.get("NASDAQ Symbol") or row.get("NASDAQ Official Symbol")
        name = row.get("Security Name", "")
        mcat = row.get("Market Category", "")
        finstat = row.get("Financial Status", "")
        roundlot = row.get("Round Lot Size") or row.get("Round Lot") or ""
        test_issue = (row.get("Test Issue", "").upper() == "Y")
        etf_flag = (row.get("ETF", "").upper() == "Y")
        nextshares = (row.get("NextShares", "").upper() == "Y")

        out.append({
            "ticker": symbol,
            "exchange": "NASDAQ",
            "security_name": name,
            "market_category": mcat or None,
            "listing_tier_desc": {"Q": "Nasdaq Global Select", "G": "Nasdaq Global", "S": "Nasdaq Capital"}.get(mcat, None),
            "financial_status": finstat or None,
            "round_lot_size": int(roundlot) if roundlot.isdigit() else None,
            "is_test_issue": test_issue,
            "is_etf": bool(etf_flag or ("ETF" in (name or "").upper())),
            "is_nextshares": nextshares,
            "act_symbol": symbol,
            "cqs_symbol": symbol,
            "_src": {"file": "nasdaqlisted.txt", "file_creation_time": fc, "url": NASDAQ_LISTED_URL},
        })
    return out, fc


def load_other_listed():
    r = http_get(OTHER_LISTED_URL)
    rows, fc = parse_pipe_table(r.text)
    out = []
    for row in rows:
        act = row.get("ACT Symbol", "").strip()
        cqs = row.get("CQS Symbol", "").strip() or act
        exch_code = row.get("Exchange", "").strip()
        exch_name = EXCH_CODE_MAP.get(exch_code, exch_code)
        name = row.get("Security Name", "")
        roundlot = row.get("Round Lot Size") or ""
        test_issue = (row.get("Test Issue", "").upper() == "Y")
        etf_flag = (row.get("ETF", "").upper() == "Y")
        nasdaq_symbol = row.get("NASDAQ Symbol", "").strip() or None

        out.append({
            "ticker": cqs or act,
            "exchange": exch_name,
            "security_name": name,
            "market_category": None,
            "listing_tier_desc": None,
            "financial_status": None,
            "round_lot_size": int(roundlot) if roundlot.isdigit() else None,
            "is_test_issue": test_issue,
            "is_etf": bool(etf_flag or ("ETF" in (name or "").upper())),
            "is_nextshares": False,
            "act_symbol": act or None,
            "cqs_symbol": cqs or None,
            "nasdaq_symbol": nasdaq_symbol,
            "_src": {"file": "otherlisted.txt", "file_creation_time": fc, "url": OTHER_LISTED_URL},
        })
    return out, fc


def load_sec_company_tickers():
    """
    Returns by_ticker: {
        'AAPL': {'cik': '0000320193', 'company_name_sec': 'Apple Inc.', 'sec_exchange': 'Nasdaq'}
    }
    """
    try:
        r = http_get(SEC_COMPANY_TICKERS_URL)
        data = r.json()
    except Exception as e:
        print(f"[WARN] SEC company_tickers.json failed: {e}. Continuing without SEC enrichment.", file=sys.stderr)
        return {}

    # Accept multiple possible shapes
    if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        fields = data.get("fields")
        dat = data.get("data")
        items = [dict(zip(fields, row)) for row in dat] if fields and dat else []

    by_ticker = {}
    for it in items:
        tkr = (it.get("ticker") or it.get("Ticker") or "").strip().upper()
        if not tkr:
            continue
        cik_val = it.get("cik_str") or it.get("CIK") or ""
        try:
            cik_str = str(int(cik_val)).zfill(10) if str(cik_val).strip() != "" else None
        except Exception:
            cik_str = None
        by_ticker[tkr] = {
            "cik": cik_str,
            "company_name_sec": it.get("title") or it.get("Title") or None,
            "sec_exchange": it.get("exchange") or it.get("Exchange") or None,
        }
    return by_ticker


def enrich_types(rec):
    name_u = (rec.get("security_name") or "").upper()
    for (_, pat, key) in TYPE_PATTERNS:
        if re.search(pat, name_u, flags=re.IGNORECASE):
            rec[key] = True
    rec["is_common_stock_like"] = any(hint.upper() in name_u for hint in COMMON_STOCK_HINTS)
    rec["is_adr"] = bool(rec.get("is_adr"))
    rec["is_etf"] = bool(rec.get("is_etf") or rec.get("is_etf_name"))
    if rec["is_etf"]:
        rec["security_type"] = "ETF"
    elif rec.get("is_etn"):
        rec["security_type"] = "ETN"
    elif rec.get("is_warrant"):
        rec["security_type"] = "WARRANT"
    elif rec.get("is_unit"):
        rec["security_type"] = "UNIT"
    elif rec.get("is_preferred"):
        rec["security_type"] = "PREFERRED"
    elif rec["is_adr"]:
        rec["security_type"] = "ADR"
    elif rec["is_common_stock_like"]:
        rec["security_type"] = "COMMON_LIKE"
    else:
        rec["security_type"] = None
    return rec


def build_record(r, sec_map):
    tkr_u = (r.get("ticker") or "").upper()
    sec_info = sec_map.get(tkr_u, {})
    r["cik"] = sec_info.get("cik")
    r["company_name_sec"] = sec_info.get("company_name_sec")
    r["sec_exchange"] = sec_info.get("sec_exchange")

    r = enrich_types(r)

    return {
        "ticker": r["ticker"],
        "exchange": r["exchange"],
        "security_name": r.get("security_name"),
        "market_category": r.get("market_category"),
        "listing_tier_desc": r.get("listing_tier_desc"),
        "round_lot_size": r.get("round_lot_size"),
        "financial_status": r.get("financial_status"),
        "is_etf": r.get("is_etf"),
        "is_adr": r.get("is_adr"),
        "security_type": r.get("security_type"),
        "act_symbol": r.get("act_symbol"),
        "cqs_symbol": r.get("cqs_symbol"),
        "nasdaq_symbol": r.get("nasdaq_symbol"),
        "cik": r.get("cik"),
        "company_name_sec": r.get("company_name_sec"),
        "sec_exchange": r.get("sec_exchange"),
        "asof": now_et().isoformat(),
        "source": {
            "nasdaqtrader_file_times": r.get("_src", {}),
        },
    }


def run(force=False, out_dir=DATA_DIR):
    today_et = today_str_et()
    outfile = out_dir / f"us-listings-{today_et}.json"
    latest = out_dir / "us-listings-latest.json"

    if outfile.exists() and not force:
        print(f"[OK] {outfile} already exists (US/Eastern {today_et}); nothing to do.")
        return 0

    print("[*] Downloading Nasdaq Trader symbol directories...")
    nasdaq_rows, nas_fc = load_nasdaq_listed()
    other_rows, other_fc = load_other_listed()

    # Stamp their file times into the rows for provenance
    for r in nasdaq_rows:
        r["_src"]["nasdaqlisted.txt"] = nas_fc
    for r in other_rows:
        r["_src"]["otherlisted.txt"] = other_fc

    print("[*] Downloading SEC company_tickers.json...")
    sec_map = load_sec_company_tickers()

    print("[*] Merging and filtering (NASDAQ / NYSE / NYSE American; excluding test issues)...")
    all_rows = nasdaq_rows + other_rows

    merged = []
    for r in all_rows:
        if r.get("is_test_issue"):
            continue
        exch = r.get("exchange")
        if exch not in TARGET_EXCHANGES:
            continue
        merged.append(build_record(r, sec_map))

    # Save files
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Quick stats
    by_exch = {}
    for rec in merged:
        by_exch[rec["exchange"]] = by_exch.get(rec["exchange"], 0) + 1

    print(f"[OK] Wrote {outfile} with {len(merged)} records; also updated {latest}")
    for k, v in sorted(by_exch.items()):
        print(f"    {k:14s}: {v}")
    return 0


def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(description="Fetch US listings (NASDAQ/NYSE/NYSE American) to JSON.")
    p.add_argument("--force", action="store_true", help="Re-download even if today's file exists.")
    p.add_argument("--outdir", type=Path, default=DATA_DIR, help="Output directory (default: ./data)")
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    try:
        return run(force=args.force, out_dir=args.outdir)
    except requests.HTTPError as e:
        print(f"[HTTPError] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
