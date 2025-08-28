#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py — Verify predictions from signals_*.jsonl.

Features
- Finds all JSONL files under data/trade-log (latest -> oldest) and pulls signals
  with revised_confidence above configurable thresholds (separate for BUY/SELL).
- Runs one worker per ticker (threaded by default) to:
  * Fetch current price from multiple free sources (Yahoo Finance + Stooq).
  * Fetch historical bars from Yahoo Finance to estimate entry price at the
    signal timestamp and to compute MFE/MAE since entry.
  * Evaluate expected_price (when present and >= $2) and comment on accuracy.
- Prints a concise report and writes a JSONL summary to
  data/trade-log/verification_YYYYMMDD_HHMMSS.jsonl

Notes
- Uses only Python stdlib networking (urllib) for portability.
- Timezone handling assumes timestamps in the JSONL are local Adelaide time if
  naive (no timezone). Adjust with --tz if needed.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import glob
import json
import math
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib import request, parse, error

try:
    from zoneinfo import ZoneInfo  # Py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# --------------------------- Config & CLI ---------------------------

DEFAULT_TZ = "Australia/Adelaide"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={sym}"
YAHOO_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval={interval}&events=div%2Csplit"
)
STOOQ_CSV_URL = (
    "https://stooq.com/q/l/?s={sym}&f=sd2t2ohlcv&h&e=csv"
)  # daily OHLC, last close (not intraday)

# --------------------------- Models ---------------------------

@dataclass
class Signal:
    file_path: str
    file_mtime: float
    line_no: int
    timestamp: dt.datetime  # aware (UTC)
    action: str  # BUY/SELL
    ticker: str
    revised_confidence: float
    expected_price: Optional[float]
    raw: dict

@dataclass
class SourcePrice:
    source: str
    price: Optional[float]
    as_of: Optional[int]  # epoch seconds (UTC) if known
    note: str = ""

@dataclass
class Verification:
    ticker: str
    signal_time_utc: int
    action: str
    revised_confidence: float
    expected_price: Optional[float]
    expected_comment: Optional[str]
    entry_price: Optional[float]
    entry_bar_time: Optional[int]
    current_price: Optional[float]
    current_prices_by_source: Dict[str, float]
    pnl_abs: Optional[float]
    pnl_pct: Optional[float]
    max_win_abs: Optional[float]
    max_win_pct: Optional[float]
    max_win_time: Optional[int]
    max_loss_abs: Optional[float]
    max_loss_pct: Optional[float]
    max_loss_time: Optional[int]
    interval_used: Optional[str]

# --------------------------- HTTP Helpers ---------------------------

def http_get(url: str, timeout: float = 8.0, headers: Optional[Dict[str, str]] = None) -> bytes:
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }
    if headers:
        hdrs.update(headers)
    req = request.Request(url, headers=hdrs)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except error.HTTPError as e:
        # Some Yahoo endpoints occasionally want cookies; retry once without UA
        if 400 <= e.code < 500:
            req = request.Request(url)
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        raise

# --------------------------- Data Utils ---------------------------

def to_aware_local(dt_str: str, tz_name: str) -> dt.datetime:
    tz = ZoneInfo(tz_name) if ZoneInfo else None
    try:
        d = dt.datetime.fromisoformat(dt_str)
    except Exception:
        # Try without microseconds
        d = dt.datetime.strptime(dt_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    if d.tzinfo is None:
        if tz is None:
            raise RuntimeError("Python lacks zoneinfo; provide timezone-aware timestamps or install Python>=3.9")
        d = d.replace(tzinfo=tz)
    return d


def to_utc_epoch_seconds(d: dt.datetime) -> int:
    if d.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(d.astimezone(dt.timezone.utc).timestamp())


def round_safe(x: Optional[float], n: int = 4) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(x, n)

# --------------------------- Source Fetchers ---------------------------

def yahoo_quote_price(symbol: str) -> SourcePrice:
    url = YAHOO_QUOTE_URL.format(sym=parse.quote(symbol))
    try:
        data = json.loads(http_get(url))
        quote = (data.get("quoteResponse", {}) or {}).get("result", [{}])[0]
        price = quote.get("regularMarketPrice")
        as_of = quote.get("regularMarketTime")  # epoch seconds
        return SourcePrice("yahoo_quote", float(price) if price is not None else None, int(as_of) if as_of else None)
    except Exception as e:  # pragma: no cover
        return SourcePrice("yahoo_quote", None, None, note=f"err={e}")


def stooq_last_close(symbol: str) -> SourcePrice:
    # Stooq expects .us suffix for US tickers. Rough heuristic.
    stooq_sym = symbol.lower()
    if "." not in stooq_sym:
        stooq_sym = f"{stooq_sym}.us"
    url = STOOQ_CSV_URL.format(sym=parse.quote(stooq_sym))
    try:
        raw = http_get(url).decode("utf-8", errors="ignore")
        # CSV has headers: Symbol,Date,Time,Open,High,Low,Close,Volume
        reader = csv.DictReader(raw.splitlines())
        row = next(reader, None)
        if not row:
            return SourcePrice("stooq", None, None, note="no data")
        close = row.get("Close")
        date_str = row.get("Date")
        time_str = row.get("Time")
        # Stooq time is exchange local EOD; treat as naive
        as_of = None
        try:
            if date_str:
                if time_str and time_str != "":
                    as_of = int(dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
                else:
                    as_of = int(dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
        except Exception:
            as_of = None
        price = float(close) if close not in (None, "-", "") else None
        return SourcePrice("stooq_close", price, as_of, note="EOD close; not intraday")
    except Exception as e:  # pragma: no cover
        return SourcePrice("stooq_close", None, None, note=f"err={e}")


def choose_interval(seconds_span: int) -> str:
    # Yahoo intraday windows:
    # 1m up to ~7d, 2m up to ~60d, 5m/15m beyond; otherwise 1d.
    days = seconds_span / 86400.0
    if days <= 7:
        return "1m"
    if days <= 30:
        return "5m"
    if days <= 60:
        return "15m"
    if days <= 365:
        return "1d"
    return "1d"


def yahoo_bars(symbol: str, start_ts: int, end_ts: int, interval: Optional[str] = None) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    if end_ts <= start_ts:
        end_ts = start_ts + 3600
    if interval is None:
        interval = choose_interval(end_ts - start_ts)
    url = YAHOO_CHART_URL.format(sym=parse.quote(symbol), p1=start_ts, p2=end_ts, interval=parse.quote(interval))
    data = json.loads(http_get(url))
    result = (data.get("chart", {}) or {}).get("result", [])
    if not result:
        return [], [], [], [], []
    r0 = result[0]
    timestamps = r0.get("timestamp") or []
    q = (r0.get("indicators", {}) or {}).get("quote", [{}])[0]
    opens = q.get("open") or []
    highs = q.get("high") or []
    lows = q.get("low") or []
    closes = q.get("close") or []
    # Clean None values by forward/backward fill where possible
    def _clean(vals: List[Optional[float]]) -> List[float]:
        out: List[float] = []
        last = None
        for v in vals:
            if v is None:
                out.append(last if last is not None else math.nan)
            else:
                last = v
                out.append(v)
        # Replace leading NaNs with first non-nan
        first = next((x for x in out if not math.isnan(x)), math.nan)
        out = [first if math.isnan(x) else x for x in out]
        return out

    return (
        [int(t) for t in timestamps],
        _clean(opens),
        _clean(highs),
        _clean(lows),
        _clean(closes),
    )

# --------------------------- Core Logic ---------------------------

def median_price(sources: List[SourcePrice]) -> Optional[float]:
    vals = [s.price for s in sources if s.price is not None and not math.isnan(s.price)]
    if not vals:
        return None
    try:
        return float(statistics.median(vals))
    except Exception:
        return float(vals[0])


def compute_entry_and_excursions(symbol: str, entry_ts: int) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[float], Optional[int], Optional[float], Optional[float], Optional[int], str]:
    # Fetch bars from 30 minutes before entry to now
    now_ts = int(dt.datetime.now(dt.timezone.utc).timestamp())
    start_ts = max(entry_ts - 30 * 60, entry_ts - 2 * 3600)
    interval = choose_interval(now_ts - start_ts)
    ts, o, h, l, c = yahoo_bars(symbol, start_ts, now_ts, interval)
    if not ts:
        return (None, None, None, None, None, None, None, None, interval)
    # Find index at/after entry
    idx = None
    for i, t in enumerate(ts):
        if t >= entry_ts:
            idx = i
            break
    if idx is None:
        idx = len(ts) - 1
    # Estimate entry price as close at idx (fallback to open)
    entry = c[idx] if not math.isnan(c[idx]) else (o[idx] if not math.isnan(o[idx]) else None)
    entry_time = ts[idx]
    # Compute MFE/MAE since entry (inclusive)
    highs = [x for x in h[idx:] if not math.isnan(x)]
    lows = [x for x in l[idx:] if not math.isnan(x)]
    if not highs or not lows or entry is None:
        return (entry, entry_time, None, None, None, None, None, None, interval)
    max_high = max(highs)
    min_low = min(lows)
    # Times of extremes
    try:
        max_i = h[idx:].index(max_high)
        max_time = ts[idx + max_i]
    except Exception:
        max_time = None
    try:
        min_i = l[idx:].index(min_low)
        min_time = ts[idx + min_i]
    except Exception:
        min_time = None
    # For BUY: MFE = max_high - entry; MAE = entry - min_low
    # SELL is handled by sign downstream
    return (entry, entry_time, max_high, min_low, max_time, None, None, min_time, interval)


def pnl_for_action(action: str, entry: float, current: float) -> Tuple[float, float]:
    if action.upper() == "BUY":
        diff = current - entry
    else:  # SELL
        diff = entry - current
    pct = diff / entry * 100.0 if entry else math.nan
    return diff, pct


def mfe_mae_for_action(action: str, entry: float, max_high: Optional[float], min_low: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if max_high is None or min_low is None or entry is None:
        return None, None
    if action.upper() == "BUY":
        mfe = max_high - entry
        mae = entry - min_low
    else:  # SELL
        mfe = entry - min_low
        mae = max_high - entry
    return mfe, mae


def comment_expected_price(expected: float, current: float, max_high: Optional[float], min_low: Optional[float]) -> str:
    if expected is None or expected < 2:
        return "(ignored)"
    diff_pct = (current - expected) / expected * 100.0 if expected else math.nan
    band = abs(diff_pct)
    if (min_low is not None and expected < min_low) or (max_high is not None and expected > max_high):
        range_note = "outside-seen-range"
    elif (min_low is not None and expected <= max_high and expected >= min_low):
        range_note = "hit-within-range"
    else:
        range_note = "unknown-range"
    if band <= 2:
        qual = "accurate (≤2%)"
    elif band <= 5:
        qual = "close (≤5%)"
    elif band <= 10:
        qual = "loose (≤10%)"
    else:
        qual = f"off by {round(band, 2)}%"
    return f"{qual}; {range_note}"

# --------------------------- Loading & Filtering ---------------------------

def load_signals(base_dir: str, tz_name: str, buy_thr: float, sell_thr: float) -> Dict[str, List[Signal]]:
    paths = sorted(glob.glob(os.path.join(base_dir, "signals_*.jsonl")), key=os.path.getmtime, reverse=True)
    by_ticker: Dict[str, List[Signal]] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    action = (obj.get("action") or obj.get("signal") or "").upper()
                    if action not in ("BUY", "SELL"):
                        continue
                    conf = float(obj.get("revised_confidence", obj.get("confidence", 0)))
                    thr = buy_thr if action == "BUY" else sell_thr
                    if conf < thr:
                        continue
                    ts_str = obj.get("timestamp")
                    if not ts_str:
                        continue
                    try:
                        local_dt = to_aware_local(ts_str, tz_name)
                    except Exception:
                        continue
                    utc_dt = local_dt.astimezone(dt.timezone.utc)
                    ticker = str(obj.get("ticker") or obj.get("symbol") or "").strip().upper()
                    if not ticker:
                        continue
                    expected_price = None
                    ep = obj.get("expected_price")
                    if ep is not None:
                        try:
                            expected_price = float(ep)
                        except Exception:
                            expected_price = None
                    s = Signal(
                        file_path=p,
                        file_mtime=os.path.getmtime(p),
                        line_no=i,
                        timestamp=utc_dt,
                        action=action,
                        ticker=ticker,
                        revised_confidence=conf,
                        expected_price=expected_price,
                        raw=obj,
                    )
                    by_ticker.setdefault(ticker, []).append(s)
        except FileNotFoundError:
            continue
    # Sort signals per ticker by time desc
    for t in by_ticker:
        by_ticker[t].sort(key=lambda s: s.timestamp, reverse=True)
    return by_ticker

# --------------------------- Worker ---------------------------

def process_ticker(ticker: str, signals: List[Signal], now_utc: int) -> List[Verification]:
    # Fetch multi-source current price once per ticker
    sources = [yahoo_quote_price(ticker), stooq_last_close(ticker)]
    current = median_price(sources)
    prices_by_source = {s.source: s.price for s in sources if s.price is not None}

    verifications: List[Verification] = []
    for s in signals:
        entry, entry_time, max_high, min_low, max_time, _x1, _x2, min_time, interval = compute_entry_and_excursions(
            ticker, to_utc_epoch_seconds(s.timestamp)
        )
        if entry is None or current is None:
            pnl_abs = pnl_pct = None
            mfe = mae = None
        else:
            pnl_abs, pnl_pct = pnl_for_action(s.action, entry, current)
            mfe, mae = mfe_mae_for_action(s.action, entry, max_high, min_low)
        expected_comment = None
        if s.expected_price is not None and s.expected_price >= 2 and current is not None:
            expected_comment = comment_expected_price(s.expected_price, current, max_high, min_low)
        v = Verification(
            ticker=ticker,
            signal_time_utc=to_utc_epoch_seconds(s.timestamp),
            action=s.action,
            revised_confidence=s.revised_confidence,
            expected_price=round_safe(s.expected_price, 4) if s.expected_price is not None else None,
            expected_comment=expected_comment,
            entry_price=round_safe(entry, 4),
            entry_bar_time=entry_time,
            current_price=round_safe(current, 4),
            current_prices_by_source={k: round_safe(v, 4) for k, v in prices_by_source.items()},
            pnl_abs=round_safe(pnl_abs, 4) if pnl_abs is not None else None,
            pnl_pct=round_safe(pnl_pct, 4) if pnl_pct is not None else None,
            max_win_abs=round_safe(mfe, 4) if mfe is not None else None,
            max_win_pct=round_safe((mfe / entry * 100.0) if (mfe is not None and entry) else None, 4),
            max_win_time=max_time,
            max_loss_abs=round_safe(mae, 4) if mae is not None else None,
            max_loss_pct=round_safe((mae / entry * 100.0) if (mae is not None and entry) else None, 4),
            max_loss_time=min_time,
            interval_used=interval,
        )
        verifications.append(v)
    return verifications

# --------------------------- Reporting ---------------------------

def fmt_epoch(ts: Optional[int]) -> str:
    if not ts:
        return "-"
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_report(ticker: str, verifications: List[Verification]) -> None:
    if not verifications:
        return
    print(f"\n=== {ticker} ===")
    for v in verifications:
        print(
            f"{fmt_epoch(v.signal_time_utc)} | {v.action} | conf={v.revised_confidence:.1f} | "
            f"entry={v.entry_price} (bar {fmt_epoch(v.entry_bar_time)}) | current={v.current_price} "
            f"sources={v.current_prices_by_source}"
        )
        print(
            f" -> PnL now: {v.pnl_abs} ({v.pnl_pct}%) | "
            f"MaxWin: {v.max_win_abs} ({v.max_win_pct}%) at {fmt_epoch(v.max_win_time)} | "
            f"MaxLoss: {v.max_loss_abs} ({v.max_loss_pct}%) at {fmt_epoch(v.max_loss_time)} | interval={v.interval_used}"
        )
        if v.expected_price is not None:
            print(f" -> expected_price={v.expected_price} comment={v.expected_comment}")


def write_jsonl(out_path: str, ticker: str, verifications: List[Verification]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for v in verifications:
            f.write(json.dumps(dataclasses.asdict(v), ensure_ascii=False) + "\n")

# --------------------------- Main ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify predictions from signals JSONL logs")
    parser.add_argument("--dir", default="data/trade-log", help="Directory containing signals_*.jsonl")
    parser.add_argument("--buy-threshold", type=float, default=80.0, help="Min revised_confidence for BUY")
    parser.add_argument("--sell-threshold", type=float, default=80.0, help="Min revised_confidence for SELL")
    parser.add_argument("--tz", default=DEFAULT_TZ, help="Timezone for naive timestamps in logs")
    parser.add_argument("--max-workers", type=int, default=min(8, (os.cpu_count() or 4)), help="Worker threads")
    parser.add_argument("--out", default=None, help="Optional JSONL output path (default auto under data/trade-log)")
    args = parser.parse_args()

    now_utc = int(dt.datetime.now(dt.timezone.utc).timestamp())
    by_ticker = load_signals(args.dir, args.tz, args.buy_threshold, args.sell_threshold)

    if not by_ticker:
        print("No qualifying signals found.")
        return

    ts_label = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(args.dir, f"verification_{ts_label}.jsonl")

    # One thread per ticker
    print(f"Verifying {len(by_ticker)} tickers with up to {args.max_workers} workers ...")
    futures = {}
    results: Dict[str, List[Verification]] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for ticker, signals in by_ticker.items():
            futures[ex.submit(process_ticker, ticker, signals, now_utc)] = ticker
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                verifications = fut.result()
            except Exception as e:  # pragma: no cover
                print(f"[ERROR] {ticker}: {e}")
                continue
            with lock:
                results[ticker] = verifications
                print_report(ticker, verifications)
                write_jsonl(out_path, ticker, verifications)

    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
