#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py — Verify refined trading signals from signals_*.jsonl files using Stooq data

Features:
- Scans data/trade-log/ for all signals_*.jsonl files (or TRADE_LOG.jsonl fallback)
- Loads signals with revised_confidence >= REVISED_CONFIDENCE_THRESHOLD (70%)
- Only verifies enriched signals (those that went through 2-stage LLM processing)
- Fetches current and historical prices from Stooq (free, no API key required)
- Calculates PnL, maximum favorable/adverse excursions
- Evaluates expected_price accuracy when present
- Outputs readable reports and verification summary
"""

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request, parse, error

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore

# Import configuration from args.py
from args import (
    REVISED_CONFIDENCE_THRESHOLD,
    VERIFY_TIMEZONE,
    STOOQ_MAX_CONCURRENCY,
    STOOQ_TIMEOUT,
    VERIFY_MAX_WORKERS,
    VERIFY_OUTPUT_DIR
)

# --------------------------- Models ---------------------------

@dataclass
class Signal:
    """Trading signal from log file"""
    timestamp: dt.datetime  # UTC aware
    action: str  # BUY/SELL
    ticker: str
    confidence: float
    revised_confidence: float
    expected_price: Optional[float]
    reason: Optional[str]
    refined_reason: Optional[str]

@dataclass
class PriceData:
    """Price data from Stooq"""
    current_price: Optional[float]
    last_close: Optional[float]
    as_of_date: Optional[str]
    daily_bars: Optional[Dict]  # Contains dates, opens, highs, lows, closes

@dataclass
class Verification:
    """Verification result for a signal"""
    ticker: str
    signal_time: str
    action: str
    confidence: float
    revised_confidence: float
    expected_price: Optional[float]
    current_price: Optional[float]
    entry_price: Optional[float]
    pnl_dollars: Optional[float]
    pnl_percent: Optional[float]
    max_favorable: Optional[float]  # MFE
    max_adverse: Optional[float]  # MAE
    expected_accuracy: Optional[str]
    data_source: str

# --------------------------- Stooq Data Fetcher ---------------------------

class StooqFetcher:
    """Fetches price data from Stooq"""
    
    CURRENT_URL = "https://stooq.com/q/l/?s={sym}&f=sd2t2ohlcv&h&e=csv"
    DAILY_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"
    
    def __init__(self):
        self.semaphore = threading.Semaphore(STOOQ_MAX_CONCURRENCY)
    
    def _http_get(self, url: str) -> bytes:
        """HTTP GET with rate limiting and retries"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Connection": "close"
        }
        
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            with self.semaphore:
                try:
                    req = request.Request(url, headers=headers)
                    with request.urlopen(req, timeout=STOOQ_TIMEOUT) as resp:
                        return resp.read()
                except (error.HTTPError, error.URLError) as e:
                    if attempt == max_attempts:
                        raise
                    time.sleep(0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2))
        
        raise RuntimeError(f"Failed to fetch: {url}")
    
    def _format_symbol(self, ticker: str) -> str:
        """Format ticker for Stooq (adds .us suffix if needed)"""
        ticker = ticker.lower()
        if "." not in ticker:
            ticker = f"{ticker}.us"
        return ticker
    
    def fetch_current(self, ticker: str) -> Tuple[Optional[float], Optional[str]]:
        """Fetch current/last close price"""
        sym = self._format_symbol(ticker)
        url = self.CURRENT_URL.format(sym=parse.quote(sym))
        
        try:
            raw = self._http_get(url).decode("utf-8", errors="ignore")
            reader = csv.DictReader(raw.splitlines())
            row = next(reader, None)
            
            if not row:
                return None, None
            
            close = row.get("Close")
            date = row.get("Date")
            
            if close and close not in ("-", ""):
                return float(close), date
            
            return None, None
            
        except Exception:
            return None, None
    
    def fetch_daily_bars(self, ticker: str, days_back: int = 30) -> Optional[Dict]:
        """Fetch daily OHLC bars"""
        sym = self._format_symbol(ticker)
        url = self.DAILY_URL.format(sym=parse.quote(sym))
        
        try:
            raw = self._http_get(url).decode("utf-8", errors="ignore")
            reader = csv.DictReader(raw.splitlines())
            
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            
            for row in reader:
                date = row.get("Date")
                if not date or date == "-":
                    continue
                
                dates.append(date)
                opens.append(float(row.get("Open", "nan")))
                highs.append(float(row.get("High", "nan")))
                lows.append(float(row.get("Low", "nan")))
                closes.append(float(row.get("Close", "nan")))
            
            if not dates:
                return None
            
            # Return last N days
            n = min(days_back, len(dates))
            return {
                "dates": dates[-n:],
                "opens": opens[-n:],
                "highs": highs[-n:],
                "lows": lows[-n:],
                "closes": closes[-n:]
            }
            
        except Exception:
            return None

# --------------------------- Analysis Functions ---------------------------

def calculate_pnl(action: str, entry: float, current: float) -> Tuple[float, float]:
    """Calculate profit/loss for a position"""
    if action.upper() == "BUY":
        dollars = current - entry
    else:  # SELL
        dollars = entry - current
    
    percent = (dollars / entry * 100.0) if entry else 0.0
    return round(dollars, 2), round(percent, 2)

def calculate_mfe_mae(action: str, entry: float, highs: List[float], lows: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate Maximum Favorable/Adverse Excursions"""
    if not highs or not lows:
        return None, None
    
    valid_highs = [h for h in highs if not math.isnan(h)]
    valid_lows = [l for l in lows if not math.isnan(l)]
    
    if not valid_highs or not valid_lows:
        return None, None
    
    max_high = max(valid_highs)
    min_low = min(valid_lows)
    
    if action.upper() == "BUY":
        mfe = max_high - entry  # Max profit
        mae = entry - min_low   # Max loss
    else:  # SELL
        mfe = entry - min_low   # Max profit
        mae = max_high - entry  # Max loss
    
    return round(mfe, 2) if mfe else None, round(mae, 2) if mae else None

def evaluate_expected_price(expected: float, current: float, highs: List[float], lows: List[float]) -> str:
    """Evaluate accuracy of expected price prediction"""
    if not expected or expected < 2 or not current:
        return "N/A"
    
    diff_pct = abs((current - expected) / expected * 100.0)
    
    # Check if price was hit historically
    hit = False
    if highs and lows:
        valid_highs = [h for h in highs if not math.isnan(h)]
        valid_lows = [l for l in lows if not math.isnan(l)]
        if valid_highs and valid_lows:
            max_high = max(valid_highs)
            min_low = min(valid_lows)
            hit = min_low <= expected <= max_high
    
    # Accuracy rating
    if diff_pct <= 2:
        accuracy = "✓ Excellent (<2%)"
    elif diff_pct <= 5:
        accuracy = "✓ Good (<5%)"
    elif diff_pct <= 10:
        accuracy = "~ Fair (<10%)"
    else:
        accuracy = f"✗ Off by {diff_pct:.1f}%"
    
    if hit:
        accuracy += " [HIT]"
    
    return accuracy

# --------------------------- Signal Loading ---------------------------

def load_signals_from_file(log_path: Path, threshold: float) -> List[Signal]:
    """Load and filter signals from a single JSONL file based on revised_confidence"""
    signals = []
    
    if not log_path.exists():
        return signals
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Extract fields
                action = data.get("action", "").upper()
                if action not in ("BUY", "SELL"):
                    continue
                
                confidence = float(data.get("confidence", 0))
                revised = float(data.get("revised_confidence", 0))
                
                # Only verify signals with revised_confidence above threshold
                # If no revised_confidence, skip (we only care about refined signals)
                if revised < threshold:
                    continue
                
                # Parse timestamp
                timestamp_str = data.get("timestamp")
                if not timestamp_str:
                    continue
                
                # Handle timezone
                try:
                    if ZoneInfo:
                        tz = ZoneInfo(VERIFY_TIMEZONE)
                        timestamp = dt.datetime.fromisoformat(timestamp_str)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=tz)
                    else:
                        timestamp = dt.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except Exception:
                    continue
                
                ticker = data.get("ticker", "").upper()
                if not ticker:
                    continue
                
                expected = data.get("expected_price")
                if expected:
                    try:
                        expected = float(expected)
                    except (ValueError, TypeError):
                        expected = None
                
                signal = Signal(
                    timestamp=timestamp,
                    action=action,
                    ticker=ticker,
                    confidence=confidence,
                    revised_confidence=revised,
                    expected_price=expected,
                    reason=data.get("reason"),
                    refined_reason=data.get("refined_reason")
                )
                
                signals.append(signal)
                
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    
    return signals

def load_all_signals(trade_log_dir: Path, threshold: float) -> List[Signal]:
    """Load signals from all signals_*.jsonl files in the trade-log directory"""
    all_signals = []
    
    # Find all signals_*.jsonl files
    signal_files = sorted(trade_log_dir.glob("signals_*.jsonl"))
    
    if not signal_files:
        # Fallback to TRADE_LOG.jsonl if no signals_ files found
        trade_log = trade_log_dir / "TRADE_LOG.jsonl"
        if trade_log.exists():
            print(f"📂 Loading from: {trade_log}")
            return load_signals_from_file(trade_log, threshold)
        return []
    
    print(f"📂 Found {len(signal_files)} signal files in {trade_log_dir}")
    
    # Load from all files
    for file_path in signal_files:
        file_signals = load_signals_from_file(file_path, threshold)
        if file_signals:
            print(f"  ✓ {file_path.name}: {len(file_signals)} signals")
            all_signals.extend(file_signals)
        else:
            print(f"  - {file_path.name}: no qualifying signals")
    
    # Sort all signals by timestamp (newest first)
    all_signals.sort(key=lambda s: s.timestamp, reverse=True)
    
    return all_signals

# --------------------------- Verification Worker ---------------------------

def verify_signal(signal: Signal, fetcher: StooqFetcher) -> Verification:
    """Verify a single signal"""
    
    # Fetch current price
    current_price, as_of_date = fetcher.fetch_current(signal.ticker)
    
    # Fetch historical bars
    bars = fetcher.fetch_daily_bars(signal.ticker, days_back=30)
    
    entry_price = None
    mfe = None
    mae = None
    
    if bars and bars["closes"]:
        # Use last close as entry (simplified - could match by date)
        entry_price = bars["closes"][-1] if not math.isnan(bars["closes"][-1]) else None
        
        # Calculate MFE/MAE
        if entry_price:
            mfe, mae = calculate_mfe_mae(signal.action, entry_price, bars["highs"], bars["lows"])
    
    # Calculate PnL
    pnl_dollars = None
    pnl_percent = None
    if entry_price and current_price:
        pnl_dollars, pnl_percent = calculate_pnl(signal.action, entry_price, current_price)
    
    # Evaluate expected price
    expected_accuracy = None
    if signal.expected_price and current_price and bars:
        expected_accuracy = evaluate_expected_price(
            signal.expected_price, 
            current_price,
            bars.get("highs", []),
            bars.get("lows", [])
        )
    
    return Verification(
        ticker=signal.ticker,
        signal_time=signal.timestamp.strftime("%Y-%m-%d %H:%M"),
        action=signal.action,
        confidence=signal.confidence,
        revised_confidence=signal.revised_confidence,
        expected_price=signal.expected_price,
        current_price=current_price,
        entry_price=entry_price,
        pnl_dollars=pnl_dollars,
        pnl_percent=pnl_percent,
        max_favorable=mfe,
        max_adverse=mae,
        expected_accuracy=expected_accuracy,
        data_source="Stooq"
    )

# --------------------------- Reporting ---------------------------

def print_summary(verifications: List[Verification]):
    """Print organized summary report"""
    
    if not verifications:
        print("\n📊 No signals to verify")
        return
    
    print("\n" + "=" * 80)
    print("📊 HERMES SIGNAL VERIFICATION REPORT")
    print("=" * 80)
    
    # Group by ticker
    by_ticker = {}
    for v in verifications:
        by_ticker.setdefault(v.ticker, []).append(v)
    
    # Statistics
    total = len(verifications)
    with_pnl = sum(1 for v in verifications if v.pnl_percent is not None)
    profitable = sum(1 for v in verifications if v.pnl_percent and v.pnl_percent > 0)
    
    print(f"\n📈 Summary Statistics:")
    print(f"  • Total Refined Signals: {total}")
    print(f"  • Signals with PnL: {with_pnl}")
    if with_pnl > 0:
        win_rate = (profitable / with_pnl) * 100
        print(f"  • Win Rate: {win_rate:.1f}% ({profitable}/{with_pnl})")
    
    # Detailed results by ticker
    for ticker in sorted(by_ticker.keys()):
        signals = by_ticker[ticker]
        print(f"\n{'─' * 60}")
        print(f"🎯 {ticker} ({len(signals)} signal{'s' if len(signals) > 1 else ''})")
        print(f"{'─' * 60}")
        
        for v in signals:
            # Signal info - show revised confidence (all should have it)
            conf_str = f"{v.revised_confidence:.0f}% (refined)"
            
            print(f"\n  📅 {v.signal_time} | {v.action} | Confidence: {conf_str}")
            
            # Price info
            if v.current_price:
                print(f"  💰 Current: ${v.current_price:.2f}", end="")
                if v.entry_price:
                    print(f" | Entry: ${v.entry_price:.2f}", end="")
                print()
            else:
                print(f"  ⚠️  No current price data available")
            
            # PnL
            if v.pnl_percent is not None:
                emoji = "🟢" if v.pnl_percent > 0 else "🔴" if v.pnl_percent < 0 else "⚪"
                sign = "+" if v.pnl_dollars > 0 else ""
                print(f"  {emoji} PnL: {sign}${v.pnl_dollars:.2f} ({sign}{v.pnl_percent:.2f}%)")
            
            # MFE/MAE
            if v.max_favorable is not None:
                print(f"  📊 Max Favorable: ${v.max_favorable:.2f} | Max Adverse: ${v.max_adverse:.2f}")
            
            # Expected price accuracy
            if v.expected_price and v.expected_accuracy != "N/A":
                print(f"  🎯 Expected: ${v.expected_price:.2f} → {v.expected_accuracy}")
    
    print(f"\n{'=' * 80}")
    print(f"Data source: {verifications[0].data_source if verifications else 'Stooq'}")
    print(f"Report generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def save_verification(verifications: List[Verification], output_dir: Path):
    """Save verification results to JSON file"""
    
    if not verifications:
        return
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"verification_{timestamp}.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = [dataclasses.asdict(v) for v in verifications]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")

# --------------------------- Main ---------------------------

def main():
    """Main verification function"""
    
    parser = argparse.ArgumentParser(
        description="Verify Hermes trading signals using Stooq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all signals_*.jsonl files with default threshold (>= 70%)
  python verify.py
  
  # Verify with custom threshold (two equivalent ways)
  python verify.py --threshold 60
  python verify.py --revised-confidence 60
  
  # Use more workers for faster processing
  python verify.py --max-workers 8
  
  # Specify custom directory
  python verify.py --trade-log-dir data/trade-log
        """
    )
    
    parser.add_argument(
        "--trade-log-dir",
        type=Path,
        default=VERIFY_OUTPUT_DIR,
        help=f"Directory containing signals_*.jsonl files (default: {VERIFY_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=REVISED_CONFIDENCE_THRESHOLD,
        help=f"Minimum revised_confidence for signals (default: {REVISED_CONFIDENCE_THRESHOLD}%)"
    )
    parser.add_argument(
        "--revised-confidence",
        type=float,
        dest="threshold",  # Maps to the same destination as --threshold
        help=f"Minimum revised_confidence for signals (alias for --threshold)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=VERIFY_MAX_WORKERS,
        help=f"Maximum concurrent workers (default: {VERIFY_MAX_WORKERS})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VERIFY_OUTPUT_DIR,
        help="Directory for output files"
    )
    
    args = parser.parse_args()
    
    # Load signals from all signals_*.jsonl files
    print(f"📂 Scanning directory: {args.trade_log_dir}")
    print(f"🎯 Filtering: revised_confidence ≥ {args.threshold}%")
    signals = load_all_signals(args.trade_log_dir, args.threshold)
    
    if not signals:
        print("❌ No qualifying signals found")
        print(f"   Only signals with revised_confidence ≥ {args.threshold}% are verified")
        print("   (Signals without enrichment are skipped)")
        return
    
    # Group by ticker for efficient processing
    by_ticker = {}
    for signal in signals:
        by_ticker.setdefault(signal.ticker, []).append(signal)
    
    print(f"📊 Found {len(signals)} refined signals across {len(by_ticker)} tickers")
    print(f"🔄 Verifying refined signals with {args.max_workers} workers...")
    
    # Initialize fetcher
    fetcher = StooqFetcher()
    
    # Process signals
    verifications = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Take only the most recent signal per ticker
        futures = {}
        for ticker, ticker_signals in by_ticker.items():
            # Process most recent signal for each ticker
            futures[executor.submit(verify_signal, ticker_signals[0], fetcher)] = ticker
        
        # Collect results
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                verification = future.result()
                verifications.append(verification)
                
                # Show progress
                emoji = "✓" if verification.current_price else "✗"
                print(f"  {emoji} {ticker}", end=" ", flush=True)
                
            except Exception as e:
                print(f"\n  ✗ {ticker}: {e}")
    
    print("\n")
    
    # Print summary
    print_summary(verifications)
    
    # Save results
    save_verification(verifications, args.output_dir)

if __name__ == "__main__":
    main()