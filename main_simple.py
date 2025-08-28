#!/usr/bin/env python3
"""
Simplified IBKR-BOT main orchestrator without subprocesses.
Direct function calls = simpler, more reliable, and better performance.
"""

import signal
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sqlite3
import argparse
from typing import Optional, Dict, List, Any

# Direct imports of our modules - no subprocess overhead!
import news_harvester
import fetch_us_listings
from llm import process_news_item, load_listings, LMStudioClient
from news_checker import get_smart_schedule
from smart_scheduler import SmartScheduler
from args import get_args

# Paths
DATA_DIR = Path("data")
NEWS_DIR = DATA_DIR / "news"
LLM_DIR = DATA_DIR / "llm"
RESULT_DIR = DATA_DIR / "result"
TRADE_LOG_DIR = DATA_DIR / "trade-log"
LISTING_DIR = DATA_DIR / "us-stock-listing"
LOGS_DIR = Path("logs")

# Create directories
for dir_path in [DATA_DIR, NEWS_DIR, LLM_DIR, RESULT_DIR, TRADE_LOG_DIR, LISTING_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global flag for clean shutdown
STOP = False


def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    global STOP
    STOP = True
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    print(f"\n[SHUTDOWN] Shutting down gracefully...", file=sys.stderr)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # File handler with daily rotation
    log_file = LOGS_DIR / f"ibkr_bot_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler]
    )


def update_listings(force: bool = False) -> bool:
    """Update stock listings if needed."""
    latest_file = LISTING_DIR / "us-listings-latest.json"
    
    # Check if update needed
    if not force and latest_file.exists():
        # Check age - update if older than 24 hours
        age = time.time() - latest_file.stat().st_mtime
        if age < 86400:  # 24 hours
            logging.debug("Stock listings are up to date")
            return True
    
    logging.info("Updating stock listings...")
    try:
        result = fetch_us_listings.run(force=True, out_dir=str(LISTING_DIR))
        if result == 0:
            logging.info("Stock listings updated successfully")
            return True
        else:
            logging.error("Failed to update stock listings")
            return False
    except Exception as e:
        logging.error(f"Error updating listings: {e}")
        return False


def harvest_news(timeout: float = 20.0, verbose: bool = False) -> Optional[Path]:
    """Harvest news for current cycle."""
    if STOP:
        return None
        
    try:
        logging.debug(f"Starting news harvest with {timeout:.0f}s timeout...")
        
        # Get feed list
        feeds = news_harvester.DEFAULT_FEEDS
        
        # Run harvester
        news_file = news_harvester.harvest_single_cycle(
            feeds=feeds,
            threads=4,
            request_timeout=10.0,
            cycle_budget=timeout,
            out_dir=str(NEWS_DIR),
            verbose=verbose
        )
        
        if news_file:
            # Count items
            with open(news_file, 'r') as f:
                news_data = json.load(f)
                count = len(news_data) if isinstance(news_data, list) else 0
            logging.info(f"Harvested {count} news items → {news_file.name}")
            return news_file
        else:
            logging.warning("No news harvested")
            return None
            
    except Exception as e:
        logging.error(f"Error harvesting news: {e}")
        return None


def process_news_batch(news_file: Path, args: argparse.Namespace, timeout: float = 40.0) -> List[Dict]:
    """Process news batch through LLM."""
    if STOP:
        return []
        
    try:
        logging.debug(f"Processing news file: {news_file}")
        
        # Load news
        with open(news_file, 'r') as f:
            news_items = json.load(f)
        
        if not news_items:
            logging.warning("No news items to process")
            return []
        
        # Load stock listings
        listings = load_listings()
        if not listings:
            logging.error("No stock listings available")
            return []
        
        # Initialize LLM client
        client = LMStudioClient(server_host=args.llm_host, verbose=args.verbose)
        
        # Process each news item
        results = []
        start_time = time.time()
        
        for i, news_item in enumerate(news_items):
            if STOP:
                break
                
            # Check timeout
            if time.time() - start_time > timeout:
                logging.warning(f"LLM processing timeout after {i}/{len(news_items)} items")
                break
            
            try:
                # Process item
                result = process_news_item(
                    client=client,
                    news_item=news_item,
                    listings=listings,
                    confidence_threshold=args.confidence_threshold,
                    news_days=args.news_days,
                    price_days=args.price_days,
                    verbose=args.verbose
                )
                
                if result and result.get('decisions'):
                    results.append(result)
                    
                    # Log high confidence signals
                    for decision in result['decisions']:
                        if decision.get('confidence', 0) >= args.high_confidence:
                            logging.info(
                                f"HIGH CONFIDENCE: {decision['ticker']} {decision['action']} "
                                f"@ {decision['confidence']}% confidence"
                            )
                            
            except Exception as e:
                logging.error(f"Error processing news item {i}: {e}")
                continue
        
        logging.info(f"Processed {len(results)} items with decisions")
        return results
        
    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        return []


def log_trading_signals(results: List[Dict], args: argparse.Namespace):
    """Log trading signals to file."""
    if not results:
        return
        
    trade_log_file = TRADE_LOG_DIR / "TRADE_LOG.jsonl"
    
    try:
        with open(trade_log_file, 'a') as f:
            for result in results:
                for decision in result.get('decisions', []):
                    # Create trade signal record
                    signal = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'news_id': result.get('id', 'unknown'),
                        'ticker': decision['ticker'],
                        'action': decision['action'],
                        'confidence': decision['confidence'],
                        'reason': decision.get('reason', ''),
                        'enriched': decision.get('enriched', False)
                    }
                    
                    # Add enrichment data if available
                    if decision.get('enriched'):
                        signal.update({
                            'expected_high_price': decision.get('expected_high_price'),
                            'horizon_hours': decision.get('horizon_hours')
                        })
                    
                    # Write as JSONL
                    f.write(json.dumps(signal) + '\n')
                    
        logging.debug(f"Logged {sum(len(r.get('decisions', [])) for r in results)} signals")
        
    except Exception as e:
        logging.error(f"Error logging signals: {e}")


def run_cycle(args: argparse.Namespace, cycle_num: int, previous_news_count: int = 0,
              recent_timeouts: int = 0, scheduler: Optional[SmartScheduler] = None) -> tuple[int, bool]:
    """Run one complete processing cycle."""
    
    cycle_start = datetime.now()
    logging.debug(f"Starting cycle {cycle_num}")
    
    # Smart scheduling
    if args.smart_scheduling and not STOP:
        try:
            schedule = get_smart_schedule(
                previous_news_count=previous_news_count,
                recent_timeouts=recent_timeouts,
                verbose=args.verbose
            )
            harvest_time = schedule['harvest_time']
            llm_time = schedule['llm_time']
            estimated_news = schedule['estimated_news']
            
            logging.info(
                f"Smart schedule: {estimated_news} estimated news, "
                f"{harvest_time:.0f}s harvest, {llm_time:.0f}s LLM"
            )
        except Exception as e:
            logging.debug(f"Smart scheduling failed: {e}, using defaults")
            harvest_time = 20.0
            llm_time = 35.0
            estimated_news = -1
    else:
        harvest_time = 20.0
        llm_time = 35.0
        estimated_news = -1
    
    # Phase 1: Harvest news
    news_file = harvest_news(timeout=harvest_time, verbose=args.verbose)
    
    if not news_file or not news_file.exists():
        logging.warning("No news file generated")
        return 0, False
    
    # Count actual news
    try:
        with open(news_file, 'r') as f:
            news_data = json.load(f)
            actual_news_count = len(news_data) if isinstance(news_data, list) else 0
    except:
        actual_news_count = 0
    
    # Phase 2: Process with LLM
    elapsed = (datetime.now() - cycle_start).total_seconds()
    remaining = max(10, 60 - elapsed)
    llm_timeout = min(llm_time, remaining)
    
    results = process_news_batch(news_file, args, timeout=llm_timeout)
    llm_timed_out = (results is None)
    
    # Phase 3: Log results
    if results:
        log_trading_signals(results, args)
        
        # Summary
        total_decisions = sum(len(r.get('decisions', [])) for r in results)
        high_conf = sum(
            1 for r in results 
            for d in r.get('decisions', [])
            if d.get('confidence', 0) >= args.high_confidence
        )
        logging.info(f"Cycle {cycle_num}: {total_decisions} decisions, {high_conf} high confidence")
    else:
        logging.warning(f"No results for cycle {cycle_num}")
    
    # Record performance for scheduler
    if scheduler and estimated_news >= 0:
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        scheduler.record_estimation(
            estimated=estimated_news,
            actual=actual_news_count,
            harvest_time=harvest_time,
            llm_time=llm_timeout,
            cycle_duration=cycle_duration,
            llm_success=bool(results)
        )
    
    return actual_news_count, llm_timed_out


def sleep_until_next_minute():
    """Sleep until the start of the next minute."""
    global STOP
    now = datetime.now()
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    sleep_seconds = (next_minute - now).total_seconds()
    
    if sleep_seconds > 0:
        # Check STOP flag frequently
        sleep_end = time.time() + sleep_seconds
        while not STOP and time.time() < sleep_end:
            time.sleep(0.1)


def main():
    """Main orchestrator loop."""
    global STOP
    
    # Parse arguments
    args = get_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Setup logging
    setup_logging(args.verbose)
    
    logging.info("="*60)
    logging.info("IBKR-BOT STARTED (Simplified Architecture)")
    logging.info(f"LM Studio: {args.llm_host}")
    logging.info(f"Confidence: {args.confidence_threshold}%")
    logging.info("="*60)
    
    # Initial setup
    if not update_listings(force=args.force_clean):
        logging.error("Failed to get stock listings")
        return 1
    
    # Initialize scheduler if using smart scheduling
    scheduler = SmartScheduler() if args.smart_scheduling else None
    
    # Main loop variables
    cycle_num = 0
    previous_news_count = 0
    recent_timeouts = 0
    
    # Main processing loop
    while not STOP:
        try:
            cycle_num += 1
            
            # Run cycle
            news_count, timed_out = run_cycle(
                args, cycle_num, previous_news_count, 
                recent_timeouts, scheduler
            )
            
            # Update tracking
            previous_news_count = news_count
            if timed_out:
                recent_timeouts = min(recent_timeouts + 1, 5)
            else:
                recent_timeouts = max(0, recent_timeouts - 1)
            
            # Update listings daily (at 3 AM)
            if datetime.now().hour == 3 and datetime.now().minute == 0:
                update_listings(force=True)
            
            # Sleep until next minute
            if not STOP:
                sleep_until_next_minute()
                
        except KeyboardInterrupt:
            STOP = True
            break
        except Exception as e:
            if STOP:
                break
            logging.error(f"Error in main loop: {e}", exc_info=True)
            if not args.retry_failures:
                break
            # Brief sleep before retry
            for _ in range(50):
                if STOP:
                    break
                time.sleep(0.2)
    
    # Shutdown
    logging.info("="*60)
    logging.info("IBKR-BOT SHUTDOWN")
    logging.info(f"Completed {cycle_num} cycles")
    logging.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())