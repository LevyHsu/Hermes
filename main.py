#!/usr/bin/env python3
"""
IBKR-BOT Main Orchestrator - Simplified Architecture
No subprocesses = Better signal handling, simpler code, better performance
"""

import signal
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import argparse
from typing import Optional, Dict, List, Any, Tuple

# Direct imports - no subprocess overhead!
import news_harvester
import fetch_us_listings
from llm import process_news_item, load_listings, LMStudioClient
from news_checker import get_smart_schedule  
from smart_scheduler import SmartScheduler
from args import get_args

# Directories
DATA_DIR = Path("data")
NEWS_DIR = DATA_DIR / "news"
LLM_DIR = DATA_DIR / "llm"
RESULT_DIR = DATA_DIR / "result"
TRADE_LOG_DIR = DATA_DIR / "trade-log"
LISTING_DIR = DATA_DIR / "us-stock-listing"
LOGS_DIR = Path("logs")

# Ensure directories exist
for directory in [DATA_DIR, NEWS_DIR, LLM_DIR, RESULT_DIR, TRADE_LOG_DIR, LISTING_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Global flags and resources for graceful shutdown
STOP = False
LLM_CLIENT = None


def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    global STOP, LLM_CLIENT
    STOP = True
    try:
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        # Close LLM client if it exists
        if 'LLM_CLIENT' in globals() and LLM_CLIENT:
            logging.info("Closing LLM client connection...")
            LLM_CLIENT.close()
    except:
        pass
    print(f"\n[SHUTDOWN] Received signal {signum}, shutting down gracefully...", file=sys.stderr)


def setup_logging(args: argparse.Namespace):
    """Setup logging configuration."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(simple_formatter)
    
    # File handlers
    detailed_log = LOGS_DIR / "detailed.log"
    file_handler = logging.FileHandler(detailed_log)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)
    
    # High confidence trade logger
    high_conf_handler = logging.FileHandler(LOGS_DIR / "high_confidence_trades.log")
    high_conf_handler.setLevel(logging.INFO)
    high_conf_handler.setFormatter(detailed_formatter)
    high_conf_logger = logging.getLogger("high_confidence")
    high_conf_logger.addHandler(high_conf_handler)


def update_stock_listings(force: bool = False) -> bool:
    """Update stock listings if needed."""
    global STOP
    if STOP:
        return False
        
    latest_file = LISTING_DIR / "us-listings-latest.json"
    
    # Check if update needed
    if not force and latest_file.exists():
        # Check if file is recent (less than 24 hours old)
        age_hours = (time.time() - latest_file.stat().st_mtime) / 3600
        if age_hours < 24:
            logging.debug(f"Stock listings are recent ({age_hours:.1f} hours old)")
            return True
    
    logging.info("Updating stock listings...")
    try:
        # Call the fetch function directly - no subprocess!
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
    global STOP
    if STOP:
        return None
    
    try:
        # Call harvest function directly - no subprocess!
        news_file = news_harvester.harvest_single_cycle(
            feeds=news_harvester.DEFAULT_FEEDS,
            threads=4,
            request_timeout=10.0,
            cycle_budget=timeout,
            out_dir=str(NEWS_DIR),
            verbose=verbose
        )
        
        if news_file and news_file.exists():
            # Count items
            with open(news_file, 'r') as f:
                news_data = json.load(f)
                count = len(news_data) if isinstance(news_data, list) else 0
            logging.info(f"News harvested: {news_file.name} ({count} items)")
            return news_file
        else:
            logging.warning("No news harvested")
            return None
            
    except Exception as e:
        if not STOP:  # Only log if not shutting down
            logging.error(f"Error harvesting news: {e}")
        return None


def process_with_llm(news_file: Path, args: argparse.Namespace, client: LMStudioClient, timeout: float = 40.0) -> Optional[List[Dict]]:
    """Process news through LLM for trading decisions."""
    global STOP
    if STOP:
        return None
    
    try:
        # Load news items
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
        
        # Process each news item
        results = []
        start_time = time.time()
        
        for i, news_item in enumerate(news_items):
            if STOP:
                break
                
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logging.warning(f"LLM timeout after processing {i}/{len(news_items)} items")
                break
            
            try:
                # Process item directly - no subprocess!
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
                            msg = (f"HIGH CONFIDENCE: {decision['ticker']} {decision['action']} "
                                  f"@ {decision['confidence']}% confidence")
                            logging.info(msg)
                            logging.getLogger("high_confidence").info(msg)
                            
            except Exception as e:
                if not STOP:
                    logging.error(f"Error processing news item {i}: {e}")
                continue
        
        if results:
            logging.info(f"LLM processed {len(results)} items with decisions")
        return results
        
    except Exception as e:
        if not STOP:
            logging.error(f"Error in LLM processing: {e}")
        return None


def log_trading_signals(results: List[Dict], minute_key: str):
    """Log trading signals to JSONL file."""
    if not results:
        return
    
    trade_log_file = TRADE_LOG_DIR / "TRADE_LOG.jsonl"
    
    try:
        with open(trade_log_file, 'a') as f:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            for result in results:
                for decision in result.get('decisions', []):
                    signal = {
                        'timestamp': timestamp,
                        'minute': minute_key,
                        'news_id': result.get('id', 'unknown'),
                        'ticker': decision['ticker'],
                        'action': decision['action'],
                        'confidence': decision['confidence'],
                        'reason': decision.get('reason', '')[:500],  # Truncate long reasons
                        'enriched': decision.get('enriched', False)
                    }
                    
                    # Add enrichment data if available
                    if decision.get('enriched'):
                        signal['expected_high_price'] = decision.get('expected_high_price')
                        signal['horizon_hours'] = decision.get('horizon_hours')
                    
                    f.write(json.dumps(signal) + '\n')
                    
    except Exception as e:
        logging.error(f"Error logging signals: {e}")


def run_minute_cycle(args: argparse.Namespace, minute: int, 
                    llm_client: LMStudioClient,
                    previous_news_count: int = 0,
                    recent_timeouts: int = 0) -> Tuple[int, bool]:
    """Run one complete minute cycle."""
    global STOP
    
    if STOP:
        return 0, False
    
    start_time = datetime.now()
    minute_key = start_time.strftime("%y%m%d%H%M")
    logging.debug(f"Starting minute cycle {minute}")
    
    # Smart scheduling
    harvest_time = 20.0
    llm_time = 35.0
    estimated_news = -1
    
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
            
            logging.info(f"Smart schedule: {estimated_news} estimated news, "
                        f"{harvest_time:.0f}s harvest, {llm_time:.0f}s LLM")
        except Exception as e:
            logging.debug(f"Smart scheduling failed: {e}, using defaults")
    
    # Step 1: Harvest news
    news_file = harvest_news(timeout=harvest_time, verbose=args.verbose)
    
    if not news_file:
        logging.warning("No news file generated")
        return 0, False
    
    # Count actual news
    actual_news_count = 0
    try:
        with open(news_file, 'r') as f:
            news_data = json.load(f)
            actual_news_count = len(news_data) if isinstance(news_data, list) else 0
    except:
        pass
    
    # Check if we're approaching minute boundary (55s limit)
    elapsed = (datetime.now() - start_time).total_seconds()
    if elapsed > 55:
        logging.warning(f"Minute boundary approaching ({elapsed:.1f}s), skipping LLM processing")
        return actual_news_count, False
    
    # Step 2: Process with LLM with strict time boundary
    remaining = max(5, 55 - elapsed)  # Hard stop at 55s into minute
    llm_timeout = min(llm_time, remaining)
    
    # Add minute boundary check
    minute_start = start_time.replace(second=0, microsecond=0)
    current_minute = minute_start.strftime("%H:%M")
    
    results = process_with_llm(news_file, args, llm_client, timeout=llm_timeout)
    llm_timed_out = (results is None)
    
    # Check if we've crossed minute boundary
    now = datetime.now()
    if now.minute != minute_start.minute:
        logging.warning(f"Processing for {current_minute} crossed into next minute, abandoning")
        return actual_news_count, True
    
    # Step 3: Log results
    if results:
        log_trading_signals(results, minute_key)
        
        # Summary
        total_decisions = sum(len(r.get('decisions', [])) for r in results)
        high_conf = sum(
            1 for r in results
            for d in r.get('decisions', [])
            if d.get('confidence', 0) >= args.high_confidence
        )
        logging.info(f"Cycle {minute} complete: {total_decisions} decisions, {high_conf} high confidence")
    else:
        logging.warning(f"No LLM results for cycle {minute}")
    
    # Record for smart scheduler if enabled
    if args.smart_scheduling and estimated_news >= 0:
        try:
            scheduler = SmartScheduler()
            cycle_duration = (datetime.now() - start_time).total_seconds()
            scheduler.record_estimation(
                estimated=estimated_news,
                actual=actual_news_count,
                harvest_time=harvest_time,
                llm_time=llm_timeout,
                cycle_duration=cycle_duration,
                llm_success=bool(results)
            )
        except:
            pass  # Don't fail if scheduler has issues
    
    return actual_news_count, llm_timed_out


def sleep_until_next_minute():
    """Sleep until the start of the next minute, checking STOP frequently."""
    global STOP
    
    now = datetime.now()
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    sleep_seconds = (next_minute - now).total_seconds()
    
    if sleep_seconds > 0:
        sleep_end = time.time() + sleep_seconds
        while not STOP and time.time() < sleep_end:
            time.sleep(0.1)  # Check every 100ms


def main():
    """Main orchestration loop."""
    global STOP
    
    # Parse arguments
    args = get_args()
    
    # Handle clean operation if requested
    if args.clean or args.force_clean:
        if not args.force_clean:
            response = input("This will delete all data files and logs. Are you sure? (y/N): ")
            if response.lower() != 'y':
                print("Clean operation cancelled")
                return 0
        
        print("Cleaning all data directories and logs...")
        import shutil
        
        # Clean directories (including logs)
        dirs_to_clean = [NEWS_DIR, LLM_DIR, RESULT_DIR, TRADE_LOG_DIR, LOGS_DIR]
        for directory in dirs_to_clean:
            if directory.exists():
                shutil.rmtree(directory)
            # Recreate the directory
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  Cleaned: {directory}")
        
        # Clean database files
        db_files = [
            DATA_DIR / ".scheduler_history.db",  # Smart scheduler history
            DATA_DIR / ".feed_stats.db",  # Feed statistics
            NEWS_DIR / ".seen.db",  # Seen news items
            LLM_DIR / ".decisions.db"  # LLM decisions
        ]
        for db_file in db_files:
            if db_file.exists():
                db_file.unlink()
                print(f"  Removed: {db_file}")
        
        print("Clean complete!")
        return 0  # Exit after any clean operation
    
    # Register signal handlers EARLY
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Setup logging
    setup_logging(args)
    
    logging.info("="*60)
    logging.info("IBKR-BOT STARTED")
    logging.info(f"Configuration: LLM={args.llm_host}, High Conf={args.high_confidence}%, "
                f"Revised={args.revised_confidence}%")
    logging.info("="*60)
    
    # Initial setup - update stock listings
    if not update_stock_listings():
        # Try to download if missing
        if not (LISTING_DIR / "us-listings-latest.json").exists():
            logging.warning("No stock listings found - downloading...")
            if not update_stock_listings(force=True):
                logging.error("Cannot proceed without stock listings")
                return 1
    
    # Initialize LLM client once
    logging.info("Initializing LLM client...")
    global LLM_CLIENT
    try:
        LLM_CLIENT = LMStudioClient(server_host=args.llm_host, verbose=args.verbose)
        llm_client = LLM_CLIENT  # Keep local reference for compatibility
    except Exception as e:
        logging.error(f"Failed to initialize LLM client: {e}")
        return 1
    
    # Perform initial backfill (mark recent news as seen)
    logging.info("Performing initial news backfill (2 minutes)...")
    backfill_end = datetime.now() + timedelta(minutes=2)
    backfill_cycles = 0
    
    while not STOP and datetime.now() < backfill_end:
        backfill_cycles += 1
        cycle_start = datetime.now()
        
        # Just harvest to populate seen.db, don't process
        news_file = harvest_news(timeout=20.0, verbose=args.verbose)
        if news_file:
            try:
                with open(news_file, 'r') as f:
                    items = json.load(f)
                logging.info(f"Backfill cycle {backfill_cycles}: marked {len(items)} items as seen")
            except:
                pass
        
        # Sleep until next minute if time remains
        elapsed = (datetime.now() - cycle_start).total_seconds()
        if elapsed < 60 and not STOP:
            sleep_until_next_minute()
    
    if STOP:
        # Clean up LLM client when interrupted during backfill
        try:
            if LLM_CLIENT:
                logging.info("Closing LLM client connection...")
                LLM_CLIENT.close()
                logging.info("LLM client closed successfully")
        except Exception as e:
            logging.error(f"Error closing LLM client: {e}")
        return 0
    
    logging.info(f"Backfill complete after {backfill_cycles} cycles")
    
    # Main loop variables
    minute_counter = 0
    previous_news_count = 0
    recent_timeouts = 0
    
    # Main processing loop
    logging.info("Starting main processing loop...")
    
    while not STOP:
        try:
            minute_counter += 1
            cycle_start = datetime.now()
            
            # Check for daily listing update (3 AM)
            if cycle_start.hour == 3 and cycle_start.minute == 0:
                update_stock_listings(force=True)
            
            # Run the minute cycle
            news_count, timed_out = run_minute_cycle(
                args, minute_counter, llm_client, previous_news_count, recent_timeouts
            )
            
            # Update tracking
            previous_news_count = news_count
            if timed_out:
                recent_timeouts = min(recent_timeouts + 1, 5)
            else:
                recent_timeouts = max(0, recent_timeouts - 1)
            
            # Health check every 10 cycles
            if minute_counter % 10 == 0:
                logging.debug(f"Health check: {minute_counter} cycles completed")
            
            # Sleep until next minute or skip if we're behind
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            if cycle_duration < 60 and not STOP:
                sleep_until_next_minute()
            elif not STOP:
                logging.warning(f"Cycle {minute_counter} took {cycle_duration:.1f}s (>60s)")
                # Skip to next minute boundary if we're significantly behind
                if cycle_duration > 90:
                    logging.error(f"Cycle severely delayed ({cycle_duration:.1f}s), realigning to next minute")
                    sleep_until_next_minute()
                
        except KeyboardInterrupt:
            STOP = True
            logging.info("Keyboard interrupt received - shutting down...")
            break
        except Exception as e:
            if STOP:
                break  # Don't log errors during shutdown
            logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
            if not args.retry_failures:
                break
            # Brief interruptible sleep before retry
            for _ in range(100):
                if STOP:
                    break
                time.sleep(0.1)
    
    # Shutdown
    logging.info("="*60)
    logging.info("IBKR-BOT SHUTTING DOWN")
    logging.info(f"Total cycles completed: {minute_counter}")
    
    # Clean up LLM client
    try:
        if LLM_CLIENT:
            logging.info("Closing LLM client connection...")
            LLM_CLIENT.close()
            logging.info("LLM client closed successfully")
    except Exception as e:
        logging.error(f"Error closing LLM client: {e}")
    
    logging.info("="*60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        # Try to clean up LLM client on fatal error
        try:
            if 'LLM_CLIENT' in globals() and LLM_CLIENT:
                print("Closing LLM client due to fatal error...", file=sys.stderr)
                LLM_CLIENT.close()
        except:
            pass
        sys.exit(1)