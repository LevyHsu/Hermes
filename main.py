#!/usr/bin/env python3
"""
Hermes - Fast News Ingestion & Trading Signal System
Named after the Greek messenger god and patron of trade
Priority Queue System - no smart scheduling needed, queue handles everything
"""

import argparse
import json
import logging
import signal
import sys
import time
import threading
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Local imports
from args import get_args, DATA_DIR, NEWS_DIR, RESULT_DIR, LISTING_DIR, TRADE_LOG_DIR, LOGS_DIR
from status_dashboard import StatusDashboard
from fetch_us_listings import run as update_stock_listings
from llm import LMStudioClient
from news_harvester import harvest_single_cycle
from news_feed import DEFAULT_FEEDS, load_extra_feeds
from priority_queue_processor import TimeOrderedNewsQueue, NewsItem

# Additional directory not in args.py
LLM_DIR = DATA_DIR / "llm"


# Global resources
STOP = False
LLM_CLIENT = None
HARVESTER_THREAD = None
PROCESSOR_THREAD = None
NEWS_QUEUE = None
DASHBOARD = None


# Helper to compute dashboard recent limit
def _calc_recent_limit(args=None) -> int:
    """
    Compute how many rows the 'Recent Decisions' panel should try to show.
    Priority:
      1) Command-line argument: --dashboard-recent-limit
      2) Env override: DASHBOARD_RECENT_LIMIT
      3) Terminal height derived: rows - 15 (for header/footer/other panels)
      4) Fallback: DASHBOARD_DEFAULT_RECENT_DECISIONS from args.py
    """
    from args import (
        DASHBOARD_MIN_RECENT_DECISIONS, 
        DASHBOARD_MAX_RECENT_DECISIONS, 
        DASHBOARD_DEFAULT_RECENT_DECISIONS
    )
    
    # Check command-line argument first
    if args and hasattr(args, 'dashboard_recent_limit') and args.dashboard_recent_limit:
        return max(DASHBOARD_MIN_RECENT_DECISIONS, min(DASHBOARD_MAX_RECENT_DECISIONS, args.dashboard_recent_limit))
    
    # Check environment variable override
    env = os.environ.get("DASHBOARD_RECENT_LIMIT")
    if env:
        try:
            return max(DASHBOARD_MIN_RECENT_DECISIONS, min(DASHBOARD_MAX_RECENT_DECISIONS, int(env)))
        except Exception:
            pass
    
    # Calculate based on terminal size
    try:
        size = shutil.get_terminal_size(fallback=(120, 40))
        # Reserve space for header(3) + queue/stats + alerts(5) + console(8) = ~15 lines
        available = size.lines - 15
        # Allocate about 60% of available space to decisions panel
        decision_lines = int(available * 0.6)
        return max(DASHBOARD_MIN_RECENT_DECISIONS, min(DASHBOARD_MAX_RECENT_DECISIONS, decision_lines))
    except Exception:
        return DASHBOARD_DEFAULT_RECENT_DECISIONS


def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    global STOP, LLM_CLIENT, DASHBOARD
    STOP = True
    
    try:
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        
        # Stop dashboard if running
        if DASHBOARD:
            DASHBOARD.stop()
            
        # LMStudioClient doesn't need explicit cleanup (uses HTTP internally)
        if LLM_CLIENT:
            logging.info("LLM client cleanup completed")
            
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")
        
    print(f"\n[SHUTDOWN] Received signal {signum}, shutting down...", file=sys.stderr)


def setup_logging(args: argparse.Namespace):
    """Setup logging configuration."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    
    # File handler
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOGS_DIR / "bot.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)


def wait_until_next_minute():
    """Wait until the start of the next minute."""
    now = datetime.now()
    seconds_to_wait = 60 - now.second - (now.microsecond / 1_000_000)
    
    if seconds_to_wait > 0 and not STOP:
        time.sleep(seconds_to_wait)


def harvester_thread(args):
    """
    Harvester thread - runs every minute and adds news to queue.
    Simple and reliable.
    """
    global STOP, NEWS_QUEUE
    
    logging.info("Harvester thread started")
    if DASHBOARD:
        DASHBOARD.add_console_log("Harvester thread started", "info")
    feeds = list(dict.fromkeys(DEFAULT_FEEDS + load_extra_feeds(getattr(args, 'feeds_file', None), args.verbose)))
    
    while not STOP:
        try:
            # Wait until next minute
            wait_until_next_minute()
            
            if STOP:
                break
                
            # Get current minute key
            now = datetime.now()
            minute_key = now.strftime("%y%m%d%H%M")
            
            logging.debug(f"Harvesting news for {minute_key}")
            start_time = time.time()
            
            # Harvest with 20-second budget (leaves 40s for processing)
            news_file = harvest_single_cycle(
                feeds=feeds,
                threads=args.news_threads,
                request_timeout=args.news_timeout,
                cycle_budget=20.0,
                out_dir=str(NEWS_DIR),
                verbose=args.verbose
            )
            
            harvest_time = time.time() - start_time
            
            if news_file and news_file.exists():
                # Load news items
                try:
                    with open(news_file, 'r') as f:
                        items = json.load(f)
                        
                    if items:
                        # Create NewsItem and add to queue
                        news_item = NewsItem(
                            timestamp=time.time(),
                            minute_key=minute_key,
                            file_path=news_file,
                            item_count=len(items),
                            items=items,
                            harvest_time=harvest_time
                        )
                        
                        if NEWS_QUEUE.add(news_item):
                            msg = f"Added {minute_key} to queue: {len(items)} items in {harvest_time:.1f}s"
                            logging.info(msg)
                            if DASHBOARD:
                                DASHBOARD.add_console_log(msg, "success")
                            
                            # Update dashboard
                            if DASHBOARD:
                                DASHBOARD.update_queue_stats(NEWS_QUEUE.get_stats())
                        
                except Exception as e:
                    msg = f"Error processing harvested news: {e}"
                    logging.error(msg)
                    if DASHBOARD:
                        DASHBOARD.add_console_log(msg, "error")
            else:
                msg = f"No news for {minute_key}"
                logging.debug(msg)
                if DASHBOARD:
                    DASHBOARD.add_console_log(msg, "warning")
                
        except Exception as e:
            logging.error(f"Harvester error: {e}")
            if not STOP:
                time.sleep(10)  # Brief pause before retry
                
    msg = "Harvester thread stopped"
    logging.info(msg)
    if DASHBOARD:
        DASHBOARD.add_console_log(msg, "info")


def processor_thread(args):
    """
    Processor thread - continuously processes news from queue.
    Always processes newest items first.
    """
    global STOP, NEWS_QUEUE, LLM_CLIENT
    
    logging.info("Processor thread started")
    if DASHBOARD:
        DASHBOARD.add_console_log("Processor thread started", "info")
    
    # Track performance
    items_processed = 0
    total_decisions = 0
    high_conf_decisions = 0
    items_abandoned = 0
    
    current_processing = None
    
    while not STOP:
        try:
            # Get next news item (newest first)
            news_item = NEWS_QUEUE.get_next(timeout=5.0)
            
            if not news_item:
                continue
                
            # Check if there's something even newer
            if NEWS_QUEUE.has_newer_than(news_item.minute_key):
                msg = f"Skipping {news_item.minute_key} - newer news available"
                logging.info(msg)
                if DASHBOARD:
                    DASHBOARD.add_console_log(msg, "warning")
                items_abandoned += 1
                continue
                
            current_processing = news_item.minute_key
            msg = f"Processing {news_item.minute_key}: {news_item.item_count} items, age {news_item.age_seconds:.1f}s"
            logging.info(msg)
            if DASHBOARD:
                DASHBOARD.add_console_log(msg, "info")
            
            # Update dashboard
            if DASHBOARD:
                DASHBOARD.set_processing(news_item.minute_key)
            
            start_time = time.time()
            
            # Process news items with LLM
            if not news_item.items:
                # Load from file if needed
                with open(news_item.file_path, 'r') as f:
                    news_item.items = json.load(f)
                    
            results = []
            batch_size = 5
            
            for i in range(0, len(news_item.items), batch_size):
                # Check if newer news arrived
                if NEWS_QUEUE.has_newer_than(news_item.minute_key):
                    logging.info(f"Abandoning {news_item.minute_key} at item {i}/{len(news_item.items)} - newer news arrived")
                    items_abandoned += 1
                    break
                    
                # Check if we should stop
                if STOP:
                    break
                    
                # Process batch
                batch = news_item.items[i:i + batch_size]
                
                try:
                    # Process with LLM (30s timeout per batch)
                    for item in batch:
                        result = process_single_news_item(
                            item, 
                            LLM_CLIENT,
                            args.confidence_threshold,
                            timeout=10.0,
                            revised_confidence=args.revised_confidence
                        )
                        
                        if result and result.get('decisions'):
                            results.append(result)
                            
                            # Track high confidence
                            for decision in result['decisions']:
                                total_decisions += 1
                                if decision.get('confidence', 0) >= args.high_confidence:
                                    high_conf_decisions += 1
                                    
                                    ticker = decision['ticker']
                                    action = decision['action'] 
                                    confidence = decision['confidence']
                                    msg = f"HIGH CONFIDENCE: {ticker} {action} @ {confidence}%"
                                    logging.info(msg)
                                    if DASHBOARD:
                                        DASHBOARD.add_console_log(msg, "success")
                                    
                                    # Add to dashboard
                                    if DASHBOARD:
                                        DASHBOARD.add_decision(decision)
                                    
                except Exception as e:
                    logging.error(f"Batch processing error: {e}")
                    
            process_time = time.time() - start_time
            
            if results:
                items_processed += 1
                
                # Log trading signals to data/trade-log
                log_trading_signals(results, news_item.minute_key, args)
                
                # Log revised confidence signals to daily log file
                log_human_readable_decisions(results, news_item.minute_key, news_item.items)
                
                msg = f"Completed {news_item.minute_key} in {process_time:.1f}s - {len(results)} items with decisions"
                logging.info(msg)
                if DASHBOARD:
                    DASHBOARD.add_console_log(msg, "info")
            else:
                msg = f"No decisions for {news_item.minute_key}"
                logging.warning(msg)
                if DASHBOARD:
                    DASHBOARD.add_console_log(msg, "warning")
                
            current_processing = None
            
        except Exception as e:
            if not STOP:
                logging.error(f"Processor error: {e}")
                time.sleep(1)
                
    # Final stats
    msg = f"Processor stopped - Processed: {items_processed}, Decisions: {total_decisions} ({high_conf_decisions} high conf), Abandoned: {items_abandoned}"
    logging.info(msg)
    if DASHBOARD:
        DASHBOARD.add_console_log(msg, "info")


def process_single_news_item(item: Dict, llm_client, confidence_threshold: float, 
                            timeout: float = 10.0, revised_confidence: float = 70.0) -> Optional[Dict]:
    """Process a single news item with 2-stage LLM."""
    try:
        # Use the actual LLM processing from llm.py
        from llm import (
            analyze_news_initial, 
            load_listings,
            fetch_google_news,
            refine_decision,
            fetch_price_data
        )
        
        # Load listings once (should cache this)
        listings = load_listings()
        if not listings:
            return None
            
        # Stage 1: Initial analysis
        decisions = analyze_news_initial(llm_client, item, listings)
        
        if decisions:
            # Filter by confidence threshold
            filtered = [d for d in decisions if d.get('confidence', 0) >= confidence_threshold]
            
            # Stage 2: Enrich high-confidence decisions
            enriched_decisions = []
            for decision in filtered:
                if decision.get('confidence', 0) >= revised_confidence:
                    try:
                        ticker = decision['ticker']
                        company_name = decision.get('company_name', ticker)
                        
                        # Fetch additional data
                        logging.debug(f"Enriching {ticker} with confidence {decision['confidence']}")
                        google_news = fetch_google_news(company_name, ticker, days=7, max_items=10)
                        price_data = fetch_price_data(ticker)
                        
                        # Refine the decision with additional context
                        refined = refine_decision(
                            llm_client, 
                            item, 
                            decision, 
                            google_news,
                            {"google": len(google_news)},
                            price_data
                        )
                        
                        if refined:
                            enriched_decisions.append(refined)
                        else:
                            enriched_decisions.append(decision)
                    except Exception as e:
                        logging.error(f"Error enriching {ticker}: {e}")
                        enriched_decisions.append(decision)
                else:
                    enriched_decisions.append(decision)
            
            if enriched_decisions:
                return {
                    'news_id': item.get('id'),
                    'decisions': enriched_decisions
                }
                
    except Exception as e:
        logging.error(f"LLM processing error: {e}")
        
    return None


def log_human_readable_decisions(results: List[Dict], minute_key: str, news_items: List[Dict]):
    """Log only revised confidence signals to daily log file."""
    try:
        # Import threshold from args
        from args import REVISED_CONFIDENCE_THRESHOLD
        
        # Create daily high confidence log file (YYMMDD format)
        date_str = datetime.now().strftime("%y%m%d")  # e.g., "250828"
        daily_log_file = LOGS_DIR / f"{date_str}_high_conf.log"
        
        # Map news items by ID for easy lookup
        news_by_id = {item.get('id'): item for item in news_items if item.get('id')}
        
        # Filter and log only revised confidence signals
        high_conf_signals = []
        
        for result in results:
            news_id = result.get('news_id')
            news_item = news_by_id.get(news_id, {})
            
            for decision in result.get('decisions', []):
                # Only log if revised confidence meets threshold
                if decision.get('revised_confidence', 0) >= REVISED_CONFIDENCE_THRESHOLD:
                    high_conf_signals.append((decision, news_item))
        
        # Only write if we have high confidence signals
        if high_conf_signals:
            with open(daily_log_file, 'a') as f:
                # Write header if file is new
                if daily_log_file.stat().st_size == 0:
                    f.write(f"="*80 + "\n")
                    f.write(f"HIGH CONFIDENCE TRADING SIGNALS - {date_str}\n")
                    f.write(f"Threshold: Revised Confidence >= {REVISED_CONFIDENCE_THRESHOLD}%\n")
                    f.write(f"="*80 + "\n\n")
                
                # Write timestamp header
                f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] Minute: {minute_key}\n")
                f.write("-" * 60 + "\n")
                
                for decision, news_item in high_conf_signals:
                    f.write(f"\n🎯 {decision['ticker']} - {decision['action']}\n")
                    f.write(f"   Original Confidence: {decision.get('confidence', 0)}%\n")
                    f.write(f"   REVISED CONFIDENCE: {decision['revised_confidence']}%\n")
                    
                    # Add price targets if present
                    if decision.get('expected_price'):
                        f.write(f"   TARGET PRICE: ${decision['expected_price']:.2f}\n")
                    if decision.get('horizon_hours'):
                        f.write(f"   HORIZON: {decision['horizon_hours']} hours\n")
                    
                    # News source
                    if news_item:
                        f.write(f"   News: {news_item.get('title', 'N/A')[:100]}\n")
                        f.write(f"   Source: {news_item.get('source', 'N/A')}\n")
                        f.write(f"   Link: {news_item.get('link', 'N/A')}\n")
                    
                    # Refined reasoning
                    if decision.get('refined_reason'):
                        f.write(f"   Reasoning: {decision['refined_reason'][:200]}\n")
                    
                    f.write("\n")
                
                f.write("-" * 60 + "\n")
                
        # Log count for monitoring
        logging.info(f"Logged {len(high_conf_signals)} revised confidence signals to {daily_log_file.name}")
                    
    except Exception as e:
        logging.error(f"Error logging revised confidence decisions: {e}")


def log_trading_signals(results: List[Dict], minute_key: str, args):
    """Log trading signals to file."""
    try:
        trade_log_file = TRADE_LOG_DIR / f"signals_{minute_key}.jsonl"
        
        with open(trade_log_file, 'w') as f:
            for result in results:
                for decision in result.get('decisions', []):
                    signal = {
                        'timestamp': datetime.now().isoformat(),
                        'minute_key': minute_key,
                        'ticker': decision['ticker'],
                        'action': decision['action'],
                        'confidence': decision['confidence'],
                        'reason': decision.get('reason', '')[:200]  # Truncate reason
                    }
                    
                    # Add enriched fields if present (from 2nd stage LLM)
                    if decision.get('revised_confidence'):
                        signal['revised'] = True
                        signal['revised_confidence'] = decision['revised_confidence']
                        
                    if decision.get('expected_price'):
                        signal['expected_price'] = decision['expected_price']
                        
                    if decision.get('expected_timeframe'):
                        signal['expected_timeframe'] = decision['expected_timeframe']
                        
                    if decision.get('stop_loss'):
                        signal['stop_loss'] = decision['stop_loss']
                        
                    if decision.get('entry_price'):
                        signal['entry_price'] = decision['entry_price']
                        
                    if decision.get('refined_reason'):
                        signal['refined_reason'] = decision.get('refined_reason', '')[:500]
                        
                    f.write(json.dumps(signal) + '\n')
                    
    except Exception as e:
        logging.error(f"Error logging signals: {e}")


def clean_data_directories():
    """Clean all data directories and databases."""
    import shutil
    
    dirs_to_clean = [NEWS_DIR, LLM_DIR, RESULT_DIR, TRADE_LOG_DIR, LOGS_DIR]
    for directory in dirs_to_clean:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  Cleaned: {directory}")
        
    # Clean database files  
    db_files = [
        DATA_DIR / ".scheduler_history.db",
        DATA_DIR / ".feed_stats.db", 
        NEWS_DIR / ".seen.db",
        LLM_DIR / ".decisions.db"
    ]
    for db_file in db_files:
        if db_file.exists():
            db_file.unlink()
            print(f"  Removed: {db_file}")


def main():
    """Main entry point - simplified with priority queue."""
    global STOP, LLM_CLIENT, NEWS_QUEUE, HARVESTER_THREAD, PROCESSOR_THREAD, DASHBOARD
    
    # Parse arguments
    args = get_args()
    
    # Handle clean operation
    if args.clean or args.force_clean:
        if not args.force_clean:
            response = input("This will delete all data files and logs. Are you sure? (y/N): ")
            if response.lower() != 'y':
                print("Clean operation cancelled")
                return 0
                
        print("Cleaning all data directories and logs...")
        clean_data_directories()
        print("Clean complete!")
        return 0
        
    # Ensure directories exist
    for directory in [DATA_DIR, NEWS_DIR, LLM_DIR, RESULT_DIR, TRADE_LOG_DIR, LISTING_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Setup logging
    setup_logging(args)
    
    logging.info("="*60)
    logging.info("⚡ HERMES STARTED (Priority Queue Mode)")
    logging.info(f"Configuration: LLM={args.llm_host}, Queue Size=64")
    logging.info("="*60)
    
    # Update stock listings
    if not update_stock_listings():
        if not (LISTING_DIR / "us-listings-latest.json").exists():
            logging.warning("No stock listings found - downloading...")
            if not update_stock_listings(force=True):
                logging.error("Cannot proceed without stock listings")
                return 1
                
    # Initialize LLM client
    logging.info("Initializing LLM client...")
    try:
        LLM_CLIENT = LMStudioClient(server_host=args.llm_host, verbose=args.verbose)
    except Exception as e:
        logging.error(f"Failed to initialize LLM client: {e}")
        return 1
        
    # Create priority queue
    logging.info("Creating priority queue (size=64)...")
    NEWS_QUEUE = TimeOrderedNewsQueue(max_size=64)
    
    # Initialize dashboard if not in quiet mode
    if not args.quiet and sys.stdout.isatty():
        logging.info("Initializing status dashboard...")
        recent_limit = _calc_recent_limit(args)
        # Try multiple constructor signatures (recent_limit vs max_recent)
        try:
            DASHBOARD = StatusDashboard(queue=NEWS_QUEUE, recent_limit=recent_limit)
        except TypeError:
            try:
                DASHBOARD = StatusDashboard(queue=NEWS_QUEUE, max_recent=recent_limit)
            except TypeError:
                DASHBOARD = StatusDashboard(queue=NEWS_QUEUE)
                # Best-effort post-init configuration if supported
                if hasattr(DASHBOARD, "set_recent_limit"):
                    try:
                        DASHBOARD.set_recent_limit(recent_limit)
                    except Exception:
                        pass
                elif hasattr(DASHBOARD, "recent_limit"):
                    try:
                        setattr(DASHBOARD, "recent_limit", recent_limit)
                    except Exception:
                        pass
        logging.info(f"Status dashboard recent list capacity set to ~{recent_limit} rows")
        DASHBOARD.run()
    
    # Start threads
    logging.info("Starting harvester and processor threads...")
    
    HARVESTER_THREAD = threading.Thread(target=harvester_thread, args=(args,), daemon=False)
    PROCESSOR_THREAD = threading.Thread(target=processor_thread, args=(args,), daemon=False)
    
    HARVESTER_THREAD.start()
    PROCESSOR_THREAD.start()
    
    # Monitor loop
    try:
        last_status = time.time()
        
        while not STOP:
            time.sleep(10)
            
            # Log status every minute
            if time.time() - last_status >= 60:
                stats = NEWS_QUEUE.get_stats()
                logging.info(
                    f"Queue: {stats['queue_size']}/64 items, "
                    f"Added: {stats['items_added']}, "
                    f"Processed: {stats['items_processed']}, "
                    f"Dropped: {stats['items_dropped']}"
                )
                
                # Update dashboard with latest stats
                if DASHBOARD:
                    DASHBOARD.update_queue_stats(stats)
                    
                last_status = time.time()
                
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
        STOP = True
        
    # Wait for threads to finish
    logging.info("Waiting for threads to complete...")
    
    if HARVESTER_THREAD:
        HARVESTER_THREAD.join(timeout=5)
    if PROCESSOR_THREAD:
        PROCESSOR_THREAD.join(timeout=10)
        
    # Cleanup
    logging.info("="*60)
    logging.info("⚡ HERMES SHUTTING DOWN")
    
    # Stop dashboard
    if DASHBOARD:
        DASHBOARD.stop()
    
    # Final stats
    if NEWS_QUEUE:
        stats = NEWS_QUEUE.get_stats()
        logging.info(f"Final queue stats: {stats}")
        
    # LMStudioClient doesn't need explicit cleanup (uses HTTP internally)
    if LLM_CLIENT:
        logging.info("LLM client cleanup completed")
            
    logging.info("="*60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)