#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py - Main orchestrator for IBKR-BOT trading signal system

This is the main entry point that coordinates all components:
1. Updates stock listings before market open
2. Runs news harvester every minute
3. Processes news with LLM immediately after harvesting
4. Manages timeouts and error recovery
5. Maintains multiple log files for different purposes
"""

import json
import logging
import logging.handlers
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import configuration
from args import (
    get_args, 
    LOGS_DIR, 
    NEWS_DIR, 
    LISTING_DIR,
    RESULT_DIR,
    HIGH_CONFIDENCE_THRESHOLD,
    REVISED_CONFIDENCE_THRESHOLD,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    LISTING_UPDATE_MINUTES_BEFORE_OPEN
)

# Try to import timezone support
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET_TZ = ZoneInfo("America/New_York")

# Global stop flag and process tracking
STOP = False
CHILD_PROCESSES = []

# =====================================================================
# SIGNAL HANDLERS
# =====================================================================

def cleanup_child_processes():
    """Kill any remaining child processes"""
    global CHILD_PROCESSES
    for proc in CHILD_PROCESSES:
        try:
            if proc.poll() is None:  # Still running
                if os.name != 'nt':
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except:
                        proc.terminate()
                else:
                    proc.terminate()
                proc.wait(timeout=2)
        except:
            try:
                proc.kill()
            except:
                pass
    CHILD_PROCESSES.clear()

def handle_signal(signum, frame):
    """Handle shutdown signals gracefully"""
    global STOP
    STOP = True
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    print(f"\n[SHUTDOWN] Received signal {signum}, shutting down gracefully...", file=sys.stderr)
    cleanup_child_processes()

# Register signal handlers
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, handle_signal)

# =====================================================================
# LOGGING SETUP
# =====================================================================

def setup_logging(args):
    """
    Set up multiple log files with different purposes:
    1. Detailed log - Everything
    2. Simple log - Key events only
    3. High confidence trades log
    4. Revised confidence trades log
    """
    
    # Create logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    
    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatter for detailed logs
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formatter for simple logs
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. Detailed log file (everything)
    detailed_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "detailed.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    detailed_handler.setLevel(logging.DEBUG)
    detailed_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(detailed_handler)
    
    # 2. Simple log file (key events only)
    simple_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "simple.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    simple_handler.setLevel(logging.INFO)
    simple_handler.setFormatter(simple_formatter)
    simple_handler.addFilter(lambda record: record.levelno >= logging.INFO)
    root_logger.addHandler(simple_handler)
    
    # 3. Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not args.quiet else logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Create specialized loggers for trading signals
    setup_trade_loggers()
    
    logging.info("="*60)
    logging.info("IBKR-BOT STARTED")
    logging.info(f"Configuration: LLM={args.llm_host}, High Conf={args.high_confidence}%, Revised={args.revised_confidence}%")
    logging.info("="*60)

def setup_trade_loggers():
    """Set up specialized loggers for trading signals"""
    
    # High confidence trades logger
    high_conf_logger = logging.getLogger('high_confidence_trades')
    high_conf_logger.setLevel(logging.INFO)
    high_conf_logger.propagate = False  # Don't propagate to root logger
    
    high_conf_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "high_confidence_trades.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    trade_formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    high_conf_handler.setFormatter(trade_formatter)
    high_conf_logger.addHandler(high_conf_handler)
    
    # Revised confidence trades logger
    revised_conf_logger = logging.getLogger('revised_confidence_trades')
    revised_conf_logger.setLevel(logging.INFO)
    revised_conf_logger.propagate = False
    
    revised_conf_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "revised_confidence_trades.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    revised_conf_handler.setFormatter(trade_formatter)
    revised_conf_logger.addHandler(revised_conf_handler)

# =====================================================================
# MARKET HOURS CHECKING
# =====================================================================

def get_et_now():
    """Get current time in Eastern Time"""
    return datetime.now(tz=ET_TZ)

def should_update_listings():
    """
    Check if we should update stock listings.
    Updates:
    1. Immediately if no listings exist at all
    2. Once daily, N minutes before market open
    """
    # First check: Do we have ANY listings at all?
    latest_file = LISTING_DIR / "us-listings-latest.json"
    if not latest_file.exists():
        # No listings at all - must download immediately
        return True
    
    # Also check if the latest file is empty or corrupted
    try:
        if latest_file.stat().st_size < 100:  # Less than 100 bytes is definitely wrong
            return True
    except:
        return True
    
    now_et = get_et_now()
    
    # Market open time today
    market_open = now_et.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
    
    # Update time (N minutes before market open)
    update_time = market_open - timedelta(minutes=LISTING_UPDATE_MINUTES_BEFORE_OPEN)
    
    # Check if we're within 1 minute of update time
    if abs((now_et - update_time).total_seconds()) < 60:
        # Check if today's file already exists
        today_str = now_et.strftime("%Y-%m-%d")
        listing_file = LISTING_DIR / f"us-listings-{today_str}.json"
        if not listing_file.exists():
            return True
    
    return False

def is_market_hours():
    """Check if market is currently open"""
    now_et = get_et_now()
    
    # Skip weekends
    if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check time
    current_time = now_et.time()
    market_open = datetime.strptime(f"{MARKET_OPEN_HOUR:02d}:{MARKET_OPEN_MINUTE:02d}", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    
    return market_open <= current_time <= market_close

# =====================================================================
# COMPONENT RUNNERS
# =====================================================================

def run_fetch_listings(args):
    """Run the stock listings fetcher"""
    logging.info("Updating stock listings...")
    
    # Ensure the listing directory exists
    LISTING_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = [sys.executable, "fetch_us_listings.py", "--force"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        if result.returncode == 0:
            logging.info("Stock listings updated successfully")
            # Parse output to get count
            for line in result.stdout.split('\n'):
                if 'records' in line:
                    logging.info(f"Listings update: {line.strip()}")
        else:
            logging.error(f"Failed to update listings: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logging.error("Stock listings update timed out")
    except Exception as e:
        logging.error(f"Error updating listings: {e}")

def run_news_harvester(args, timeout_seconds=20):
    """
    Run the news harvester for one minute cycle.
    Returns the path to the generated news file.
    """
    global STOP
    if STOP:
        return None
        
    try:
        cmd = [
            sys.executable, "news_harvester.py",
            "-t", str(args.news_threads),
            "--request-timeout", str(args.news_timeout),
            "--cycle-budget", str(timeout_seconds),
            "--out-dir", str(NEWS_DIR)
        ]
        
        if args.verbose:
            cmd.append("-v")
        
        # Run news harvester (single cycle) with proper process handling
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if os.name != 'nt' else None  # Create new process group
        )
        CHILD_PROCESSES.append(process)
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds + 10)
            try:
                CHILD_PROCESSES.remove(process)  # Remove from tracking after completion
            except ValueError:
                pass  # Already removed
            if process.returncode == 0 and stdout:
                news_file = stdout.strip()
                if news_file:
                    return Path(news_file)
            else:
                if stderr and args.verbose:
                    logging.debug(f"News harvester stderr: {stderr}")
        except subprocess.TimeoutExpired:
            # Kill entire process group
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
            logging.warning("News harvester timed out")
    except Exception as e:
        logging.error(f"Error running news harvester: {e}")
        
    return None

def run_llm_processor(news_file: Path, args, timeout_seconds=120):
    """
    Run the LLM processor on a news file.
    Returns the processing results or None if failed/timeout.
    """
    global STOP
    if STOP:
        return None
        
    logging.debug(f"Processing news file: {news_file}")
    
    try:
        cmd = [
            sys.executable, "llm.py",
            "--server-host", args.llm_host,
            "--confidence-threshold", str(args.confidence_threshold),
            "--news-days", str(args.news_days),
            "--price-days", str(args.price_days),
            "--news-file", str(news_file)
        ]
        
        if args.verbose:
            cmd.append("-v")
        
        # Use Popen for better process control
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        CHILD_PROCESSES.append(process)
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            try:
                CHILD_PROCESSES.remove(process)  # Remove from tracking after completion
            except ValueError:
                pass  # Already removed
            result_returncode = process.returncode
            result_stdout = stdout
            result_stderr = stderr
        except subprocess.TimeoutExpired:
            # Kill entire process group
            if os.name != 'nt':
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except:
                    process.terminate()
            else:
                process.terminate()
            
            # Give it a moment to terminate gracefully
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if still running
                process.wait()
            
            logging.warning(f"LLM processing timed out after {timeout_seconds}s")
            if args.skip_timeout:
                logging.info("Skipping to next news batch due to timeout")
            return None
        
        if result_returncode == 0:
            # Read the results from the latest file
            latest_results = RESULT_DIR / "latest_results.json"
            if latest_results.exists():
                with open(latest_results, 'r') as f:
                    return json.load(f)
        else:
            if result_stderr:
                logging.error(f"LLM processor failed: {result_stderr}")
    except Exception as e:
        logging.error(f"Error running LLM processor: {e}")
        
    return None

# =====================================================================
# TRADE LOGGING
# =====================================================================

def log_trading_signals(results: List[Dict[str, Any]], args):
    """
    Log trading signals to appropriate log files based on confidence levels.
    """
    if not results:
        return
    
    high_conf_logger = logging.getLogger('high_confidence_trades')
    revised_conf_logger = logging.getLogger('revised_confidence_trades')
    
    for result in results:
        news_title = result.get("news_title", "Unknown")
        news_link = result.get("news_link", "")
        news_time = result.get("published_at", "")
        
        for decision in result.get("decisions", []):
            ticker = decision.get("ticker")
            action = decision.get("action")
            confidence = decision.get("confidence", 0)
            revised_confidence = decision.get("revised_confidence", confidence)
            reason = decision.get("reason", "")
            refined_reason = decision.get("refined_reason", reason)
            expected_price = decision.get("expected_high_price")
            horizon = decision.get("horizon_hours", 24)
            
            # Format trade info
            trade_info = {
                "ticker": ticker,
                "action": action,
                "confidence": confidence,
                "revised_confidence": revised_confidence,
                "expected_price": expected_price,
                "horizon_hours": horizon,
                "reason": refined_reason if refined_reason else reason,
                "news_title": news_title[:100],
                "news_link": news_link,
                "news_time": news_time
            }
            
            trade_json = json.dumps(trade_info, ensure_ascii=False)
            
            # Log high confidence trades
            if confidence >= args.high_confidence:
                high_conf_logger.info(trade_json)
                logging.info(f"HIGH CONFIDENCE: {ticker} {action} @ {confidence}% confidence")
            
            # Log revised confidence trades
            if revised_confidence >= args.revised_confidence:
                revised_conf_logger.info(trade_json)
                if revised_confidence != confidence:
                    logging.info(f"REVISED CONFIDENCE: {ticker} {action} @ {revised_confidence}% (was {confidence}%)")

# =====================================================================
# MAIN ORCHESTRATION LOOP
# =====================================================================

def find_latest_news_file() -> Optional[Path]:
    """Find the most recent news file"""
    news_files = sorted(NEWS_DIR.glob("*.json"))
    if news_files:
        return news_files[-1]
    return None

def run_minute_cycle(args, minute_num: int, previous_news_count: int = -1):
    """
    Run one complete minute cycle with smart scheduling:
    1. Quick check of news availability (5 seconds)
    2. Adaptive time allocation based on news volume
    3. Harvest news with allocated time
    4. Process with LLM using remaining time
    
    Args:
        args: Command line arguments
        minute_num: Current minute counter
        previous_news_count: Number of news items from previous cycle
        
    Returns:
        Number of news items processed in this cycle
    """
    start_time = datetime.now()
    logging.debug(f"Starting minute cycle {minute_num}")
    
    # Pre-check: Ensure listings are available for LLM processing
    if not (LISTING_DIR / "us-listings-latest.json").exists():
        logging.error("Stock listings not available - cannot process news with LLM")
        logging.info("Attempting to download listings...")
        run_fetch_listings(args)
        if not (LISTING_DIR / "us-listings-latest.json").exists():
            logging.error("Failed to download listings - skipping cycle")
            return 0
    
    # Step 0: Smart scheduling - Quick news check (5 seconds max)
    harvest_time = args.news_budget
    estimated_news = -1
    
    if getattr(args, 'smart_scheduling', True):  # Default to True if not set
        try:
            from news_checker import get_smart_schedule
            
            logging.debug("Checking news availability for smart scheduling...")
            schedule = get_smart_schedule(
                previous_news_count=previous_news_count,
                verbose=args.verbose
            )
            
            harvest_time = schedule['harvest_time']
            estimated_news = schedule['estimated_news']
            
            logging.info(f"Smart schedule: {estimated_news} estimated news items, "
                        f"allocating {harvest_time:.0f}s for harvest, "
                        f"{schedule['llm_time']:.0f}s for LLM")
            
            # Adjust timeouts based on smart schedule
            if estimated_news == 0 and previous_news_count <= 0:
                # Really nothing to process - quick cycle
                logging.info("No news expected and no previous batch - quick cycle")
                harvest_time = 5.0
            elif estimated_news == 0 and previous_news_count > 0:
                # No new news but have previous batch - give LLM more time
                logging.info(f"No new news but {previous_news_count} items from previous cycle - extending LLM time")
                harvest_time = 10.0
                
        except Exception as e:
            logging.debug(f"Smart scheduling failed: {e}, using defaults")
            harvest_time = args.news_budget
            estimated_news = -1
    else:
        logging.debug("Smart scheduling disabled - using fixed time allocation")
    
    # Step 1: Harvest news with adaptive timeout
    logging.debug(f"Starting news harvest with {harvest_time:.0f}s timeout...")
    news_file = run_news_harvester(args, timeout_seconds=harvest_time)
    
    if not news_file:
        # Try to find the latest file that was created
        news_file = find_latest_news_file()
        if not news_file:
            logging.warning("No news file generated or found")
            return 0
    
    # Count actual news items harvested
    actual_news_count = 0
    try:
        with open(news_file, 'r') as f:
            news_data = json.load(f)
            actual_news_count = len(news_data) if isinstance(news_data, list) else 0
    except:
        pass
    
    logging.info(f"News harvested: {news_file.name} ({actual_news_count} items, estimated was {estimated_news})")
    
    # Step 2: Process with LLM (with adaptive timeout)
    logging.debug("Starting LLM processing...")
    
    # Calculate remaining time in minute with smart allocation
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # If we have very few or no news, give LLM more time
    if actual_news_count == 0:
        # No news at all - give LLM maximum time if there's a previous batch
        if previous_news_count > 0:
            remaining = max(30, 60 - elapsed)  # At least 30 seconds
            logging.info(f"No new news - giving LLM {remaining:.0f}s to process previous batch thoroughly")
        else:
            remaining = 10  # Nothing to process, quick timeout
    elif actual_news_count <= 3:
        # Very few items - give LLM extra time
        remaining = max(25, 60 - elapsed)
        logging.info(f"Only {actual_news_count} news items - extending LLM timeout to {remaining:.0f}s")
    else:
        # Normal allocation
        remaining = max(10, 60 - elapsed)
    
    llm_timeout = min(args.llm_timeout, remaining)
    
    results = run_llm_processor(news_file, args, timeout_seconds=llm_timeout)
    
    if results:
        logging.info(f"LLM processed {len(results)} news items")
        
        # Step 3: Log trading signals
        log_trading_signals(results, args)
        
        # Summary statistics
        total_decisions = sum(len(r.get("decisions", [])) for r in results)
        high_conf = sum(
            1 for r in results 
            for d in r.get("decisions", [])
            if d.get("confidence", 0) >= args.high_confidence
        )
        
        logging.info(f"Cycle {minute_num} complete: {total_decisions} decisions, {high_conf} high confidence")
    else:
        logging.warning(f"No LLM results for cycle {minute_num}")
    
    # Return the actual news count for tracking
    return actual_news_count

def sleep_until_next_minute():
    """Sleep until the start of the next minute"""
    now = datetime.now()
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    sleep_seconds = (next_minute - now).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

def clean_data_and_logs(args, force=False):
    """
    Clean all data and log files.
    
    Args:
        args: Command line arguments
        force: If True, skip confirmation prompt
        
    Returns:
        True if cleaned, False if cancelled
    """
    # Directories to clean
    dirs_to_clean = [
        args.data_dir / "news",
        args.data_dir / "result", 
        args.data_dir / "us-stock-listing",
        args.logs_dir
    ]
    
    # Files to preserve (don't delete these)
    preserve_files = {
        ".seen.db",  # News deduplication database
        ".decisions.db"  # LLM decision tracking database
    }
    
    # Count files to be deleted
    total_files = 0
    total_size = 0
    files_to_delete = []
    
    for dir_path in dirs_to_clean:
        if not dir_path.exists():
            continue
            
        for item in dir_path.rglob("*"):
            if item.is_file():
                # Skip preserved files
                if item.name in preserve_files:
                    continue
                    
                files_to_delete.append(item)
                total_files += 1
                try:
                    total_size += item.stat().st_size
                except:
                    pass
    
    if total_files == 0:
        print("No files to clean.")
        return True
    
    # Show summary
    print("\n" + "="*60)
    print("CLEAN OPERATION SUMMARY")
    print("="*60)
    print(f"Files to delete: {total_files}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"\nDirectories affected:")
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            count = sum(1 for f in files_to_delete if str(dir_path) in str(f))
            if count > 0:
                print(f"  - {dir_path}: {count} files")
    
    print(f"\nPreserved files:")
    for preserved in preserve_files:
        print(f"  - {preserved}")
    
    # Confirmation
    if not force:
        print("\n" + "="*60)
        response = input("Are you sure you want to delete these files? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Clean operation cancelled.")
            return False
    
    # Perform deletion
    print("\nCleaning files...")
    deleted_count = 0
    errors = []
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            deleted_count += 1
            if deleted_count % 100 == 0:
                print(f"  Deleted {deleted_count}/{total_files} files...")
        except Exception as e:
            errors.append((file_path, str(e)))
    
    # Clean empty directories (but preserve the main directories)
    main_dirs = {str(d) for d in dirs_to_clean}
    for dir_path in dirs_to_clean:
        if not dir_path.exists():
            continue
            
        for subdir in sorted(dir_path.rglob("*"), reverse=True):
            if subdir.is_dir() and str(subdir) not in main_dirs:
                try:
                    if not any(subdir.iterdir()):  # Empty directory
                        subdir.rmdir()
                except:
                    pass
    
    # Report results
    print(f"\n✓ Successfully deleted {deleted_count} files")
    
    if errors:
        print(f"\n⚠ Failed to delete {len(errors)} files:")
        for file_path, error in errors[:10]:  # Show first 10 errors
            print(f"  - {file_path}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    # Ensure main directories still exist (recreate if needed)
    for dir_path in dirs_to_clean:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CLEAN OPERATION COMPLETE")
    print("="*60)
    
    return True

def main():
    """Main orchestration loop"""
    global STOP
    
    # Parse arguments
    args = get_args()
    
    # Handle clean operation
    if args.clean or args.force_clean:
        print("Starting clean operation...")
        if clean_data_and_logs(args, force=args.force_clean):
            print("Clean operation completed.")
            if not args.force_clean:
                response = input("\nContinue with normal operation? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Exiting.")
                    return 0
        else:
            return 1
    
    # Setup logging
    setup_logging(args)
    
    # Log startup information
    logging.info("Starting IBKR-BOT Trading Signal System")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"LM Studio server: {args.llm_host}")
    
    # Check for stock listings at startup
    latest_listings = LISTING_DIR / "us-listings-latest.json"
    if not latest_listings.exists() or latest_listings.stat().st_size < 100:
        logging.info("No stock listings found - downloading immediately...")
        print("\n" + "="*60)
        print("INITIAL SETUP: Downloading stock listings...")
        print("="*60)
        run_fetch_listings(args)
        
        # Verify download succeeded
        if not latest_listings.exists():
            logging.error("Failed to download stock listings - cannot continue")
            print("\nERROR: Failed to download stock listings.")
            print("Please check your internet connection and SEC_USER_AGENT setting.")
            print("Set SEC_USER_AGENT environment variable to: 'YourBot/1.0 (your-email@example.com)'")
            return 1
        else:
            logging.info("Stock listings downloaded successfully")
            print("✓ Stock listings downloaded successfully\n")
    
    # Check if LM Studio is accessible
    try:
        import requests
        response = requests.get(f"http://{args.llm_host}/health", timeout=5)
        if response.status_code == 200:
            logging.info("LM Studio server is accessible")
        else:
            logging.warning(f"LM Studio server returned status {response.status_code}")
    except Exception as e:
        logging.warning(f"Could not reach LM Studio server: {e}")
        logging.warning("Continuing anyway - will retry when processing")
    
    minute_counter = 0
    last_listing_update = None
    previous_news_count = -1  # Track news count from previous cycle
    recent_news_counts = []  # Track recent cycles for trend analysis
    
    # Main loop
    while not STOP:
        try:
            minute_counter += 1
            cycle_start = datetime.now()
            
            # Check if we should update listings (includes emergency check for missing listings)
            if should_update_listings():
                today = get_et_now().date()
                if last_listing_update != today or not (LISTING_DIR / "us-listings-latest.json").exists():
                    if not (LISTING_DIR / "us-listings-latest.json").exists():
                        logging.warning("Stock listings missing - downloading immediately")
                    else:
                        logging.info("Market open approaching - updating stock listings")
                    run_fetch_listings(args)
                    last_listing_update = today
                    
                    # Verify listings exist before continuing
                    if not (LISTING_DIR / "us-listings-latest.json").exists():
                        logging.error("Failed to download listings - skipping this cycle")
                        sleep_until_next_minute()
                        continue
            
            # Run the minute cycle (news + LLM) with smart scheduling
            if is_market_hours() or not args.dry_run:
                news_count = run_minute_cycle(args, minute_counter, previous_news_count)
                
                # Track news counts for trend analysis
                previous_news_count = news_count
                recent_news_counts.append(news_count)
                
                # Keep only last 10 cycles for trend
                if len(recent_news_counts) > 10:
                    recent_news_counts.pop(0)
                
                # Log trend if interesting
                if minute_counter % 5 == 0 and recent_news_counts:
                    avg_news = sum(recent_news_counts) / len(recent_news_counts)
                    logging.info(f"News trend: Last {len(recent_news_counts)} cycles averaged {avg_news:.1f} items/cycle")
            else:
                logging.debug(f"Outside market hours - cycle {minute_counter} skipped")
                previous_news_count = 0  # Reset when market closed
            
            # Health check every N minutes
            if minute_counter % 5 == 0:
                logging.debug(f"Health check: {minute_counter} cycles completed")
            
            # Calculate time to sleep
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            if cycle_duration < 60 and not STOP:
                # Sleep in small increments to be responsive to SIGINT
                sleep_time = 60 - cycle_duration
                sleep_start = time.time()
                while not STOP and (time.time() - sleep_start) < sleep_time:
                    time.sleep(0.5)  # Check every 0.5 seconds
            elif not STOP:
                logging.warning(f"Cycle {minute_counter} took {cycle_duration:.1f}s (>60s)")
                
        except KeyboardInterrupt:
            STOP = True
            logging.info("Keyboard interrupt received - shutting down...")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
            if not args.retry_failures:
                break
            logging.info("Retrying in 10 seconds...")
            time.sleep(10)
    
    # Shutdown
    logging.info("="*60)
    logging.info("IBKR-BOT SHUTTING DOWN")
    logging.info(f"Total cycles completed: {minute_counter}")
    
    # Final cleanup
    cleanup_child_processes()
    logging.info("All child processes terminated")
    logging.info("="*60)
    
    return 0

# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)