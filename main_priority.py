#!/usr/bin/env python3
"""
Main orchestrator using Priority Queue System
Elegant solution where latest news always gets priority
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Local imports
from args import get_args
from fetch_us_listings import update_stock_listings
from llm import LMStudioClient, process_news_file
from priority_queue_processor import (
    TimeOrderedNewsQueue,
    NewsItem,
    ContinuousNewsProcessor,
    NewsHarvester,
    PriorityQueueSystem
)

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

# Global flags
STOP = False
SYSTEM = None  # Priority queue system


def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    global STOP, SYSTEM
    STOP = True
    
    try:
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        
        # Stop the priority queue system
        if SYSTEM:
            logging.info("Stopping priority queue system...")
            SYSTEM.stop()
            
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")
        
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
    root_logger.handlers = []
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)


class EnhancedNewsProcessor(ContinuousNewsProcessor):
    """Enhanced processor that integrates with existing LLM code"""
    
    def __init__(self, news_queue: TimeOrderedNewsQueue, llm_client, args):
        super().__init__(news_queue, llm_client)
        self.args = args
        
    def _process_batch(self, batch: List[Dict], timeout: float) -> List[Dict]:
        """Process a batch of news items with actual LLM"""
        results = []
        
        for item in batch:
            try:
                # Use existing LLM processing
                result = self.llm_client.process_single_news(
                    item,
                    confidence_threshold=self.args.confidence_threshold,
                    timeout=min(timeout / len(batch), 10)
                )
                
                if result and result.get('decisions'):
                    results.append(result)
                    
                    # Log high confidence decisions
                    for decision in result['decisions']:
                        if decision.get('confidence', 0) >= self.args.high_confidence:
                            ticker = decision['ticker']
                            action = decision['action']
                            confidence = decision['confidence']
                            logging.info(f"HIGH CONFIDENCE: {ticker} {action} @ {confidence}% confidence")
                            
            except Exception as e:
                logging.error(f"Error processing item: {e}")
                
        return results
        
    def _process_news_item(self, news_item: NewsItem) -> Optional[List[Dict]]:
        """Override to add logging and trade signal recording"""
        results = super()._process_news_item(news_item)
        
        if results:
            # Log trading signals
            self.log_trading_signals(results, news_item.minute_key)
            
        return results
        
    def log_trading_signals(self, results: List[Dict], minute_key: str):
        """Log trading signals to file"""
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
                            'reason': decision.get('reason', '')
                        }
                        f.write(json.dumps(signal) + '\n')
                        
            logging.debug(f"Logged {sum(len(r.get('decisions', [])) for r in results)} signals to {trade_log_file}")
            
        except Exception as e:
            logging.error(f"Error logging signals: {e}")


def main():
    """Main entry point using priority queue system."""
    global STOP, SYSTEM
    
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
        
        # Clean directories
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
        
        print("Clean complete!")
        return 0
        
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Setup logging
    setup_logging(args)
    
    logging.info("="*60)
    logging.info("IBKR-BOT STARTED (Priority Queue Mode)")
    logging.info(f"Configuration: LLM={args.llm_host}, Queue Size=64")
    logging.info(f"High Confidence={args.high_confidence}%, Revised={args.revised_confidence}%")
    logging.info("="*60)
    
    # Initial setup - update stock listings
    if not update_stock_listings():
        if not (LISTING_DIR / "us-listings-latest.json").exists():
            logging.warning("No stock listings found - downloading...")
            if not update_stock_listings(force=True):
                logging.error("Cannot proceed without stock listings")
                return 1
                
    # Initialize LLM client
    logging.info("Initializing LLM client...")
    try:
        llm_client = LMStudioClient(server_host=args.llm_host, verbose=args.verbose)
    except Exception as e:
        logging.error(f"Failed to initialize LLM client: {e}")
        return 1
        
    # Create priority queue system
    logging.info("Initializing priority queue system...")
    
    # Create custom components
    news_queue = TimeOrderedNewsQueue(max_size=64)
    harvester = NewsHarvester(news_queue)
    processor = EnhancedNewsProcessor(news_queue, llm_client, args)
    
    # Create system
    SYSTEM = PriorityQueueSystem(llm_client, queue_size=64)
    SYSTEM.queue = news_queue
    SYSTEM.harvester = harvester
    SYSTEM.processor = processor
    
    # Start the system
    logging.info("Starting harvester and processor threads...")
    SYSTEM.start()
    
    # Monitor loop
    try:
        cycle_counter = 0
        last_status_time = time.time()
        
        while not STOP:
            time.sleep(10)
            cycle_counter += 1
            
            # Check for interrupts (newer news arrived while processing)
            SYSTEM.interrupt_if_needed()
            
            # Log status every 60 seconds
            if time.time() - last_status_time >= 60:
                status = SYSTEM.get_status()
                queue_stats = status['queue']
                proc_stats = status['processor']
                
                logging.info(
                    f"Status: Queue={queue_stats['queue_size']}/{news_queue.max_size}, "
                    f"Processed={proc_stats['items_processed']}, "
                    f"Abandoned={proc_stats['items_abandoned']}, "
                    f"Decisions={proc_stats['total_decisions']} "
                    f"(High Conf={proc_stats['high_confidence_decisions']})"
                )
                
                last_status_time = time.time()
                
            # Health check every 10 cycles (100 seconds)
            if cycle_counter % 10 == 0:
                logging.debug(f"Health check: System running for {cycle_counter * 10}s")
                
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
    finally:
        # Shutdown
        logging.info("="*60)
        logging.info("IBKR-BOT SHUTTING DOWN")
        
        if SYSTEM:
            final_status = SYSTEM.get_status()
            proc_stats = final_status['processor']
            logging.info(f"Final stats: Processed {proc_stats['items_processed']} items, "
                        f"Generated {proc_stats['total_decisions']} decisions")
            
            SYSTEM.stop()
            
            # Close LLM client
            try:
                if llm_client:
                    logging.info("Closing LLM client connection...")
                    llm_client.close()
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
        # Try to clean up
        try:
            if SYSTEM:
                SYSTEM.stop()
        except:
            pass
        sys.exit(1)