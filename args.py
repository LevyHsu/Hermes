#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
args.py - Centralized configuration and arguments for IBKR-BOT

This file contains all configuration settings and command-line arguments
for the trading bot system. Modify these settings to customize behavior.

System Components:
1. Stock Listings Fetcher - Updates daily before market open
2. News Harvester - Runs every minute to collect news
3. LLM Decision Engine - Processes news for trading signals
"""

import argparse
from pathlib import Path
from typing import Optional
import os

# =====================================================================
# DIRECTORY CONFIGURATION
# =====================================================================

# Base directories
DATA_DIR = Path("./data")
NEWS_DIR = DATA_DIR / "news"
RESULT_DIR = DATA_DIR / "result"
LISTING_DIR = DATA_DIR / "us-stock-listing"
LOGS_DIR = Path("./logs")

# Ensure directories exist
for dir_path in [DATA_DIR, NEWS_DIR, RESULT_DIR, LISTING_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =====================================================================
# MARKET HOURS CONFIGURATION
# =====================================================================

# US Market hours in Eastern Time
MARKET_OPEN_HOUR = 9   # 9:30 AM ET
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16  # 4:00 PM ET
MARKET_CLOSE_MINUTE = 0

# Update listings this many minutes before market open
LISTING_UPDATE_MINUTES_BEFORE_OPEN = 30  # Update at 9:00 AM ET

# =====================================================================
# NEWS HARVESTER CONFIGURATION
# =====================================================================

# News feed settings
NEWS_THREADS = 4  # Number of concurrent threads for fetching feeds
NEWS_REQUEST_TIMEOUT = 10.0  # Timeout for each RSS feed request (seconds)
NEWS_CYCLE_BUDGET = 20.0  # Time budget per minute for news harvesting (seconds)
NEWS_VERBOSE = True  # Enable detailed logging for news harvester

# =====================================================================
# LLM CONFIGURATION
# =====================================================================

# LM Studio server configuration
LLM_SERVER_HOST = os.getenv("LMSTUDIO_SERVER_HOST", "localhost:1234")
LLM_TIMEOUT = 120.0  # Timeout for LLM processing (seconds) - will be enforced to 2 minutes in main loop
LLM_CONFIDENCE_THRESHOLD = 70  # Minimum confidence for enrichment (0-100)
LLM_NEWS_DAYS = 30  # Days of historical news to fetch for enrichment
LLM_PRICE_DAYS = 7  # Days of price history to fetch
LLM_VERBOSE = True  # Enable detailed logging for LLM

# =====================================================================
# TRADING SIGNAL THRESHOLDS
# =====================================================================

# Confidence thresholds for logging trades
HIGH_CONFIDENCE_THRESHOLD = 70  # Log trades with confidence >= this value
REVISED_CONFIDENCE_THRESHOLD = 50  # Log trades with revised confidence >= this value

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================

# Log file names (will be created in LOGS_DIR)
DETAILED_LOG_FILE = "detailed.log"  # Full system activity log
SIMPLE_LOG_FILE = "simple.log"  # Summary log with key events only
HIGH_CONFIDENCE_LOG_FILE = "high_confidence_trades.log"  # Trades with confidence >= 70%
REVISED_CONFIDENCE_LOG_FILE = "revised_confidence_trades.log"  # Trades with revised confidence >= 50%

# Log rotation settings
LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB per log file
LOG_BACKUP_COUNT = 10  # Keep 10 backup files

# =====================================================================
# SYSTEM BEHAVIOR CONFIGURATION
# =====================================================================

# Processing behavior
SKIP_ON_TIMEOUT = True  # Skip to latest news if LLM times out
LLM_PROCESSING_TIMEOUT_MINUTES = 2  # Maximum time to process news before moving to next batch
RETRY_FAILED_PROCESSING = False  # Whether to retry failed LLM processing
MAX_RETRY_ATTEMPTS = 2  # Maximum retries for failed operations

# System health monitoring
HEALTH_CHECK_INTERVAL_MINUTES = 5  # Check system health every N minutes
ALERT_ON_CONSECUTIVE_FAILURES = 3  # Alert if N consecutive failures occur

# =====================================================================
# EXTERNAL API CONFIGURATION (Currently only using free APIs)
# =====================================================================

# SEC API configuration (required for fetching listings)
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "IBKR-BOT/1.0 (contact: admin@example.com)"  # CHANGE THIS to your email
)

# =====================================================================
# COMMAND LINE ARGUMENT PARSER
# =====================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for main.py
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="IBKR-BOT Main Orchestrator - Automated Trading Signal Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py
  
  # Run with custom LM Studio server
  python main.py --llm-host 192.168.1.100:1234
  
  # Run with lower confidence thresholds
  python main.py --high-confidence 60 --revised-confidence 40
  
  # Run in quiet mode (less logging)
  python main.py --quiet
  
  # Dry run (no actual trading signals)
  python main.py --dry-run
  
  # Run with custom data directory
  python main.py --data-dir /path/to/data

Configuration Priority:
  1. Command-line arguments (highest priority)
  2. Environment variables
  3. Default values in this file (lowest priority)
        """
    )
    
    # General options
    general = parser.add_argument_group('General Options')
    general.add_argument(
        '--data-dir', 
        type=Path, 
        default=DATA_DIR,
        help=f'Base data directory (default: {DATA_DIR})'
    )
    general.add_argument(
        '--logs-dir', 
        type=Path, 
        default=LOGS_DIR,
        help=f'Logs directory (default: {LOGS_DIR})'
    )
    general.add_argument(
        '--dry-run', 
        action='store_true',
        help='Run without generating actual trading signals'
    )
    general.add_argument(
        '--quiet', 
        action='store_true',
        help='Reduce logging verbosity'
    )
    general.add_argument(
        '--verbose', 
        action='store_true',
        help='Increase logging verbosity'
    )
    general.add_argument(
        '-c', '--clean',
        action='store_true',
        help='Clean all data and log files before starting (requires confirmation)'
    )
    general.add_argument(
        '--force-clean',
        action='store_true',
        help='Clean all data and log files without confirmation (use with caution!)'
    )
    
    # Market hours options
    market = parser.add_argument_group('Market Hours Options')
    market.add_argument(
        '--market-open',
        type=str,
        default=f"{MARKET_OPEN_HOUR:02d}:{MARKET_OPEN_MINUTE:02d}",
        help='Market open time in HH:MM format (Eastern Time)'
    )
    market.add_argument(
        '--market-close',
        type=str,
        default=f"{MARKET_CLOSE_HOUR:02d}:{MARKET_CLOSE_MINUTE:02d}",
        help='Market close time in HH:MM format (Eastern Time)'
    )
    market.add_argument(
        '--update-listings-before',
        type=int,
        default=LISTING_UPDATE_MINUTES_BEFORE_OPEN,
        help=f'Update listings N minutes before market open (default: {LISTING_UPDATE_MINUTES_BEFORE_OPEN})'
    )
    
    # News harvester options
    news = parser.add_argument_group('News Harvester Options')
    news.add_argument(
        '--news-threads',
        type=int,
        default=NEWS_THREADS,
        help=f'Number of threads for news fetching (default: {NEWS_THREADS})'
    )
    news.add_argument(
        '--news-timeout',
        type=float,
        default=NEWS_REQUEST_TIMEOUT,
        help=f'Timeout for news requests in seconds (default: {NEWS_REQUEST_TIMEOUT})'
    )
    news.add_argument(
        '--news-budget',
        type=float,
        default=NEWS_CYCLE_BUDGET,
        help=f'Time budget per minute for news harvesting (default: {NEWS_CYCLE_BUDGET})'
    )
    
    # LLM options
    llm = parser.add_argument_group('LLM Engine Options')
    llm.add_argument(
        '--llm-host',
        type=str,
        default=LLM_SERVER_HOST,
        help=f'LM Studio server host:port (default: {LLM_SERVER_HOST})'
    )
    llm.add_argument(
        '--llm-timeout',
        type=float,
        default=LLM_PROCESSING_TIMEOUT_MINUTES * 60,
        help=f'LLM processing timeout in seconds (default: {LLM_PROCESSING_TIMEOUT_MINUTES * 60})'
    )
    llm.add_argument(
        '--confidence-threshold',
        type=int,
        default=LLM_CONFIDENCE_THRESHOLD,
        help=f'Minimum confidence for enrichment (default: {LLM_CONFIDENCE_THRESHOLD})'
    )
    llm.add_argument(
        '--news-days',
        type=int,
        default=LLM_NEWS_DAYS,
        help=f'Days of news history for enrichment (default: {LLM_NEWS_DAYS})'
    )
    llm.add_argument(
        '--price-days',
        type=int,
        default=LLM_PRICE_DAYS,
        help=f'Days of price history to fetch (default: {LLM_PRICE_DAYS})'
    )
    
    # Trading signal options
    trading = parser.add_argument_group('Trading Signal Options')
    trading.add_argument(
        '--high-confidence',
        type=int,
        default=HIGH_CONFIDENCE_THRESHOLD,
        help=f'High confidence threshold for logging (default: {HIGH_CONFIDENCE_THRESHOLD})'
    )
    trading.add_argument(
        '--revised-confidence',
        type=int,
        default=REVISED_CONFIDENCE_THRESHOLD,
        help=f'Revised confidence threshold for logging (default: {REVISED_CONFIDENCE_THRESHOLD})'
    )
    
    # Processing options
    processing = parser.add_argument_group('Processing Options')
    processing.add_argument(
        '--skip-timeout',
        action='store_true',
        default=SKIP_ON_TIMEOUT,
        help='Skip to latest news if LLM times out'
    )
    processing.add_argument(
        '--no-skip-timeout',
        dest='skip_timeout',
        action='store_false',
        help='Wait for LLM even if it times out'
    )
    processing.add_argument(
        '--retry-failures',
        action='store_true',
        default=RETRY_FAILED_PROCESSING,
        help='Retry failed LLM processing'
    )
    processing.add_argument(
        '--max-retries',
        type=int,
        default=MAX_RETRY_ATTEMPTS,
        help=f'Maximum retry attempts (default: {MAX_RETRY_ATTEMPTS})'
    )
    
    return parser

def get_args():
    """
    Parse and return command-line arguments.
    
    Returns:
        Namespace object with all arguments
    """
    parser = create_parser()
    return parser.parse_args()

# =====================================================================
# USAGE INSTRUCTIONS
# =====================================================================

"""
HOW TO USE THIS CONFIGURATION:

1. In main.py:
   ```python
   from args import get_args, LOGS_DIR, HIGH_CONFIDENCE_THRESHOLD
   
   args = get_args()
   # Use args.llm_host, args.confidence_threshold, etc.
   ```

2. In individual components:
   ```python
   from args import LLM_SERVER_HOST, NEWS_THREADS
   
   # Use the constants directly
   ```

3. Override with environment variables:
   ```bash
   export LMSTUDIO_SERVER_HOST=192.168.1.100:1234
   export SEC_USER_AGENT="MyBot/1.0 (myemail@example.com)"
   python main.py
   ```

4. Override with command-line arguments:
   ```bash
   python main.py --llm-host 192.168.1.100:1234 --high-confidence 60
   ```

IMPORTANT NOTES:
- Always set SEC_USER_AGENT to your email before running in production
- Ensure LM Studio server is running before starting the bot
- Market hours are in Eastern Time (ET)
- All confidence thresholds are percentages (0-100)
- Log files will rotate automatically when they reach 50MB
"""

if __name__ == "__main__":
    # Test the argument parser
    args = get_args()
    print("Configuration loaded successfully!")
    print(f"LLM Host: {args.llm_host}")
    print(f"High Confidence Threshold: {args.high_confidence}%")
    print(f"Data Directory: {args.data_dir}")