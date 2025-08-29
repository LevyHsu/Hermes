# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hermes is a financial data pipeline that harvests news, analyzes it with LLM, and generates trading signals - named after the Greek messenger god and patron of trade.

## System Architecture

Three independent processes run continuously in sequence:
1. **Stock Listings Fetcher** → Provides ticker universe (daily refresh)
2. **News Harvester** → Minute-bucketed RSS processing (continuous)  
3. **LLM Decision Engine** → Analyzes news for trading signals (continuous)

## Commands

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install lmstudio requests feedparser readability-lxml beautifulsoup4 lxml
```

### Run Components
```bash
# 1. Fetch stock listings (run once daily)
python fetch_us_listings.py

# 2. Start news harvester (continuous)
python news_feed.py -v

# 3. Start LLM processor (requires LM Studio server running)
python llm.py --server-host localhost:1234 -v
```

### Testing
```bash
# Test refined reasoning truncation
python test_refined_reason.py
```

### Verify Trading Signals
```bash
# Verify refined signals (revised_confidence >= 70%)
python verify.py

# Custom threshold for revised_confidence (both options work)
python verify.py --threshold 60
python verify.py --revised-confidence 60

# Limit Stooq concurrency to avoid rate limits
export STOOQ_MAX_CONCURRENCY=1
python verify.py --max-workers 2
```

## Data Flow

```
NASDAQ/SEC APIs → fetch_us_listings.py → data/us-stock-listing/us-listings-latest.json
                                                           ↓
RSS Feeds → news_feed.py → data/news/YYMMDDHHMM.json → llm.py → data/trade-log/TRADE_LOG.jsonl
                ↓                                         ↑                    ↓
         data/news/.seen.db                    LM Studio Server      verify.py (Stooq data)
                                                          ↓                    ↓
                                               data/llm/.decisions.db   verification_*.json
```

## Environment Variables

- `SEC_USER_AGENT`: Required for SEC.gov API (e.g., "YourBot/1.0 (email@example.com)")
- `LMSTUDIO_SERVER_HOST`: LM Studio server address (default: "192.168.0.198:1234")

## Key Implementation Details

### News Harvester (`news_feed.py`)
- Fetches from 60+ RSS feeds every minute
- Uses Readability to extract article bodies (mandatory field)
- SQLite deduplication via `.seen.db`
- 2-minute backfill on startup
- ThreadPoolExecutor with 4 threads
- 20-second time budget per minute cycle

### LLM Engine (`llm.py`)
- Uses lmstudio-python SDK for structured responses
- Processes news items with confidence threshold (60% default)
- Enriches high-confidence decisions with Google News + Yahoo Finance
- Outputs include: ticker, action (BUY/SELL), confidence, reasoning
- Tracks processed items in `.decisions.db`

### Stock Listings (`fetch_us_listings.py`)
- Fetches from NASDAQ Trader (nasdaqlisted.txt, otherlisted.txt)
- Enriches with SEC CIK data
- Filters test issues and non-stocks
- Outputs both dated and latest symlinks

### Signal Verification (`verify.py`)
- Scans data/trade-log/ for all signals_*.jsonl files automatically
- Only verifies refined signals with revised_confidence >= REVISED_CONFIDENCE_THRESHOLD (70%)
- Skips signals without enrichment (no revised_confidence)
- Uses Stooq for free market data (no API keys required)
- Calculates PnL and Maximum Favorable/Adverse Excursions
- Evaluates expected_price accuracy with hit detection
- Outputs readable reports with emoji indicators
- Configuration uses REVISED_CONFIDENCE_THRESHOLD from `args.py`

## Critical Requirements

- Python 3.9+ (uses zoneinfo)
- LM Studio server must be running with a model loaded before starting `llm.py`
- News items without `article-body` field are dropped
- All timestamps stored in UTC with local conversions
- Common Chrome User-Agent for all HTTP requests