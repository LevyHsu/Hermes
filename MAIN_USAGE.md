# IBKR-BOT Main Orchestrator Usage Guide

## Overview

The main orchestrator (`main.py`) connects all components of the trading signal system:

1. **Stock Listings Fetcher** - Updates daily before market open
2. **News Harvester** - Collects news every minute
3. **LLM Decision Engine** - Analyzes news for trading signals

## Quick Start

### 1. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install lmstudio requests feedparser readability-lxml beautifulsoup4 lxml
```

### 2. Configure Environment

```bash
# Required: Set your SEC user agent (use your email)
export SEC_USER_AGENT="YourBot/1.0 (your-email@example.com)"

# Optional: Set LM Studio server if not on localhost
export LMSTUDIO_SERVER_HOST="192.168.1.100:1234"
```

### 3. Start LM Studio Server

Ensure LM Studio is running with a model loaded before starting the bot.

### 4. Run the Main Orchestrator

```bash
# Run with default settings
python main.py

# Run with custom settings
python main.py --llm-host 192.168.1.100:1234 --high-confidence 60

# Run in verbose mode
python main.py --verbose

# Run quietly (less logging)
python main.py --quiet

# Dry run (no actual processing)
python main.py --dry-run
```

## Configuration (args.py)

All configuration is centralized in `args.py`. Key settings:

### Market Hours
- **Market Open**: 9:30 AM ET
- **Market Close**: 4:00 PM ET
- **Listing Update**: 30 minutes before open (9:00 AM ET)

### Processing Thresholds
- **High Confidence**: 70% (trades logged to high_confidence_trades.log)
- **Revised Confidence**: 50% (trades logged to revised_confidence_trades.log)
- **LLM Timeout**: 2 minutes per news batch

### News Harvesting
- **Threads**: 4 concurrent fetchers
- **Time Budget**: 20 seconds per minute
- **Request Timeout**: 10 seconds per feed

## Log Files

Four specialized log files are maintained in `./logs/`:

1. **detailed.log** - Complete system activity with debug information
2. **simple.log** - Summary of key events and decisions
3. **high_confidence_trades.log** - Trading signals with confidence ≥70%
4. **revised_confidence_trades.log** - Trading signals with revised confidence ≥50%

### Trade Log Format

Each trade is logged as JSON with:
```json
{
  "ticker": "AAPL",
  "action": "BUY",
  "confidence": 75,
  "revised_confidence": 82,
  "expected_price": 185.50,
  "horizon_hours": 24,
  "reason": "Strong earnings beat with raised guidance...",
  "news_title": "Apple Reports Record Q4 Revenue",
  "news_link": "https://...",
  "news_time": "2025-08-26T14:30:00Z"
}
```

## Command Line Arguments

### General Options
- `--data-dir PATH` - Base data directory (default: ./data)
- `--logs-dir PATH` - Logs directory (default: ./logs)
- `--dry-run` - Run without generating trading signals
- `--quiet` - Reduce logging verbosity
- `--verbose` - Increase logging verbosity

### Market Hours Options
- `--market-open HH:MM` - Market open time in ET (default: 09:30)
- `--market-close HH:MM` - Market close time in ET (default: 16:00)
- `--update-listings-before N` - Update listings N minutes before open (default: 30)

### News Harvester Options
- `--news-threads N` - Number of fetching threads (default: 4)
- `--news-timeout SECONDS` - Request timeout (default: 10.0)
- `--news-budget SECONDS` - Time budget per minute (default: 20.0)

### LLM Engine Options
- `--llm-host HOST:PORT` - LM Studio server (default: localhost:1234)
- `--llm-timeout SECONDS` - Processing timeout (default: 120)
- `--confidence-threshold N` - Minimum confidence for enrichment (default: 70)
- `--news-days N` - Days of news history (default: 30)
- `--price-days N` - Days of price history (default: 7)

### Trading Signal Options
- `--high-confidence N` - High confidence threshold (default: 70)
- `--revised-confidence N` - Revised confidence threshold (default: 50)

### Processing Options
- `--skip-timeout` - Skip to latest news if LLM times out (default: true)
- `--no-skip-timeout` - Wait for LLM even on timeout
- `--retry-failures` - Retry failed processing
- `--max-retries N` - Maximum retry attempts (default: 2)

## System Flow

### Every Minute:
1. **News Harvest** (0-20 seconds)
   - Fetches from 60+ RSS feeds
   - Extracts article bodies
   - Saves to `data/news/YYMMDDHHMM.json`

2. **LLM Processing** (20-60 seconds)
   - Analyzes news for affected stocks
   - Generates BUY/SELL signals with confidence
   - Enriches high-confidence signals with additional data

3. **Signal Logging**
   - Logs trades based on confidence thresholds
   - Updates all 4 log files

### Daily (Before Market Open):
- Updates stock listings from NASDAQ/NYSE
- Refreshes CIK mappings from SEC

## Monitoring

### Check System Health
```bash
# View real-time logs
tail -f logs/simple.log

# Monitor high confidence trades
tail -f logs/high_confidence_trades.log

# Check for errors
grep ERROR logs/detailed.log | tail -20
```

### Extract Today's Trades
```bash
# Get all high confidence trades from today
grep "$(date +%Y-%m-%d)" logs/high_confidence_trades.log | jq '.'

# Count trades by action
grep "$(date +%Y-%m-%d)" logs/high_confidence_trades.log | jq -r '.action' | sort | uniq -c
```

## Troubleshooting

### LM Studio Connection Issues
```bash
# Test LM Studio connection
curl http://localhost:1234/health

# Check if model is loaded
curl http://localhost:1234/v1/models
```

### News Harvesting Issues
```bash
# Test news harvester standalone
python news_harvester.py -v

# Check seen database
sqlite3 data/news/.seen.db "SELECT COUNT(*) FROM seen;"
```

### LLM Processing Issues
```bash
# Test with specific news file
python llm.py --news-file data/news/2508242333.json -v

# Check results
cat data/result/latest_results.json | jq '.'
```

## Production Deployment

### Using systemd (Linux)

Create `/etc/systemd/system/ibkr-bot.service`:

```ini
[Unit]
Description=IBKR Trading Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/IBKR-BOT
Environment="SEC_USER_AGENT=YourBot/1.0 (your@email.com)"
Environment="LMSTUDIO_SERVER_HOST=localhost:1234"
ExecStart=/path/to/venv/bin/python /path/to/IBKR-BOT/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ibkr-bot
sudo systemctl start ibkr-bot
sudo systemctl status ibkr-bot
```

### Using screen/tmux

```bash
# Start in screen
screen -S ibkr-bot
source .venv/bin/activate
python main.py

# Detach: Ctrl+A, D
# Reattach: screen -r ibkr-bot
```

## Safety Notes

1. **This is for educational purposes** - Not financial advice
2. **Always validate signals** - The LLM can make mistakes
3. **Start with paper trading** - Test thoroughly before real money
4. **Monitor continuously** - Check logs for anomalies
5. **Set appropriate thresholds** - Adjust confidence levels based on testing

## Support Files

- **args.py** - All configuration constants and argument parsing
- **news_harvester.py** - Single-cycle news fetcher
- **fetch_us_listings.py** - Stock listings updater
- **llm.py** - LLM decision engine
- **test_refined_reason.py** - Test script for LLM output

## Data Flow

```
NASDAQ/SEC → fetch_us_listings.py → data/us-stock-listing/*.json
                                            ↓
RSS Feeds → news_harvester.py → data/news/YYMMDDHHMM.json
                                            ↓
                                        llm.py
                                            ↓
                            data/result/latest_results.json
                                            ↓
                                    Log Files:
                        - logs/detailed.log (everything)
                        - logs/simple.log (key events)
                        - logs/high_confidence_trades.log (≥70%)
                        - logs/revised_confidence_trades.log (≥50%)
```

## Next Steps

1. Test with paper trading account
2. Fine-tune confidence thresholds
3. Add email/Slack alerts for high-confidence trades
4. Implement position sizing logic
5. Add risk management rules
6. Create dashboard for monitoring

---

Remember: This system generates trading *signals*, not automatic trades. Always review and validate before acting on any recommendations.