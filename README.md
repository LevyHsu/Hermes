# IBKR-BOT

AI-powered trading signal system that harvests financial news, analyzes it with LLM, and generates real-time trading decisions.

## Features

- **Real-time News Harvesting**: Processes 60+ RSS feeds every minute
- **LLM Analysis**: Uses local LLM (via LM Studio) for structured trading decisions
- **Smart Scheduling**: Adaptive time allocation based on news volume
- **Graceful Shutdown**: Proper process management with Ctrl+C support
- **No API Keys Required**: Uses only free data sources (RSS feeds, Yahoo Finance)

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/IBKR-BOT.git
cd IBKR-BOT

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install lmstudio requests feedparser readability-lxml beautifulsoup4 lxml
```

### 2. Configure LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model (recommended: Mistral, Llama, or similar)
3. Start the server (default: `http://localhost:1234`)

### 3. Run the System

```bash
# Main orchestrator (runs all components)
python main.py

# Or run components individually:
python fetch_us_listings.py  # Fetch stock listings (daily)
python news_feed.py -v       # Start news harvester
python llm.py -v             # Start LLM processor
```

### 4. Monitor Output

- **Trading Signals**: `data/trade-log/TRADE_LOG.jsonl`
- **Processed News**: `data/llm/<minute>/<news_id>.json`
- **System Logs**: `logs/main_YYYYMMDD.log`

## System Architecture

```
┌─────────────────┐
│  Main Process   │
│   (main.py)     │
└────────┬────────┘
         │
    ┌────┴─────────────────┬──────────────────┐
    ▼                       ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│Stock Listings│   │News Harvester│   │LLM Decision  │
│  Fetcher     │   │(news_feed.py)│   │   Engine     │
└──────────────┘   └──────────────┘   └──────────────┘
    │                   │                   │
    ▼                   ▼                   ▼
[us-listings.json] [news/*.json]    [trade signals]
```

## Configuration

Edit `args.py` to customize:

```python
# Confidence thresholds
confidence_threshold = 80      # Primary signal threshold
confidence_low_threshold = 70  # Secondary signal threshold

# Time allocation (smart scheduling)
min_harvest_time = 10.0        # Minimum news harvest time
max_harvest_time = 25.0        # Maximum news harvest time
total_cycle_time = 55.0        # Total cycle (60s - 5s buffer)

# Server settings
server_host = "localhost:1234" # LM Studio server
```

## Command Line Options

```bash
python main.py [options]

Options:
  -v, --verbose         Enable verbose logging
  -d, --dry-run        Test mode (no actual trades)
  -c, --clean          Clean all data/logs before starting
  --force-clean        Clean without confirmation
  --confidence N       Set confidence threshold (default: 80)
  --server-host HOST   LM Studio server (default: localhost:1234)
```

## Data Structure

```
IBKR-BOT/
├── data/
│   ├── news/              # Raw news (YYMMDDHHMM.json)
│   ├── llm/               # LLM decisions by minute
│   ├── trade-log/         # Trading signals (TRADE_LOG.jsonl)
│   └── us-stock-listing/  # Stock universe
├── logs/                  # System logs
├── doc/                   # Documentation
└── test/                  # Test suites
```

## Trading Signal Format

```json
{
  "timestamp": "2024-12-19T10:30:00Z",
  "ticker": "AAPL",
  "action": "BUY",
  "confidence": 85,
  "reason": "Strong iPhone sales data with analyst upgrades...",
  "news_id": "abc123",
  "expected_high_price": 185.50,
  "horizon_hours": 24
}
```

## Testing

```bash
# Run all tests
for test in test/test_*.py; do python "$test"; done

# Individual tests
python test/test_shutdown.py         # Test graceful shutdown
python test/test_smart_scheduling.py # Test time allocation
python test/test_refined_reason.py   # Test LLM truncation
python test/test_listing_logic.py    # Test stock listings
```

## Documentation

- [Complete Usage Guide](doc/MAIN_USAGE.md)
- [Smart Scheduling Details](doc/SMART_SCHEDULING.md)
- [Shutdown Implementation](doc/SHUTDOWN_FIX.md)
- [Claude AI Instructions](doc/CLAUDE.md)

## Safety Features

- **Graceful Shutdown**: Clean process termination with Ctrl+C
- **Smart Scheduling**: Adaptive processing based on workload
- **Database Preservation**: Clean flag preserves critical databases
- **Confirmation Prompts**: Destructive operations require confirmation
- **Process Monitoring**: Automatic cleanup of zombie processes

## Requirements

- Python 3.9+ (uses zoneinfo)
- LM Studio with loaded model
- 4GB+ RAM recommended
- Stable internet connection

## Troubleshooting

### LM Studio Connection Failed
```bash
# Check if server is running
curl http://localhost:1234/v1/models

# Use custom host
python main.py --server-host 192.168.1.100:1234
```

### Missing Stock Listings
```bash
# Force re-download
rm -rf data/us-stock-listing/*
python fetch_us_listings.py
```

### Process Won't Stop
```bash
# Emergency kill
pkill -9 -f "python.*main.py"
pkill -9 -f "python.*news_feed.py"
pkill -9 -f "python.*llm.py"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Disclaimer

This software is for educational purposes only. Not financial advice. Use at your own risk. Always verify signals before trading.

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/IBKR-BOT/issues)
- Documentation: [doc/](doc/)
- Tests: [test/](test/)