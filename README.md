# ⚡ Hermes

*Named after the Greek messenger god and patron of trade*

AI-powered trading signal system that harvests financial news, analyzes it with local LLM, and generates real-time trading decisions. Hermes processes 60+ RSS feeds every minute, uses a priority queue for newest-first processing, and employs a 2-stage LLM analysis with confidence scoring and enrichment.

## Features

- **Real-time News Harvesting**: Processes 60+ RSS feeds every minute with priority queue system
- **2-Stage LLM Analysis**: Initial analysis + enrichment for high-confidence signals
- **Rich Terminal Dashboard**: Real-time status monitoring with color-coded alerts
- **Priority Queue System**: Time-ordered processing (newest first, max 64 items)
- **Refined Confidence Scoring**: Dual confidence metrics with enrichment from Google News & Yahoo Finance
- **Daily High-Confidence Logs**: Filtered signals in `logs/YYMMDD_high_conf.log`
- **Graceful Shutdown**: Clean process termination with Ctrl+C support
- **No API Keys Required**: Uses only free data sources (RSS feeds, Yahoo Finance, Google News)

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/LevyHsu/Hermes.git
cd Hermes

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install lmstudio requests feedparser readability-lxml beautifulsoup4 lxml rich
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

- **Live Dashboard**: Terminal UI with queue status, decisions, and alerts
- **Trading Signals**: `data/trade-log/TRADE_LOG.jsonl`
- **High-Confidence Signals**: `logs/YYMMDD_high_conf.log` (daily files)
- **Processed News**: `data/llm/<minute>/<news_id>.json`
- **System Logs**: `logs/bot.log`

## System Architecture

```
┌─────────────────────────────────────────────────┐
│              Main Process (main.py)             │
│         ┌────────────────────────────┐          │
│         │   Status Dashboard (Rich)   │          │
│         └────────────────────────────┘          │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│   Harvester  │ │ Priority │ │   Processor  │
│    Thread    │─►│  Queue   │─►│    Thread    │
│              │ │ (Max 64) │ │  (2-Stage LLM)│
└──────────────┘ └──────────┘ └──────────────┘
        │                             │
        ▼                             ▼
[data/news/*.json]          [data/trade-log/]
                            [logs/YYMMDD_high_conf.log]
```

## Configuration

Edit `args.py` to customize:

```python
# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 80      # High confidence signal threshold
REVISED_CONFIDENCE_THRESHOLD = 70   # Refined confidence threshold

# Priority Queue
max_queue_size = 64                 # Maximum items in priority queue

# LM Studio Server
LLM_SERVER_HOST = "localhost:1234"  # LM Studio server address
LLM_CONFIDENCE_THRESHOLD = 70       # Minimum confidence for enrichment

# News Harvesting
NEWS_THREADS = 16                   # Concurrent feed fetching threads
NEWS_CYCLE_BUDGET = 15.0           # Time budget per minute (seconds)
```

## Command Line Options

```bash
python main.py [options]

Options:
  -v, --verbose                Enable verbose logging
  --quiet                      Reduce logging verbosity
  --dry-run                    Test mode (no actual trades)
  -c, --clean                  Clean all data/logs before starting
  --force-clean                Clean without confirmation
  --llm-host HOST              LM Studio server (default: localhost:1234)
  --high-confidence N          High confidence threshold (default: 80)
  --revised-confidence N       Revised confidence threshold (default: 70)
  --confidence-threshold N     Min confidence for enrichment (default: 70)
  --news-threads N             Concurrent feed threads (default: 16)
  --smart-scheduling           Enable adaptive time allocation
  --dashboard-recent-limit N   Number of decisions to display (auto by default)
  --dashboard-compact N        Threshold for compact mode (default: 20)
```

## Data Structure

```
Hermes/
├── data/
│   ├── news/                # Raw news (YYMMDDHHMM.json)
│   │   └── .seen.db         # Deduplication database
│   ├── llm/                 # LLM decisions by minute
│   │   ├── <minute>/        # Per-minute directories
│   │   └── .decisions.db    # Processed tracking database
│   ├── trade-log/           # Trading signals
│   │   └── TRADE_LOG.jsonl  # All trading decisions
│   └── us-stock-listing/    # Stock universe
│       └── us-listings-latest.json
├── logs/                    # System & signal logs
│   ├── bot.log              # Main system log
│   └── YYMMDD_high_conf.log # Daily high-confidence signals
├── doc/                     # Documentation
└── test/                    # Test suites
```

## Trading Signal Format

```json
{
  "timestamp": "2024-12-19T10:30:00Z",
  "ticker": "AAPL",
  "action": "BUY",
  "confidence": 85,
  "revised_confidence": 92,
  "reason": "Strong iPhone sales data with analyst upgrades...",
  "refined_reason": "Multiple converging bullish signals...",
  "news_id": "abc123",
  "expected_high_price": 185.50,
  "expected_price": 185.50,
  "horizon_hours": 24,
  "source": "bloomberg",
  "news_count": 5,
  "price_change_7d": 2.3
}
```

## Terminal Dashboard

The system includes a rich terminal dashboard showing:

- **Queue Status**: Current size, items added/processed/dropped
- **Recent Decisions**: Last 10 trading signals with confidence levels
- **Statistics**: High confidence rate, revised signal counts
- **Refined Alerts**: Highlights revised high-confidence signals (≥70%)
- **Console Output**: Real-time system messages with color coding

Dashboard features:
- **Dynamic capacity**: Automatically adjusts to terminal size (10-200 decisions)
- **Compact mode**: Switches to space-efficient display above 20 decisions
- **Flexible layout**: Adjusts panel sizes based on content
- Updates every 2 seconds
- Color-coded confidence levels (green ≥80%, yellow ≥60%)
- Special alerts for refined signals
- Graceful shutdown on Ctrl+C

Configure dashboard:
```bash
# Set specific number of decisions to show
python main.py --dashboard-recent-limit 50

# Or use environment variable
export DASHBOARD_RECENT_LIMIT=100
python main.py
```

## Testing & Verification

### Run Tests
```bash
# Run all tests
for test in test/test_*.py; do python "$test"; done

# Individual tests
python test/test_shutdown.py         # Test graceful shutdown
python test/test_smart_scheduling.py # Test time allocation
python test/test_refined_reason.py   # Test LLM truncation
python test/test_listing_logic.py    # Test stock listings

# Test dashboard visualization
python test_dashboard.py              # Run dashboard with simulated data
```

### Verify Trading Signals
```bash
# Verify refined signals (revised_confidence >= 70%)
python verify.py

# With custom threshold (both work the same)
python verify.py --threshold 60
python verify.py --revised-confidence 60

# Faster verification with more workers
python verify.py --max-workers 8
```

The verification system:
- **Automatically scans** all `signals_*.jsonl` files in `data/trade-log/`
- **Only verifies refined signals** with `revised_confidence >= REVISED_CONFIDENCE_THRESHOLD (70%)`
- Uses **Stooq** for free market data (no API keys required)
- Calculates PnL, Maximum Favorable/Adverse Excursions
- Evaluates expected price accuracy
- Outputs organized reports with emojis for clarity
- Saves results to `data/trade-log/verification_YYYYMMDD_HHMMSS.json`

## Key Processing Logic

1. **Priority Queue**: Time-ordered (negative timestamp) ensuring newest news processed first
2. **2-Stage LLM**:
   - Stage 1: Initial analysis with confidence scoring
   - Stage 2: High-confidence (≥70%) signals enriched with Google News & Yahoo Finance
3. **Signal Filtering**: Only revised confidence ≥70% logged to daily files
4. **Deduplication**: SQLite databases prevent reprocessing

## Documentation

- [Complete Usage Guide](doc/MAIN_USAGE.md)
- [Smart Scheduling Details](doc/SMART_SCHEDULING.md)
- [Shutdown Implementation](doc/SHUTDOWN_FIX.md)
- [Claude AI Instructions](doc/CLAUDE.md)

## Safety Features

- **Graceful Shutdown**: Clean process termination with Ctrl+C
- **Priority Queue Management**: Automatic oldest item dropping when queue full (64 max)
- **2-Minute Processing Timeout**: Prevents LLM hanging, auto-skips to latest news
- **Database Preservation**: Clean flag preserves critical databases (.seen.db, .decisions.db)
- **Confirmation Prompts**: Destructive operations require confirmation
- **Thread Monitoring**: Separate harvester and processor threads with health checks
- **Automatic Recovery**: Continues processing even if individual items fail

## Requirements

- Python 3.9+ (uses zoneinfo)
- LM Studio with loaded model
- 96GB+ VRAM recommended for large models
- 16GB+ RAM recommended
- Stable internet connection

### Tested Platform
- **Hardware**: MacBook Pro 16-inch
  - M4 Max chip with 16-core CPU
  - 40-core GPU
  - 128GB unified memory
  - 2TB SSD storage
  - 1Gbps network connection
- **Models Tested**:
  - Llama 3.3-70B
  - GPT-OSS-120B

## Signal Verification

Hermes includes a verification system to track signal performance:

```bash
# Basic verification (uses REVISED_CONFIDENCE_THRESHOLD from args.py)
python verify.py

# Override threshold (consistent with main.py)
python verify.py --revised-confidence 60

# Custom configuration via environment
export STOOQ_MAX_CONCURRENCY=2  # Limit concurrent Stooq requests
python verify.py --max-workers 2
```

Verification features:
- **Auto-discovery**: Finds all `signals_*.jsonl` files in trade-log directory
- **Refined Signals Only**: Verifies signals with `revised_confidence >= 70%` (configurable)
- **Consistent Parameters**: Supports `--revised-confidence` like main.py for consistency
- **Stooq-only**: Uses free Stooq data, no API keys needed
- **Readable Output**: Color-coded emojis show wins/losses at a glance
- **Accuracy Tracking**: Evaluates expected_price predictions
- **MFE/MAE Analysis**: Maximum favorable and adverse price excursions
- **Centralized Config**: Uses `REVISED_CONFIDENCE_THRESHOLD` from `args.py`

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

GPL v3 License - see LICENSE file for details

## Disclaimer

This software is for educational purposes only. Not financial advice. Use at your own risk. Always verify signals before trading.

## Support

- Issues: [GitHub Issues](https://github.com/LevyHsu/Hermes/issues)
- Documentation: [doc/](doc/)
- Tests: [test/](test/)