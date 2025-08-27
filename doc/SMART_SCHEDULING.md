# Smart Scheduling Optimization

## Overview

Smart scheduling dynamically adjusts time allocation between news harvesting and LLM processing based on news volume. This optimization ensures maximum utilization of each minute cycle.

## How It Works

### 1. Quick News Check (5 seconds)
Before each cycle, the system performs a rapid check of all RSS feeds to estimate how many new items are available.

### 2. Adaptive Time Allocation
Based on the estimated news volume and previous cycle's count, time is allocated intelligently:

| News Volume | Harvest Time | LLM Time | Rationale |
|------------|--------------|----------|-----------|
| 0 items (no previous) | 5s | 50s | Ultra-quick check, minimal processing |
| 0 items (with previous) | 10s | 45s | Quick harvest, extended LLM for previous batch |
| 1-3 items | 10s | 45s | Quick harvest, more LLM time |
| 4-10 items | 15s | 40s | Standard harvest, balanced allocation |
| 11-20 items | 20s | 35s | Standard harvest, adequate LLM time |
| 20+ items | 25s | 30s | Maximum harvest, sufficient LLM time |

### 3. Volume Tracking
The system tracks news counts across cycles to identify trends and adjust accordingly.

## Key Benefits

### 1. **Optimized LLM Processing**
When news is scarce (nights, weekends), the LLM gets up to 50 seconds to thoroughly analyze existing items instead of wasting time on empty harvests.

### 2. **Efficient Resource Usage**
- No news → Minimal harvesting overhead
- Few news → Extended analysis time
- Many news → Adequate collection time

### 3. **Adaptive to Market Conditions**
- **Pre-market**: Often low volume → More LLM time
- **Market hours**: Higher volume → Balanced allocation
- **After hours**: Variable → Adaptive response

### 4. **Continuous Processing**
Even with no new news, the system continues processing the previous batch with extended time, ensuring thorough analysis.

## Configuration

### Enable/Disable Smart Scheduling

```bash
# Enable smart scheduling (default)
python main.py --smart-scheduling

# Disable smart scheduling
python main.py --no-smart-scheduling
```

### Configuration Parameters (args.py)

```python
ENABLE_SMART_SCHEDULING = True  # Enable by default
NEWS_CHECK_TIMEOUT = 5.0        # Quick check timeout (seconds)
MIN_HARVEST_TIME = 10.0         # Minimum harvest time
MAX_HARVEST_TIME = 25.0         # Maximum harvest time
EXTENDED_LLM_TIME = 30.0        # Minimum LLM time when no news
```

## Implementation Details

### Components

1. **news_checker.py** - Quick news availability checker
   - Parallel feed checking (8 threads)
   - 5-second maximum check time
   - Returns estimated item count

2. **main.py** - Orchestrator with smart scheduling
   - Calls news_checker before each cycle
   - Adjusts timeouts dynamically
   - Tracks news volume trends

3. **args.py** - Configuration management
   - Smart scheduling settings
   - Command-line arguments

## Example Scenarios

### Scenario 1: Weekend/Night (Low Activity)
```
Cycle 1: 0 new items → 5s harvest, 50s LLM
Cycle 2: 1 new item → 10s harvest, 45s LLM  
Cycle 3: 0 new items → 10s harvest, 45s LLM (process previous)
```

### Scenario 2: Market Open (High Activity)
```
Cycle 1: 25 new items → 25s harvest, 30s LLM
Cycle 2: 18 new items → 20s harvest, 35s LLM
Cycle 3: 12 new items → 20s harvest, 35s LLM
```

### Scenario 3: Major News Event
```
Cycle 1: 50+ new items → 25s harvest, 30s LLM
Cycle 2: 35 new items → 25s harvest, 30s LLM
Cycle 3: 8 new items → 15s harvest, 40s LLM (returning to normal)
```

## Performance Impact

### Without Smart Scheduling
- Fixed 20s harvest + 40s LLM every cycle
- Wasted time during low-activity periods
- Insufficient LLM time when news is scarce

### With Smart Scheduling
- Up to 2.5x more LLM time during low activity
- 25% more harvest time during news surges
- Adaptive response to changing conditions

## Monitoring

The system logs scheduling decisions for monitoring:

```
[10:15:00] Smart schedule: 2 estimated news items, allocating 10s for harvest, 45s for LLM
[10:15:10] News harvested: 2510151014.json (3 items, estimated was 2)
[10:15:11] Only 3 news items - extending LLM timeout to 45s
[10:20:00] News trend: Last 10 cycles averaged 4.2 items/cycle
```

## Testing

Run the test script to verify scheduling logic:

```bash
python test_smart_scheduling.py
```

This will test various scenarios and confirm the time allocation algorithm works correctly.

## Future Enhancements

1. **Machine Learning**: Predict news volume based on time of day and market events
2. **Provider-Specific Timing**: Adjust per-feed timeouts based on historical response times
3. **Dynamic Threading**: Increase threads during high-volume periods
4. **Predictive Caching**: Pre-fetch likely news sources before the minute boundary
5. **Quality-Based Allocation**: Give more time to high-value news sources

## Troubleshooting

### Smart Scheduling Not Working

1. Check if enabled:
   ```bash
   grep ENABLE_SMART_SCHEDULING args.py
   ```

2. Verify news_checker.py is accessible:
   ```bash
   python news_checker.py -v
   ```

3. Check logs for scheduling decisions:
   ```bash
   grep "Smart schedule" logs/detailed.log
   ```

### Incorrect Time Allocation

1. Review recent news counts:
   ```bash
   grep "News trend" logs/simple.log
   ```

2. Manually test allocation:
   ```bash
   python test_smart_scheduling.py
   ```

3. Disable if causing issues:
   ```bash
   python main.py --no-smart-scheduling
   ```