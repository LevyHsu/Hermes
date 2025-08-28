# IBKR-BOT Critical Improvements & Optimizations

## 1. LLM Processing Issues

### Problem: Malformed/Corrupted Responses
- LLM returns invalid JSON with unicode issues
- Empty reasoning fields
- Invalid ticker symbols (XXXX)

### Solution:
```python
# Add retry logic with validation
def process_with_llm_robust(news_item, client, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.predict(...)
            # Validate response structure
            if validate_response(response):
                return response
            # If invalid, retry with simplified prompt
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            logger.warning(f"LLM attempt {attempt+1} failed: {e}")
    return None

def validate_response(response):
    """Strict validation of LLM response"""
    if not response or 'decisions' not in response:
        return False
    for decision in response['decisions']:
        if len(decision.get('reason', '')) < 20:
            return False
        if decision.get('ticker') == 'XXXX':
            return False
    return True
```

## 2. Timeout Management

### Problem: LLM Timeouts Cause Incomplete Processing
- Current: Hard timeout cuts off processing mid-batch
- Result: Lost opportunities

### Solution: Adaptive Batch Processing
```python
class AdaptiveBatchProcessor:
    def __init__(self):
        self.avg_process_time = 5.0  # Track average per item
        self.timeout_buffer = 5.0
        
    def process_batch(self, items, time_budget):
        processed = []
        deadline = time.time() + time_budget - self.timeout_buffer
        
        for item in items:
            if time.time() + self.avg_process_time > deadline:
                logger.info(f"Stopping batch early to avoid timeout ({len(processed)}/{len(items)})")
                break
                
            start = time.time()
            result = process_item(item)
            self.avg_process_time = 0.7 * self.avg_process_time + 0.3 * (time.time() - start)
            
            if result:
                processed.append(result)
                
        return processed
```

## 3. Feed Efficiency Crisis

### Problem: 99% Deduplication Waste
- Fetching 935 items, saving 3-7
- No feed prioritization
- No incremental fetching

### Solution: Smart Feed Manager
```python
class SmartFeedManager:
    def __init__(self):
        self.feed_stats = {}  # Track productivity per feed
        self.last_fetch = {}  # Last modified headers
        
    def fetch_feeds(self, feeds, budget_seconds):
        # Prioritize by historical productivity
        sorted_feeds = sorted(feeds, key=lambda f: self.get_productivity(f), reverse=True)
        
        results = []
        deadline = time.time() + budget_seconds
        
        for feed_url in sorted_feeds:
            if time.time() > deadline - 2:
                break
                
            # Use conditional GET with If-Modified-Since
            headers = {}
            if feed_url in self.last_fetch:
                headers['If-Modified-Since'] = self.last_fetch[feed_url]
                
            try:
                response = requests.get(feed_url, headers=headers, timeout=3)
                if response.status_code == 304:
                    continue  # Not modified
                    
                items = parse_feed(response.content)
                new_items = filter_new_items(items)
                
                # Update stats
                self.update_stats(feed_url, len(items), len(new_items))
                results.extend(new_items)
                
            except Exception as e:
                self.penalize_feed(feed_url)
                
        return results
```

## 4. Ticker Detection False Positives

### Problem: Overly Aggressive Pattern Matching
- "Meta" in text triggers META ticker
- No context awareness

### Solution: Context-Aware NER
```python
class ContextAwareTicker Extractor:
    def __init__(self):
        # Common false positive patterns
        self.false_patterns = [
            r'meta[- ]?(data|analysis|tag)',
            r'apple\s+(pie|juice|tree)',
            r'ford\s+(focus|fiesta|mustang)'  # Car models not stock
        ]
        
    def extract_tickers(self, text, article_title):
        candidates = []
        
        # Only consider capitalized mentions
        pattern = r'\b([A-Z]{2,5})\b|\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        
        for match in re.finditer(pattern, text):
            context = text[max(0, match.start()-50):match.end()+50]
            
            # Check if it's a false positive
            if self.is_false_positive(match.group(), context):
                continue
                
            # Require financial context keywords nearby
            if self.has_financial_context(context):
                candidates.append(match.group())
                
        return self.resolve_to_tickers(candidates)
```

## 5. Smart Scheduling Improvements

### Current Issues:
- Fixed 15s minimum even when no news
- No learning from actual processing times
- No consideration for feed response times

### Enhanced Smart Scheduler:
```python
class EnhancedSmartScheduler(SmartScheduler):
    def __init__(self):
        super().__init__()
        self.feed_response_times = {}
        self.llm_items_per_second = 0.5  # Track actual throughput
        
    def calculate_optimal_allocation(self, context):
        # Get feed statistics
        avg_feed_time = self.get_avg_feed_response_time()
        expected_new_items = self.predict_new_items(context['hour'])
        
        if expected_new_items == 0:
            # Minimal checking mode
            return {
                'harvest_time': max(10, avg_feed_time + 2),  # Just enough to check
                'llm_time': 45,  # More time for any surprises
            }
        
        # Calculate based on actual throughput
        required_llm_time = expected_new_items / self.llm_items_per_second
        required_harvest_time = avg_feed_time + 5  # Buffer
        
        # Balance within 55s cycle
        if required_llm_time + required_harvest_time > 55:
            # Scale back proportionally
            scale = 55 / (required_llm_time + required_harvest_time)
            return {
                'harvest_time': required_harvest_time * scale,
                'llm_time': required_llm_time * scale,
                'warning': 'Scaled back due to time constraints'
            }
            
        return {
            'harvest_time': required_harvest_time,
            'llm_time': required_llm_time
        }
```

## 6. Database Optimizations

### Current: Multiple Separate SQLite Files
- `.seen.db`, `.decisions.db`, `.scheduler_history.db`
- No connection pooling
- No indexes

### Solution: Unified Database with Proper Indexes
```python
class UnifiedDatabase:
    def __init__(self, db_path):
        self.pool = sqlite3.connect(db_path, check_same_thread=False)
        self.init_schema()
        
    def init_schema(self):
        # Create tables with proper indexes
        self.pool.executescript("""
            CREATE TABLE IF NOT EXISTS seen_items (
                id TEXT PRIMARY KEY,
                first_seen_utc DATETIME,
                feed_url TEXT,
                INDEX idx_first_seen (first_seen_utc),
                INDEX idx_feed (feed_url)
            );
            
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY,
                news_id TEXT,
                ticker TEXT,
                action TEXT,
                confidence INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_ticker (ticker),
                INDEX idx_confidence (confidence),
                INDEX idx_created (created_at)
            );
            
            -- Add table for feed statistics
            CREATE TABLE IF NOT EXISTS feed_stats (
                feed_url TEXT PRIMARY KEY,
                total_fetched INTEGER DEFAULT 0,
                new_items INTEGER DEFAULT 0,
                avg_response_time REAL,
                last_productive DATETIME,
                productivity_score REAL DEFAULT 1.0
            );
        """)
```

## 7. Parallel Processing Pipeline

### Current: Sequential Processing
- Harvest → Save → Process → Repeat

### Solution: Pipeline Architecture
```python
class PipelineProcessor:
    def __init__(self):
        self.harvest_queue = Queue()
        self.process_queue = Queue()
        self.decision_queue = Queue()
        
    def run(self):
        # Start parallel workers
        threads = [
            Thread(target=self.harvester_worker),
            Thread(target=self.processor_worker),
            Thread(target=self.decision_worker),
            Thread(target=self.enrichment_worker)
        ]
        
        for t in threads:
            t.start()
            
    def harvester_worker(self):
        """Continuously harvest news"""
        while not self.stop:
            news = harvest_feeds()
            for item in news:
                self.harvest_queue.put(item)
                
    def processor_worker(self):
        """Process with LLM"""
        batch = []
        while not self.stop:
            try:
                item = self.harvest_queue.get(timeout=1)
                batch.append(item)
                
                if len(batch) >= 5 or self.harvest_queue.empty():
                    decisions = process_batch_with_llm(batch)
                    for d in decisions:
                        self.process_queue.put(d)
                    batch = []
            except Empty:
                continue
```

## 8. Monitoring & Alerting

### Add Real-time Metrics
```python
class SystemMonitor:
    def __init__(self):
        self.metrics = {
            'feeds_responding': 0,
            'new_items_per_minute': 0,
            'llm_success_rate': 0,
            'high_confidence_decisions': 0,
            'processing_latency': 0
        }
        
    def alert_on_degradation(self):
        if self.metrics['llm_success_rate'] < 0.5:
            self.send_alert("LLM success rate critically low")
            
        if self.metrics['feeds_responding'] < 10:
            self.send_alert("Many feeds not responding")
```

## 9. Configuration Management

### Current: Hardcoded Values Everywhere
### Solution: Centralized Config
```yaml
# config.yaml
scheduler:
  cycle_time: 55
  min_harvest: 10
  max_harvest: 25
  
llm:
  timeout: 30
  batch_size: 5
  retry_count: 3
  temperature: 0.3
  
feeds:
  timeout: 3
  max_concurrent: 10
  prioritize_by: productivity
  
monitoring:
  alert_threshold: 0.5
  metrics_interval: 60
```

## 10. Testing Infrastructure

### Add Comprehensive Tests
```python
# test_llm_processor.py
class TestLLMProcessor(unittest.TestCase):
    def test_malformed_response_handling(self):
        """Test handling of corrupted LLM responses"""
        malformed = {
            "decisions": [{
                "ticker": "XXXX",
                "reason": "\u00a0\u00a0..."
            }]
        }
        result = process_llm_response(malformed)
        self.assertEqual(len(result), 0)
        
    def test_timeout_recovery(self):
        """Test graceful timeout handling"""
        with mock.patch('llm.predict', side_effect=TimeoutError):
            result = process_with_timeout(news_items, timeout=5)
            self.assertIsNotNone(result)
```

## Implementation Priority

1. **Immediate** (Fix breaking issues):
   - LLM response validation
   - Timeout management
   - False positive ticker detection

2. **Short-term** (Performance):
   - Feed prioritization
   - Database optimization
   - Parallel processing

3. **Long-term** (Architecture):
   - Pipeline architecture
   - Monitoring system
   - Configuration management

## Expected Improvements

- **Processing Speed**: 3-5x faster
- **Accuracy**: 90% reduction in false positives  
- **Reliability**: 99% uptime with auto-recovery
- **Efficiency**: 80% reduction in wasted API calls
- **Scalability**: Handle 10x more feeds