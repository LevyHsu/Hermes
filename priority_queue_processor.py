#!/usr/bin/env python3
"""
Priority Queue Based News Processor

Elegant solution using a time-ordered priority queue where:
- Latest news always gets highest priority
- Queue has configurable max size (default 64)
- LLM always processes from the front (latest news)
- Old news automatically gets dropped when queue is full
"""

import threading
import time
import logging
import heapq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from collections import deque
import queue

logger = logging.getLogger(__name__)


@dataclass(order=True)
class NewsItem:
    """News item with timestamp-based priority (newer = higher priority)"""
    timestamp: float = field(compare=True)  # Negative for reverse order
    minute_key: str = field(compare=False)
    file_path: Path = field(compare=False) 
    item_count: int = field(compare=False, default=0)
    items: List[Dict] = field(compare=False, default_factory=list)
    harvest_time: float = field(compare=False, default=0.0)
    
    def __post_init__(self):
        # Make timestamp negative for reverse ordering (newest first)
        if self.timestamp > 0:
            self.timestamp = -self.timestamp
            
    @property
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return time.time() - abs(self.timestamp)
        
    @property
    def is_stale(self) -> bool:
        """Check if news is too old to process (>3 minutes)"""
        return self.age_seconds > 180


class TimeOrderedNewsQueue:
    """
    Thread-safe priority queue for news items.
    Maintains time-based ordering with newest items first.
    """
    
    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self._queue = []
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._processed = set()  # Track processed minute_keys
        
        # Statistics
        self.stats = {
            'items_added': 0,
            'items_dropped': 0,
            'items_processed': 0,
            'items_skipped': 0
        }
        
    def add(self, news_item: NewsItem) -> bool:
        """
        Add news item to queue in time-order.
        Returns True if added, False if duplicate or rejected.
        """
        with self._lock:
            # Skip if already processed
            if news_item.minute_key in self._processed:
                self.stats['items_skipped'] += 1
                logger.debug(f"Skipping already processed {news_item.minute_key}")
                return False
                
            # Check if we already have this minute
            for item in self._queue:
                if item.minute_key == news_item.minute_key:
                    logger.debug(f"Duplicate minute {news_item.minute_key} ignored")
                    return False
                    
            # Add to queue
            heapq.heappush(self._queue, news_item)
            self.stats['items_added'] += 1
            
            # Maintain max size - remove oldest items
            while len(self._queue) > self.max_size:
                dropped = heapq.heappop(self._queue)
                self.stats['items_dropped'] += 1
                logger.debug(f"Dropped old item {dropped.minute_key} (queue full)")
                
            # Clean up stale items
            self._cleanup_stale()
            
            # Signal that new item is available
            self._not_empty.notify()
            
            logger.info(f"Added {news_item.minute_key} to queue "
                       f"({news_item.item_count} items, position {self.get_position(news_item.minute_key)}/{len(self._queue)})")
            return True
            
    def get_next(self, timeout: Optional[float] = None) -> Optional[NewsItem]:
        """
        Get the highest priority (newest) news item.
        Blocks if queue is empty unless timeout is specified.
        """
        with self._lock:
            # Wait for items if queue is empty
            if not self._queue:
                if timeout is None:
                    self._not_empty.wait()
                else:
                    if not self._not_empty.wait(timeout):
                        return None
                        
            # Clean up stale items first
            self._cleanup_stale()
            
            if self._queue:
                # Get newest item (front of priority queue)
                item = heapq.heappop(self._queue)
                self._processed.add(item.minute_key)
                self.stats['items_processed'] += 1
                
                # Clean up old processed items periodically
                if len(self._processed) > 100:
                    self._cleanup_processed()
                    
                return item
            return None
            
    def peek(self) -> Optional[NewsItem]:
        """Peek at the highest priority item without removing it"""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None
            
    def has_newer_than(self, minute_key: str) -> bool:
        """Check if there's newer news than the given minute"""
        with self._lock:
            if not self._queue:
                return False
                
            newest = self._queue[0]
            return newest.minute_key > minute_key
            
    def get_position(self, minute_key: str) -> int:
        """Get position of minute_key in queue (1-based, 0 if not found)"""
        with self._lock:
            for i, item in enumerate(self._queue):
                if item.minute_key == minute_key:
                    return i + 1
            return 0
            
    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._queue)
            
    def clear(self):
        """Clear the queue"""
        with self._lock:
            self._queue.clear()
            self._processed.clear()
            
    def _cleanup_stale(self):
        """Remove stale items from queue (must hold lock)"""
        cleaned = []
        for item in self._queue:
            if not item.is_stale:
                cleaned.append(item)
            else:
                self.stats['items_dropped'] += 1
                logger.debug(f"Removed stale item {item.minute_key}")
                
        if len(cleaned) < len(self._queue):
            heapq.heapify(cleaned)
            self._queue = cleaned
            
    def _cleanup_processed(self):
        """Clean up old processed items (must hold lock)"""
        # Keep only recent processed items (last 50)
        if len(self._processed) > 50:
            # This is a simple cleanup - in production, you'd want time-based cleanup
            to_remove = len(self._processed) - 50
            for _ in range(to_remove):
                self._processed.pop()
                
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        with self._lock:
            return {
                **self.stats,
                'queue_size': len(self._queue),
                'processed_count': len(self._processed),
                'oldest_age': self._queue[-1].age_seconds if self._queue else 0,
                'newest_age': self._queue[0].age_seconds if self._queue else 0
            }


class ContinuousNewsProcessor:
    """
    Continuously processes news from the priority queue.
    Always processes newest items first.
    """
    
    def __init__(self, news_queue: TimeOrderedNewsQueue, llm_client):
        self.queue = news_queue
        self.llm_client = llm_client
        self.running = False
        self.current_processing = None
        self.stop_current = threading.Event()
        self.thread = None
        
        # Performance tracking
        self.metrics = {
            'items_processed': 0,
            'items_abandoned': 0,
            'total_decisions': 0,
            'high_confidence_decisions': 0,
            'processing_time_total': 0.0
        }
        
    def start(self):
        """Start the processor thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_loop, daemon=True)
            self.thread.start()
            logger.info("News processor started")
            
    def stop(self):
        """Stop the processor thread"""
        self.running = False
        self.stop_current.set()
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("News processor stopped")
        
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get next item (blocks if queue is empty)
                news_item = self.queue.get_next(timeout=1.0)
                
                if not news_item:
                    continue
                    
                # Check if there's something newer
                if self.queue.has_newer_than(news_item.minute_key):
                    logger.info(f"Skipping {news_item.minute_key} - newer news available")
                    self.metrics['items_abandoned'] += 1
                    continue
                    
                # Process this item
                self.current_processing = news_item.minute_key
                self.stop_current.clear()
                
                logger.info(f"Processing {news_item.minute_key} ({news_item.item_count} items, age: {news_item.age_seconds:.1f}s)")
                
                start_time = time.time()
                results = self._process_news_item(news_item)
                process_time = time.time() - start_time
                
                if results:
                    self.metrics['items_processed'] += 1
                    self.metrics['processing_time_total'] += process_time
                    
                    # Count decisions
                    for result in results:
                        decisions = result.get('decisions', [])
                        self.metrics['total_decisions'] += len(decisions)
                        self.metrics['high_confidence_decisions'] += sum(
                            1 for d in decisions if d.get('confidence', 0) >= 80
                        )
                        
                    logger.info(f"Completed {news_item.minute_key} in {process_time:.1f}s - "
                               f"{len(results)} results")
                else:
                    logger.warning(f"No results for {news_item.minute_key}")
                    
                self.current_processing = None
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(1)
                
    def _process_news_item(self, news_item: NewsItem) -> Optional[List[Dict]]:
        """
        Process a single news item with interruption checks.
        """
        if not news_item.items:
            # Load from file if needed
            try:
                with open(news_item.file_path, 'r') as f:
                    news_item.items = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {news_item.file_path}: {e}")
                return None
                
        results = []
        batch_size = 5
        items_to_process = news_item.items
        
        for i in range(0, len(items_to_process), batch_size):
            # Check if we should stop
            if self.stop_current.is_set():
                logger.info(f"Stopping processing of {news_item.minute_key}")
                self.metrics['items_abandoned'] += 1
                break
                
            # Check if there's newer news
            if self.queue.has_newer_than(news_item.minute_key):
                logger.info(f"Abandoning {news_item.minute_key} - newer news arrived")
                self.metrics['items_abandoned'] += 1
                break
                
            # Process batch
            batch = items_to_process[i:i + batch_size]
            
            try:
                # Calculate timeout based on remaining items
                items_remaining = len(items_to_process) - i
                timeout = min(30, max(5, items_remaining * 2))  # 2s per item, max 30s
                
                batch_results = self._process_batch(batch, timeout)
                if batch_results:
                    results.extend(batch_results)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                
        return results if results else None
        
    def _process_batch(self, batch: List[Dict], timeout: float) -> List[Dict]:
        """Process a batch of news items with LLM"""
        # This calls the actual LLM processing
        # For now, returning mock results
        try:
            from llm import process_with_llm
            # Note: In real implementation, this would call the actual LLM
            results = []
            for item in batch:
                # Mock processing - replace with actual LLM call
                result = {
                    'news_id': item.get('id'),
                    'decisions': []
                }
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            return []
            
    def interrupt_current(self):
        """Interrupt current processing to handle newer items"""
        if self.current_processing:
            logger.info(f"Interrupting processing of {self.current_processing}")
            self.stop_current.set()
            
    def get_metrics(self) -> Dict:
        """Get processor metrics"""
        avg_time = (self.metrics['processing_time_total'] / 
                   max(1, self.metrics['items_processed']))
        return {
            **self.metrics,
            'avg_processing_time': avg_time,
            'currently_processing': self.current_processing
        }


class NewsHarvester:
    """
    Continuously harvests news every minute and adds to queue.
    Runs independently of processing.
    """
    
    def __init__(self, news_queue: TimeOrderedNewsQueue):
        self.queue = news_queue
        self.running = False
        self.thread = None
        
    def start(self):
        """Start harvester thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._harvest_loop, daemon=True)
            self.thread.start()
            logger.info("News harvester started")
            
    def stop(self):
        """Stop harvester thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("News harvester stopped")
        
    def _harvest_loop(self):
        """Main harvest loop - runs every minute"""
        while self.running:
            try:
                # Wait until start of next minute
                self._wait_for_next_minute()
                
                if not self.running:
                    break
                    
                # Harvest news for current minute
                now = datetime.now()
                minute_key = now.strftime("%y%m%d%H%M")
                
                logger.debug(f"Starting harvest for {minute_key}")
                start_time = time.time()
                
                # Import here to avoid circular dependency
                from news_harvester import harvest_single_cycle
                from news_feed import DEFAULT_FEEDS, load_extra_feeds
                
                feeds = list(dict.fromkeys(DEFAULT_FEEDS + load_extra_feeds(None, False)))
                
                # Harvest with 20 second budget
                news_file = harvest_single_cycle(
                    feeds=feeds,
                    threads=4,
                    cycle_budget=20.0,
                    verbose=False
                )
                
                harvest_time = time.time() - start_time
                
                if news_file:
                    # Load news to get count
                    try:
                        with open(news_file, 'r') as f:
                            items = json.load(f)
                            
                        # Create NewsItem
                        news_item = NewsItem(
                            timestamp=time.time(),
                            minute_key=minute_key,
                            file_path=news_file,
                            item_count=len(items),
                            items=items,
                            harvest_time=harvest_time
                        )
                        
                        # Add to queue
                        self.queue.add(news_item)
                        
                    except Exception as e:
                        logger.error(f"Failed to process harvested news: {e}")
                else:
                    logger.debug(f"No news harvested for {minute_key}")
                    
            except Exception as e:
                logger.error(f"Harvest error: {e}")
                time.sleep(10)  # Wait before retry
                
    def _wait_for_next_minute(self):
        """Wait until the start of next minute"""
        now = datetime.now()
        seconds_to_wait = 60 - now.second - (now.microsecond / 1_000_000)
        
        if seconds_to_wait > 0:
            time.sleep(seconds_to_wait)


class PriorityQueueSystem:
    """
    Complete priority queue based system.
    Manages harvester, processor, and queue.
    """
    
    def __init__(self, llm_client, queue_size: int = 64):
        self.queue = TimeOrderedNewsQueue(max_size=queue_size)
        self.harvester = NewsHarvester(self.queue)
        self.processor = ContinuousNewsProcessor(self.queue, llm_client)
        self.running = False
        
    def start(self):
        """Start the system"""
        logger.info(f"Starting priority queue system (queue size: {self.queue.max_size})")
        self.running = True
        self.harvester.start()
        self.processor.start()
        
    def stop(self):
        """Stop the system"""
        logger.info("Stopping priority queue system")
        self.running = False
        self.processor.stop()
        self.harvester.stop()
        
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'running': self.running,
            'queue': self.queue.get_stats(),
            'processor': self.processor.get_metrics()
        }
        
    def interrupt_if_needed(self):
        """Check if processor should be interrupted for newer news"""
        if self.processor.current_processing:
            if self.queue.has_newer_than(self.processor.current_processing):
                self.processor.interrupt_current()


# Integration functions for main.py

def create_priority_queue_system(llm_client, queue_size: int = 64) -> PriorityQueueSystem:
    """Create a priority queue based processing system"""
    return PriorityQueueSystem(llm_client, queue_size)


def run_with_priority_queue(llm_client, args) -> None:
    """
    Run the complete priority queue based system.
    This replaces the minute-by-minute cycle approach.
    """
    # Create system
    system = create_priority_queue_system(llm_client, queue_size=64)
    
    # Start processing
    system.start()
    
    try:
        # Monitor loop
        while True:
            time.sleep(10)
            
            # Check for interrupts
            system.interrupt_if_needed()
            
            # Log status every minute
            if int(time.time()) % 60 < 10:
                status = system.get_status()
                logger.info(f"System status: Queue={status['queue']['queue_size']}, "
                           f"Processed={status['processor']['items_processed']}, "
                           f"Decisions={status['processor']['total_decisions']}")
                           
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        system.stop()


if __name__ == "__main__":
    # Test the queue system
    logging.basicConfig(level=logging.DEBUG)
    
    # Create queue
    queue = TimeOrderedNewsQueue(max_size=10)
    
    # Add some test items
    for i in range(5):
        item = NewsItem(
            timestamp=time.time() - i * 60,  # Older items
            minute_key=f"202408281{i:02d}",
            file_path=Path(f"test_{i}.json"),
            item_count=10 + i
        )
        queue.add(item)
        time.sleep(0.1)
        
    # Show queue state
    print(f"Queue stats: {queue.get_stats()}")
    
    # Get items in priority order
    print("\nItems in priority order:")
    while queue.size() > 0:
        item = queue.get_next()
        print(f"  {item.minute_key}: {item.item_count} items, age={item.age_seconds:.1f}s")