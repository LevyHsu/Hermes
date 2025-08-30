#!/usr/bin/env python3
"""
Priority Queue Based News Processor

Time-ordered priority queue for news processing where:
- Latest news always gets highest priority
- Queue has configurable max size (default 64)
- LLM always processes from the front (latest news)
- Old news automatically gets dropped when queue is full
"""

import threading
import time
import logging
import heapq
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

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