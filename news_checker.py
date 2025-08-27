#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
news_checker.py - Quick news availability checker for adaptive scheduling

This module provides a fast way to check how many new news items are available
without doing full article body extraction. Used by main.py to optimize time
allocation between news harvesting and LLM processing.
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple
import time

from news_feed import (
    DEFAULT_FEEDS,
    SEEN_DB_PATH,
    compute_entry_id,
    safe_get,
    load_extra_feeds
)
import feedparser

def quick_check_feed(feed_url: str, seen_ids: Set[str], timeout: float = 2.0) -> int:
    """
    Quickly check a single feed for new items.
    
    Args:
        feed_url: RSS feed URL
        seen_ids: Set of already-seen item IDs
        timeout: Request timeout in seconds
        
    Returns:
        Number of new (unseen) items in the feed
    """
    try:
        data = safe_get(feed_url, timeout=timeout)
        if not data:
            return 0
            
        parsed = feedparser.parse(data)
        new_count = 0
        
        for entry in parsed.get("entries", []):
            # Compute ID same way as harvester
            entry_id = compute_entry_id(entry)
            if entry_id not in seen_ids:
                new_count += 1
                
        return new_count
    except:
        return 0

def estimate_available_news(
    feeds_file: str = None,
    max_check_time: float = 5.0,
    verbose: bool = False
) -> Tuple[int, Dict[str, int]]:
    """
    Quickly estimate how many new news items are available across all feeds.
    
    Args:
        feeds_file: Optional path to extra feeds file
        max_check_time: Maximum time to spend checking (seconds)
        verbose: Enable verbose output
        
    Returns:
        Tuple of (total_new_items, feed_counts_dict)
    """
    start_time = time.time()
    
    # Load feeds
    feeds = list(dict.fromkeys(DEFAULT_FEEDS + load_extra_feeds(feeds_file or "", False)))
    
    # Load seen IDs from database
    seen_ids = set()
    if SEEN_DB_PATH.exists():
        try:
            con = sqlite3.connect(SEEN_DB_PATH, timeout=1.0)
            cursor = con.execute("SELECT id FROM seen")
            for row in cursor:
                seen_ids.add(row[0])
            con.close()
        except:
            pass
    
    # Quick parallel check with short timeout
    total_new = 0
    feed_counts = {}
    checked_feeds = 0
    
    # Calculate timeout per feed
    timeout_per_feed = min(2.0, max_check_time / max(len(feeds), 1))
    
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all feed checks
            future_to_feed = {
                executor.submit(quick_check_feed, feed, seen_ids, timeout_per_feed): feed
                for feed in feeds
            }
            
            # Collect results with time limit
            for future in as_completed(future_to_feed, timeout=max_check_time):
                feed_url = future_to_feed[future]
                try:
                    new_count = future.result(timeout=0.1)
                    if new_count > 0:
                        feed_counts[feed_url] = new_count
                        total_new += new_count
                    checked_feeds += 1
                    
                    # Stop if we're out of time
                    if time.time() - start_time > max_check_time:
                        break
                except:
                    pass
                    
    except:
        pass
    
    if verbose and checked_feeds > 0:
        elapsed = time.time() - start_time
        print(f"[news_checker] Checked {checked_feeds}/{len(feeds)} feeds in {elapsed:.1f}s, found {total_new} new items")
    
    return total_new, feed_counts

def calculate_time_allocation(
    estimated_news: int,
    previous_cycle_news: int = -1,
    min_harvest_time: float = 10.0,
    max_harvest_time: float = 25.0,
    total_cycle_time: float = 55.0
) -> Tuple[float, float]:
    """
    Calculate optimal time allocation between news harvesting and LLM processing.
    
    Args:
        estimated_news: Estimated number of new news items
        previous_cycle_news: Number of items from previous cycle (-1 if unknown)
        min_harvest_time: Minimum time for news harvesting
        max_harvest_time: Maximum time for news harvesting  
        total_cycle_time: Total time available in the cycle
        
    Returns:
        Tuple of (harvest_time, llm_time)
    """
    # Base allocation on estimated news volume
    if estimated_news == 0:
        # No news - minimal harvest, maximum LLM time
        if previous_cycle_news > 0:
            # Still process previous batch thoroughly
            harvest_time = min_harvest_time
        else:
            # Really nothing to do - ultra quick check
            harvest_time = 5.0
    elif estimated_news <= 3:
        # Very few items - quick harvest
        harvest_time = min_harvest_time
    elif estimated_news <= 10:
        # Moderate items - standard harvest
        harvest_time = 15.0
    elif estimated_news <= 20:
        # Many items - standard harvest
        harvest_time = 20.0
    else:
        # Lots of items - give more time to harvest
        harvest_time = max_harvest_time
    
    # Ensure we don't exceed limits (but allow 5s quick check when nothing to do)
    if estimated_news == 0 and previous_cycle_news <= 0:
        harvest_time = 5.0  # Allow ultra-quick check when truly nothing to do
    else:
        harvest_time = max(min_harvest_time, min(harvest_time, max_harvest_time))
    
    # Calculate LLM time as remainder
    llm_time = total_cycle_time - harvest_time
    
    return harvest_time, llm_time

def get_smart_schedule(
    feeds_file: str = None,
    previous_news_count: int = -1,
    verbose: bool = False,
    recent_timeouts: int = 0
) -> Dict[str, float]:
    """
    Get smart scheduling recommendation based on news availability.
    
    Args:
        feeds_file: Optional path to extra feeds file
        previous_news_count: Number of news items from previous cycle
        verbose: Enable verbose output
        recent_timeouts: Number of recent LLM timeouts
        
    Returns:
        Dictionary with scheduling recommendations:
        {
            'estimated_news': int,
            'harvest_time': float,
            'llm_time': float,
            'check_time': float,
            'confidence': str  # 'high', 'medium', 'low'
        }
    """
    start = time.time()
    
    # Quick check (5 seconds max)
    raw_estimate, feed_counts = estimate_available_news(
        feeds_file=feeds_file,
        max_check_time=5.0,
        verbose=verbose
    )
    
    check_time = time.time() - start
    
    # Try to use enhanced scheduler if available
    try:
        from smart_scheduler import integrate_smart_scheduler
        enhanced = integrate_smart_scheduler(
            raw_estimate, 
            previous_news_count,
            recent_timeouts,
            verbose
        )
        
        estimated_news = enhanced['estimated_news']
        harvest_time = enhanced['harvest_time']
        llm_time = enhanced['llm_time']
        
        # Convert confidence percentage to high/medium/low
        conf_pct = enhanced.get('confidence', 0)
        if conf_pct > 0.7:
            confidence = 'high'
        elif conf_pct > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
            
        if verbose and raw_estimate != estimated_news:
            print(f"[smart_schedule] Adjusted estimate: {raw_estimate} -> {estimated_news}")
            
    except ImportError:
        # Fallback to original calculation
        estimated_news = raw_estimate
        harvest_time, llm_time = calculate_time_allocation(
            estimated_news=estimated_news,
            previous_cycle_news=previous_news_count
        )
        
        # Determine confidence in estimate
        if check_time < 2.0 and len(feed_counts) > 10:
            confidence = 'high'
        elif check_time < 4.0 and len(feed_counts) > 5:
            confidence = 'medium'
        else:
            confidence = 'low'
    
    schedule = {
        'estimated_news': estimated_news,
        'harvest_time': harvest_time,
        'llm_time': llm_time,
        'check_time': check_time,
        'confidence': confidence,
        'active_feeds': len(feed_counts),
        'raw_estimate': raw_estimate
    }
    
    if verbose:
        print(f"[smart_schedule] News estimate: {estimated_news} items, "
              f"Harvest: {harvest_time:.0f}s, LLM: {llm_time:.0f}s, "
              f"Confidence: {confidence}")
    
    return schedule

def main():
    """Test the news checker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick news availability checker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--feeds-file", type=str, help="Extra feeds file")
    args = parser.parse_args()
    
    print("Checking news availability...")
    print("="*60)
    
    # Get smart schedule
    schedule = get_smart_schedule(
        feeds_file=args.feeds_file,
        previous_news_count=10,  # Simulate having processed 10 items before
        verbose=args.verbose
    )
    
    print(f"\nSmart Schedule Recommendation:")
    print(f"  Estimated new items: {schedule['estimated_news']}")
    print(f"  Active feeds: {schedule['active_feeds']}")
    print(f"  Check time: {schedule['check_time']:.1f}s")
    print(f"  Confidence: {schedule['confidence']}")
    print(f"\nTime Allocation:")
    print(f"  News harvest: {schedule['harvest_time']:.0f}s")
    print(f"  LLM processing: {schedule['llm_time']:.0f}s")
    print(f"  Total cycle: {schedule['harvest_time'] + schedule['llm_time']:.0f}s")
    
    # Test different scenarios
    print("\n" + "="*60)
    print("Scenario Testing:")
    print("="*60)
    
    scenarios = [
        (0, "No news"),
        (2, "Very few items"),
        (5, "Few items"),
        (10, "Moderate items"),
        (25, "Many items"),
        (50, "Lots of items")
    ]
    
    for count, desc in scenarios:
        harvest, llm = calculate_time_allocation(count)
        print(f"{desc:20s} ({count:3d} items): Harvest {harvest:4.0f}s, LLM {llm:4.0f}s")

if __name__ == "__main__":
    main()