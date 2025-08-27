#!/usr/bin/env python3
"""Test smart scheduling logic without requiring feedparser"""

import sys
from pathlib import Path

# Test the time allocation logic
def calculate_time_allocation(
    estimated_news: int,
    previous_cycle_news: int = -1,
    min_harvest_time: float = 10.0,
    max_harvest_time: float = 25.0,
    total_cycle_time: float = 55.0
):
    """Calculate optimal time allocation between news harvesting and LLM processing."""
    
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

print("="*60)
print("SMART SCHEDULING TEST")
print("="*60)

print("\nScenario 1: No news, no previous batch")
harvest, llm = calculate_time_allocation(0, 0)
print(f"  Harvest: {harvest:.0f}s, LLM: {llm:.0f}s")
assert harvest == 5.0, "Should do quick check only"

print("\nScenario 2: No news, but previous batch exists")
harvest, llm = calculate_time_allocation(0, 10)
print(f"  Harvest: {harvest:.0f}s, LLM: {llm:.0f}s")
assert harvest == 10.0, "Should do minimal harvest"

print("\nScenario 3: Very few news items (2)")
harvest, llm = calculate_time_allocation(2, 5)
print(f"  Harvest: {harvest:.0f}s, LLM: {llm:.0f}s")
assert harvest == 10.0, "Should do quick harvest"

print("\nScenario 4: Moderate news (8 items)")
harvest, llm = calculate_time_allocation(8, 10)
print(f"  Harvest: {harvest:.0f}s, LLM: {llm:.0f}s")
assert harvest == 15.0, "Should do standard harvest"

print("\nScenario 5: Many news items (15)")
harvest, llm = calculate_time_allocation(15, 12)
print(f"  Harvest: {harvest:.0f}s, LLM: {llm:.0f}s")
assert harvest == 20.0, "Should do standard harvest"

print("\nScenario 6: Lots of news (30 items)")
harvest, llm = calculate_time_allocation(30, 20)
print(f"  Harvest: {harvest:.0f}s, LLM: {llm:.0f}s")
assert harvest == 25.0, "Should do max harvest"

print("\n" + "="*60)
print("ALL TESTS PASSED")
print("="*60)

print("\n## Summary of Smart Scheduling Logic:")
print("1. **No news + No previous**: 5s harvest (quick check), 50s LLM")
print("2. **No news + Previous batch**: 10s harvest, 45s LLM (more time for LLM)")
print("3. **1-3 items**: 10s harvest, 45s LLM")
print("4. **4-10 items**: 15s harvest, 40s LLM")
print("5. **11-20 items**: 20s harvest, 35s LLM")
print("6. **20+ items**: 25s harvest, 30s LLM")

print("\n## Benefits:")
print("✓ When news is scarce, LLM gets more time to analyze thoroughly")
print("✓ When news is abundant, adequate time for harvesting")
print("✓ Previous batch processing continues even with no new news")
print("✓ Adaptive allocation maximizes resource utilization")
print("✓ Total cycle time stays within 55 seconds (leaving 5s buffer)")