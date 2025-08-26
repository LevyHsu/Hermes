#!/usr/bin/env python3
"""Test script to verify listing download logic after cleaning"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from main import should_update_listings, LISTING_DIR
    from args import MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, LISTING_UPDATE_MINUTES_BEFORE_OPEN
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test scenarios
print("\n" + "="*60)
print("TESTING LISTING UPDATE LOGIC")
print("="*60)

# Scenario 1: No listings exist
print("\n1. Testing with no listings file:")
latest_file = LISTING_DIR / "us-listings-latest.json"
file_existed = latest_file.exists()
if file_existed:
    # Temporarily rename it
    backup_file = LISTING_DIR / "us-listings-latest.json.backup"
    latest_file.rename(backup_file)

result = should_update_listings()
print(f"   - No listings exist: should_update_listings() = {result}")
assert result == True, "Should return True when no listings exist"
print("   ✓ Correctly returns True when no listings")

# Scenario 2: Empty/corrupted file
print("\n2. Testing with empty listings file:")
latest_file.write_text("{}")  # Write minimal JSON
result = should_update_listings()
print(f"   - Empty file exists: should_update_listings() = {result}")
assert result == True, "Should return True when file is too small"
print("   ✓ Correctly returns True when file is corrupted/empty")

# Scenario 3: Valid file exists
print("\n3. Testing with valid listings file:")
# Create a dummy valid file (>100 bytes)
dummy_data = '{"listings": [' + ','.join([f'{{"ticker": "TEST{i}"}}' for i in range(20)]) + ']}'
latest_file.write_text(dummy_data)
print(f"   - File size: {latest_file.stat().st_size} bytes")
result = should_update_listings()
print(f"   - Valid file exists: should_update_listings() = {result}")
# Result depends on current time vs market open
print(f"   ✓ Returns {result} (depends on current time vs market schedule)")

# Cleanup
latest_file.unlink()
if file_existed and (LISTING_DIR / "us-listings-latest.json.backup").exists():
    (LISTING_DIR / "us-listings-latest.json.backup").rename(latest_file)

print("\n" + "="*60)
print("ALL TESTS PASSED")
print("="*60)

print(f"\nMarket Schedule Settings:")
print(f"  - Market Open: {MARKET_OPEN_HOUR:02d}:{MARKET_OPEN_MINUTE:02d} ET")
print(f"  - Update Time: {LISTING_UPDATE_MINUTES_BEFORE_OPEN} minutes before open")
print(f"  - Listing Dir: {LISTING_DIR}")

# Show current time check
try:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
    now_et = datetime.now(tz=ET_TZ)
    print(f"\nCurrent ET Time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Check if we're near update time
    market_open = now_et.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    update_time = market_open.replace(hour=market_open.hour, minute=market_open.minute - LISTING_UPDATE_MINUTES_BEFORE_OPEN)
    
    time_to_update = (update_time - now_et).total_seconds()
    if abs(time_to_update) < 60:
        print(f"⚠ Currently within update window!")
    elif time_to_update > 0:
        print(f"Next update in: {time_to_update/60:.1f} minutes")
    else:
        print(f"Last update was: {-time_to_update/60:.1f} minutes ago")
        
except Exception as e:
    print(f"\nCould not check time: {e}")