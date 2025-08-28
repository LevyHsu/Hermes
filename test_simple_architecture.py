#!/usr/bin/env python3
"""
Test that the simplified architecture handles signals correctly.
"""

import signal
import sys
import time
import logging

# Test basic signal handling
STOP = False

def handle_signal(signum, frame):
    global STOP
    STOP = True
    print(f"\n✅ Signal {signum} received! Setting STOP=True", file=sys.stderr)

# Register handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

print("Testing signal handling in simplified architecture...")
print("Press Ctrl+C to test...")

# Simple loop that checks STOP frequently
counter = 0
while not STOP and counter < 100:
    counter += 1
    print(f"Working... {counter}", end='\r')
    
    # Simulate work with interruptible sleep
    for _ in range(10):
        if STOP:
            break
        time.sleep(0.1)

if STOP:
    print("\n✅ SUCCESS: Signal handled correctly!")
    sys.exit(0)
else:
    print("\n❌ Timed out")
    sys.exit(1)