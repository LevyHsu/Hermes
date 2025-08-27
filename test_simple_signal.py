#!/usr/bin/env python3
"""
Simple test of signal handling without process groups.
"""

import subprocess
import time
import signal
import sys

def test():
    print("Starting main.py...")
    
    # Start without process group
    process = subprocess.Popen(
        [sys.executable, "main.py", "--dry-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    print(f"PID: {process.pid}")
    print("Waiting 2 seconds...")
    time.sleep(2)
    
    print("Sending SIGINT...")
    process.send_signal(signal.SIGINT)
    
    print("Waiting for shutdown...")
    try:
        process.wait(timeout=3)
        print(f"✓ Process terminated with code {process.returncode}")
    except subprocess.TimeoutExpired:
        print("✗ Process still running after 3 seconds")
        process.kill()
        
if __name__ == "__main__":
    test()