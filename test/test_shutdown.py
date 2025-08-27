#!/usr/bin/env python3
"""Test script to verify graceful shutdown is working"""

import subprocess
import time
import signal
import os
import sys
from pathlib import Path

def test_shutdown():
    """Test that main.py shuts down gracefully with Ctrl+C"""
    
    print("Starting main.py in test mode...")
    print("Will send SIGINT (Ctrl+C) after 5 seconds...")
    print("-" * 60)
    
    # Start main.py (go up one directory)
    main_path = Path(__file__).parent.parent / "main.py"
    process = subprocess.Popen(
        [sys.executable, str(main_path), "--dry-run", "--verbose"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid if os.name != 'nt' else None
    )
    
    # Let it run for 5 seconds
    start_time = time.time()
    timeout = 5
    
    try:
        # Read output for 5 seconds
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                print("Process ended prematurely")
                break
            time.sleep(0.1)
        
        # Send SIGINT (Ctrl+C)
        print("\n" + "-" * 60)
        print("Sending SIGINT (Ctrl+C) to process...")
        
        if os.name != 'nt':
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        else:
            process.send_signal(signal.CTRL_C_EVENT)
        
        # Wait for graceful shutdown (max 10 seconds)
        print("Waiting for graceful shutdown...")
        shutdown_start = time.time()
        shutdown_timeout = 10
        
        while process.poll() is None and (time.time() - shutdown_start) < shutdown_timeout:
            time.sleep(0.1)
        
        if process.poll() is None:
            print("WARNING: Process did not shut down gracefully, force killing...")
            process.kill()
            process.wait()
            return False
        else:
            print(f"✓ Process shut down gracefully in {time.time() - shutdown_start:.1f} seconds")
            return True
            
    except Exception as e:
        print(f"Error during test: {e}")
        try:
            process.kill()
        except:
            pass
        return False
    
    finally:
        # Ensure process is dead
        if process.poll() is None:
            try:
                process.kill()
                process.wait()
            except:
                pass

if __name__ == "__main__":
    print("="*60)
    print("TESTING GRACEFUL SHUTDOWN")
    print("="*60)
    
    success = test_shutdown()
    
    print("\n" + "="*60)
    if success:
        print("TEST PASSED: Graceful shutdown is working correctly")
        sys.exit(0)
    else:
        print("TEST FAILED: Graceful shutdown not working properly")
        sys.exit(1)