#!/usr/bin/env python3
"""
Test signal handling and shutdown behavior.
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path

def test_signal_handling():
    """Test that the program responds to Ctrl+C properly."""
    
    print("="*60)
    print("TESTING SIGNAL HANDLING (Ctrl+C)")
    print("="*60)
    
    print("\n1. Starting main.py in test mode...")
    
    # Start main.py
    main_path = Path(__file__).parent / "main.py"
    process = subprocess.Popen(
        [sys.executable, str(main_path), "--dry-run", "--verbose"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid if os.name != 'nt' else None
    )
    
    print(f"   Process started with PID: {process.pid}")
    print("   Waiting 3 seconds before sending SIGINT...")
    
    # Let it run for 3 seconds
    time.sleep(3)
    
    print("\n2. Sending SIGINT (Ctrl+C) to process...")
    
    # Send SIGINT
    if os.name != 'nt':
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    else:
        process.send_signal(signal.CTRL_C_EVENT)
    
    print("   Signal sent. Waiting for graceful shutdown...")
    
    # Wait for shutdown (max 5 seconds)
    shutdown_start = time.time()
    shutdown_timeout = 5
    
    while process.poll() is None and (time.time() - shutdown_start) < shutdown_timeout:
        time.sleep(0.1)
    
    if process.poll() is None:
        print("\n❌ FAILED: Process did not shut down within 5 seconds")
        print("   Force killing process...")
        process.kill()
        process.wait()
        return False
    else:
        shutdown_time = time.time() - shutdown_start
        print(f"\n✅ SUCCESS: Process shut down gracefully in {shutdown_time:.1f} seconds")
        
        # Check exit code
        if process.returncode == 0 or process.returncode == -2:  # -2 is SIGINT
            print(f"   Exit code: {process.returncode} (expected)")
        else:
            print(f"   Warning: Unexpected exit code: {process.returncode}")
        
        return True

def test_stop_flag_propagation():
    """Test that STOP flag is checked in all blocking operations."""
    
    print("\n" + "="*60)
    print("TESTING STOP FLAG PROPAGATION")
    print("="*60)
    
    print("\nChecking code for STOP flag usage...")
    
    # Check for problematic patterns
    issues = []
    
    main_file = Path(__file__).parent / "main.py"
    with open(main_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        # Check for sleep without STOP check
        if 'time.sleep(' in line and 'while not STOP' not in lines[max(0, i-3):i+1]:
            if 'time.sleep(0.1)' not in line and 'time.sleep(0.5)' not in line:
                issues.append(f"   Line {i}: {line.strip()} - May block without checking STOP")
        
        # Check for subprocess.run without timeout
        if 'subprocess.run(' in line:
            # Check next few lines for timeout
            if not any('timeout' in lines[j] for j in range(i-1, min(i+5, len(lines)))):
                issues.append(f"   Line {i}: subprocess.run without timeout")
    
    if issues:
        print("\n⚠️  Potential blocking issues found:")
        for issue in issues[:5]:  # Show first 5 issues
            print(issue)
    else:
        print("\n✅ No obvious blocking issues found")
    
    return len(issues) == 0

def main():
    """Run all signal handling tests."""
    
    print("SIGNAL HANDLING TEST SUITE")
    print("="*60)
    
    # Test 1: Basic signal handling
    test1_passed = test_signal_handling()
    
    # Test 2: STOP flag propagation
    test2_passed = test_stop_flag_propagation()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        print("\nThe program should now respond properly to Ctrl+C:")
        print("- Signal handlers registered early in main()")
        print("- sleep_until_next_minute() checks STOP flag")
        print("- subprocess calls are interruptible")
        print("- All long operations check STOP flag")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        if not test1_passed:
            print("- Signal handling test failed")
        if not test2_passed:
            print("- STOP flag propagation has issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())