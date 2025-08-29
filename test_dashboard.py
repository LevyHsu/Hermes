#!/usr/bin/env python3
"""Test script for the status dashboard with console output."""

from status_dashboard import StatusDashboard
import time
import random
from datetime import datetime

def main():
    dashboard = StatusDashboard()
    
    # Start the dashboard
    dashboard.run()
    
    try:
        # Simulate activity
        for i in range(30):
            time.sleep(1)
            
            # Add console logs
            if i % 3 == 0:
                dashboard.add_console_log(f"Processing minute 250828{100+i:03d}", "info")
            
            if i % 5 == 0:
                dashboard.add_console_log(f"Added news to queue: {random.randint(5, 20)} items", "success")
                
            if i % 7 == 0:
                dashboard.add_console_log("Warning: Queue nearly full", "warning")
                
            if i % 11 == 0:
                dashboard.add_console_log("Error: Failed to fetch price data", "error")
            
            # Update queue stats
            dashboard.update_queue_stats({
                'queue_size': random.randint(0, 64),
                'items_added': i * 3,
                'items_processed': i * 2,
                'items_dropped': max(0, i - 25)
            })
            
            # Add random decisions
            if random.random() > 0.7:
                ticker = random.choice(['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'])
                confidence = random.randint(60, 95)
                revised = random.randint(70, 95) if confidence >= 80 else 0
                
                dashboard.add_decision({
                    'ticker': ticker,
                    'action': random.choice(['BUY', 'SELL']),
                    'confidence': confidence,
                    'revised_confidence': revised,
                    'expected_price': random.uniform(100, 500) if revised > 0 else None,
                    'reason': f"Test reason for {ticker}"
                })
                
                # Log high confidence to console
                if confidence >= 80:
                    dashboard.add_console_log(f"HIGH CONFIDENCE: {ticker} @ {confidence}%", "success")
                if revised >= 70:
                    dashboard.add_console_log(f"REFINED SIGNAL: {ticker} @ {revised}%", "warning")
            
            # Set processing
            dashboard.set_processing(f"250828{100+i:03d}")
            
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        dashboard.stop()

if __name__ == "__main__":
    main()