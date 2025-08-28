#!/usr/bin/env python3
"""
Rich terminal status dashboard for Hermes
Shows queue status, current processing, and trading decisions with colors
"""

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box


class StatusDashboard:
    """Terminal dashboard for monitoring bot status."""
    
    def __init__(self, queue=None):
        self.console = Console()
        self.queue = queue
        self.current_processing = None
        self.recent_decisions = deque(maxlen=10)
        self.console_logs = deque(maxlen=15)  # Keep last 15 console messages
        self.stats = {
            'items_processed': 0,
            'high_confidence': 0,
            'revised_decisions': 0,
            'total_decisions': 0,
            'queue_size': 0,
            'items_added': 0,
            'items_dropped': 0
        }
        self.running = False
        self.thread = None
        
    def create_header(self) -> Panel:
        """Create header panel."""
        header_text = Text()
        header_text.append("⚡ Hermes ", style="bold cyan")
        header_text.append("Priority Queue Trading System", style="white")
        header_text.append(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        
        return Panel(
            Align.center(header_text),
            style="bold blue",
            box=box.DOUBLE
        )
    
    def create_queue_status(self) -> Panel:
        """Create queue status panel."""
        table = Table(show_header=False, box=None, padding=0)
        table.add_column("Label", style="cyan", width=20)
        table.add_column("Value", style="bold")
        
        # Queue metrics
        queue_pct = (self.stats['queue_size'] / 64) * 100 if self.stats['queue_size'] <= 64 else 100
        queue_color = "green" if queue_pct < 50 else "yellow" if queue_pct < 80 else "red"
        
        table.add_row("Queue Size:", f"[{queue_color}]{self.stats['queue_size']}/64 ({queue_pct:.0f}%)[/{queue_color}]")
        table.add_row("Items Added:", f"[green]{self.stats['items_added']}[/green]")
        table.add_row("Items Processed:", f"[blue]{self.stats['items_processed']}[/blue]")
        table.add_row("Items Dropped:", f"[red]{self.stats['items_dropped']}[/red]" if self.stats['items_dropped'] > 0 else "0")
        
        if self.current_processing:
            table.add_row("", "")  # Spacer
            table.add_row("Processing:", f"[yellow]{self.current_processing}[/yellow]")
        
        return Panel(
            table,
            title="📊 Queue Status",
            border_style="cyan"
        )
    
    def create_decisions_panel(self) -> Panel:
        """Create recent decisions panel."""
        # Import thresholds from args
        try:
            from args import HIGH_CONFIDENCE_THRESHOLD, REVISED_CONFIDENCE_THRESHOLD
            high_threshold = HIGH_CONFIDENCE_THRESHOLD
            revised_threshold = REVISED_CONFIDENCE_THRESHOLD
        except ImportError:
            high_threshold = 80
            revised_threshold = 70
            
        table = Table(show_header=True, box=box.SIMPLE, padding=0)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Ticker", style="cyan", width=6)
        table.add_column("Action", width=6)
        table.add_column("Conf", width=4)
        table.add_column("Rev", width=4)
        table.add_column("Target", width=8)
        
        for decision in self.recent_decisions:
            time_str = decision.get('time', '')
            ticker = decision.get('ticker', '')
            action = decision.get('action', '')
            confidence = decision.get('confidence', 0)
            revised = decision.get('revised_confidence', 0)
            target = decision.get('expected_price', '')
            
            # Color coding for action
            action_color = "green" if action == "BUY" else "red"
            action_text = f"[{action_color}]{action}[/{action_color}]"
            
            # Color coding for confidence based on HIGH_CONFIDENCE_THRESHOLD
            conf_color = "bold green" if confidence >= high_threshold else "yellow" if confidence >= 60 else "white"
            conf_text = f"[{conf_color}]{confidence}%[/{conf_color}]"
            
            # Revised confidence based on REVISED_CONFIDENCE_THRESHOLD
            rev_text = ""
            if revised > 0:
                rev_color = "bold magenta" if revised >= revised_threshold else "cyan"
                rev_text = f"[{rev_color}]{revised}%[/{rev_color}]"
            
            # Target price
            target_text = f"${target:.2f}" if target else "-"
            
            table.add_row(time_str, ticker, action_text, conf_text, rev_text, target_text)
        
        return Panel(
            table,
            title="📈 Recent Decisions",
            border_style="green"
        )
    
    def create_statistics_panel(self) -> Panel:
        """Create statistics panel."""
        # Import thresholds from args
        try:
            from args import HIGH_CONFIDENCE_THRESHOLD, REVISED_CONFIDENCE_THRESHOLD
            high_threshold = HIGH_CONFIDENCE_THRESHOLD
            revised_threshold = REVISED_CONFIDENCE_THRESHOLD
        except ImportError:
            high_threshold = 80
            revised_threshold = 70
            
        stats_text = Text()
        
        # Decision statistics
        stats_text.append("Trading Statistics\n", style="bold cyan")
        stats_text.append(f"Total Decisions: ", style="white")
        stats_text.append(f"{self.stats['total_decisions']}\n", style="bold white")
        
        stats_text.append(f"High Confidence (≥{high_threshold}%): ", style="white")
        stats_text.append(f"{self.stats['high_confidence']}\n", style="bold green")
        
        stats_text.append(f"Revised (≥{revised_threshold}%): ", style="white")
        stats_text.append(f"{self.stats['revised_decisions']}\n", style="bold magenta")
        
        # Success rate if available
        if self.stats['total_decisions'] > 0:
            high_conf_pct = (self.stats['high_confidence'] / self.stats['total_decisions']) * 100
            stats_text.append(f"\nHigh Conf Rate: ", style="white")
            color = "green" if high_conf_pct > 30 else "yellow" if high_conf_pct > 15 else "red"
            stats_text.append(f"{high_conf_pct:.1f}%", style=f"bold {color}")
        
        return Panel(
            stats_text,
            title="📊 Statistics",
            border_style="yellow"
        )
    
    def create_alerts_panel(self) -> Panel:
        """Create alerts panel for revised high confidence decisions."""
        alerts = []
        
        # Import thresholds from args
        try:
            from args import REVISED_CONFIDENCE_THRESHOLD
            threshold = REVISED_CONFIDENCE_THRESHOLD
        except ImportError:
            threshold = 70  # Fallback default
        
        # Check for recent revised high confidence decisions
        for decision in list(self.recent_decisions)[-3:]:  # Last 3 decisions
            # Only alert on REVISED confidence that meets threshold
            if decision.get('revised_confidence', 0) >= threshold:
                alert_text = Text()
                alert_text.append("⚠️ REVISED SIGNAL: ", style="bold magenta")  # No blink
                alert_text.append(f"{decision['ticker']} ", style="bold cyan")
                alert_text.append(f"{decision['action']} ", style="bold green" if decision['action'] == 'BUY' else "bold red")
                alert_text.append(f"@ {decision['revised_confidence']}%", style="bold yellow")
                if decision.get('expected_price'):
                    alert_text.append(f" Target: ${decision['expected_price']:.2f}", style="cyan")
                alerts.append(alert_text)
        
        if not alerts:
            alerts.append(Text("No revised high confidence signals", style="dim"))
        
        return Panel(
            Columns(alerts, padding=(0, 1)),
            title="🎯 Refined Alerts",
            border_style="magenta",
            box=box.DOUBLE if alerts and "REVISED SIGNAL" in str(alerts[0]) else box.ROUNDED
        )
    
    def create_console_panel(self) -> Panel:
        """Create console output panel."""
        console_text = Text()
        
        if self.console_logs:
            for log_entry in self.console_logs:
                # Parse log entry for styling
                if isinstance(log_entry, dict):
                    msg = log_entry.get('message', '')
                    level = log_entry.get('level', 'info')
                    timestamp = log_entry.get('timestamp', '')
                    
                    # Add timestamp if present
                    if timestamp:
                        console_text.append(f"[{timestamp}] ", style="dim cyan")
                    
                    # Style based on level
                    if level == 'error':
                        console_text.append(msg, style="red")
                    elif level == 'warning':
                        console_text.append(msg, style="yellow")
                    elif level == 'success':
                        console_text.append(msg, style="green")
                    else:
                        console_text.append(msg, style="white")
                else:
                    # Plain text log
                    console_text.append(str(log_entry), style="white")
                    
                console_text.append("\n")
        else:
            console_text.append("No console output yet...", style="dim")
        
        return Panel(
            console_text,
            title="📟 Console Output",
            border_style="blue",
            height=8,
            box=box.ROUNDED
        )
    
    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="alerts", size=5),
            Layout(name="console", size=8)  # Console output at bottom
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right", ratio=2)
        )
        
        layout["left"].split_column(
            Layout(name="queue", size=10),
            Layout(name="stats")
        )
        
        return layout
    
    def update_display(self) -> Layout:
        """Update the entire display."""
        layout = self.create_layout()
        
        layout["header"].update(self.create_header())
        layout["queue"].update(self.create_queue_status())
        layout["stats"].update(self.create_statistics_panel())
        layout["right"].update(self.create_decisions_panel())
        layout["alerts"].update(self.create_alerts_panel())
        layout["console"].update(self.create_console_panel())
        
        return layout
    
    def add_decision(self, decision: Dict[str, Any]):
        """Add a new decision to the dashboard."""
        # Import thresholds from args
        try:
            from args import HIGH_CONFIDENCE_THRESHOLD, REVISED_CONFIDENCE_THRESHOLD
            high_threshold = HIGH_CONFIDENCE_THRESHOLD
            revised_threshold = REVISED_CONFIDENCE_THRESHOLD
        except ImportError:
            high_threshold = 80
            revised_threshold = 70
            
        decision['time'] = datetime.now().strftime('%H:%M:%S')
        self.recent_decisions.append(decision)
        
        # Update statistics
        self.stats['total_decisions'] += 1
        if decision.get('confidence', 0) >= high_threshold:
            self.stats['high_confidence'] += 1
        if decision.get('revised_confidence', 0) >= revised_threshold:
            self.stats['revised_decisions'] += 1
    
    def update_queue_stats(self, stats: Dict[str, Any]):
        """Update queue statistics."""
        self.stats.update(stats)
    
    def set_processing(self, minute_key: Optional[str]):
        """Set current processing item."""
        self.current_processing = minute_key
    
    def add_console_log(self, message: str, level: str = "info", timestamp: str = None):
        """Add a console log message."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%H:%M:%S')
        
        self.console_logs.append({
            'message': message,
            'level': level,
            'timestamp': timestamp
        })
    
    def run(self):
        """Run the dashboard in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_display, daemon=True)
        self.thread.start()
    
    def _run_display(self):
        """Internal method to run the display loop."""
        try:
            with Live(self.update_display(), refresh_per_second=2, console=self.console) as live:
                while self.running:
                    time.sleep(0.5)
                    live.update(self.update_display())
        except Exception as e:
            self.console.print(f"[red]Dashboard error: {e}[/red]")
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


def create_simple_status_bar(queue_size: int, processing: str, high_conf: int) -> str:
    """Create a simple status bar string for logging."""
    bar = f"[Queue: {queue_size}/64 | Processing: {processing} | High Conf: {high_conf}]"
    return bar


if __name__ == "__main__":
    # Test the dashboard
    import random
    
    dashboard = StatusDashboard()
    dashboard.run()
    
    try:
        # Simulate some activity
        for i in range(60):
            time.sleep(2)
            
            # Update queue stats
            dashboard.update_queue_stats({
                'queue_size': random.randint(0, 64),
                'items_added': i * 2,
                'items_processed': i,
                'items_dropped': max(0, i - 50)
            })
            
            # Add random decision
            if random.random() > 0.5:
                dashboard.add_decision({
                    'ticker': random.choice(['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']),
                    'action': random.choice(['BUY', 'SELL']),
                    'confidence': random.randint(50, 95),
                    'revised_confidence': random.randint(70, 95) if random.random() > 0.5 else 0,
                    'expected_price': random.uniform(100, 500) if random.random() > 0.5 else None
                })
            
            # Set processing
            dashboard.set_processing(f"2508282{100+i:03d}")
            
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        dashboard.stop()