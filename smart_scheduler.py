#!/usr/bin/env python3
"""
Enhanced Smart Scheduling with Historical Learning and Adaptive Correction.

This module improves scheduling accuracy by:
1. Tracking historical estimation accuracy
2. Using exponentially weighted moving averages (EWMA)
3. Learning time-of-day patterns
4. Applying adaptive correction factors
5. Feed-specific productivity tracking
"""

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import math

DATA_DIR = Path("data")
SCHEDULER_DB = DATA_DIR / ".scheduler_history.db"


class SmartScheduler:
    """Adaptive scheduler with historical learning."""
    
    def __init__(self, db_path: Path = SCHEDULER_DB):
        """Initialize scheduler with database for historical tracking."""
        self.db_path = db_path
        self.ensure_database()
        
        # Configuration
        self.min_harvest_time = 10.0
        self.max_harvest_time = 25.0
        self.total_cycle_time = 55.0
        self.llm_time_buffer = 5.0  # Reserve for LLM startup
        
        # EWMA parameters
        self.alpha = 0.3  # Weight for new observations
        self.correction_alpha = 0.4  # Faster adaptation for correction factor
        
    def ensure_database(self):
        """Create database tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        con = sqlite3.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS estimation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                estimated_count INTEGER,
                actual_count INTEGER,
                harvest_time REAL,
                llm_time REAL,
                cycle_duration REAL,
                llm_success BOOLEAN
            )
        """)
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS hourly_patterns (
                hour INTEGER PRIMARY KEY,
                avg_news_count REAL,
                avg_error_ratio REAL,
                sample_count INTEGER,
                last_updated DATETIME
            )
        """)
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS feed_productivity (
                feed_url TEXT PRIMARY KEY,
                total_items INTEGER DEFAULT 0,
                check_count INTEGER DEFAULT 0,
                avg_items REAL DEFAULT 0,
                last_productive DATETIME,
                reliability_score REAL DEFAULT 1.0
            )
        """)
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        con.commit()
        con.close()
        
    def record_estimation(self, estimated: int, actual: int, harvest_time: float,
                         llm_time: float, cycle_duration: float, llm_success: bool):
        """Record actual vs estimated for learning."""
        now = datetime.now()
        hour = now.hour
        dow = now.weekday()
        
        con = sqlite3.connect(self.db_path)
        con.execute("""
            INSERT INTO estimation_history 
            (hour_of_day, day_of_week, estimated_count, actual_count,
             harvest_time, llm_time, cycle_duration, llm_success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (hour, dow, estimated, actual, harvest_time, llm_time, cycle_duration, llm_success))
        
        # Update hourly pattern
        self._update_hourly_pattern(con, hour, actual, estimated)
        
        # Update correction factor
        self._update_correction_factor(con, estimated, actual)
        
        con.commit()
        con.close()
        
    def _update_hourly_pattern(self, con, hour: int, actual: int, estimated: int):
        """Update hourly pattern statistics using EWMA."""
        cursor = con.execute(
            "SELECT avg_news_count, avg_error_ratio, sample_count FROM hourly_patterns WHERE hour = ?",
            (hour,)
        )
        row = cursor.fetchone()
        
        error_ratio = actual / max(estimated, 1) if estimated > 0 else 1.0
        
        if row:
            old_avg, old_error, samples = row
            # EWMA update
            new_avg = self.alpha * actual + (1 - self.alpha) * old_avg
            new_error = self.alpha * error_ratio + (1 - self.alpha) * old_error
            new_samples = samples + 1
            
            con.execute("""
                UPDATE hourly_patterns 
                SET avg_news_count = ?, avg_error_ratio = ?, 
                    sample_count = ?, last_updated = CURRENT_TIMESTAMP
                WHERE hour = ?
            """, (new_avg, new_error, new_samples, hour))
        else:
            con.execute("""
                INSERT INTO hourly_patterns (hour, avg_news_count, avg_error_ratio, sample_count)
                VALUES (?, ?, ?, 1)
            """, (hour, actual, error_ratio))
            
    def _update_correction_factor(self, con, estimated: int, actual: int):
        """Update global correction factor based on recent accuracy."""
        if estimated <= 0:
            return
            
        error_ratio = actual / estimated
        
        # Get current correction factor
        cursor = con.execute("SELECT value FROM scheduler_state WHERE key = 'correction_factor'")
        row = cursor.fetchone()
        
        if row:
            old_factor = float(row[0])
            # Apply bounded update to prevent wild swings
            new_factor = old_factor * (1 - self.correction_alpha) + error_ratio * self.correction_alpha
            new_factor = max(0.5, min(3.0, new_factor))  # Bound between 0.5x and 3x
            
            con.execute(
                "UPDATE scheduler_state SET value = ?, updated = CURRENT_TIMESTAMP WHERE key = 'correction_factor'",
                (str(new_factor),)
            )
        else:
            con.execute(
                "INSERT INTO scheduler_state (key, value) VALUES ('correction_factor', '1.0')"
            )
            
    def get_correction_factor(self) -> float:
        """Get current correction factor for estimates."""
        con = sqlite3.connect(self.db_path)
        cursor = con.execute("SELECT value FROM scheduler_state WHERE key = 'correction_factor'")
        row = cursor.fetchone()
        con.close()
        
        return float(row[0]) if row else 1.0
        
    def get_hourly_multiplier(self, hour: int) -> float:
        """Get hour-specific multiplier based on historical patterns."""
        con = sqlite3.connect(self.db_path)
        cursor = con.execute(
            "SELECT avg_error_ratio FROM hourly_patterns WHERE hour = ?",
            (hour,)
        )
        row = cursor.fetchone()
        con.close()
        
        return row[0] if row else 1.0
        
    def calculate_adaptive_allocation(self, raw_estimate: int, 
                                     previous_news_count: int = -1,
                                     recent_timeouts: int = 0) -> Dict:
        """
        Calculate time allocation with adaptive corrections.
        
        Args:
            raw_estimate: Raw estimated news count
            previous_news_count: Items from previous cycle
            recent_timeouts: Number of recent LLM timeouts
            
        Returns:
            Dict with scheduling parameters
        """
        now = datetime.now()
        hour = now.hour
        
        # Apply corrections to raw estimate
        correction = self.get_correction_factor()
        hourly_mult = self.get_hourly_multiplier(hour)
        
        # Adjusted estimate with bounded correction
        adjusted_estimate = int(raw_estimate * correction * hourly_mult)
        adjusted_estimate = max(0, min(adjusted_estimate, 200))  # Cap at reasonable max
        
        # Check if we're in market hours (9:30 AM - 4:00 PM ET)
        et_hour = (hour - 5) % 24  # Simple ET conversion (adjust for DST)
        is_market_hours = 9 <= et_hour <= 16
        
        # Base time calculation with market hours boost
        if adjusted_estimate == 0 and previous_news_count <= 0:
            # Truly nothing to do
            harvest_time = 5.0
        elif adjusted_estimate <= 5:
            harvest_time = 10.0
        elif adjusted_estimate <= 15:
            harvest_time = 15.0
        elif adjusted_estimate <= 30:
            harvest_time = 20.0
        else:
            harvest_time = self.max_harvest_time
            
        # Market hours adjustment
        if is_market_hours and adjusted_estimate > 10:
            harvest_time = min(harvest_time + 3.0, self.max_harvest_time)
            
        # Timeout adjustment - reduce harvest time if LLM is timing out
        if recent_timeouts > 0:
            timeout_penalty = min(recent_timeouts * 2.0, 5.0)
            harvest_time = max(self.min_harvest_time, harvest_time - timeout_penalty)
            
        # Calculate LLM time
        llm_time = self.total_cycle_time - harvest_time
        
        # Get recent success rate
        success_rate = self._get_recent_success_rate()
        
        return {
            'harvest_time': harvest_time,
            'llm_time': llm_time,
            'raw_estimate': raw_estimate,
            'adjusted_estimate': adjusted_estimate,
            'correction_factor': correction,
            'hourly_multiplier': hourly_mult,
            'is_market_hours': is_market_hours,
            'recent_timeouts': recent_timeouts,
            'success_rate': success_rate,
            'confidence': self._calculate_confidence()
        }
        
    def _get_recent_success_rate(self, window_hours: int = 1) -> float:
        """Get success rate for recent cycles."""
        since = datetime.now() - timedelta(hours=window_hours)
        
        con = sqlite3.connect(self.db_path)
        cursor = con.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN llm_success = 1 THEN 1 ELSE 0 END) as successes
            FROM estimation_history
            WHERE timestamp > ?
        """, (since,))
        
        row = cursor.fetchone()
        con.close()
        
        if row and row[0] > 0:
            return row[1] / row[0]
        return 1.0  # Assume success if no data
        
    def _calculate_confidence(self) -> float:
        """Calculate confidence in scheduling decision based on historical data."""
        con = sqlite3.connect(self.db_path)
        
        # Check sample size
        cursor = con.execute("SELECT COUNT(*) FROM estimation_history")
        samples = cursor.fetchone()[0]
        
        # Check recent accuracy
        cursor = con.execute("""
            SELECT AVG(ABS(estimated_count - actual_count) * 1.0 / MAX(actual_count, 1))
            FROM estimation_history
            WHERE timestamp > datetime('now', '-1 hour')
        """)
        row = cursor.fetchone()
        recent_error = row[0] if row and row[0] else 1.0
        
        con.close()
        
        # Confidence based on sample size and recent accuracy
        sample_confidence = min(samples / 100.0, 1.0)  # Max confidence at 100 samples
        accuracy_confidence = max(0, 1.0 - recent_error)
        
        return sample_confidence * accuracy_confidence
        
    def get_statistics(self) -> Dict:
        """Get scheduler performance statistics."""
        con = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Overall stats
        cursor = con.execute("""
            SELECT COUNT(*) as cycles,
                   AVG(estimated_count) as avg_estimated,
                   AVG(actual_count) as avg_actual,
                   AVG(ABS(estimated_count - actual_count)) as avg_error,
                   SUM(CASE WHEN llm_success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM estimation_history
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        
        row = cursor.fetchone()
        if row:
            stats['24h'] = {
                'cycles': row[0],
                'avg_estimated': row[1],
                'avg_actual': row[2],
                'avg_error': row[3],
                'success_rate': row[4]
            }
            
        # Current correction factor
        cursor = con.execute("SELECT value FROM scheduler_state WHERE key = 'correction_factor'")
        row = cursor.fetchone()
        stats['correction_factor'] = float(row[0]) if row else 1.0
        
        con.close()
        return stats


def integrate_smart_scheduler(estimated_news: int, previous_news_count: int = -1,
                            recent_timeouts: int = 0, verbose: bool = False) -> Dict:
    """
    Integration function for existing codebase.
    
    Returns dict compatible with current system.
    """
    scheduler = SmartScheduler()
    result = scheduler.calculate_adaptive_allocation(
        estimated_news, 
        previous_news_count,
        recent_timeouts
    )
    
    if verbose:
        print(f"[SmartScheduler] Raw: {result['raw_estimate']}, "
              f"Adjusted: {result['adjusted_estimate']} "
              f"(x{result['correction_factor']:.2f} correction, "
              f"x{result['hourly_multiplier']:.2f} hourly), "
              f"Confidence: {result['confidence']:.1%}")
              
    return {
        'estimated_news': result['adjusted_estimate'],
        'harvest_time': result['harvest_time'],
        'llm_time': result['llm_time'],
        'confidence': result['confidence'],
        'raw_estimate': result['raw_estimate']
    }