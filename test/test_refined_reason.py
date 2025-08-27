#!/usr/bin/env python3
"""Test script to verify refined_reason handling without needing LM Studio"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import llm module (go up one level from test/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import truncate_with_ellipsis, clean_text

# Simulate a refinement response with various reasoning lengths
test_responses = [
    {
        "ticker": "AAPL",
        "action": "BUY", 
        "previous_confidence": 75,
        "revised_confidence": 82,
        "reasoning": "Strong iPhone sales data combined with positive analyst upgrades suggest continued momentum in the near term despite macro headwinds",
        "horizon_hours": 24,
        "expected_high_price": 185.5
    },
    {
        "ticker": "MSFT",
        "action": "SELL",
        "previous_confidence": 60,
        "revised_confidence": 45,
        "reasoning": "Recent price action shows resistance at current levels. Google News headlines indicate mixed sentiment with concerns about Azure growth rates slowing. Technical indicators suggest a pullback is likely before the next leg up. However, long-term fundamentals remain strong so this is a short-term trading call only based on the immediate news catalyst and price action...",
        "horizon_hours": 12,
        "expected_high_price": 420.0
    },
    {
        "ticker": "NVDA",
        "action": "BUY",
        "previous_confidence": 90,
        "revised_confidence": 95,
        "reasoning": "AI boom continues unabated with new customer wins announced. Price momentum very strong, breaking all resistance levels. Multiple expansion justified by growth acceleration. Institutional buying detected in recent sessions. Supply constraints easing which should boost margins. Management guidance likely conservative given historical patterns. Technical setup remains bullish with no signs of exhaustion yet despite extended rally from recent lows. Options flow heavily skewed bullish.",
        "horizon_hours": 48,
        "expected_high_price": 135.0
    }
]

print("Testing refined_reason truncation:\n")
print("=" * 80)

for i, refined in enumerate(test_responses, 1):
    ticker = refined["ticker"]
    reasoning = refined.get("reasoning", "")
    
    # Simulate the actual processing logic from llm.py
    reason_text = reasoning
    cleaned_reason = clean_text(str(reason_text))
    truncated_reason = truncate_with_ellipsis(cleaned_reason, 240)
    
    print(f"\nTest {i}: {ticker}")
    print(f"Original length: {len(reasoning)} chars")
    print(f"Original text: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
    print(f"Truncated length: {len(truncated_reason)} chars")
    print(f"Truncated text: {truncated_reason}")
    
    # Check if truncation added ellipsis properly
    if len(reasoning) > 240:
        assert truncated_reason.endswith("..."), "Should end with ellipsis when truncated"
        assert len(truncated_reason) <= 240, "Should not exceed max length"
        print("✓ Properly truncated with ellipsis")
    else:
        assert truncated_reason == cleaned_reason, "Should not modify text under limit"
        print("✓ Text under limit, not modified")
    
    # Simulate JSON output
    decision_json = {
        "ticker": ticker,
        "refined_confidence": refined["revised_confidence"],
        "refined_reason": truncated_reason,
    }
    
    print(f"JSON output: {json.dumps(decision_json, indent=2)[:200]}...")

print("\n" + "=" * 80)
print("All tests passed! The refined_reason field is properly handled.")
print("\nKey improvements:")
print("1. Added maxLength constraint to schema (500 chars)")
print("2. Using truncate_with_ellipsis() for intelligent truncation")
print("3. Truncation happens at word boundaries when possible")
print("4. Proper ellipsis (...) added when text is truncated")