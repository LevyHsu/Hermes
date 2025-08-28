#!/usr/bin/env python3
"""
Robust LLM processor with validation and retry logic.
Fixes malformed responses, timeouts, and validation issues.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    ticker: str
    action: str
    confidence: int
    reason: str
    
    def is_valid(self) -> bool:
        """Validate decision fields"""
        # Check ticker format
        if not re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', self.ticker):
            return False
        if self.ticker in ['XXXX', 'TEST', 'DUMMY']:
            return False
            
        # Check action
        if self.action not in ['BUY', 'SELL']:
            return False
            
        # Check confidence
        if not 0 <= self.confidence <= 100:
            return False
            
        # Check reason quality
        if len(self.reason) < 20:
            return False
        # Check for corrupted unicode
        if '\u00a0' in self.reason or '...' * 3 in self.reason:
            return False
        # Check for placeholder text
        if self.reason.strip() in ['N/A', 'None', 'TBD', '']:
            return False
            
        return True


class RobustLLMProcessor:
    """LLM processor with validation and retry logic"""
    
    def __init__(self, client, max_retries: int = 3):
        self.client = client
        self.max_retries = max_retries
        self.validation_stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed_validation': 0,
            'timeouts': 0
        }
        
    def process_news_item(self, news_item: Dict, timeout: float = 30.0) -> Optional[List[Decision]]:
        """Process news item with retries and validation"""
        
        for attempt in range(self.max_retries):
            self.validation_stats['total_attempts'] += 1
            
            try:
                # Attempt to get response
                response = self._call_llm_with_timeout(news_item, timeout)
                
                if not response:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    continue
                    
                # Validate and parse response
                decisions = self._parse_and_validate_response(response)
                
                if decisions:
                    self.validation_stats['successful'] += 1
                    return decisions
                    
                # Invalid response, retry with simpler prompt
                if attempt < self.max_retries - 1:
                    logger.warning(f"Invalid response on attempt {attempt + 1}, retrying...")
                    self.validation_stats['failed_validation'] += 1
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    news_item = self._simplify_prompt(news_item)
                    
            except TimeoutError:
                self.validation_stats['timeouts'] += 1
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    
        logger.error(f"Failed to process news item after {self.max_retries} attempts")
        return None
        
    def _call_llm_with_timeout(self, news_item: Dict, timeout: float) -> Optional[Dict]:
        """Call LLM with timeout handling"""
        # This would integrate with your existing LLM client
        # Simplified for demonstration
        start = time.time()
        
        try:
            # Add timeout to the actual LLM call
            response = self.client.predict(
                news_item,
                timeout=timeout
            )
            
            elapsed = time.time() - start
            logger.debug(f"LLM response received in {elapsed:.1f}s")
            
            return response
            
        except Exception as e:
            if 'timeout' in str(e).lower():
                raise TimeoutError(f"LLM timeout after {timeout}s")
            raise
            
    def _parse_and_validate_response(self, response: Any) -> Optional[List[Decision]]:
        """Parse and validate LLM response"""
        
        # Handle string response
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return None
                
        # Check structure
        if not isinstance(response, dict):
            logger.error(f"Response is not a dict: {type(response)}")
            return None
            
        if 'decisions' not in response:
            logger.error("Response missing 'decisions' field")
            return None
            
        decisions_data = response.get('decisions', [])
        if not isinstance(decisions_data, list):
            logger.error("'decisions' is not a list")
            return None
            
        # Parse and validate each decision
        valid_decisions = []
        for idx, decision_data in enumerate(decisions_data):
            try:
                decision = Decision(
                    ticker=decision_data.get('ticker', '').strip().upper(),
                    action=decision_data.get('action', '').strip().upper(),
                    confidence=int(decision_data.get('confidence', 0)),
                    reason=self._clean_reason(decision_data.get('reason', ''))
                )
                
                if decision.is_valid():
                    valid_decisions.append(decision)
                else:
                    logger.warning(f"Invalid decision at index {idx}: {decision}")
                    
            except Exception as e:
                logger.warning(f"Failed to parse decision at index {idx}: {e}")
                
        return valid_decisions if valid_decisions else None
        
    def _clean_reason(self, reason: str) -> str:
        """Clean and fix reasoning text"""
        if not reason:
            return ""
            
        # Fix common unicode issues
        reason = reason.replace('\u00a0', ' ')  # Non-breaking space
        reason = re.sub(r'\s+', ' ', reason)  # Multiple spaces
        reason = re.sub(r'\.{4,}', '...', reason)  # Too many dots
        
        # Remove placeholder patterns
        if re.match(r'^(The|This|A)\s*\.{3,}', reason):
            return ""
            
        return reason.strip()
        
    def _simplify_prompt(self, news_item: Dict) -> Dict:
        """Simplify prompt for retry"""
        # Reduce complexity for retry
        simplified = news_item.copy()
        
        # Truncate body if too long
        if 'body' in simplified and len(simplified['body']) > 1000:
            simplified['body'] = simplified['body'][:1000] + "..."
            
        # Reduce sample tickers
        if 'sample_tickers' in simplified and len(simplified['sample_tickers']) > 50:
            simplified['sample_tickers'] = simplified['sample_tickers'][:50]
            
        return simplified
        
    def get_stats(self) -> Dict:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_attempts']
            stats['validation_failure_rate'] = stats['failed_validation'] / stats['total_attempts']
            stats['timeout_rate'] = stats['timeouts'] / stats['total_attempts']
        return stats


class AdaptiveBatchProcessor:
    """Process batches with adaptive timeout management"""
    
    def __init__(self, processor: RobustLLMProcessor):
        self.processor = processor
        self.avg_process_time = 5.0  # Initial estimate
        self.timeout_buffer = 5.0
        
    def process_batch(self, items: List[Dict], time_budget: float) -> List[Decision]:
        """Process batch within time budget"""
        processed = []
        deadline = time.time() + time_budget - self.timeout_buffer
        
        for idx, item in enumerate(items):
            # Check if we have time for another item
            time_remaining = deadline - time.time()
            if time_remaining < self.avg_process_time:
                logger.info(f"Stopping batch early to avoid timeout ({idx}/{len(items)} processed)")
                break
                
            # Process item
            start = time.time()
            try:
                decisions = self.processor.process_news_item(
                    item,
                    timeout=min(time_remaining - 2, 30)  # Leave 2s buffer
                )
                
                if decisions:
                    processed.extend(decisions)
                    
            except Exception as e:
                logger.error(f"Failed to process item {idx}: {e}")
                
            # Update average processing time (EWMA)
            elapsed = time.time() - start
            self.avg_process_time = 0.7 * self.avg_process_time + 0.3 * elapsed
            
        logger.info(f"Processed {len(processed)} decisions from {len(items)} items")
        return processed


def integrate_robust_processor(llm_client) -> RobustLLMProcessor:
    """Factory function to create robust processor"""
    return RobustLLMProcessor(llm_client, max_retries=3)


if __name__ == "__main__":
    # Test validation
    test_decision = Decision(
        ticker="AAPL",
        action="BUY",
        confidence=85,
        reason="Apple announced strong iPhone sales and raised guidance for next quarter."
    )
    assert test_decision.is_valid()
    
    # Test invalid cases
    invalid_decision = Decision(
        ticker="XXXX",
        action="BUY",
        confidence=0,
        reason="The..."
    )
    assert not invalid_decision.is_valid()
    
    print("Validation tests passed!")