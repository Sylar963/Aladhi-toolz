#!/usr/bin/env python3
"""
Volatility Regime Detection Module
==================================

This module classifies current market conditions into volatility regimes to help
with forward-looking volatility expectations for options pricing.

Unlike static Black-Scholes assumptions, this provides dynamic volatility context
that's crucial for straddle buying decisions.
"""

import math
import statistics
import time
from typing import Dict, List, Optional, Tuple, NamedTuple
from collections import deque
from enum import Enum
from dataclasses import dataclass
import logging

import numpy as np
from scipy import stats

from financial_math import FinancialMath, TradingConstants


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    UNKNOWN = "unknown"
    LOW = "low"           # < 25th percentile
    NORMAL = "normal"     # 25th-75th percentile  
    HIGH = "high"         # 75th-90th percentile
    EXTREME = "extreme"   # > 90th percentile
    CRISIS = "crisis"     # > 95th percentile with additional criteria


@dataclass
class RegimeMetrics:
    """Metrics for volatility regime analysis"""
    current_vol: float
    short_term_vol: float  # 1-7 days
    medium_term_vol: float # 7-30 days
    long_term_vol: float   # 30-90 days
    
    vol_momentum: float    # Recent vol trend
    vol_mean_reversion: float  # Distance from long-term mean
    vol_percentile: float  # Current vol vs historical distribution
    
    regime: VolatilityRegime
    confidence: float
    
    # Additional context
    price_trend: float     # Recent price momentum
    vol_price_correlation: float  # Correlation between vol and price changes
    time_since_regime_change: float  # Hours since last regime change


class VolatilityRegimeDetector:
    """
    Detects and classifies volatility regimes for crypto options markets
    
    Key features:
    - Multi-timeframe analysis (1d, 7d, 30d, 90d)
    - Volatility momentum and mean reversion tracking
    - Price-volatility correlation analysis
    - Regime persistence modeling
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.max_observations = lookback_days * 24  # Hourly data
        
        # Price and return history
        self.price_history = deque(maxlen=self.max_observations)
        self.timestamp_history = deque(maxlen=self.max_observations)
        self.return_history = deque(maxlen=self.max_observations)
        
        # Volatility history (calculated periodically)
        self.vol_history = deque(maxlen=lookback_days)
        self.vol_timestamps = deque(maxlen=lookback_days)
        
        # Regime tracking
        self.current_regime = VolatilityRegime.UNKNOWN
        self.regime_start_time: Optional[float] = None
        self.regime_confidence = 0.0
        
        # Historical regime statistics for percentile calculations
        self.historical_vols: List[float] = []
        self.vol_percentiles: Dict[str, float] = {}
        
        logging.info(f"Initialized VolatilityRegimeDetector with {lookback_days}d lookback")
    
    def add_price_observation(self, price: float, timestamp: Optional[float] = None) -> bool:
        """
        Add new price observation and calculate returns
        
        Args:
            price: Asset price
            timestamp: Unix timestamp (current time if None)
            
        Returns:
            True if return was calculated, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()
            
        if price <= 0:
            logging.warning(f"Invalid price: {price}")
            return False
        
        # Calculate return if we have previous data
        return_calculated = False
        if len(self.price_history) > 0:
            prev_price = self.price_history[-1]
            prev_timestamp = self.timestamp_history[-1]
            
            if prev_price > 0 and timestamp > prev_timestamp:
                log_return = math.log(price / prev_price)
                
                # Sanity check for extreme moves (likely data errors)
                if abs(log_return) < 0.5:  # < 50% instantaneous move
                    self.return_history.append(log_return)
                    return_calculated = True
                else:
                    logging.warning(f"Extreme return rejected: {log_return:.4f}")
        
        self.price_history.append(price)
        self.timestamp_history.append(timestamp)
        
        return return_calculated
    
    def calculate_multi_timeframe_volatility(self) -> Dict[str, float]:
        """
        Calculate volatility across multiple timeframes
        
        Returns:
            Dictionary with volatility estimates for different periods
        """
        current_time = time.time()
        
        timeframes = {
            'short': 7 * 24 * 3600,      # 7 days in seconds
            'medium': 30 * 24 * 3600,    # 30 days
            'long': 90 * 24 * 3600       # 90 days
        }
        
        volatilities = {}
        
        for name, seconds in timeframes.items():
            cutoff_time = current_time - seconds
            
            # Get returns within timeframe
            recent_returns = []
            if len(self.timestamp_history) > 0:
                for i, ts in enumerate(self.timestamp_history):
                    if ts >= cutoff_time and i < len(self.return_history):
                        recent_returns.append(self.return_history[i])
            
            if len(recent_returns) >= 5:  # Minimum for volatility calculation
                vol = FinancialMath.annualize_volatility(recent_returns, "hourly")
                volatilities[name] = vol
            else:
                volatilities[name] = 0.0
                
        return volatilities
    
    def calculate_volatility_momentum(self, window_hours: int = 24) -> float:
        """
        Calculate volatility momentum (recent vol trend)
        
        Args:
            window_hours: Hours to look back for trend calculation
            
        Returns:
            Volatility momentum (-1 to 1, where 1 is strong upward trend)
        """
        if len(self.vol_history) < 3:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - (window_hours * 3600)
        
        # Get recent volatility observations
        recent_vols = []
        recent_times = []
        
        for i, ts in enumerate(self.vol_timestamps):
            if ts >= cutoff_time and i < len(self.vol_history):
                recent_vols.append(self.vol_history[i])
                recent_times.append(ts)
        
        if len(recent_vols) < 3:
            return 0.0
        
        # Calculate slope of volatility trend
        try:
            slope, _, r_value, _, _ = stats.linregress(recent_times, recent_vols)
            
            # Normalize slope to -1 to 1 range
            # Scale by typical volatility to make it meaningful
            avg_vol = statistics.mean(recent_vols)
            if avg_vol > 0:
                normalized_slope = slope / avg_vol
                return max(-1.0, min(1.0, normalized_slope * 1000))  # Scale factor
            
        except Exception:
            pass
        
        return 0.0
    
    def calculate_mean_reversion_distance(self) -> float:
        """
        Calculate how far current volatility is from long-term mean
        
        Returns:
            Standard deviations from long-term mean (positive = above mean)
        """
        if len(self.historical_vols) < 20:
            return 0.0
        
        current_vol = self.get_current_volatility()
        if current_vol == 0.0:
            return 0.0
        
        long_term_mean = statistics.mean(self.historical_vols)
        long_term_std = statistics.stdev(self.historical_vols)
        
        if long_term_std == 0:
            return 0.0
        
        return (current_vol - long_term_mean) / long_term_std
    
    def get_current_volatility(self, window_hours: int = 24) -> float:
        """Get current volatility estimate"""
        current_time = time.time()
        cutoff_time = current_time - (window_hours * 3600)
        
        recent_returns = []
        for i, ts in enumerate(self.timestamp_history):
            if ts >= cutoff_time and i < len(self.return_history):
                recent_returns.append(self.return_history[i])
        
        if len(recent_returns) >= 5:
            return FinancialMath.annualize_volatility(recent_returns, "hourly")
        
        return 0.0
    
    def update_volatility_percentiles(self):
        """Update historical volatility percentiles for regime classification"""
        if len(self.historical_vols) < 50:
            return
        
        self.vol_percentiles = {
            '25th': np.percentile(self.historical_vols, 25),
            '50th': np.percentile(self.historical_vols, 50),
            '75th': np.percentile(self.historical_vols, 75),
            '90th': np.percentile(self.historical_vols, 90),
            '95th': np.percentile(self.historical_vols, 95)
        }
    
    def classify_regime(self, current_vol: float) -> Tuple[VolatilityRegime, float]:
        """
        Classify current volatility regime
        
        Args:
            current_vol: Current volatility estimate
            
        Returns:
            Tuple of (regime, confidence)
        """
        if not self.vol_percentiles or current_vol == 0.0:
            return VolatilityRegime.UNKNOWN, 0.0
        
        # Base classification on percentiles
        if current_vol >= self.vol_percentiles['95th']:
            base_regime = VolatilityRegime.CRISIS
        elif current_vol >= self.vol_percentiles['90th']:
            base_regime = VolatilityRegime.EXTREME
        elif current_vol >= self.vol_percentiles['75th']:
            base_regime = VolatilityRegime.HIGH
        elif current_vol >= self.vol_percentiles['25th']:
            base_regime = VolatilityRegime.NORMAL
        else:
            base_regime = VolatilityRegime.LOW
        
        # Calculate confidence based on how far we are from threshold
        confidence = 0.7  # Base confidence
        
        # Adjust confidence based on distance from regime thresholds
        if base_regime == VolatilityRegime.CRISIS:
            distance = current_vol - self.vol_percentiles['95th']
            relative_distance = distance / self.vol_percentiles['95th']
            confidence = min(0.95, 0.8 + relative_distance)
        elif base_regime == VolatilityRegime.LOW:
            distance = self.vol_percentiles['25th'] - current_vol
            relative_distance = distance / self.vol_percentiles['25th']
            confidence = min(0.95, 0.8 + relative_distance)
        
        # Additional criteria for CRISIS regime
        if base_regime == VolatilityRegime.CRISIS:
            vol_momentum = self.calculate_volatility_momentum()
            if vol_momentum > 0.5:  # Strong upward vol momentum
                confidence = min(0.95, confidence + 0.1)
        
        return base_regime, confidence
    
    def calculate_vol_price_correlation(self, window_hours: int = 168) -> float:
        """
        Calculate correlation between volatility changes and price changes
        
        Args:
            window_hours: Analysis window (default 1 week)
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(self.return_history) < 20:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - (window_hours * 3600)
        
        # Get recent data
        recent_returns = []
        recent_timestamps = []
        
        for i, ts in enumerate(self.timestamp_history):
            if ts >= cutoff_time and i < len(self.return_history):
                recent_returns.append(self.return_history[i])
                recent_timestamps.append(ts)
        
        if len(recent_returns) < 10:
            return 0.0
        
        # Calculate rolling volatility changes
        vol_changes = []
        price_changes = []
        
        window_size = 6  # 6-hour rolling windows
        for i in range(window_size, len(recent_returns)):
            # Volatility of current window vs previous window
            current_window = recent_returns[i-window_size:i]
            prev_window = recent_returns[i-window_size*2:i-window_size] if i >= window_size*2 else None
            
            if prev_window and len(prev_window) == window_size:
                current_vol = statistics.stdev(current_window) if len(current_window) > 1 else 0
                prev_vol = statistics.stdev(prev_window) if len(prev_window) > 1 else 0
                
                if prev_vol > 0:
                    vol_change = (current_vol - prev_vol) / prev_vol
                    price_change = sum(recent_returns[i-window_size:i])  # Price move over window
                    
                    vol_changes.append(vol_change)
                    price_changes.append(price_change)
        
        if len(vol_changes) < 5:
            return 0.0
        
        try:
            correlation, _ = stats.pearsonr(vol_changes, price_changes)
            return correlation if not math.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def analyze_regime(self) -> Optional[RegimeMetrics]:
        """
        Perform comprehensive regime analysis
        
        Returns:
            RegimeMetrics object with full analysis or None if insufficient data
        """
        if len(self.return_history) < 50:
            logging.debug("Insufficient data for regime analysis")
            return None
        
        # Calculate multi-timeframe volatilities
        volatilities = self.calculate_multi_timeframe_volatility()
        
        current_vol = self.get_current_volatility()
        if current_vol == 0.0:
            return None
        
        # Update historical data and percentiles
        self.historical_vols.append(current_vol)
        if len(self.historical_vols) > self.lookback_days:
            self.historical_vols.pop(0)
        
        self.update_volatility_percentiles()
        
        # Store volatility history
        self.vol_history.append(current_vol)
        self.vol_timestamps.append(time.time())
        
        # Classify regime
        regime, confidence = self.classify_regime(current_vol)
        
        # Calculate additional metrics
        vol_momentum = self.calculate_volatility_momentum()
        mean_reversion_distance = self.calculate_mean_reversion_distance()
        vol_percentile = self._calculate_current_percentile(current_vol)
        vol_price_correlation = self.calculate_vol_price_correlation()
        
        # Calculate price trend
        price_trend = self._calculate_price_trend()
        
        # Track regime changes
        time_since_change = 0.0
        if regime != self.current_regime:
            self.regime_start_time = time.time()
            self.current_regime = regime
        elif self.regime_start_time:
            time_since_change = (time.time() - self.regime_start_time) / 3600  # Hours
        
        return RegimeMetrics(
            current_vol=current_vol,
            short_term_vol=volatilities.get('short', 0.0),
            medium_term_vol=volatilities.get('medium', 0.0),
            long_term_vol=volatilities.get('long', 0.0),
            vol_momentum=vol_momentum,
            vol_mean_reversion=mean_reversion_distance,
            vol_percentile=vol_percentile,
            regime=regime,
            confidence=confidence,
            price_trend=price_trend,
            vol_price_correlation=vol_price_correlation,
            time_since_regime_change=time_since_change
        )
    
    def _calculate_current_percentile(self, current_vol: float) -> float:
        """Calculate what percentile the current volatility represents"""
        if len(self.historical_vols) < 20:
            return 50.0  # Default to median
        
        sorted_vols = sorted(self.historical_vols)
        position = 0
        
        for vol in sorted_vols:
            if current_vol > vol:
                position += 1
            else:
                break
        
        return (position / len(sorted_vols)) * 100
    
    def _calculate_price_trend(self, window_hours: int = 24) -> float:
        """Calculate recent price momentum"""
        if len(self.return_history) < 10:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - (window_hours * 3600)
        
        recent_returns = []
        for i, ts in enumerate(self.timestamp_history):
            if ts >= cutoff_time and i < len(self.return_history):
                recent_returns.append(self.return_history[i])
        
        if len(recent_returns) < 5:
            return 0.0
        
        # Sum of recent returns as trend indicator
        total_return = sum(recent_returns)
        return total_return
    
    def get_regime_implications_for_straddles(self) -> Dict[str, str]:
        """
        Get regime-specific implications for straddle trading
        
        Returns:
            Dictionary with trading implications
        """
        if self.current_regime == VolatilityRegime.UNKNOWN:
            return {"status": "insufficient_data", "implication": "Need more data for analysis"}
        
        implications = {
            VolatilityRegime.LOW: {
                "status": "vol_expansion_possible",
                "implication": "Low vol may expand - favorable for straddle buying",
                "risk": "Vol could stay low (unfavorable)",
                "strategy": "Consider buying straddles on expected vol expansion"
            },
            VolatilityRegime.NORMAL: {
                "status": "neutral",
                "implication": "Normal vol environment - monitor for regime changes",
                "risk": "Could move either direction",
                "strategy": "Focus on relative pricing vs theoretical"
            },
            VolatilityRegime.HIGH: {
                "status": "vol_contraction_risk",
                "implication": "High vol may contract - risky for straddle buying",
                "risk": "Vol contraction would hurt long straddles",
                "strategy": "Consider selling straddles or wait for lower vol"
            },
            VolatilityRegime.EXTREME: {
                "status": "vol_contraction_likely",
                "implication": "Extreme vol likely to revert - very risky for buying",
                "risk": "Strong mean reversion expected",
                "strategy": "Avoid buying straddles - consider selling"
            },
            VolatilityRegime.CRISIS: {
                "status": "crisis_vol",
                "implication": "Crisis vol - extremely risky for straddle buying",
                "risk": "Severe vol contraction when crisis passes",
                "strategy": "Avoid straddle buying entirely"
            }
        }
        
        return implications.get(self.current_regime, {"status": "unknown"})
    
    def log_regime_summary(self, metrics: RegimeMetrics):
        """Log detailed regime analysis summary"""
        logging.info("="*60)
        logging.info("VOLATILITY REGIME ANALYSIS")
        logging.info("="*60)
        logging.info(f"Current Regime: {metrics.regime.value.upper()} (confidence: {metrics.confidence:.1%})")
        logging.info(f"Current Vol: {metrics.current_vol:.1%}")
        logging.info(f"Vol Percentile: {metrics.vol_percentile:.0f}th")
        logging.info(f"Short-term Vol: {metrics.short_term_vol:.1%}")
        logging.info(f"Medium-term Vol: {metrics.medium_term_vol:.1%}")
        logging.info(f"Long-term Vol: {metrics.long_term_vol:.1%}")
        logging.info(f"Vol Momentum: {metrics.vol_momentum:.2f}")
        logging.info(f"Mean Reversion Distance: {metrics.vol_mean_reversion:.1f} std devs")
        logging.info(f"Vol-Price Correlation: {metrics.vol_price_correlation:.2f}")
        logging.info(f"Price Trend: {metrics.price_trend:.3f}")
        logging.info(f"Time Since Regime Change: {metrics.time_since_regime_change:.1f} hours")
        
        implications = self.get_regime_implications_for_straddles()
        logging.info(f"Straddle Implication: {implications.get('implication', 'Unknown')}")
        logging.info("="*60)