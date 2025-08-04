#!/usr/bin/env python3
"""
Statistical Analysis Module
Provides robust statistical analysis without artificial manipulation
"""

import math
import statistics
from typing import List, Optional, Dict, Tuple
from collections import deque
import numpy as np
from scipy import stats
from financial_math import TradingConstants

class DistributionAnalyzer:
    """
    Robust distribution analysis without artificial scaling
    """
    
    @staticmethod
    def calculate_mad_from_median(data: List[float]) -> float:
        """
        Calculate Mean Absolute Deviation from median
        
        Args:
            data: List of data points
            
        Returns:
            MAD from median
        """
        if len(data) < 2:
            return 0.0
            
        median = statistics.median(data)
        mad = statistics.mean(abs(x - median) for x in data)
        return mad
    
    @staticmethod 
    def calculate_mad_from_mean(data: List[float]) -> float:
        """
        Calculate Mean Absolute Deviation from mean
        
        Args:
            data: List of data points
            
        Returns:
            MAD from mean
        """
        if len(data) < 2:
            return 0.0
            
        mean = statistics.mean(data)
        mad = statistics.mean(abs(x - mean) for x in data)
        return mad
    
    @staticmethod
    def calculate_distribution_metrics(data: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive distribution metrics
        
        Args:
            data: List of data points
            
        Returns:
            Dictionary of distribution metrics
        """
        if len(data) < 2:
            return {
                'mean': 0.0, 'median': 0.0, 'std_dev': 0.0,
                'mad_from_median': 0.0, 'mad_from_mean': 0.0,
                'mad_sd_ratio': 0.0, 'skewness': 0.0, 'kurtosis': 0.0,
                'sample_size': len(data)
            }
        
        # Basic statistics
        mean = statistics.mean(data)
        median = statistics.median(data)
        std_dev = statistics.stdev(data) if len(data) > 1 else 0.0
        
        # MAD calculations
        mad_median = DistributionAnalyzer.calculate_mad_from_median(data)
        mad_mean = DistributionAnalyzer.calculate_mad_from_mean(data)
        
        # MAD/SD ratio (theoretical value for normal distribution is ~0.7979)
        mad_sd_ratio = mad_median / std_dev if std_dev > 0 else 0.0
        
        # Higher-order moments
        if len(data) >= 3:
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)  # Excess kurtosis (0 for normal)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        return {
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'mad_from_median': mad_median,
            'mad_from_mean': mad_mean,
            'mad_sd_ratio': mad_sd_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sample_size': len(data)
        }
    
    @staticmethod
    def assess_distribution_normality(data: List[float]) -> Dict[str, any]:
        """
        Assess if data follows normal distribution
        
        Args:
            data: List of data points
            
        Returns:
            Dictionary with normality test results
        """
        if len(data) < 8:  # Minimum for Jarque-Bera test
            return {
                'is_normal': None,
                'test_statistic': None,
                'p_value': None,
                'assessment': 'Insufficient data for normality test'
            }
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_p_value = stats.jarque_bera(data)
            
            # Significance level 0.05
            is_normal = jb_p_value > 0.05
            
            if jb_p_value > 0.10:
                assessment = "Likely normal distribution"
            elif jb_p_value > 0.05:
                assessment = "Possibly normal distribution"
            elif jb_p_value > 0.01:
                assessment = "Likely non-normal distribution"
            else:
                assessment = "Definitely non-normal distribution"
                
            return {
                'is_normal': is_normal,
                'test_statistic': jb_stat,
                'p_value': jb_p_value,
                'assessment': assessment
            }
            
        except Exception:
            return {
                'is_normal': None,
                'test_statistic': None,
                'p_value': None,
                'assessment': 'Error in normality test'
            }
    
    @staticmethod
    def classify_distribution_shape(mad_sd_ratio: float) -> Tuple[str, str]:
        """
        Classify distribution shape based on MAD/SD ratio
        WITHOUT artificial scaling - use genuine statistical interpretation
        
        Args:
            mad_sd_ratio: Genuine MAD/Standard deviation ratio
            
        Returns:
            Tuple of (classification, description)
        """
        theoretical_normal = TradingConstants.NORMAL_DISTRIBUTION_MAD_SD_RATIO
        
        if mad_sd_ratio < 0.50:
            return "extreme_tails", "Extreme heavy tails - very high tail risk"
        elif mad_sd_ratio < 0.65:
            return "heavy_tails", "Heavy tails - significant tail risk"
        elif mad_sd_ratio < 0.73:
            return "moderate_tails", "Moderate tails - elevated tail risk"
        elif 0.73 <= mad_sd_ratio <= 0.83:  # ±0.05 around theoretical 0.7979
            return "near_normal", "Near-normal distribution"
        elif mad_sd_ratio <= 0.90:
            return "light_tails", "Light tails - lower than normal tail risk"
        else:
            return "compressed", "Unusually compressed distribution"

class TimeSeriesAnalyzer:
    """
    Time series analysis for financial data
    """
    
    def __init__(self, max_history: int = 1000):
        self.price_history = deque(maxlen=max_history)
        self.return_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
    
    def add_price_point(self, price: float, timestamp: float) -> bool:
        """
        Add price point and calculate return
        
        Args:
            price: Asset price
            timestamp: Unix timestamp
            
        Returns:
            True if return was calculated, False otherwise
        """
        if price <= 0 or timestamp <= 0:
            return False
        
        # Calculate return if we have previous data
        return_calculated = False
        if len(self.price_history) > 0:
            prev_price = self.price_history[-1]
            prev_timestamp = self.timestamp_history[-1]
            
            if prev_price > 0 and timestamp > prev_timestamp:
                # Calculate log return
                log_return = math.log(price / prev_price)
                
                # Sanity check: reject extreme returns (likely data errors)
                if abs(log_return) < 0.2:  # < 20% instantaneous move
                    self.return_history.append(log_return)
                    return_calculated = True
        
        self.price_history.append(price)
        self.timestamp_history.append(timestamp)
        
        return return_calculated
    
    def get_recent_returns(self, max_age_hours: Optional[float] = None) -> List[float]:
        """
        Get recent returns within specified time window
        
        Args:
            max_age_hours: Maximum age of data in hours (None for all data)
            
        Returns:
            List of recent returns
        """
        if max_age_hours is None:
            return list(self.return_history)
        
        current_time = self.timestamp_history[-1] if self.timestamp_history else 0
        cutoff_time = current_time - (max_age_hours * 3600)
        
        # Find returns within time window
        recent_returns = []
        returns_list = list(self.return_history)
        timestamps_list = list(self.timestamp_history)
        
        # Match returns with their corresponding timestamps
        # Returns correspond to timestamps[1:] since they're calculated from price differences
        if len(timestamps_list) > len(returns_list):
            relevant_timestamps = timestamps_list[-len(returns_list):]
        else:
            relevant_timestamps = timestamps_list
        
        for i, timestamp in enumerate(relevant_timestamps):
            if timestamp >= cutoff_time and i < len(returns_list):
                recent_returns.append(returns_list[i])
        
        return recent_returns
    
    def calculate_rolling_statistics(self, window_hours: float = 24) -> Optional[Dict]:
        """
        Calculate rolling statistics over specified window
        
        Args:
            window_hours: Rolling window in hours
            
        Returns:
            Dictionary of rolling statistics or None
        """
        recent_returns = self.get_recent_returns(window_hours)
        
        if len(recent_returns) < 3:
            return None
        
        # Calculate distribution metrics
        metrics = DistributionAnalyzer.calculate_distribution_metrics(recent_returns)
        
        # Add rolling-specific metrics
        metrics['window_hours'] = window_hours
        metrics['data_points'] = len(recent_returns)
        
        return metrics
    
    def estimate_confidence_level(self, sample_size: int) -> float:
        """
        Estimate statistical confidence level based on sample size
        
        Args:
            sample_size: Number of data points
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        if sample_size < 3:
            return TradingConstants.CONFIDENCE_LEVELS["low"]
        elif sample_size < 10:
            return TradingConstants.CONFIDENCE_LEVELS["medium"]
        elif sample_size < 30:
            return TradingConstants.CONFIDENCE_LEVELS["good"]
        else:
            return TradingConstants.CONFIDENCE_LEVELS["high"]

class VolatilityEstimator:
    """
    Robust volatility estimation methods
    """
    
    @staticmethod
    def estimate_realized_volatility(returns: List[float], frequency: str = "intraday") -> float:
        """
        Estimate realized volatility from returns without artificial adjustments
        
        Args:
            returns: List of log returns
            frequency: Data frequency for proper annualization
            
        Returns:
            Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate sample standard deviation
        volatility = statistics.stdev(returns)
        
        # Proper annualization based on frequency
        if frequency == "daily":
            annualization_factor = math.sqrt(TradingConstants.TRADING_DAYS_PER_YEAR)
        elif frequency == "hourly":
            annualization_factor = math.sqrt(
                TradingConstants.TRADING_DAYS_PER_YEAR * TradingConstants.HOURS_PER_TRADING_DAY
            )
        elif frequency == "intraday":
            # For high-frequency intraday data, estimate based on actual data frequency
            # Assume roughly 1 observation per minute during trading hours
            observations_per_day = 60 * TradingConstants.HOURS_PER_TRADING_DAY
            observations_per_year = observations_per_day * TradingConstants.TRADING_DAYS_PER_YEAR
            annualization_factor = math.sqrt(observations_per_year)
        else:
            # Default to daily
            annualization_factor = math.sqrt(TradingConstants.TRADING_DAYS_PER_YEAR)
        
        return volatility * annualization_factor
    
    @staticmethod
    def estimate_garch_volatility(returns: List[float], alpha: float = 0.1, beta: float = 0.8) -> float:
        """
        Simple GARCH(1,1) volatility estimate
        
        Args:
            returns: List of returns
            alpha: ARCH parameter
            beta: GARCH parameter
            
        Returns:
            Current conditional volatility estimate
        """
        if len(returns) < 10:
            return VolatilityEstimator.estimate_realized_volatility(returns)
        
        # Initialize with unconditional variance
        long_run_var = statistics.variance(returns)
        omega = long_run_var * (1 - alpha - beta)
        
        # Iterate through returns to build conditional variance
        conditional_var = long_run_var
        
        for ret in returns:
            conditional_var = omega + alpha * (ret ** 2) + beta * conditional_var
        
        return math.sqrt(conditional_var * TradingConstants.TRADING_DAYS_PER_YEAR)