#!/usr/bin/env python3
"""
MAD Straddle Analyzer - Advanced straddle analysis with Mean Absolute Deviation theory
Integrates MAD vs Standard Deviation analysis for better risk assessment
Enhanced with proper Black-Scholes implementation and robust statistical analysis
"""
import asyncio
import json
import logging
import time
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

# Configure matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import thalex_py.thalex as th
import keys
from StraddlePositionCalculator import InteractiveStraddleSelector

# Import enhanced financial mathematics and statistical analysis
from financial_math import FinancialMath, TradingConstants
from statistical_analysis import DistributionAnalyzer, TimeSeriesAnalyzer, VolatilityEstimator

# Configuration
UNDERLYING = "BTCUSD"
NETWORK = th.Network.TEST
CID_LOGIN = 1000
CID_TICKER = 1001

class SimpleOption:
    """Simple option data"""
    def __init__(self, name: str, strike: float, option_type: str, expiry_ts: int):
        self.name = name
        self.strike = strike
        self.option_type = option_type
        self.expiry_ts = expiry_ts
        self.mark_price = 0.0
        
    def is_call(self) -> bool:
        return self.option_type == "call"
        
    def is_put(self) -> bool:
        return self.option_type == "put"

class MADAnalysis:
    """Enhanced MAD analysis results with proper statistical interpretation"""
    def __init__(self, mad: float, std_dev: float, mad_sd_ratio: float, 
                 theoretical_straddle: float, actual_straddle: float, 
                 distribution_metrics: Dict = None):
        self.mad = mad
        self.std_dev = std_dev
        self.mad_sd_ratio = mad_sd_ratio
        self.theoretical_straddle = theoretical_straddle
        self.actual_straddle = actual_straddle
        self.efficiency_ratio = actual_straddle / theoretical_straddle if theoretical_straddle > 0 else 0
        
        # Enhanced distribution information
        self.distribution_metrics = distribution_metrics or {}
        
        # Benchmarking attributes
        self.bs_theoretical = 0.0
        self.bs_efficiency_ratio = 0.0
        
        # Enhanced volatility and pricing attributes
        self.bs_implied_vol = 0.0
        self.bs_theoretical_implied = 0.0
        self.bs_implied_efficiency_ratio = 0.0
        self.estimated_vol = 0.0
        
        # Statistical confidence
        self.confidence_level = 0.0
        
    def get_tail_adjusted_thresholds(self) -> Tuple[float, float]:
        """Get dynamic efficiency thresholds based on genuine statistical tail risk"""
        # Use proper statistical classification without artificial manipulation
        shape_class, _ = DistributionAnalyzer.classify_distribution_shape(self.mad_sd_ratio)
        
        # Threshold adjustments based on genuine distribution characteristics
        if shape_class == "extreme_tails":
            return (0.65, 1.50)  # Very conservative for extreme tail risk
        elif shape_class == "heavy_tails":
            return (0.70, 1.40)  # Conservative for heavy tails
        elif shape_class == "moderate_tails":
            return (0.75, 1.30)  # Moderate adjustment
        elif shape_class == "near_normal":
            return (0.80, 1.25)  # Standard thresholds for normal distribution
        elif shape_class == "light_tails":
            return (0.85, 1.20)  # Less conservative for light tails
        else:  # compressed
            return (0.90, 1.15)  # Least conservative for compressed distribution
    
    def get_distribution_assessment(self) -> str:
        """Assess distribution shape based on genuine statistical analysis"""
        shape_class, description = DistributionAnalyzer.classify_distribution_shape(self.mad_sd_ratio)
        return description
            
    def get_straddle_assessment(self) -> str:
        """Assess straddle pricing efficiency with dynamic tail-risk adjustment"""
        underpriced_threshold, overpriced_threshold = self.get_tail_adjusted_thresholds()
        
        if self.efficiency_ratio < underpriced_threshold:
            return f"UNDERPRICED - Potential buying opportunity (threshold: {underpriced_threshold:.2f})"
        elif self.efficiency_ratio > overpriced_threshold:
            tail_context = "heavy tail risk" if self.mad_sd_ratio < 0.65 else "moderate tail risk" if self.mad_sd_ratio < 0.75 else "normal conditions"
            return f"OVERPRICED - Potential selling opportunity (threshold: {overpriced_threshold:.2f}, {tail_context})"
        else:
            return f"Fair value - normally priced ({underpriced_threshold:.2f} - {overpriced_threshold:.2f})"
            
    def get_risk_warnings(self) -> List[str]:
        """Generate tail-adjusted risk warnings based on dynamic analysis"""
        warnings = []
        underpriced_threshold, overpriced_threshold = self.get_tail_adjusted_thresholds()
        
        # Tail risk warnings
        if self.mad_sd_ratio < 0.6:
            warnings.append("ALERT: EXTREME TAIL RISK: Use minimal position sizing")
        elif self.mad_sd_ratio < 0.7:
            warnings.append("WARNING: HEAVY TAILS DETECTED: Reduce standard position sizes")
        elif self.mad_sd_ratio < 0.75:
            warnings.append("CAUTION: MODERATE TAIL RISK: Consider smaller positions")
        
        # Dynamic threshold-based opportunities/warnings
        if self.efficiency_ratio < underpriced_threshold:
            warnings.append(f"OPPORTUNITY: UNDERPRICED (below {underpriced_threshold:.2f}): Consider buying")
        elif self.efficiency_ratio > overpriced_threshold:
            tail_premium = "HIGH PREMIUM DEMANDED" if overpriced_threshold >= 1.30 else "STANDARD PREMIUM"
            warnings.append(f"SELLING OPPORTUNITY: OVERPRICED (above {overpriced_threshold:.2f}, {tail_premium})")
        
        return warnings

class ExpirationData:
    """Enhanced expiration data with expiry-specific MAD analysis"""
    def __init__(self, expiry_ts: int, expiry_date: str):
        self.expiry_ts = expiry_ts
        self.expiry_date = expiry_date
        self.atm_call: Optional[SimpleOption] = None
        self.atm_put: Optional[SimpleOption] = None
        self.days_to_expiry = (expiry_ts - time.time()) / 86400
        self.mad_analysis: Optional[MADAnalysis] = None
        
        # Create expiry-specific MAD analyzer
        self.mad_analyzer = ExpirySpecificMADAnalyzer(expiry_ts, self.days_to_expiry)
        
    def has_straddle(self) -> bool:
        return (self.atm_call is not None and self.atm_put is not None and 
                self.atm_call.mark_price > 0 and self.atm_put.mark_price > 0)
                
    def get_straddle_price(self) -> float:
        if self.has_straddle():
            return self.atm_call.mark_price + self.atm_put.mark_price
        return 0.0
        
    def get_breakeven_range(self) -> Tuple[float, float]:
        if self.has_straddle():
            straddle_price = self.get_straddle_price()
            strike = self.atm_call.strike
            return (strike - straddle_price, strike + straddle_price)
        return (0.0, 0.0)

class ExpirySpecificMADAnalyzer:
    """Enhanced expiry-specific analyzer using robust statistical methods"""
    
    def __init__(self, expiry_timestamp: int, days_to_expiry: float):
        self.expiry_timestamp = expiry_timestamp
        self.days_to_expiry = days_to_expiry
        
        # Use enhanced time series analyzer
        max_history = max(50, min(500, int(days_to_expiry * 5)))
        self.time_series_analyzer = TimeSeriesAnalyzer(max_history)
        
        # Determine minimum sample size based on expiry characteristics using constants
        if days_to_expiry < 1.0:
            self.min_sample_size = TradingConstants.MIN_SAMPLE_SIZES["very_short"]
        elif days_to_expiry < 7.0:
            self.min_sample_size = TradingConstants.MIN_SAMPLE_SIZES["short"]
        elif days_to_expiry < 30.0:
            self.min_sample_size = TradingConstants.MIN_SAMPLE_SIZES["medium"]
        else:
            self.min_sample_size = TradingConstants.MIN_SAMPLE_SIZES["long"]
            
        logging.debug(f"Created enhanced analyzer for {days_to_expiry:.1f}d expiry, min_samples={self.min_sample_size}")
    
    def _get_relevant_data_window(self) -> float:
        """Get relevant data window in hours based on expiry time horizon"""
        if self.days_to_expiry < 1.0:
            return 2.0  # 2 hours for very short-term options
        elif self.days_to_expiry < 3.0:
            return 12.0  # 12 hours for short-term options  
        elif self.days_to_expiry < 7.0:
            return 24.0  # 1 day for weekly options
        elif self.days_to_expiry < 14.0:
            return 48.0  # 2 days for bi-weekly options
        elif self.days_to_expiry < 30.0:
            return 120.0  # 5 days for monthly options
        elif self.days_to_expiry < 90.0:
            return 240.0  # 10 days for quarterly options
        else:
            return min(360.0, self.days_to_expiry * 4.0)  # Longer for far-dated options, max 15 days
    
    def _calculate_confidence_level(self, sample_size: int) -> float:
        """Calculate confidence level based on sample size"""
        if sample_size < 5:
            return 0.5  # Low confidence
        elif sample_size < 10:
            return 0.7  # Medium confidence
        elif sample_size < 20:
            return 0.85  # Good confidence
        else:
            return 0.95  # High confidence
    
    def _get_term_structure_adjustment(self) -> float:
        """Get volatility term structure adjustment based on time to expiry"""
        if self.days_to_expiry < 7:
            # Short-term: Higher vol premium due to gamma risk and event risk
            return 1.25
        elif self.days_to_expiry < 30:
            # Medium-term: Standard vol level
            return 1.0
        elif self.days_to_expiry < 90:  
            # Longer-term: Slight vol discount due to mean reversion
            return 0.95
        else:
            # Very long-term: Vol discount for long-term mean reversion
            return 0.85
        
    def add_price_point(self, price: float, timestamp: Optional[float] = None):
        """Add new price point using enhanced time series analyzer"""
        if timestamp is None:
            timestamp = time.time()
        
        # Validate inputs
        if price <= 0:
            logging.warning(f"Invalid price {price} for expiry {self.days_to_expiry:.1f} days")
            return
            
        if timestamp <= 0:
            logging.warning(f"Invalid timestamp {timestamp}")
            return
        
        # Use the enhanced time series analyzer
        return_added = self.time_series_analyzer.add_price_point(price, timestamp)
        
        if return_added:
            logging.debug(f"Added price point for {self.days_to_expiry:.1f}d expiry: ${price:.2f}")
        
        return return_added
        
    def calculate_expiry_specific_metrics(self) -> Optional[Dict]:
        """Calculate robust expiry-specific metrics using enhanced statistical analysis"""
        # Get relevant returns based on time window
        max_data_age_hours = self._get_relevant_data_window()
        returns_list = self.time_series_analyzer.get_recent_returns(max_data_age_hours)
        
        if len(returns_list) < self.min_sample_size:
            logging.debug(f"Insufficient data for {self.days_to_expiry:.1f}d expiry: {len(returns_list)}/{self.min_sample_size}")
            return None
        
        # Data quality validation: ensure sufficient unique values
        unique_returns = len(set(round(r, 8) for r in returns_list))  # Round to avoid float precision issues
        if unique_returns < max(3, self.min_sample_size // 2):
            logging.warning(f"[{self.days_to_expiry:.1f}d] Insufficient unique data points: {unique_returns}/{len(returns_list)} - data may be stale")
            return None
        
        # Validate data range - returns should show some variability
        returns_range = max(returns_list) - min(returns_list)
        if returns_range < 1e-8:  # Essentially zero variance
            logging.warning(f"[{self.days_to_expiry:.1f}d] No price movement detected: range={returns_range:.2e} - data may be stale")
            return None
        
        logging.info(f"[{self.days_to_expiry:.1f}d] Using {len(returns_list)} returns (window: {max_data_age_hours:.0f}h)")
        
        # Calculate comprehensive distribution metrics without artificial scaling
        distribution_metrics = DistributionAnalyzer.calculate_distribution_metrics(returns_list)
        
        # Calculate proper annualized volatility
        annualized_vol = VolatilityEstimator.estimate_realized_volatility(returns_list, "intraday")
        
        # Time-scaled volatility for this specific expiry
        time_factor = self.days_to_expiry / 365.0
        expiry_scaled_vol = annualized_vol * (time_factor ** 0.5)
        
        # Statistical confidence
        confidence_level = self.time_series_analyzer.estimate_confidence_level(len(returns_list))
        
        # Log genuine metrics without artificial manipulation
        logging.info(f"[{self.days_to_expiry:.1f}d] DATA WINDOW: {max_data_age_hours:.1f}h, "
                    f"SAMPLES: {len(returns_list)}/{self.time_series_analyzer.return_history.maxlen}")
        logging.info(f"[{self.days_to_expiry:.1f}d] UNIQUE MAD/SD={distribution_metrics['mad_sd_ratio']:.3f} "
                    f"(MAD={distribution_metrics['mad_from_median']:.6f}, SD={distribution_metrics['std_dev']:.6f}, "
                    f"conf={confidence_level:.2f})")
        
        # Enhanced debugging for data differentiation
        if len(returns_list) > 0:
            sample_returns = returns_list[:5] if len(returns_list) >= 5 else returns_list
            logging.info(f"[{self.days_to_expiry:.1f}d] SAMPLE RETURNS: {[f'{r:.6f}' for r in sample_returns]}")
            logging.info(f"[{self.days_to_expiry:.1f}d] RETURN RANGE: {min(returns_list):.6f} to {max(returns_list):.6f}")
        
        # Combine all metrics
        combined_metrics = {
            **distribution_metrics,
            'annualized_vol': annualized_vol,
            'expiry_scaled_vol': expiry_scaled_vol,
            'time_factor': time_factor,
            'days_to_expiry': self.days_to_expiry,
            'confidence_level': confidence_level,
            'data_window_hours': max_data_age_hours
        }
        
        return combined_metrics
        
    def analyze_straddle_efficiency(self, straddle_price: float, spot_price: float, strike_price: float) -> Optional[MADAnalysis]:
        """Analyze straddle efficiency using proper Black-Scholes with actual strike prices"""
        metrics = self.calculate_expiry_specific_metrics()
        if not metrics or spot_price <= 0 or straddle_price <= 0 or strike_price <= 0:
            return None
        
        # Time to expiry in years
        time_to_expiry = self.days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            return None
        
        # Use annualized volatility for Black-Scholes pricing with near-expiry adjustments
        base_annual_vol = metrics['annualized_vol']
        
        # Enhanced volatility estimation with market-realistic floors
        # Problem: intraday price ticks severely underestimate true volatility
        
        # Set minimum volatility floors based on market reality for crypto options
        if self.days_to_expiry < 7.0:
            min_vol = 0.30   # 30% minimum for short-term crypto options
        elif self.days_to_expiry < 30.0:
            min_vol = 0.25   # 25% minimum for medium-term
        else:
            min_vol = 0.20   # 20% minimum for long-term
        
        # Use the higher of estimated vol or minimum realistic vol
        market_adjusted_vol = max(base_annual_vol, min_vol)
        
        # Apply near-expiry volatility adjustments for options <7 days
        if self.days_to_expiry < 7.0:
            # Very short-term options have higher effective volatility due to:
            # 1. Gamma risk amplification  
            # 2. Event risk (news, announcements)
            # 3. Microstructure effects
            if self.days_to_expiry < 2.0:
                vol_multiplier = 1.5   # 150% for <2 days (more realistic)
            elif self.days_to_expiry < 7.0:
                vol_multiplier = 1.2   # 120% for 2-7 days
            else:
                vol_multiplier = 1.0
                
            annual_vol = market_adjusted_vol * vol_multiplier
            
            if market_adjusted_vol != base_annual_vol or vol_multiplier != 1.0:
                logging.info(f"  Vol adjustments: Raw={base_annual_vol:.1%} -> Floor={market_adjusted_vol:.1%} -> Final={annual_vol:.1%} (x{vol_multiplier})")
        else:
            annual_vol = market_adjusted_vol
            if market_adjusted_vol != base_annual_vol:
                logging.info(f"  Vol floor applied: {base_annual_vol:.1%} -> {annual_vol:.1%}")
            
        risk_free_rate = TradingConstants.DEFAULT_RISK_FREE_RATE
        
        # First calculate implied volatility from market straddle price
        implied_vol = self._calculate_implied_volatility_from_straddle(
            straddle_price, spot_price, strike_price, time_to_expiry, risk_free_rate
        )
        
        # Calculate Black-Scholes prices using both estimated and implied volatility
        bs_theoretical_estimated = FinancialMath.black_scholes_straddle(
            spot=spot_price,
            strike=strike_price,  # Use actual option strike, not ATM assumption
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=annual_vol,
            dividend_yield=0.0  # No dividends for crypto
        )
        
        # Also calculate Black-Scholes price using implied volatility for comparison
        bs_theoretical_implied = FinancialMath.black_scholes_straddle(
            spot=spot_price,
            strike=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=implied_vol,
            dividend_yield=0.0
        )
        
        # Enhanced MAD-based theoretical pricing (use estimated vol Black-Scholes as base)
        # Use MAD/SD ratio to adjust for distribution characteristics  
        mad_adjustment = self._get_mad_based_adjustment(metrics['mad_sd_ratio'])
        mad_enhanced_theoretical = bs_theoretical_estimated * mad_adjustment
        
        # Convert MAD metrics to dollar terms
        mad_dollars = metrics['mad_from_median'] * spot_price
        std_dev_dollars = metrics['std_dev'] * spot_price
        
        # Debug logging with full parameter details
        moneyness = strike_price / spot_price
        logging.info(f"[{self.days_to_expiry:.1f}d] PRICING PARAMETERS:")
        logging.info(f"  Spot: ${spot_price:.0f}, Strike: ${strike_price:.0f}, Moneyness: {moneyness:.3f}")
        logging.info(f"  Time to Expiry: {time_to_expiry:.4f} years ({self.days_to_expiry:.1f} days)")
        logging.info(f"  Estimated Annual Vol: {annual_vol:.1%} (from {metrics['sample_size']} returns)")
        logging.info(f"  Market Implied Vol: {implied_vol:.1%} ({'EXTREME' if implied_vol > 2.0 else 'HIGH' if implied_vol > 1.0 else 'NORMAL'})")
        logging.info(f"  Vol Ratio (Market/Estimated): {implied_vol/annual_vol:.1f}x")
        logging.info(f"  Risk-Free Rate: {risk_free_rate:.1%}")
        logging.info(f"  MAD/SD Ratio: {metrics['mad_sd_ratio']:.3f}")
        logging.info(f"[{self.days_to_expiry:.1f}d] PRICING COMPARISON:")
        logging.info(f"  Market Straddle Price: ${straddle_price:.2f}")
        logging.info(f"  BS Estimated Vol Price: ${bs_theoretical_estimated:.2f} (ratio: {straddle_price/bs_theoretical_estimated:.1f}x)")
        logging.info(f"  BS Implied Vol Price: ${bs_theoretical_implied:.2f} (ratio: {straddle_price/bs_theoretical_implied:.1f}x)")
        logging.info(f"  MAD-Enhanced Price: ${mad_enhanced_theoretical:.2f} (adj: {mad_adjustment:.2f}, ratio: {straddle_price/mad_enhanced_theoretical:.1f}x)")
        
        # Assessment of pricing accuracy
        implied_ratio = straddle_price / bs_theoretical_implied if bs_theoretical_implied > 0 else float('inf')
        if 0.9 <= implied_ratio <= 1.1:
            accuracy_assessment = " EXCELLENT (implied vol pricing)"
        elif 0.8 <= implied_ratio <= 1.2:
            accuracy_assessment = " GOOD (implied vol pricing)"
        else:
            estimated_ratio = straddle_price / bs_theoretical_estimated if bs_theoretical_estimated > 0 else float('inf')
            if 0.8 <= estimated_ratio <= 1.2:
                accuracy_assessment = "  ESTIMATED VOL CLOSER TO MARKET"
            else:
                accuracy_assessment = " SIGNIFICANT MISPRICING DETECTED"
        logging.info(f"  Accuracy Assessment: {accuracy_assessment}")
        
        # Create enhanced analysis
        analysis = MADAnalysis(
            mad=mad_dollars,
            std_dev=std_dev_dollars,
            mad_sd_ratio=metrics['mad_sd_ratio'],
            theoretical_straddle=mad_enhanced_theoretical,
            actual_straddle=straddle_price,
            distribution_metrics=metrics
        )
        
        # Add comprehensive Black-Scholes benchmarking
        analysis.bs_theoretical = bs_theoretical_estimated  # Use estimated vol as primary BS benchmark
        analysis.bs_efficiency_ratio = straddle_price / bs_theoretical_estimated if bs_theoretical_estimated > 0 else 0
        
        # Add implied volatility comparison
        analysis.bs_implied_vol = implied_vol
        analysis.bs_theoretical_implied = bs_theoretical_implied  
        analysis.bs_implied_efficiency_ratio = straddle_price / bs_theoretical_implied if bs_theoretical_implied > 0 else 0
        
        # Add volatility metrics
        analysis.estimated_vol = annual_vol
        analysis.confidence_level = metrics['confidence_level']
        
        return analysis
    
    def _get_mad_based_adjustment(self, mad_sd_ratio: float) -> float:
        """
        Calculate MAD-based adjustment factor for theoretical pricing
        Based on genuine distribution characteristics
        """
        shape_class, _ = DistributionAnalyzer.classify_distribution_shape(mad_sd_ratio)
        
        # Adjustment factors based on distribution shape
        adjustments = {
            "extreme_tails": 1.35,    # Higher premium for extreme tail risk
            "heavy_tails": 1.25,      # Higher premium for heavy tails
            "moderate_tails": 1.15,   # Moderate premium for some tail risk
            "near_normal": 1.0,       # Standard pricing for normal distribution
            "light_tails": 0.90,      # Discount for light tails
            "compressed": 0.80        # Larger discount for compressed distribution
        }
        
        return adjustments.get(shape_class, 1.0)
    
    def _calculate_implied_volatility_from_straddle(self, market_straddle_price: float, 
                                                   spot_price: float, strike_price: float,
                                                   time_to_expiry: float, risk_free_rate: float) -> float:
        """
        Calculate implied volatility from market straddle price using Newton-Raphson method
        """
        try:
            # Use the Newton-Raphson method for straddle implied volatility
            # Start with a reasonable initial guess (higher for short-term options)
            if time_to_expiry < 0.02:  # Less than ~7 days
                initial_vol = 1.0    # 100% starting guess for short-term
            elif time_to_expiry < 0.08:  # Less than ~30 days  
                initial_vol = 0.8    # 80% starting guess
            else:
                initial_vol = 0.5    # 50% starting guess for longer-term
            
            # Try to find implied volatility that matches market straddle price
            vol = initial_vol
            tolerance = 0.001  # 0.1% tolerance
            max_iterations = 50
            
            for i in range(max_iterations):
                # Calculate theoretical straddle price
                bs_straddle = FinancialMath.black_scholes_straddle(
                    spot_price, strike_price, time_to_expiry, 
                    risk_free_rate, vol, dividend_yield=0.0
                )
                
                # Calculate vega (sensitivity to volatility) for straddle
                call_greeks = FinancialMath.calculate_greeks(
                    spot_price, strike_price, time_to_expiry,
                    risk_free_rate, vol, dividend_yield=0.0, option_type="call"
                )
                put_greeks = FinancialMath.calculate_greeks(
                    spot_price, strike_price, time_to_expiry,
                    risk_free_rate, vol, dividend_yield=0.0, option_type="put"
                )
                
                # Total vega for straddle (call vega + put vega)
                straddle_vega = (call_greeks['vega'] + put_greeks['vega']) * 100  # Convert from per 1% to per 1
                
                if abs(straddle_vega) < 1e-10:  # Avoid division by zero
                    break
                
                # Newton-Raphson update
                price_diff = bs_straddle - market_straddle_price
                vol_new = vol - price_diff / straddle_vega
                
                # Keep volatility within reasonable bounds
                vol_new = max(0.01, min(vol_new, 10.0))  # Between 1% and 1000%
                
                if abs(vol_new - vol) < tolerance:
                    return vol_new
                
                vol = vol_new
            
            # If convergence failed, return a reasonable estimate based on price ratio
            logging.warning(f"Implied vol convergence failed, using price-based estimate")
            return self._estimate_implied_vol_from_price_ratio(
                market_straddle_price, spot_price, time_to_expiry
            )
            
        except Exception as e:
            logging.error(f"Error calculating implied volatility: {e}")
            return self._estimate_implied_vol_from_price_ratio(
                market_straddle_price, spot_price, time_to_expiry
            )
    
    def _estimate_implied_vol_from_price_ratio(self, straddle_price: float, 
                                             spot_price: float, time_to_expiry: float) -> float:
        """
        Rough estimate of implied volatility from straddle price using approximation
        Straddle ≈ 0.8 * Spot * Vol * sqrt(Time)
        Therefore: Vol ≈ Straddle / (0.8 * Spot * sqrt(Time))
        """
        if time_to_expiry <= 0 or spot_price <= 0:
            return 1.0  # Default 100% vol
        
        import math
        estimated_vol = straddle_price / (0.8 * spot_price * math.sqrt(time_to_expiry))
        
        # Keep within reasonable bounds
        return max(0.1, min(estimated_vol, 5.0))  # Between 10% and 500%


class MADStraddleAnalyzer:
    """Enhanced straddle analyzer with expiry-specific MAD analysis"""
    
    def __init__(self):
        self.thalex = th.Thalex(NETWORK)
        self.btc_price = 0.0
        self.login_success = False
        self.expirations: Dict[str, ExpirationData] = {}
        
        # Global price history for sharing across analyzers
        self.global_price_history = deque(maxlen=200)
        self.global_timestamps = deque(maxlen=200)
        
        # Market regime tracking
        self.current_market_regime = "UNKNOWN"
        self.regime_confidence = 0.0
        
        # API credentials
        self.key_id = keys.key_ids[NETWORK]
        self.private_key = keys.private_keys[NETWORK]
        
    async def connect_and_authenticate(self) -> bool:
        """Connect and authenticate"""
        try:
            logging.info("Connecting to Thalex API...")
            await self.thalex.connect()
            
            await self.thalex.login(self.key_id, self.private_key, id=CID_LOGIN)
            
            # Wait for login
            response = json.loads(await self.thalex.receive())
            if response.get('id') == CID_LOGIN and 'result' in response:
                self.login_success = True
                logging.info("Successfully authenticated")
                return True
            else:
                logging.error(f"Login failed: {response}")
                return False
                
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False
            
    async def get_btc_price(self):
        """Get current BTC price and update all expiry-specific MAD analyzers"""
        try:
            await self.thalex.ticker("BTC-PERPETUAL", id=CID_TICKER)
            response = json.loads(await self.thalex.receive())
            
            if response.get('id') == CID_TICKER and 'result' in response:
                new_price = float(response['result']['mark_price'])
                current_timestamp = time.time()
                
                # Update global price history
                self.global_price_history.append(new_price)
                self.global_timestamps.append(current_timestamp)
                self.btc_price = new_price
                
                # Distribute genuine price updates to all expiry-specific analyzers
                for expiry_date, expiry_data in self.expirations.items():
                    # No artificial manipulation - use genuine price data for all expirations
                    expiry_data.mad_analyzer.add_price_point(new_price, current_timestamp)
                
                print(f"Current BTC Price: ${self.btc_price:.2f}")
            else:
                print("Failed to get BTC price - invalid response")
        except Exception as e:
            print(f"Failed to get BTC price: {e}")
            self.btc_price = 0.0
            
    async def load_expiration_data(self):
        """Load 2 ATM options per expiration with MAD analysis"""
        try:
            # Get instruments
            await self.thalex.instruments()
            response = json.loads(await self.thalex.receive())
            
            if 'result' not in response:
                print("Failed to load instruments - invalid response")
                return
                
            instruments = response["result"]
            
            # Group options by expiry
            options_by_expiry = {}
            current_time = time.time()
        
            for instrument in instruments:
                if (instrument["underlying"] == UNDERLYING and 
                    instrument["type"] == "option" and
                    instrument.get("expiration_timestamp", 0) > current_time):
                    
                    expiry_ts = instrument["expiration_timestamp"]
                    expiry_date = datetime.fromtimestamp(expiry_ts).strftime("%Y-%m-%d")
                    
                    if expiry_date not in options_by_expiry:
                        options_by_expiry[expiry_date] = []
                        
                    option = SimpleOption(
                        name=instrument["instrument_name"],
                        strike=instrument.get("strike_price", 0),
                        option_type=instrument.get("option_type", ""),
                        expiry_ts=expiry_ts
                    )
                    options_by_expiry[expiry_date].append(option)
            
            print(f"\nFound {len(options_by_expiry)} active expirations")
            
            # For each expiry, find the 2 closest ATM options
            for expiry_date, options in options_by_expiry.items():
                expiry_ts = options[0].expiry_ts
                expiry_data = ExpirationData(expiry_ts, expiry_date)
                
                # Find closest call and put to BTC price
                calls = [opt for opt in options if opt.is_call()]
                puts = [opt for opt in options if opt.is_put()]
                
                if calls and puts and self.btc_price > 0:
                    # Find strikes that have BOTH call and put (proper straddle)
                    call_strikes = set(call.strike for call in calls)
                    put_strikes = set(put.strike for put in puts)
                    available_strikes = call_strikes & put_strikes  # Intersection
                    
                    if available_strikes:
                        # Find ATM strike from available pairs
                        atm_strike = min(available_strikes, key=lambda x: abs(x - self.btc_price))
                        
                        # Get call and put at the same strike
                        atm_call = next(call for call in calls if call.strike == atm_strike)
                        atm_put = next(put for put in puts if put.strike == atm_strike)
                        
                        expiry_data.atm_call = atm_call
                        expiry_data.atm_put = atm_put
                    else:
                        # No matching strikes - skip this expiration
                        print(f"DEBUG: No matching call/put strikes for {expiry_date}")
                        continue
                    
                    # Request prices for these 2 options
                    await self._get_option_price(atm_call)
                    await self._get_option_price(atm_put)
                    
                    # Initialize expiry analyzer with existing price history
                    self._initialize_expiry_analyzer(expiry_data)
                    
                    # Calculate expiry-specific MAD analysis
                    if expiry_data.has_straddle() and expiry_data.atm_call is not None:
                        straddle_price = expiry_data.get_straddle_price()
                        strike_price = expiry_data.atm_call.strike  # Use actual strike price
                        expiry_data.mad_analysis = expiry_data.mad_analyzer.analyze_straddle_efficiency(
                            straddle_price, self.btc_price, strike_price
                        )
                    
                    self.expirations[expiry_date] = expiry_data
                    
            print(f"Loaded data for {len(self.expirations)} expirations")
        
        except Exception as e:
            print(f"Failed to load expiration data: {e}")
            self.expirations = {}
        
    async def _get_option_price(self, option: SimpleOption):
        """Get price for a single option"""
        try:
            await self.thalex.ticker(option.name, id=CID_TICKER + 1)
            response = json.loads(await self.thalex.receive())
            
            if response.get('id') == CID_TICKER + 1 and 'result' in response:
                option.mark_price = response['result'].get('mark_price', 0.0)
                
        except Exception as e:
            logging.error(f"Failed to get price for {option.name}: {e}")
    
    def _initialize_expiry_analyzer(self, expiry_data: ExpirationData):
        """Initialize expiry analyzer with existing global price history"""
        if len(self.global_price_history) > 0 and len(self.global_timestamps) > 0:
            # Add existing price points to the expiry-specific analyzer
            for price, timestamp in zip(self.global_price_history, self.global_timestamps):
                expiry_data.mad_analyzer.add_price_point(price, timestamp)
    
    def _detect_market_regime(self) -> Tuple[str, float]:
        """Detect current market regime based on MAD/SD patterns across expirations"""
        if not self.expirations:
            return "UNKNOWN", 0.0
            
        # Collect MAD/SD ratios from all valid expirations
        mad_sd_ratios = []
        for exp_data in self.expirations.values():
            if exp_data.mad_analysis and exp_data.mad_analysis.mad_sd_ratio > 0:
                mad_sd_ratios.append(exp_data.mad_analysis.mad_sd_ratio)
        
        if len(mad_sd_ratios) < 2:
            return "INSUFFICIENT_DATA", 0.0
            
        # Calculate average MAD/SD ratio across expirations
        avg_mad_sd = statistics.mean(mad_sd_ratios)
        ratio_std = statistics.stdev(mad_sd_ratios) if len(mad_sd_ratios) > 1 else 0
        
        # Calculate confidence based on consistency across expirations
        confidence = max(0.5, 1.0 - (ratio_std * 2))  # Higher consistency = higher confidence
        
        # Regime classification based on average tail risk
        if avg_mad_sd < 0.4:
            regime = "CRISIS"  # Extreme tail risk across all expirations
        elif avg_mad_sd < 0.55:
            regime = "VOLATILE"  # High tail risk, elevated stress
        elif avg_mad_sd < 0.70:
            regime = "TRANSITIONAL"  # Moderate tail risk, changing conditions
        else:
            regime = "NORMAL"  # Low tail risk, stable conditions
            
        return regime, min(confidence, 0.95)
            
    def show_available_expirations(self):
        """Show available expirations with MAD analysis"""
        # Update market regime detection
        self.current_market_regime, self.regime_confidence = self._detect_market_regime()
        
        print("\n" + "="*105)  
        print("AVAILABLE EXPIRATION DATES - TAIL-RISK ADJUSTED ANALYSIS")
        print(f"MARKET REGIME: {self.current_market_regime} (Confidence: {self.regime_confidence:.1%})")
        print("="*105)
        print(f"{'#':<3} {'Date':<12} {'Days':<6} {'Strike':<8} {'Straddle':<10} {'MAD/SD':<8} {'Threshold':<10} {'Status':<15} {'Risk Level'}")
        print("-"*105)
        
        valid_expirations = []
        for i, (date, exp_data) in enumerate(sorted(self.expirations.items()), 1):
            strike = f"${exp_data.atm_call.strike:.0f}" if exp_data.atm_call else "N/A"
            
            if exp_data.has_straddle():
                straddle_price = f"${exp_data.get_straddle_price():.2f}"
                
                # Enhanced MAD analysis info with tail-risk awareness
                if exp_data.mad_analysis:
                    mad_sd = f"{exp_data.mad_analysis.mad_sd_ratio:.2f}"
                    underpriced_threshold, overpriced_threshold = exp_data.mad_analysis.get_tail_adjusted_thresholds()
                    threshold_display = f"{overpriced_threshold:.2f}"
                    
                    # Get tail-adjusted status
                    if exp_data.mad_analysis.efficiency_ratio > overpriced_threshold:
                        status = "🔴 SELL"
                        risk_level = "HIGH PREMIUM" if overpriced_threshold >= 1.30 else "STANDARD"
                    elif exp_data.mad_analysis.efficiency_ratio < underpriced_threshold:
                        status = "🟢 BUY"
                        risk_level = "UNDERPRICED"
                    else:
                        status = "🟡 FAIR"
                        risk_level = "NORMAL"
                    
                    # Add tail risk indicator with realistic bands
                    if exp_data.mad_analysis.mad_sd_ratio < 0.50:
                        risk_level += " (EXTREME TAILS)"
                    elif exp_data.mad_analysis.mad_sd_ratio < 0.60:
                        risk_level += " (HEAVY TAILS)"
                    elif exp_data.mad_analysis.mad_sd_ratio < 0.70:
                        risk_level += " (MOD TAILS)"
                    # else: normal distribution, no tail risk suffix
                else:
                    mad_sd = "N/A"
                    threshold_display = "N/A"
                    status = "⚪ NO DATA"
                    risk_level = "INSUFFICIENT DATA"
                    
                valid_expirations.append((i, date, exp_data))
            else:
                straddle_price = "N/A"
                mad_sd = "N/A"
                threshold_display = "N/A"
                status = "✗ INCOMPLETE"
                risk_level = "NO STRADDLE"
                
            print(f"{i:<3} {date:<12} {exp_data.days_to_expiry:<6.1f} {strike:<8} {straddle_price:<10} {mad_sd:<8} {threshold_display:<10} {status:<15} {risk_level}")
            
        return valid_expirations
        
    def plot_enhanced_straddle_chart(self, expiry_data: ExpirationData):
        """Plot enhanced straddle chart with MAD analysis"""
        try:
            print(f"DEBUG: Plotting enhanced chart for {expiry_data.expiry_date}")
            
            if not expiry_data.has_straddle():
                print("Cannot plot: No complete straddle data")
                return
            
            # Setup enhanced chart layout
            fig, (ax_main, ax_analysis) = plt.subplots(2, 1, figsize=(14, 10), 
                                                     height_ratios=[3, 1])
            
            # Main straddle chart (top)
            ax_main.set_title(f"BTC Straddle Analysis with MAD Theory - {expiry_data.expiry_date}", 
                            fontsize=14, pad=20)
            ax_main.set_ylabel("Price ($)")
            ax_main.grid(True, alpha=0.3)
            
            # Get data
            call = expiry_data.atm_call
            put = expiry_data.atm_put
            straddle_price = expiry_data.get_straddle_price()
            lower_breakeven, upper_breakeven = expiry_data.get_breakeven_range()
            
            # Plot current BTC price
            ax_main.axhline(y=self.btc_price, color='blue', linewidth=2, 
                          label=f'BTC Price: ${self.btc_price:.0f}')
            
            # Plot breakeven lines
            ax_main.axhline(y=upper_breakeven, color='red', linestyle='--', 
                          linewidth=1.5, label=f'Upper Breakeven: ${upper_breakeven:.0f}')
            ax_main.axhline(y=lower_breakeven, color='red', linestyle='--', 
                          linewidth=1.5, label=f'Lower Breakeven: ${lower_breakeven:.0f}')
            
            # Plot strike price
            if call is not None:
                ax_main.axhline(y=call.strike, color='green', linestyle=':', 
                              linewidth=1, alpha=0.7, label=f'ATM Strike: ${call.strike:.0f}')
            
            # Enhanced info box with MAD analysis
            info_text = self._create_enhanced_info_text(expiry_data)
            
            ax_main.text(0.02, 0.98, info_text, transform=ax_main.transAxes, 
                        verticalalignment='top', 
                        bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
                        fontsize=9)
            
            # Set y-axis limits
            padding = (upper_breakeven - lower_breakeven) * 0.1
            ax_main.set_ylim(lower_breakeven - padding, upper_breakeven + padding)
            ax_main.legend(loc='upper right')
            
            # MAD Analysis panel (bottom)
            self._plot_mad_analysis_panel(ax_analysis, expiry_data)
            
            plt.tight_layout()
            plt.show()
            print("DEBUG: Enhanced chart displayed successfully!")
            
        except Exception as e:
            print(f"ERROR plotting enhanced chart: {e}")
            import traceback
            traceback.print_exc()
            
    def _create_enhanced_info_text(self, expiry_data: ExpirationData) -> str:
        """Create enhanced info text with MAD analysis"""
        if not expiry_data.has_straddle():
            return "Incomplete straddle data"
            
        call = expiry_data.atm_call
        put = expiry_data.atm_put
        
        # Type safety checks
        if call is None or put is None:
            return "Missing option data"
            
        straddle_price = expiry_data.get_straddle_price()
        lower_breakeven, upper_breakeven = expiry_data.get_breakeven_range()
        
        info_lines = [
            f"Call: {call.name} = ${call.mark_price:.2f}",
            f"Put: {put.name} = ${put.mark_price:.2f}",
            f"Straddle Price: ${straddle_price:.2f}",
            f"Range Width: ${upper_breakeven - lower_breakeven:.0f}",
            f"Days to Expiry: {expiry_data.days_to_expiry:.1f}",
            ""
        ]
        
        # Add MAD analysis if available
        if expiry_data.mad_analysis:
            mad_analysis = expiry_data.mad_analysis
            info_lines.extend([
                "MAD ANALYSIS:",
                f"MAD/SD Ratio: {mad_analysis.mad_sd_ratio:.3f}",
                f"Distribution: {mad_analysis.get_distribution_assessment()}",
                f"MAD-Enhanced Price: ${mad_analysis.theoretical_straddle:.2f}",
                f"MAD Efficiency: {mad_analysis.efficiency_ratio:.2f}",
                "",
                "BENCHMARKING:",
                f"Black-Scholes Price: ${getattr(mad_analysis, 'bs_theoretical', 0):.2f}",
                f"BS Efficiency: {getattr(mad_analysis, 'bs_efficiency_ratio', 0):.2f}",
                f"Assessment: {mad_analysis.get_straddle_assessment()}"
            ])
        else:
            info_lines.append("MAD Analysis: Insufficient data")
            
        return "\n".join(info_lines)
        
    def _plot_mad_analysis_panel(self, ax, expiry_data: ExpirationData):
        """Plot MAD analysis panel"""
        ax.set_title("Risk Analysis Dashboard", fontsize=12)
        ax.axis('off')
        
        if expiry_data.mad_analysis:
            mad_analysis = expiry_data.mad_analysis
            warnings = mad_analysis.get_risk_warnings()
            
            # Create analysis text
            analysis_lines = [
                f"DATA: MAD/SD Ratio: {mad_analysis.mad_sd_ratio:.3f} ({'Normal' if 0.75 <= mad_analysis.mad_sd_ratio <= 0.85 else 'Abnormal'})",
                f"PRICE: Straddle Efficiency: {mad_analysis.efficiency_ratio:.2f} ({mad_analysis.get_straddle_assessment()})",
                f"DISTRIBUTION: Distribution Shape: {mad_analysis.get_distribution_assessment()}",
                ""
            ]
            
            if warnings:
                analysis_lines.append("WARNINGS: RISK WARNINGS:")
                analysis_lines.extend([f"   {warning}" for warning in warnings])
            else:
                analysis_lines.append("STATUS: No significant risk warnings detected")
                
            analysis_text = "\n".join(analysis_lines)
            
            ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "MAD Analysis: Insufficient price history data\nCollecting data points...", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
        
    async def run_interactive_session(self):
        """Run enhanced interactive session with MAD analysis"""
        print("Starting MAD Enhanced Straddle Analyzer...")
        print("Collecting price data for MAD analysis...")
        
        # Connect and get basic data
        if not await self.connect_and_authenticate():
            print("Failed to connect")
            return
            
        # Collect substantial initial price points for MAD analysis
        print("Collecting initial price data...")
        for i in range(60):  # Increased to 60 to ensure sufficient data for filtering
            await self.get_btc_price()
            if i < 59:  # Don't sleep on last iteration
                await asyncio.sleep(0.5)  # Faster collection for more data points
        
        await self.load_expiration_data()
        
        # Interactive loop
        while True:
            valid_expirations = self.show_available_expirations()
            
            if not valid_expirations:
                print("\nNo valid expirations found with complete data")
                break
                
            print("\nOptions:")
            print("  Enter expiration number (1-{}) to view enhanced chart".format(len(valid_expirations)))
            print("  'position' to calculate position sizing for selling straddles")
            print("  'refresh' to reload data and update MAD analysis")
            print("  'quit' to exit")
            
            choice = input("\nYour choice: ").strip()
            
            if choice.lower() == 'quit':
                break
            elif choice.lower() == 'position':
                print("\nStarting Position Calculator...")
                try:
                    position_selector = InteractiveStraddleSelector(self)
                    position_selector.interactive_position_calculator()
                except Exception as e:
                    print(f" Error in position calculator: {e}")
                    logging.error(f"Position calculator error: {e}")
                continue
            elif choice.lower() == 'refresh':
                print("\nRefreshing data and updating MAD analysis...")
                
                # Clean disconnect old connection
                if hasattr(self.thalex, 'disconnect'):
                    try:
                        await self.thalex.disconnect()
                    except:
                        pass
                
                # Create fresh connection and reload all data
                if await self.connect_and_authenticate():
                    # Collect fresh price data
                    for i in range(3):
                        await self.get_btc_price()
                        if i < 2:
                            await asyncio.sleep(1)
                    await self.load_expiration_data()
                    print("Data and MAD analysis refreshed successfully!")
                else:
                    print("Failed to refresh - connection error")
                continue
            else:
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(valid_expirations):
                        _, date, exp_data = valid_expirations[choice_num - 1]
                        print(f"\nShowing enhanced analysis for {date}...")
                        self.plot_enhanced_straddle_chart(exp_data)
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        # Cleanup
        if hasattr(self.thalex, 'disconnect'):
            await self.thalex.disconnect()

async def main():
    """Main entry point"""
    # Enable debug logging to see MAD calculation details
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    analyzer = MADStraddleAnalyzer()
    try:
        await analyzer.run_interactive_session()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())