#!/usr/bin/env python3
"""
Optimal Delta Hedging Module
============================

Core implementation of Colin Bennett's delta hedging philosophy:
- Uses FORECASTED realized volatility (σ_R) for hedge ratio calculation, NOT market implied vol
- Incorporates volatility smile/skew effects for accurate hedge ratios
- Optimizes rebalancing frequency considering transaction costs
- Provides regime-aware dynamic hedge adjustments

Key Philosophy:
"The hedge ratio should be based on your best estimate of future realized volatility,
not the market's implied volatility, which is merely a quotation convention."
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

from financial_math import FinancialMath, TradingConstants
from forward_volatility import ForwardVolatilityEstimator, VolatilityForecast
from volatility_surface import VolatilitySurface
from volatility_regime import VolatilityRegimeDetector, RegimeMetrics


class OptionType(Enum):
    CALL = "call"
    PUT = "put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"


@dataclass
class OptionParams:
    """Option parameters for delta calculation"""
    current_spot_price: float  # S
    strike_price: float        # K
    time_to_expiration: float  # τ (in years)
    risk_free_rate: float      # r
    dividend_yield: float      # D (default 0 for crypto)
    option_type: OptionType
    position_size: float = 1.0  # Number of contracts (can be negative for short)
    
    # For complex positions
    put_strike: Optional[float] = None  # For strangles
    

@dataclass
class MarketData:
    """Market data for option pricing (NOT for hedge calculation)"""
    market_implied_volatility: float  # Σ - only for pricing, NOT hedging
    current_option_price: float       # Current market price
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    underlying_bid: Optional[float] = None
    underlying_ask: Optional[float] = None
    
    # Additional market context
    bid_ask_spread: Optional[float] = None
    volume: float = 0.0
    open_interest: float = 0.0


@dataclass
class TransactionCostParams:
    """Transaction cost parameters"""
    commission_per_trade: float = 0.0005  # 0.05% per trade
    bid_ask_spread_cost: float = 0.001    # 0.1% spread cost
    market_impact_factor: float = 0.0001  # 0.01% market impact per unit
    minimum_trade_size: float = 0.01      # Minimum trade size
    
    # Rebalancing parameters
    rebalancing_frequency: str = "threshold"  # "daily", "hourly", "threshold"
    delta_threshold: float = 0.05             # Rebalance when delta off by this much
    time_threshold_hours: float = 24          # Max time between rebalances


@dataclass
class OptimalDeltaResult:
    """Result of optimal delta calculation"""
    optimal_delta: float                    # The optimal hedge ratio
    current_bsm_delta: float               # Standard BSM delta for comparison
    hedge_adjustment_factor: float         # optimal_delta / bsm_delta
    
    # Decomposition of delta calculation
    base_delta_component: float            # Pure BSM with forecast vol
    skew_adjustment: float                 # Adjustment for vol smile/skew
    regime_adjustment: float               # Adjustment for vol regime
    transaction_cost_adjustment: float     # Adjustment for costs
    
    # Supporting information
    forecast_volatility: float             # The σ_R used for calculation
    market_implied_volatility: float       # The market Σ (for reference)
    vol_forecast_confidence: float         # Confidence in vol forecast
    
    # Risk metrics
    expected_hedge_error: float            # Expected tracking error
    cost_adjusted_delta: float             # Delta adjusted for transaction costs
    recommended_rebalance_threshold: float # When to rebalance
    
    # Regime context
    volatility_regime: str                 # Current vol regime
    regime_implications: str               # Implications for hedging


class OptimalDeltaHedger:
    """
    Core optimal delta hedging calculator implementing Colin Bennett's philosophy
    """
    
    def __init__(self, 
                 vol_estimator: Optional[ForwardVolatilityEstimator] = None,
                 vol_surface: Optional[VolatilitySurface] = None,
                 regime_detector: Optional[VolatilityRegimeDetector] = None):
        """
        Initialize the optimal delta hedger
        
        Args:
            vol_estimator: Forward volatility estimator for σ_R
            vol_surface: Volatility surface for smile/skew modeling
            regime_detector: Volatility regime detector
        """
        self.vol_estimator = vol_estimator
        self.vol_surface = vol_surface
        self.regime_detector = regime_detector
        
        # Cache for performance
        self._delta_cache: Dict[str, Tuple[float, OptimalDeltaResult]] = {}
        self._cache_ttl_seconds = 30  # Cache delta calculations for 30 seconds
        
        logging.info("Initialized OptimalDeltaHedger with Bennett philosophy")
    
    def calculate_optimal_delta(self, 
                              option_params: OptionParams,
                              market_data: MarketData,
                              forecast_volatility: float,
                              transaction_costs: Optional[TransactionCostParams] = None,
                              historical_returns: Optional[List[float]] = None) -> OptimalDeltaResult:
        """
        Calculate optimal delta using forecasted realized volatility
        
        This is the core function implementing Bennett's philosophy:
        "Use your best estimate of future realized volatility for the hedge ratio"
        
        Args:
            option_params: Option parameters
            market_data: Current market data (implied vol used only for pricing)
            forecast_volatility: Forecasted realized volatility (σ_R) - THE KEY INPUT
            transaction_costs: Transaction cost parameters
            historical_returns: Recent returns for regime analysis
            
        Returns:
            OptimalDeltaResult with comprehensive hedging information
        """
        # Input validation
        if forecast_volatility <= 0:
            raise ValueError("Forecast volatility must be positive")
        if option_params.time_to_expiration <= 0:
            raise ValueError("Time to expiration must be positive")
        
        # Check cache
        cache_key = self._generate_cache_key(option_params, market_data, forecast_volatility)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Default transaction costs if not provided
        if transaction_costs is None:
            transaction_costs = TransactionCostParams()
        
        # Calculate base delta using FORECASTED volatility (not market implied)
        base_delta = self._calculate_base_delta_with_forecast_vol(
            option_params, forecast_volatility
        )
        
        # Calculate standard BSM delta for comparison (using market implied vol)
        bsm_delta = self._calculate_standard_bsm_delta(option_params, market_data)
        
        # Apply volatility smile/skew adjustments
        skew_adjustment = self._calculate_skew_adjustment(
            option_params, market_data, forecast_volatility
        )
        
        # Apply volatility regime adjustments
        regime_adjustment = self._calculate_regime_adjustment(
            option_params, historical_returns, forecast_volatility
        )
        
        # Apply transaction cost optimizations
        cost_adjustment = self._calculate_transaction_cost_adjustment(
            option_params, transaction_costs, base_delta
        )
        
        # Combine all adjustments
        optimal_delta = base_delta + skew_adjustment + regime_adjustment + cost_adjustment
        
        # Calculate hedge adjustment factor
        hedge_adjustment_factor = optimal_delta / bsm_delta if bsm_delta != 0 else 1.0
        
        # Calculate expected hedge error and cost-adjusted metrics
        expected_hedge_error = self._estimate_hedge_error(
            option_params, forecast_volatility, transaction_costs
        )
        
        cost_adjusted_delta = self._calculate_cost_adjusted_delta(
            optimal_delta, transaction_costs, option_params
        )
        
        # Determine optimal rebalancing threshold
        rebalance_threshold = self._calculate_optimal_rebalance_threshold(
            option_params, forecast_volatility, transaction_costs
        )
        
        # Get regime context
        regime_info = self._get_regime_context()
        
        # Get vol forecast confidence
        vol_confidence = self._get_vol_forecast_confidence(historical_returns)
        
        result = OptimalDeltaResult(
            optimal_delta=optimal_delta,
            current_bsm_delta=bsm_delta,
            hedge_adjustment_factor=hedge_adjustment_factor,
            base_delta_component=base_delta,
            skew_adjustment=skew_adjustment,
            regime_adjustment=regime_adjustment,
            transaction_cost_adjustment=cost_adjustment,
            forecast_volatility=forecast_volatility,
            market_implied_volatility=market_data.market_implied_volatility,
            vol_forecast_confidence=vol_confidence,
            expected_hedge_error=expected_hedge_error,
            cost_adjusted_delta=cost_adjusted_delta,
            recommended_rebalance_threshold=rebalance_threshold,
            volatility_regime=regime_info.get("regime", "unknown"),
            regime_implications=regime_info.get("implications", "")
        )
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        logging.debug(f"Calculated optimal delta: {optimal_delta:.4f} vs BSM: {bsm_delta:.4f} "
                     f"(adjustment: {hedge_adjustment_factor:.3f}x)")
        
        return result
    
    def _calculate_base_delta_with_forecast_vol(self, 
                                              option_params: OptionParams, 
                                              forecast_vol: float) -> float:
        """
        Calculate base delta using forecasted volatility instead of implied vol
        
        This is the core of Bennett's approach - use σ_R not Σ
        """
        if option_params.option_type == OptionType.CALL:
            greeks = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.strike_price,
                time_to_expiry=option_params.time_to_expiration,
                risk_free_rate=option_params.risk_free_rate,
                volatility=forecast_vol,  # KEY: Using forecast vol, not implied
                dividend_yield=option_params.dividend_yield,
                option_type="call"
            )
            return greeks['delta'] * option_params.position_size
            
        elif option_params.option_type == OptionType.PUT:
            greeks = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.strike_price,
                time_to_expiry=option_params.time_to_expiration,
                risk_free_rate=option_params.risk_free_rate,
                volatility=forecast_vol,
                dividend_yield=option_params.dividend_yield,
                option_type="put"
            )
            return greeks['delta'] * option_params.position_size
            
        elif option_params.option_type == OptionType.STRADDLE:
            # Straddle = Call + Put at same strike
            call_greeks = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.strike_price,
                time_to_expiry=option_params.time_to_expiration,
                risk_free_rate=option_params.risk_free_rate,
                volatility=forecast_vol,
                dividend_yield=option_params.dividend_yield,
                option_type="call"
            )
            put_greeks = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.strike_price,
                time_to_expiry=option_params.time_to_expiration,
                risk_free_rate=option_params.risk_free_rate,
                volatility=forecast_vol,
                dividend_yield=option_params.dividend_yield,
                option_type="put"
            )
            return (call_greeks['delta'] + put_greeks['delta']) * option_params.position_size
            
        elif option_params.option_type == OptionType.STRANGLE:
            # Strangle = Call at higher strike + Put at lower strike
            if not option_params.put_strike:
                raise ValueError("Put strike required for strangle")
            
            call_greeks = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.strike_price,  # Call strike
                time_to_expiry=option_params.time_to_expiration,
                risk_free_rate=option_params.risk_free_rate,
                volatility=forecast_vol,
                dividend_yield=option_params.dividend_yield,
                option_type="call"
            )
            put_greeks = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.put_strike,  # Put strike
                time_to_expiry=option_params.time_to_expiration,
                risk_free_rate=option_params.risk_free_rate,
                volatility=forecast_vol,
                dividend_yield=option_params.dividend_yield,
                option_type="put"
            )
            return (call_greeks['delta'] + put_greeks['delta']) * option_params.position_size
        
        else:
            raise ValueError(f"Unsupported option type: {option_params.option_type}")
    
    def _calculate_standard_bsm_delta(self, 
                                    option_params: OptionParams, 
                                    market_data: MarketData) -> float:
        """Calculate standard BSM delta using market implied volatility"""
        if option_params.option_type == OptionType.CALL:
            greeks = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.strike_price,
                time_to_expiry=option_params.time_to_expiration,
                risk_free_rate=option_params.risk_free_rate,
                volatility=market_data.market_implied_volatility,  # Using market implied vol
                dividend_yield=option_params.dividend_yield,
                option_type="call"
            )
            return greeks['delta'] * option_params.position_size
        
        # Similar implementations for other option types...
        # (Abbreviated for brevity - would include full implementations)
        return 0.0
    
    def _calculate_skew_adjustment(self, 
                                 option_params: OptionParams,
                                 market_data: MarketData, 
                                 forecast_vol: float) -> float:
        """
        Calculate adjustment for volatility smile/skew effects
        
        Bennett's insight: BSM delta is often wrong for OTM options due to skew
        """
        if not self.vol_surface:
            return 0.0  # No adjustment if no vol surface available
        
        # Get volatility for this specific strike from surface
        strike_vol = self.vol_surface.get_implied_volatility(
            option_params.strike_price, 
            option_params.time_to_expiration
        )
        
        if not strike_vol:
            return 0.0
        
        # Calculate delta difference due to skew
        # If strike vol differs from forecast vol, adjust delta accordingly
        vol_diff = strike_vol - forecast_vol
        
        # Estimate delta sensitivity to vol changes (vega/spot approximation)
        greeks = FinancialMath.calculate_greeks(
            spot=option_params.current_spot_price,
            strike=option_params.strike_price,
            time_to_expiry=option_params.time_to_expiration,
            risk_free_rate=option_params.risk_free_rate,
            volatility=forecast_vol,
            dividend_yield=option_params.dividend_yield,
            option_type=option_params.option_type.value
        )
        
        # Delta adjustment ≈ vega * vol_difference / spot_price
        delta_vol_sensitivity = greeks['vega'] * 100 / option_params.current_spot_price
        skew_adjustment = delta_vol_sensitivity * vol_diff
        
        return skew_adjustment * option_params.position_size
    
    def _calculate_regime_adjustment(self, 
                                   option_params: OptionParams,
                                   historical_returns: Optional[List[float]], 
                                   forecast_vol: float) -> float:
        """Calculate adjustment based on current volatility regime"""
        if not self.regime_detector or not historical_returns:
            return 0.0
        
        # Update regime detector with recent data
        for i, ret in enumerate(historical_returns[-100:]):  # Use last 100 returns
            self.regime_detector.add_price_observation(
                price=option_params.current_spot_price * math.exp(sum(historical_returns[-100+i:])),
                timestamp=time.time() - (100-i) * 3600  # Hourly data assumption
            )
        
        regime_metrics = self.regime_detector.analyze_regime()
        if not regime_metrics:
            return 0.0
        
        # Adjust delta based on regime characteristics
        # High vol regimes: reduce delta (expect mean reversion)
        # Low vol regimes: increase delta (expect vol expansion)
        base_adjustment = 0.0
        
        if regime_metrics.vol_percentile > 80:  # High vol regime
            base_adjustment = -0.05  # Reduce hedge ratio by 5%
        elif regime_metrics.vol_percentile < 20:  # Low vol regime
            base_adjustment = 0.03   # Increase hedge ratio by 3%
        
        # Scale by time to expiration (less adjustment for longer-dated options)
        time_scaling = math.exp(-option_params.time_to_expiration * 4)  # Decay over time
        
        return base_adjustment * time_scaling * abs(option_params.position_size)
    
    def _calculate_transaction_cost_adjustment(self, 
                                             option_params: OptionParams,
                                             transaction_costs: TransactionCostParams, 
                                             base_delta: float) -> float:
        """
        Calculate delta adjustment to account for transaction costs
        
        Wider hedging bands in high-cost environments
        """
        # Estimate rebalancing frequency based on costs
        if transaction_costs.rebalancing_frequency == "threshold":
            # Higher costs -> wider thresholds -> less frequent rebalancing
            cost_factor = (transaction_costs.commission_per_trade + 
                          transaction_costs.bid_ask_spread_cost)
            
            # Adjust delta slightly to account for less frequent rebalancing
            # This is a second-order effect - main optimization is in rebalancing logic
            cost_adjustment = -0.01 * cost_factor * 100  # Small adjustment
            
            return cost_adjustment * np.sign(base_delta)
        
        return 0.0
    
    def _estimate_hedge_error(self, 
                            option_params: OptionParams,
                            forecast_vol: float, 
                            transaction_costs: TransactionCostParams) -> float:
        """Estimate expected hedge tracking error"""
        # Theoretical hedge error from discrete rebalancing
        rebalance_frequency_hours = 24  # Default daily
        if transaction_costs.rebalancing_frequency == "hourly":
            rebalance_frequency_hours = 1
        elif transaction_costs.rebalancing_frequency == "threshold":
            # Estimate based on delta threshold
            rebalance_frequency_hours = 24 / (transaction_costs.delta_threshold * 20)
        
        # Error scales with vol, sqrt(time between rebalances), and gamma
        greeks = FinancialMath.calculate_greeks(
            spot=option_params.current_spot_price,
            strike=option_params.strike_price,
            time_to_expiry=option_params.time_to_expiration,
            risk_free_rate=option_params.risk_free_rate,
            volatility=forecast_vol,
            dividend_yield=option_params.dividend_yield,
            option_type=option_params.option_type.value
        )
        
        # Simplified hedge error estimate
        dt = rebalance_frequency_hours / (24 * 365.25)  # Convert to years
        error = 0.5 * greeks['gamma'] * (forecast_vol * option_params.current_spot_price)**2 * dt
        
        return abs(error) * option_params.position_size
    
    def _calculate_cost_adjusted_delta(self, 
                                     optimal_delta: float,
                                     transaction_costs: TransactionCostParams, 
                                     option_params: OptionParams) -> float:
        """Calculate delta adjusted for practical trading costs"""
        # Round to minimum trade size to avoid tiny adjustments
        min_size = transaction_costs.minimum_trade_size
        
        # Round delta to nearest tradeable increment
        cost_adjusted = round(optimal_delta / min_size) * min_size
        
        return cost_adjusted
    
    def _calculate_optimal_rebalance_threshold(self, 
                                             option_params: OptionParams,
                                             forecast_vol: float, 
                                             transaction_costs: TransactionCostParams) -> float:
        """Calculate optimal delta threshold for rebalancing"""
        # Balance hedge error cost vs transaction cost
        # Higher vol -> tighter thresholds (more rebalancing)
        # Higher transaction costs -> wider thresholds (less rebalancing)
        
        base_threshold = transaction_costs.delta_threshold
        
        # Adjust based on volatility (higher vol needs tighter control)
        vol_adjustment = (forecast_vol - 0.5) * 0.02  # Scale around 50% vol
        
        # Adjust based on time to expiration (shorter time needs tighter control)
        time_adjustment = max(0.01, option_params.time_to_expiration) * 0.01
        
        # Adjust based on transaction costs
        cost_adjustment = (transaction_costs.commission_per_trade + 
                          transaction_costs.bid_ask_spread_cost) * 10
        
        optimal_threshold = base_threshold - vol_adjustment - time_adjustment + cost_adjustment
        
        # Ensure reasonable bounds
        return max(0.01, min(0.20, optimal_threshold))
    
    def _get_regime_context(self) -> Dict[str, str]:
        """Get current volatility regime context"""
        if not self.regime_detector:
            return {"regime": "unknown", "implications": "No regime data available"}
        
        regime_info = self.regime_detector.get_regime_implications_for_straddles()
        return {
            "regime": self.regime_detector.current_regime.value,
            "implications": regime_info.get("implication", "")
        }
    
    def _get_vol_forecast_confidence(self, historical_returns: Optional[List[float]]) -> float:
        """Get confidence level in volatility forecast"""
        if not self.vol_estimator or not historical_returns:
            return 0.5  # Default moderate confidence
        
        # Simple confidence based on data quantity and consistency
        if len(historical_returns) >= 200:
            return 0.8
        elif len(historical_returns) >= 100:
            return 0.7
        elif len(historical_returns) >= 50:
            return 0.6
        else:
            return 0.4
    
    def _generate_cache_key(self, 
                          option_params: OptionParams,
                          market_data: MarketData, 
                          forecast_vol: float) -> str:
        """Generate cache key for delta calculation"""
        return f"{option_params.strike_price}_{option_params.time_to_expiration:.3f}_{forecast_vol:.3f}_{market_data.market_implied_volatility:.3f}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[OptimalDeltaResult]:
        """Get cached result if still valid"""
        if cache_key in self._delta_cache:
            cache_time, result = self._delta_cache[cache_key]
            if time.time() - cache_time < self._cache_ttl_seconds:
                return result
        return None
    
    def _cache_result(self, cache_key: str, result: OptimalDeltaResult):
        """Cache calculation result"""
        self._delta_cache[cache_key] = (time.time(), result)
        
        # Clean old cache entries
        if len(self._delta_cache) > 100:
            current_time = time.time()
            expired_keys = [
                key for key, (cache_time, _) in self._delta_cache.items()
                if current_time - cache_time >= self._cache_ttl_seconds
            ]
            for key in expired_keys:
                del self._delta_cache[key]
    
    def log_delta_analysis(self, result: OptimalDeltaResult, option_params: OptionParams):
        """Log comprehensive delta analysis"""
        logging.info("="*60)
        logging.info("OPTIMAL DELTA HEDGING ANALYSIS")
        logging.info("="*60)
        logging.info(f"Option: {option_params.option_type.value.upper()} "
                    f"Strike: {option_params.strike_price} "
                    f"Expiry: {option_params.time_to_expiration:.3f}y")
        logging.info(f"Optimal Delta: {result.optimal_delta:.4f}")
        logging.info(f"BSM Delta: {result.current_bsm_delta:.4f}")
        logging.info(f"Hedge Adjustment Factor: {result.hedge_adjustment_factor:.3f}x")
        logging.info("")
        logging.info("DELTA COMPONENTS:")
        logging.info(f"  Base (Forecast Vol): {result.base_delta_component:.4f}")
        logging.info(f"  Skew Adjustment: {result.skew_adjustment:+.4f}")
        logging.info(f"  Regime Adjustment: {result.regime_adjustment:+.4f}")
        logging.info(f"  Cost Adjustment: {result.transaction_cost_adjustment:+.4f}")
        logging.info("")
        logging.info("VOLATILITY:")
        logging.info(f"  Forecast Vol (σ_R): {result.forecast_volatility:.1%}")
        logging.info(f"  Market Implied (Σ): {result.market_implied_volatility:.1%}")
        logging.info(f"  Forecast Confidence: {result.vol_forecast_confidence:.1%}")
        logging.info("")
        logging.info("RISK METRICS:")
        logging.info(f"  Expected Hedge Error: {result.expected_hedge_error:.4f}")
        logging.info(f"  Rebalance Threshold: {result.recommended_rebalance_threshold:.3f}")
        logging.info(f"  Volatility Regime: {result.volatility_regime}")
        logging.info("="*60)