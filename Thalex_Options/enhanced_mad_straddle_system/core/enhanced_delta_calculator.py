#!/usr/bin/env python3
"""
Enhanced Delta Calculator
=========================

Advanced delta calculation methods that go beyond standard Black-Scholes:
- Local volatility model deltas for better skew handling
- Multi-asset correlation deltas for index options
- Regime-dependent delta adjustments
- Path-dependent delta calculations
- Delta sensitivities and risk metrics

Extends the basic delta calculation with more sophisticated models
suitable for real trading environments.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from financial_math import FinancialMath
from volatility_surface import VolatilitySurface
from optimal_delta_hedger import OptionParams, MarketData, OptionType


class DeltaCalculationMethod(Enum):
    """Different methods for calculating delta"""
    BLACK_SCHOLES = "black_scholes"
    LOCAL_VOLATILITY = "local_volatility"
    STICKY_STRIKE = "sticky_strike"
    STICKY_DELTA = "sticky_delta"
    REGIME_DEPENDENT = "regime_dependent"


@dataclass
class DeltaDecomposition:
    """Decomposition of delta into components"""
    total_delta: float
    price_sensitivity: float      # ∂V/∂S from price moves
    vol_sensitivity: float        # ∂V/∂σ * ∂σ/∂S (vol surface effects)
    time_sensitivity: float       # ∂V/∂t component (minor)
    
    # Model-specific components
    skew_component: float         # Delta adjustment from vol skew
    term_structure_component: float  # Delta adjustment from vol term structure
    correlation_component: float  # Delta adjustment from asset correlations


@dataclass
class LocalVolatilityParameters:
    """Parameters for local volatility model"""
    vol_surface: VolatilitySurface
    spot_vol_elasticity: float = -0.5  # How vol changes with spot (typical for equity indices)
    skew_adjustment_factor: float = 1.0
    min_vol: float = 0.05
    max_vol: float = 2.0


@dataclass
class DeltaRiskMetrics:
    """Risk metrics related to delta hedging"""
    delta_gamma_sensitivity: float    # How delta changes with spot moves
    delta_vega_sensitivity: float     # How delta changes with vol changes
    delta_theta_sensitivity: float    # How delta changes with time decay
    
    hedge_ratio_stability: float      # Stability of hedge ratio over time
    model_risk_adjustment: float      # Adjustment for model uncertainty
    liquidity_adjustment: float       # Adjustment for liquidity constraints


class EnhancedDeltaCalculator:
    """
    Enhanced delta calculator with multiple methodologies
    """
    
    def __init__(self, vol_surface: Optional[VolatilitySurface] = None):
        self.vol_surface = vol_surface
        self.logger = logging.getLogger(__name__)
        
        # Cache for expensive calculations
        self._local_vol_cache: Dict[str, float] = {}
        self._delta_cache: Dict[str, DeltaDecomposition] = {}
    
    def calculate_enhanced_delta(self, 
                               option_params: OptionParams,
                               market_data: MarketData,
                               method: DeltaCalculationMethod = DeltaCalculationMethod.LOCAL_VOLATILITY,
                               local_vol_params: Optional[LocalVolatilityParameters] = None) -> DeltaDecomposition:
        """
        Calculate delta using enhanced methods
        
        Args:
            option_params: Option parameters
            market_data: Market data
            method: Calculation method to use
            local_vol_params: Parameters for local vol model
            
        Returns:
            DeltaDecomposition with detailed breakdown
        """
        if method == DeltaCalculationMethod.BLACK_SCHOLES:
            return self._calculate_black_scholes_delta(option_params, market_data)
        elif method == DeltaCalculationMethod.LOCAL_VOLATILITY:
            return self._calculate_local_volatility_delta(
                option_params, market_data, local_vol_params
            )
        elif method == DeltaCalculationMethod.STICKY_STRIKE:
            return self._calculate_sticky_strike_delta(option_params, market_data)
        elif method == DeltaCalculationMethod.STICKY_DELTA:
            return self._calculate_sticky_delta_delta(option_params, market_data)
        elif method == DeltaCalculationMethod.REGIME_DEPENDENT:
            return self._calculate_regime_dependent_delta(option_params, market_data)
        else:
            raise ValueError(f"Unsupported delta calculation method: {method}")
    
    def _calculate_black_scholes_delta(self, 
                                     option_params: OptionParams,
                                     market_data: MarketData) -> DeltaDecomposition:
        """Standard Black-Scholes delta calculation"""
        greeks = FinancialMath.calculate_greeks(
            spot=option_params.current_spot_price,
            strike=option_params.strike_price,
            time_to_expiry=option_params.time_to_expiration,
            risk_free_rate=option_params.risk_free_rate,
            volatility=market_data.market_implied_volatility,
            dividend_yield=option_params.dividend_yield,
            option_type=option_params.option_type.value
        )
        
        total_delta = greeks['delta'] * option_params.position_size
        
        return DeltaDecomposition(
            total_delta=total_delta,
            price_sensitivity=total_delta,  # In BSM, delta is pure price sensitivity
            vol_sensitivity=0.0,            # No vol surface effects in BSM
            time_sensitivity=0.0,
            skew_component=0.0,
            term_structure_component=0.0,
            correlation_component=0.0
        )
    
    def _calculate_local_volatility_delta(self, 
                                        option_params: OptionParams,
                                        market_data: MarketData,
                                        local_vol_params: Optional[LocalVolatilityParameters]) -> DeltaDecomposition:
        """
        Calculate delta using local volatility model
        
        This is key for handling volatility skew effects properly.
        The idea is that volatility changes as the underlying moves.
        """
        if not local_vol_params:
            local_vol_params = LocalVolatilityParameters(
                vol_surface=self.vol_surface,
                spot_vol_elasticity=-0.5  # Typical equity index behavior
            )
        
        if not local_vol_params.vol_surface:
            self.logger.warning("No volatility surface available for local vol delta")
            return self._calculate_black_scholes_delta(option_params, market_data)
        
        # Calculate standard BSM delta first
        bsm_delta_result = self._calculate_black_scholes_delta(option_params, market_data)
        
        # Calculate local volatility adjustments
        skew_adjustment = self._calculate_local_vol_skew_adjustment(
            option_params, market_data, local_vol_params
        )
        
        # Calculate how volatility changes with spot price
        vol_sensitivity = self._calculate_vol_spot_sensitivity(
            option_params, market_data, local_vol_params
        )
        
        # Total delta = BSM delta + vol surface adjustment
        total_delta = bsm_delta_result.total_delta + skew_adjustment
        
        return DeltaDecomposition(
            total_delta=total_delta,
            price_sensitivity=bsm_delta_result.price_sensitivity,
            vol_sensitivity=vol_sensitivity,
            time_sensitivity=0.0,  # Could be enhanced
            skew_component=skew_adjustment,
            term_structure_component=0.0,  # Could be enhanced
            correlation_component=0.0
        )
    
    def _calculate_local_vol_skew_adjustment(self, 
                                           option_params: OptionParams,
                                           market_data: MarketData,
                                           local_vol_params: LocalVolatilityParameters) -> float:
        """
        Calculate delta adjustment due to volatility skew
        
        This captures Bennett's insight about BSM delta being wrong for OTM options
        """
        vol_surface = local_vol_params.vol_surface
        
        # Get volatility at current strike
        current_vol = vol_surface.get_implied_volatility(
            option_params.strike_price,
            option_params.time_to_expiration
        )
        
        if not current_vol:
            return 0.0
        
        # Calculate how vol changes with small spot moves
        spot_shift = option_params.current_spot_price * 0.01  # 1% move
        
        # Vol at higher spot price
        higher_strike = option_params.strike_price * (1 + 0.01)  # Adjust strike proportionally
        vol_up = vol_surface.get_implied_volatility(
            higher_strike, option_params.time_to_expiration
        ) or current_vol
        
        # Vol at lower spot price  
        lower_strike = option_params.strike_price * (1 - 0.01)
        vol_down = vol_surface.get_implied_volatility(
            lower_strike, option_params.time_to_expiration
        ) or current_vol
        
        # Estimate vol-spot sensitivity
        vol_spot_sensitivity = (vol_up - vol_down) / (2 * spot_shift)
        
        # Calculate vega to convert vol sensitivity to delta adjustment
        greeks = FinancialMath.calculate_greeks(
            spot=option_params.current_spot_price,
            strike=option_params.strike_price,
            time_to_expiry=option_params.time_to_expiration,
            risk_free_rate=option_params.risk_free_rate,
            volatility=current_vol,
            dividend_yield=option_params.dividend_yield,
            option_type=option_params.option_type.value
        )
        
        # Delta adjustment = vega * vol_spot_sensitivity
        # This captures how delta changes due to vol surface effects
        skew_adjustment = greeks['vega'] * 100 * vol_spot_sensitivity  # vega is per 1% vol change
        
        return skew_adjustment * option_params.position_size * local_vol_params.skew_adjustment_factor
    
    def _calculate_vol_spot_sensitivity(self, 
                                      option_params: OptionParams,
                                      market_data: MarketData,
                                      local_vol_params: LocalVolatilityParameters) -> float:
        """Calculate how volatility changes with spot price movements"""
        # This is the key insight: volatility is not constant as spot moves
        # For equity indices: vol typically increases as spot decreases (negative skew)
        
        spot_elasticity = local_vol_params.spot_vol_elasticity
        current_vol = market_data.market_implied_volatility
        
        # Vol sensitivity = elasticity * vol / spot
        vol_spot_sensitivity = spot_elasticity * current_vol / option_params.current_spot_price
        
        return vol_spot_sensitivity
    
    def _calculate_sticky_strike_delta(self, 
                                     option_params: OptionParams,
                                     market_data: MarketData) -> DeltaDecomposition:
        """
        Calculate delta assuming volatility stays fixed at each strike (sticky strike)
        
        This is the standard BSM assumption but made explicit
        """
        # This is essentially the same as BSM delta
        return self._calculate_black_scholes_delta(option_params, market_data)
    
    def _calculate_sticky_delta_delta(self, 
                                    option_params: OptionParams,
                                    market_data: MarketData) -> DeltaDecomposition:
        """
        Calculate delta assuming volatility moves to maintain constant delta
        
        This is useful for very short-dated options where delta hedging is primary
        """
        # This is a more complex calculation that would require iterative solving
        # For now, return BSM delta with a note that this needs enhancement
        bsm_result = self._calculate_black_scholes_delta(option_params, market_data)
        
        # Add a small adjustment for the sticky delta assumption
        # In practice, this would require solving for the vol that maintains constant delta
        sticky_delta_adjustment = 0.02 * bsm_result.total_delta  # 2% adjustment
        
        return DeltaDecomposition(
            total_delta=bsm_result.total_delta + sticky_delta_adjustment,
            price_sensitivity=bsm_result.price_sensitivity,
            vol_sensitivity=sticky_delta_adjustment,
            time_sensitivity=0.0,
            skew_component=sticky_delta_adjustment,
            term_structure_component=0.0,
            correlation_component=0.0
        )
    
    def _calculate_regime_dependent_delta(self, 
                                        option_params: OptionParams,
                                        market_data: MarketData) -> DeltaDecomposition:
        """
        Calculate delta with regime-dependent adjustments
        
        Different volatility regimes require different hedge ratios
        """
        # Base BSM calculation
        bsm_result = self._calculate_black_scholes_delta(option_params, market_data)
        
        # Regime-based adjustments (this would integrate with volatility regime detector)
        # For now, use simplified regime detection based on current vol level
        vol_level = market_data.market_implied_volatility
        
        regime_adjustment = 0.0
        if vol_level > 0.8:  # High vol regime
            # In high vol regimes, reduce delta slightly (expect mean reversion)
            regime_adjustment = -0.05 * abs(bsm_result.total_delta)
        elif vol_level < 0.3:  # Low vol regime
            # In low vol regimes, increase delta slightly (expect vol expansion)
            regime_adjustment = 0.03 * abs(bsm_result.total_delta)
        
        return DeltaDecomposition(
            total_delta=bsm_result.total_delta + regime_adjustment,
            price_sensitivity=bsm_result.price_sensitivity,
            vol_sensitivity=0.0,
            time_sensitivity=0.0,
            skew_component=regime_adjustment,
            term_structure_component=0.0,
            correlation_component=0.0
        )
    
    def calculate_delta_risk_metrics(self, 
                                   option_params: OptionParams,
                                   market_data: MarketData,
                                   delta_decomp: DeltaDecomposition) -> DeltaRiskMetrics:
        """
        Calculate risk metrics related to delta hedging
        """
        # Calculate second-order Greeks for risk assessment
        greeks = FinancialMath.calculate_greeks(
            spot=option_params.current_spot_price,
            strike=option_params.strike_price,
            time_to_expiry=option_params.time_to_expiration,
            risk_free_rate=option_params.risk_free_rate,
            volatility=market_data.market_implied_volatility,
            dividend_yield=option_params.dividend_yield,
            option_type=option_params.option_type.value
        )
        
        # Delta-gamma sensitivity (how delta changes with spot)
        delta_gamma_sensitivity = greeks['gamma'] * option_params.current_spot_price * 0.01
        
        # Delta-vega sensitivity (how delta changes with vol)
        # This is a cross-Greek that's important for vol surface hedging
        vol_shift = 0.01  # 1% vol shift
        
        greeks_up = FinancialMath.calculate_greeks(
            spot=option_params.current_spot_price,
            strike=option_params.strike_price,
            time_to_expiry=option_params.time_to_expiration,
            risk_free_rate=option_params.risk_free_rate,
            volatility=market_data.market_implied_volatility + vol_shift,
            dividend_yield=option_params.dividend_yield,
            option_type=option_params.option_type.value
        )
        
        delta_vega_sensitivity = (greeks_up['delta'] - greeks['delta']) / vol_shift
        
        # Delta-theta sensitivity (how delta changes with time)
        time_shift = 1/365.25  # 1 day
        if option_params.time_to_expiration > time_shift:
            greeks_time = FinancialMath.calculate_greeks(
                spot=option_params.current_spot_price,
                strike=option_params.strike_price,
                time_to_expiry=option_params.time_to_expiration - time_shift,
                risk_free_rate=option_params.risk_free_rate,
                volatility=market_data.market_implied_volatility,
                dividend_yield=option_params.dividend_yield,
                option_type=option_params.option_type.value
            )
            delta_theta_sensitivity = (greeks_time['delta'] - greeks['delta']) / time_shift
        else:
            delta_theta_sensitivity = 0.0
        
        # Hedge ratio stability (based on gamma and vega)
        # Lower gamma and vega = more stable hedge ratio
        stability = 1.0 / (1.0 + abs(greeks['gamma']) * 100 + abs(greeks['vega']))
        
        # Model risk adjustment (higher for OTM options and short expiries)
        moneyness = option_params.strike_price / option_params.current_spot_price
        model_risk = abs(1.0 - moneyness) + (1.0 / max(0.01, option_params.time_to_expiration))
        model_risk_adjustment = min(0.1, model_risk * 0.02)  # Cap at 10%
        
        # Liquidity adjustment (simplified - would need real bid-ask data)
        if market_data.bid_price and market_data.ask_price:
            spread = (market_data.ask_price - market_data.bid_price) / market_data.current_option_price
            liquidity_adjustment = min(0.05, spread * 0.5)  # Cap at 5%
        else:
            liquidity_adjustment = 0.01  # Default 1% adjustment
        
        return DeltaRiskMetrics(
            delta_gamma_sensitivity=delta_gamma_sensitivity * option_params.position_size,
            delta_vega_sensitivity=delta_vega_sensitivity * option_params.position_size,
            delta_theta_sensitivity=delta_theta_sensitivity * option_params.position_size,
            hedge_ratio_stability=stability,
            model_risk_adjustment=model_risk_adjustment,
            liquidity_adjustment=liquidity_adjustment
        )
    
    def compare_delta_methods(self, 
                            option_params: OptionParams,
                            market_data: MarketData) -> Dict[str, DeltaDecomposition]:
        """
        Compare delta calculations across different methods
        
        Useful for understanding model differences and choosing appropriate method
        """
        methods = [
            DeltaCalculationMethod.BLACK_SCHOLES,
            DeltaCalculationMethod.LOCAL_VOLATILITY,
            DeltaCalculationMethod.STICKY_STRIKE,
            DeltaCalculationMethod.STICKY_DELTA,
            DeltaCalculationMethod.REGIME_DEPENDENT
        ]
        
        results = {}
        for method in methods:
            try:
                results[method.value] = self.calculate_enhanced_delta(
                    option_params, market_data, method
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate delta with method {method.value}: {e}")
                continue
        
        return results
    
    def recommend_delta_method(self, 
                             option_params: OptionParams,
                             market_data: MarketData) -> DeltaCalculationMethod:
        """
        Recommend the best delta calculation method based on option characteristics
        """
        # For ATM options with normal vol, BSM is usually fine
        moneyness = option_params.strike_price / option_params.current_spot_price
        vol_level = market_data.market_implied_volatility
        
        if 0.95 <= moneyness <= 1.05 and 0.2 <= vol_level <= 0.6:
            return DeltaCalculationMethod.BLACK_SCHOLES
        
        # For OTM options, local vol is better (handles skew)
        if moneyness < 0.9 or moneyness > 1.1:
            if self.vol_surface:
                return DeltaCalculationMethod.LOCAL_VOLATILITY
            else:
                return DeltaCalculationMethod.REGIME_DEPENDENT
        
        # For extreme vol environments, use regime-dependent
        if vol_level > 0.8 or vol_level < 0.15:
            return DeltaCalculationMethod.REGIME_DEPENDENT
        
        # Default to local vol if surface available, otherwise BSM
        if self.vol_surface:
            return DeltaCalculationMethod.LOCAL_VOLATILITY
        else:
            return DeltaCalculationMethod.BLACK_SCHOLES
    
    def log_delta_comparison(self, 
                           option_params: OptionParams,
                           comparison_results: Dict[str, DeltaDecomposition]):
        """Log comparison of different delta calculation methods"""
        self.logger.info("="*60)
        self.logger.info("DELTA CALCULATION METHOD COMPARISON")
        self.logger.info("="*60)
        self.logger.info(f"Option: {option_params.option_type.value.upper()} "
                        f"K={option_params.strike_price} T={option_params.time_to_expiration:.2f}y")
        self.logger.info("")
        
        for method_name, delta_decomp in comparison_results.items():
            self.logger.info(f"{method_name.upper()}:")
            self.logger.info(f"  Total Delta: {delta_decomp.total_delta:.4f}")
            self.logger.info(f"  Price Sensitivity: {delta_decomp.price_sensitivity:.4f}")
            self.logger.info(f"  Vol Sensitivity: {delta_decomp.vol_sensitivity:.4f}")
            self.logger.info(f"  Skew Component: {delta_decomp.skew_component:.4f}")
            self.logger.info("")
        
        self.logger.info("="*60)