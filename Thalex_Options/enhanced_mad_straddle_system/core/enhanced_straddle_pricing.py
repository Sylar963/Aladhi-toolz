#!/usr/bin/env python3
"""
Enhanced Straddle Pricing Model
===============================

This module provides dynamic straddle pricing that goes beyond Black-Scholes by incorporating:
- Forward volatility expectations (not constant vol)
- Volatility regime awareness
- Volatility smile/skew effects
- Time-varying correlation structures

Critical for identifying true straddle buying opportunities where vol expansion is expected.
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.stats import norm

from financial_math import FinancialMath, TradingConstants
from volatility_regime import VolatilityRegimeDetector, VolatilityRegime
from forward_volatility import ForwardVolatilityEstimator
from volatility_surface import VolatilitySurface


@dataclass
class EnhancedStraddlePricing:
    """Enhanced straddle pricing with multiple volatility models"""
    
    # Basic parameters
    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    
    # Black-Scholes baseline
    bs_call_price: float
    bs_put_price: float
    bs_straddle_price: float
    bs_implied_vol: float
    
    # Enhanced volatility models
    forward_vol_expected: float
    regime_adjusted_vol: float
    surface_implied_vol: float
    
    # Enhanced pricing
    forward_vol_straddle_price: float
    regime_adjusted_straddle_price: float
    surface_adjusted_straddle_price: float
    
    # Composite model (weighted combination)
    composite_expected_vol: float
    composite_straddle_price: float
    
    # Risk metrics
    vol_forecast_confidence: float
    pricing_confidence: float
    
    # Volatility analysis
    vol_expansion_probability: float
    expected_vol_change: float
    vol_regime_impact: float


@dataclass
class StraddleBuyingSignal:
    """Signal for straddle buying opportunities"""
    
    signal_strength: float  # 0-100 score
    recommendation: str     # "STRONG_BUY", "BUY", "HOLD", "AVOID"
    
    # Key factors
    vol_expansion_expected: bool
    current_vol_regime: VolatilityRegime
    price_vs_enhanced_fair_value: float
    
    # Supporting evidence
    forward_vol_evidence: str
    regime_evidence: str
    surface_evidence: str
    
    # Risk warnings
    risk_factors: List[str]
    confidence_level: float


class EnhancedStraddlePricingModel:
    """
    Enhanced straddle pricing model integrating multiple volatility approaches
    """
    
    def __init__(self, regime_detector: VolatilityRegimeDetector,
                 forward_vol_estimator: ForwardVolatilityEstimator,
                 volatility_surface: VolatilitySurface):
        
        self.regime_detector = regime_detector
        self.forward_vol_estimator = forward_vol_estimator
        self.volatility_surface = volatility_surface
        
        # Model weighting parameters (can be calibrated)
        self.forward_vol_weight = 0.5
        self.regime_weight = 0.3
        self.surface_weight = 0.2
        
        # Pricing confidence thresholds
        self.min_data_confidence = 0.6
        self.high_confidence_threshold = 0.8
        
        logging.info("Initialized EnhancedStraddlePricingModel")
    
    def price_straddle(self, spot_price: float, strike_price: float, 
                      time_to_expiry: float, market_straddle_price: float,
                      current_returns: List[float]) -> Optional[EnhancedStraddlePricing]:
        """
        Generate enhanced straddle pricing analysis
        
        Args:
            spot_price: Current underlying price
            strike_price: Straddle strike
            time_to_expiry: Time to expiry in years
            market_straddle_price: Current market price
            current_returns: Recent return history
            
        Returns:
            EnhancedStraddlePricing object or None if insufficient data
        """
        if time_to_expiry <= 0 or len(current_returns) < 50:
            logging.debug("Insufficient data for enhanced straddle pricing")
            return None
        
        risk_free_rate = TradingConstants.DEFAULT_RISK_FREE_RATE
        
        # 1. Calculate Black-Scholes baseline with current implied volatility
        bs_implied_vol = self._calculate_implied_vol_from_market_price(
            market_straddle_price, spot_price, strike_price, time_to_expiry, risk_free_rate
        )
        
        bs_call_price = FinancialMath.black_scholes_call(
            spot_price, strike_price, time_to_expiry, risk_free_rate, bs_implied_vol
        )
        bs_put_price = FinancialMath.black_scholes_put(
            spot_price, strike_price, time_to_expiry, risk_free_rate, bs_implied_vol
        )
        bs_straddle_price = bs_call_price + bs_put_price
        
        # 2. Get forward volatility forecast
        forward_vol_forecast = self.forward_vol_estimator.forecast_volatility(
            time_to_expiry * 365, current_returns
        )
        
        if not forward_vol_forecast:
            logging.warning("No forward volatility forecast available")
            return None
        
        forward_vol_expected = forward_vol_forecast.expected_volatility
        vol_forecast_confidence = forward_vol_forecast.forecast_confidence
        
        # 3. Get regime-adjusted volatility
        regime_metrics = self.regime_detector.analyze_regime()
        regime_adjusted_vol = bs_implied_vol  # Default fallback
        
        if regime_metrics:
            regime_vol_adjustment = self._get_regime_vol_adjustment(
                regime_metrics, time_to_expiry
            )
            regime_adjusted_vol = bs_implied_vol * (1 + regime_vol_adjustment)
        
        # 4. Get volatility surface implied volatility
        surface_implied_vol = self.volatility_surface.get_implied_volatility(
            strike_price, time_to_expiry
        )
        
        if surface_implied_vol is None:
            surface_implied_vol = bs_implied_vol  # Fallback
        
        # 5. Calculate enhanced straddle prices
        forward_vol_straddle_price = FinancialMath.black_scholes_straddle(
            spot_price, strike_price, time_to_expiry, risk_free_rate, forward_vol_expected
        )
        
        regime_adjusted_straddle_price = FinancialMath.black_scholes_straddle(
            spot_price, strike_price, time_to_expiry, risk_free_rate, regime_adjusted_vol
        )
        
        surface_adjusted_straddle_price = FinancialMath.black_scholes_straddle(
            spot_price, strike_price, time_to_expiry, risk_free_rate, surface_implied_vol
        )
        
        # 6. Create composite model (weighted average)
        composite_expected_vol = (
            forward_vol_expected * self.forward_vol_weight +
            regime_adjusted_vol * self.regime_weight +
            surface_implied_vol * self.surface_weight
        )
        
        composite_straddle_price = FinancialMath.black_scholes_straddle(
            spot_price, strike_price, time_to_expiry, risk_free_rate, composite_expected_vol
        )
        
        # 7. Calculate volatility analysis metrics
        expected_vol_change = (forward_vol_expected - bs_implied_vol) / bs_implied_vol
        vol_expansion_probability = self._calculate_vol_expansion_probability(
            forward_vol_forecast, regime_metrics
        )
        
        regime_impact = 0.0
        if regime_metrics:
            regime_impact = regime_metrics.vol_momentum * 0.1  # Scale factor
        
        # 8. Calculate overall pricing confidence
        pricing_confidence = self._calculate_pricing_confidence(
            vol_forecast_confidence, regime_metrics, len(current_returns)
        )
        
        return EnhancedStraddlePricing(
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            
            bs_call_price=bs_call_price,
            bs_put_price=bs_put_price,
            bs_straddle_price=bs_straddle_price,
            bs_implied_vol=bs_implied_vol,
            
            forward_vol_expected=forward_vol_expected,
            regime_adjusted_vol=regime_adjusted_vol,
            surface_implied_vol=surface_implied_vol,
            
            forward_vol_straddle_price=forward_vol_straddle_price,
            regime_adjusted_straddle_price=regime_adjusted_straddle_price,
            surface_adjusted_straddle_price=surface_adjusted_straddle_price,
            
            composite_expected_vol=composite_expected_vol,
            composite_straddle_price=composite_straddle_price,
            
            vol_forecast_confidence=vol_forecast_confidence,
            pricing_confidence=pricing_confidence,
            
            vol_expansion_probability=vol_expansion_probability,
            expected_vol_change=expected_vol_change,
            vol_regime_impact=regime_impact
        )
    
    def generate_straddle_buying_signal(self, enhanced_pricing: EnhancedStraddlePricing,
                                      market_price: float) -> StraddleBuyingSignal:
        """
        Generate straddle buying signal based on enhanced pricing analysis
        
        Args:
            enhanced_pricing: Enhanced pricing analysis
            market_price: Current market price of straddle
            
        Returns:
            StraddleBuyingSignal with recommendation
        """
        # Calculate key metrics
        composite_fair_value = enhanced_pricing.composite_straddle_price
        price_vs_fair_value = (market_price - composite_fair_value) / composite_fair_value
        
        vol_expansion_expected = enhanced_pricing.expected_vol_change > 0.05  # 5% threshold
        
        # Get regime context
        regime_metrics = self.regime_detector.analyze_regime()
        current_regime = regime_metrics.regime if regime_metrics else VolatilityRegime.UNKNOWN
        
        # Base signal calculation
        signal_components = []
        
        # 1. Volatility expansion component (40% of signal)
        if enhanced_pricing.vol_expansion_probability > 0.7:
            vol_signal = 40 * enhanced_pricing.vol_expansion_probability
        elif enhanced_pricing.vol_expansion_probability > 0.5:
            vol_signal = 20 + 20 * enhanced_pricing.vol_expansion_probability
        else:
            vol_signal = 0  # No vol expansion expected
        signal_components.append(("vol_expansion", vol_signal))
        
        # 2. Price attractiveness component (30% of signal)
        if price_vs_fair_value < -0.10:  # >10% underpriced
            price_signal = 30
        elif price_vs_fair_value < -0.05:  # 5-10% underpriced
            price_signal = 20
        elif price_vs_fair_value < 0.05:   # Fair value
            price_signal = 10
        else:  # Overpriced
            price_signal = 0
        signal_components.append(("price_attractiveness", price_signal))
        
        # 3. Regime favorability component (20% of signal)
        regime_favorability = self._get_regime_signal_component(current_regime)
        signal_components.append(("regime_favorability", regime_favorability))
        
        # 4. Confidence component (10% of signal)
        confidence_signal = enhanced_pricing.pricing_confidence * 10
        signal_components.append(("confidence", confidence_signal))
        
        # Calculate total signal
        total_signal = sum(component[1] for component in signal_components)
        
        # Generate recommendation
        if total_signal >= 80:
            recommendation = "STRONG_BUY"
        elif total_signal >= 60:
            recommendation = "BUY"
        elif total_signal >= 40:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID"
        
        # Generate supporting evidence
        forward_vol_evidence = self._generate_forward_vol_evidence(enhanced_pricing)
        regime_evidence = self._generate_regime_evidence(current_regime, regime_metrics)
        surface_evidence = self._generate_surface_evidence(enhanced_pricing)
        
        # Generate risk factors
        risk_factors = self._identify_risk_factors(enhanced_pricing, current_regime, price_vs_fair_value)
        
        return StraddleBuyingSignal(
            signal_strength=total_signal,
            recommendation=recommendation,
            vol_expansion_expected=vol_expansion_expected,
            current_vol_regime=current_regime,
            price_vs_enhanced_fair_value=price_vs_fair_value,
            forward_vol_evidence=forward_vol_evidence,
            regime_evidence=regime_evidence,
            surface_evidence=surface_evidence,
            risk_factors=risk_factors,
            confidence_level=enhanced_pricing.pricing_confidence
        )
    
    def _calculate_implied_vol_from_market_price(self, market_price: float, spot: float,
                                               strike: float, time_to_expiry: float,
                                               risk_free_rate: float) -> float:
        """Calculate implied volatility from market straddle price"""
        # Use Newton-Raphson method
        vol = 0.3  # Initial guess (30%)
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            bs_price = FinancialMath.black_scholes_straddle(
                spot, strike, time_to_expiry, risk_free_rate, vol
            )
            
            if abs(bs_price - market_price) < tolerance:
                break
            
            # Calculate vega for Newton-Raphson
            call_greeks = FinancialMath.calculate_greeks(
                spot, strike, time_to_expiry, risk_free_rate, vol, option_type="call"
            )
            put_greeks = FinancialMath.calculate_greeks(
                spot, strike, time_to_expiry, risk_free_rate, vol, option_type="put"
            )
            
            total_vega = (call_greeks['vega'] + put_greeks['vega']) * 100
            
            if abs(total_vega) < 1e-10:
                break
            
            # Newton-Raphson update
            vol_new = vol - (bs_price - market_price) / total_vega
            vol = max(0.01, min(vol_new, 5.0))  # Keep reasonable bounds
        
        return vol
    
    def _get_regime_vol_adjustment(self, regime_metrics, time_to_expiry: float) -> float:
        """Get volatility adjustment based on regime"""
        regime = regime_metrics.regime
        
        # Base adjustments by regime
        regime_adjustments = {
            VolatilityRegime.LOW: 0.20,      # Expect significant vol expansion
            VolatilityRegime.NORMAL: 0.0,    # No adjustment
            VolatilityRegime.HIGH: -0.15,    # Expect vol contraction
            VolatilityRegime.EXTREME: -0.25, # Strong mean reversion
            VolatilityRegime.CRISIS: -0.10   # May persist longer
        }
        
        base_adjustment = regime_adjustments.get(regime, 0.0)
        
        # Adjust for volatility momentum
        momentum_adjustment = regime_metrics.vol_momentum * 0.1
        
        # Time decay for regime effects
        time_decay = math.exp(-time_to_expiry * 4)  # Stronger for longer expiries
        
        return (base_adjustment + momentum_adjustment) * time_decay
    
    def _calculate_vol_expansion_probability(self, forward_vol_forecast, regime_metrics) -> float:
        """Calculate probability of volatility expansion"""
        if not forward_vol_forecast:
            return 0.5  # Default neutral
        
        # Base probability from forward vol forecast
        vol_change = forward_vol_forecast.expected_volatility / (forward_vol_forecast.current_vol_component or 0.3) - 1
        
        if vol_change > 0.20:
            base_prob = 0.8
        elif vol_change > 0.10:
            base_prob = 0.7
        elif vol_change > 0.0:
            base_prob = 0.6
        else:
            base_prob = 0.4
        
        # Adjust for regime
        if regime_metrics:
            if regime_metrics.regime == VolatilityRegime.LOW:
                base_prob += 0.1  # More likely in low vol regime
            elif regime_metrics.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                base_prob -= 0.1  # Less likely in high vol regime
        
        # Adjust for forecast confidence
        confidence_adjustment = (forward_vol_forecast.forecast_confidence - 0.5) * 0.2
        
        return max(0.1, min(0.9, base_prob + confidence_adjustment))
    
    def _calculate_pricing_confidence(self, vol_forecast_confidence: float,
                                    regime_metrics, sample_size: int) -> float:
        """Calculate overall pricing confidence"""
        # Base confidence from data quality
        if sample_size >= 200:
            base_confidence = 0.8
        elif sample_size >= 100:
            base_confidence = 0.7
        else:
            base_confidence = 0.6
        
        # Adjust for forward vol forecast confidence
        base_confidence = (base_confidence + vol_forecast_confidence) / 2
        
        # Adjust for regime detection confidence
        if regime_metrics:
            regime_confidence_boost = regime_metrics.confidence * 0.1
            base_confidence += regime_confidence_boost
        
        # Adjust for surface quality
        surface_metrics = self.volatility_surface.calculate_surface_quality_metrics()
        surface_confidence_boost = surface_metrics['surface_quality_score'] * 0.1
        base_confidence += surface_confidence_boost
        
        return max(0.3, min(0.95, base_confidence))
    
    def _get_regime_signal_component(self, regime: VolatilityRegime) -> float:
        """Get regime contribution to buy signal (0-20 points)"""
        regime_signals = {
            VolatilityRegime.LOW: 20,        # Very favorable for buying
            VolatilityRegime.NORMAL: 10,     # Neutral
            VolatilityRegime.HIGH: 5,        # Slightly unfavorable
            VolatilityRegime.EXTREME: 0,     # Unfavorable
            VolatilityRegime.CRISIS: 0,      # Very unfavorable
            VolatilityRegime.UNKNOWN: 5      # Neutral default
        }
        
        return regime_signals.get(regime, 5)
    
    def _generate_forward_vol_evidence(self, pricing: EnhancedStraddlePricing) -> str:
        """Generate forward volatility evidence string"""
        if pricing.expected_vol_change > 0.15:
            return f"Strong vol expansion expected (+{pricing.expected_vol_change:.1%})"
        elif pricing.expected_vol_change > 0.05:
            return f"Moderate vol expansion expected (+{pricing.expected_vol_change:.1%})"
        elif pricing.expected_vol_change > -0.05:
            return f"Vol stability expected ({pricing.expected_vol_change:+.1%})"
        else:
            return f"Vol contraction expected ({pricing.expected_vol_change:+.1%})"
    
    def _generate_regime_evidence(self, regime: VolatilityRegime, regime_metrics) -> str:
        """Generate regime evidence string"""
        regime_descriptions = {
            VolatilityRegime.LOW: "Low vol regime - expansion likely",
            VolatilityRegime.NORMAL: "Normal vol regime - stable conditions",
            VolatilityRegime.HIGH: "High vol regime - contraction risk",
            VolatilityRegime.EXTREME: "Extreme vol regime - mean reversion expected",
            VolatilityRegime.CRISIS: "Crisis vol regime - high uncertainty",
            VolatilityRegime.UNKNOWN: "Regime unclear - insufficient data"
        }
        
        base_desc = regime_descriptions.get(regime, "Unknown regime")
        
        if regime_metrics and regime_metrics.vol_momentum:
            momentum = regime_metrics.vol_momentum
            if momentum > 0.3:
                momentum_desc = " with strong upward momentum"
            elif momentum > 0.1:
                momentum_desc = " with moderate upward momentum"
            elif momentum < -0.3:
                momentum_desc = " with strong downward momentum"
            elif momentum < -0.1:
                momentum_desc = " with moderate downward momentum"
            else:
                momentum_desc = " with stable momentum"
            
            return base_desc + momentum_desc
        
        return base_desc
    
    def _generate_surface_evidence(self, pricing: EnhancedStraddlePricing) -> str:
        """Generate volatility surface evidence string"""
        surface_vol = pricing.surface_implied_vol
        bs_vol = pricing.bs_implied_vol
        
        vol_diff = (surface_vol - bs_vol) / bs_vol if bs_vol > 0 else 0
        
        if abs(vol_diff) < 0.02:
            return "Surface vol consistent with market"
        elif vol_diff > 0.05:
            return f"Surface suggests higher vol (+{vol_diff:.1%})"
        else:
            return f"Surface suggests lower vol ({vol_diff:+.1%})"
    
    def _identify_risk_factors(self, pricing: EnhancedStraddlePricing,
                             regime: VolatilityRegime, price_vs_fair: float) -> List[str]:
        """Identify key risk factors for straddle position"""
        risks = []
        
        # High volatility regime risks
        if regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME, VolatilityRegime.CRISIS]:
            risks.append("High vol regime - mean reversion risk")
        
        # Pricing risks
        if price_vs_fair > 0.1:
            risks.append("Straddle appears overpriced vs enhanced fair value")
        
        # Time decay risks
        if pricing.time_to_expiry < 0.1:  # < ~36 days
            risks.append("Short time to expiry - high theta decay")
        
        # Forecast confidence risks
        if pricing.vol_forecast_confidence < 0.6:
            risks.append("Low confidence in volatility forecast")
        
        # Volatility contraction risks
        if pricing.expected_vol_change < -0.1:
            risks.append("Significant vol contraction expected")
        
        # Model disagreement risks
        model_spread = max(pricing.forward_vol_straddle_price, pricing.regime_adjusted_straddle_price, 
                          pricing.surface_adjusted_straddle_price) - \
                      min(pricing.forward_vol_straddle_price, pricing.regime_adjusted_straddle_price,
                          pricing.surface_adjusted_straddle_price)
        
        if model_spread / pricing.composite_straddle_price > 0.15:
            risks.append("High disagreement between pricing models")
        
        return risks
    
    def log_enhanced_pricing_summary(self, pricing: EnhancedStraddlePricing, market_price: float):
        """Log comprehensive enhanced pricing summary"""
        logging.info("="*80)
        logging.info("ENHANCED STRADDLE PRICING ANALYSIS")
        logging.info("="*80)
        logging.info(f"Spot: ${pricing.spot_price:.0f}, Strike: ${pricing.strike_price:.0f}, "
                    f"Time: {pricing.time_to_expiry:.3f}y ({pricing.time_to_expiry*365:.0f}d)")
        logging.info(f"Market Price: ${market_price:.2f}")
        logging.info("")
        
        logging.info("VOLATILITY ANALYSIS:")
        logging.info(f"  Current Implied Vol: {pricing.bs_implied_vol:.1%}")
        logging.info(f"  Forward Vol Expected: {pricing.forward_vol_expected:.1%}")
        logging.info(f"  Regime Adjusted Vol: {pricing.regime_adjusted_vol:.1%}")
        logging.info(f"  Surface Implied Vol: {pricing.surface_implied_vol:.1%}")
        logging.info(f"  Composite Expected Vol: {pricing.composite_expected_vol:.1%}")
        logging.info(f"  Expected Vol Change: {pricing.expected_vol_change:+.1%}")
        logging.info("")
        
        logging.info("PRICING MODELS:")
        logging.info(f"  Black-Scholes Price: ${pricing.bs_straddle_price:.2f}")
        logging.info(f"  Forward Vol Price: ${pricing.forward_vol_straddle_price:.2f}")
        logging.info(f"  Regime Adjusted Price: ${pricing.regime_adjusted_straddle_price:.2f}")
        logging.info(f"  Surface Adjusted Price: ${pricing.surface_adjusted_straddle_price:.2f}")
        logging.info(f"  Composite Fair Value: ${pricing.composite_straddle_price:.2f}")
        logging.info("")
        
        price_vs_composite = (market_price - pricing.composite_straddle_price) / pricing.composite_straddle_price
        logging.info("MARKET ASSESSMENT:")
        logging.info(f"  Price vs Composite Fair Value: {price_vs_composite:+.1%}")
        logging.info(f"  Vol Expansion Probability: {pricing.vol_expansion_probability:.1%}")
        logging.info(f"  Pricing Confidence: {pricing.pricing_confidence:.1%}")
        
        logging.info("="*80)