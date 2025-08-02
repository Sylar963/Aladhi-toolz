#!/usr/bin/env python3
"""
Forward Volatility Estimation Module
====================================

This module estimates forward-looking volatility over option lifetimes using:
- GARCH models for volatility clustering and mean reversion
- Regime-aware adjustments based on current market conditions
- Time-varying volatility expectations (not constant like Black-Scholes)

Critical for straddle buying decisions where you need volatility to EXPAND.
"""

import math
import statistics
import time
from typing import Dict, List, Optional, Tuple, NamedTuple
from collections import deque
from dataclasses import dataclass
import logging

import numpy as np
from scipy.optimize import minimize
from scipy import stats

from financial_math import FinancialMath, TradingConstants
from volatility_regime import VolatilityRegime, VolatilityRegimeDetector


@dataclass
class GARCHParameters:
    """GARCH(1,1) model parameters"""
    omega: float  # Constant term
    alpha: float  # ARCH parameter (sensitivity to recent shocks)
    beta: float   # GARCH parameter (persistence)
    
    # Model diagnostics
    log_likelihood: float
    aic: float
    convergence: bool
    
    def is_stationary(self) -> bool:
        """Check if GARCH process is stationary"""
        return (self.alpha + self.beta) < 1.0
    
    def persistence(self) -> float:
        """Calculate volatility persistence"""
        return self.alpha + self.beta
    
    def long_run_variance(self) -> float:
        """Calculate long-run unconditional variance"""
        if self.is_stationary():
            return self.omega / (1 - self.alpha - self.beta)
        return float('inf')


@dataclass
class VolatilityForecast:
    """Forward volatility forecast for specific time horizon"""
    time_horizon_days: float
    expected_volatility: float
    volatility_confidence_interval: Tuple[float, float]
    
    # Decomposition of forecast
    current_vol_component: float    # Current vol persistence
    mean_reversion_component: float # Pull toward long-run mean
    regime_adjustment: float        # Adjustment for current regime
    
    # Risk metrics
    vol_of_vol: float              # Uncertainty in vol forecast
    skew_adjustment: float         # Adjustment for vol asymmetry
    forecast_confidence: float     # Overall forecast confidence


class ForwardVolatilityEstimator:
    """
    Estimates forward-looking volatility using GARCH models and regime information.
    
    Key differences from Black-Scholes:
    - Volatility is NOT constant over option life
    - Accounts for volatility clustering and mean reversion
    - Adjusts for current volatility regime
    - Provides confidence intervals around forecasts
    """
    
    def __init__(self, regime_detector: Optional[VolatilityRegimeDetector] = None):
        self.regime_detector = regime_detector
        
        # GARCH model state
        self.garch_params: Optional[GARCHParameters] = None
        self.conditional_variance_history = deque(maxlen=500)
        self.last_garch_fit_time = 0.0
        self.refit_interval_hours = 24  # Refit GARCH daily
        
        # Volatility forecasting history
        self.forecast_history = deque(maxlen=100)
        self.realized_vol_history = deque(maxlen=100)
        
        # Model calibration parameters
        self.min_observations = 100
        self.max_observations = 500
        
        logging.info("Initialized ForwardVolatilityEstimator")
    
    def fit_garch_model(self, returns: List[float]) -> Optional[GARCHParameters]:
        """
        Fit GARCH(1,1) model to return series
        
        Args:
            returns: List of log returns
            
        Returns:
            GARCHParameters object or None if fitting fails
        """
        if len(returns) < self.min_observations:
            logging.warning(f"Insufficient data for GARCH fitting: {len(returns)} < {self.min_observations}")
            return None
        
        # Use most recent data (but not too much for crypto markets)
        recent_returns = returns[-min(len(returns), self.max_observations):]
        
        try:
            # Initial parameter guesses
            initial_params = [
                0.00001,  # omega (small constant)
                0.1,      # alpha (moderate ARCH effect)
                0.8       # beta (high persistence)
            ]
            
            # Parameter bounds
            bounds = [
                (1e-8, 1.0),    # omega > 0
                (1e-6, 0.99),   # 0 < alpha < 1
                (1e-6, 0.99)    # 0 < beta < 1
            ]
            
            # Constraint: alpha + beta < 1 (stationarity)
            constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - (x[1] + x[2])}
            
            # Maximize log-likelihood
            result = minimize(
                self._garch_negative_log_likelihood,
                initial_params,
                args=(recent_returns,),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                logging.warning(f"GARCH optimization failed: {result.message}")
                return self._fallback_garch_params()
            
            omega, alpha, beta = result.x
            log_likelihood = -result.fun
            
            # Calculate AIC
            aic = 2 * 3 - 2 * log_likelihood  # 3 parameters
            
            params = GARCHParameters(
                omega=omega,
                alpha=alpha,
                beta=beta,
                log_likelihood=log_likelihood,
                aic=aic,
                convergence=result.success
            )
            
            logging.info(f"GARCH(1,1) fitted: ω={omega:.6f}, α={alpha:.3f}, β={beta:.3f}, persist={params.persistence():.3f}")
            
            # Update conditional variance history
            self._update_conditional_variances(recent_returns, params)
            
            return params
            
        except Exception as e:
            logging.error(f"GARCH fitting error: {e}")
            return self._fallback_garch_params()
    
    def _garch_negative_log_likelihood(self, params: List[float], returns: List[float]) -> float:
        """GARCH(1,1) negative log-likelihood function"""
        omega, alpha, beta = params
        
        # Initialize conditional variance
        unconditional_var = np.var(returns)
        conditional_variances = [unconditional_var]
        
        log_likelihood = 0.0
        
        for i, ret in enumerate(returns):
            if i == 0:
                h_t = unconditional_var
            else:
                h_t = omega + alpha * (returns[i-1] ** 2) + beta * conditional_variances[i-1]
            
            conditional_variances.append(h_t)
            
            # Avoid numerical issues
            h_t = max(h_t, 1e-8)
            
            # Log-likelihood contribution
            log_likelihood += -0.5 * (math.log(2 * math.pi) + math.log(h_t) + (ret ** 2) / h_t)
        
        return -log_likelihood
    
    def _fallback_garch_params(self) -> GARCHParameters:
        """Fallback GARCH parameters for crypto markets"""
        return GARCHParameters(
            omega=0.00001,
            alpha=0.15,      # Higher alpha for crypto volatility clustering
            beta=0.80,       # High persistence
            log_likelihood=0.0,
            aic=float('inf'),
            convergence=False
        )
    
    def _update_conditional_variances(self, returns: List[float], params: GARCHParameters):
        """Update conditional variance history with new parameters"""
        if len(returns) < 2:
            return
        
        # Initialize with unconditional variance
        unconditional_var = np.var(returns)
        current_var = unconditional_var
        
        # Calculate conditional variances
        for ret in returns[-50:]:  # Keep last 50 for efficiency
            current_var = params.omega + params.alpha * (ret ** 2) + params.beta * current_var
            self.conditional_variance_history.append(current_var)
    
    def update_garch_model(self, returns: List[float]) -> bool:
        """
        Update GARCH model if enough time has passed or significant regime change
        
        Args:
            returns: Current return series
            
        Returns:
            True if model was updated, False otherwise
        """
        current_time = time.time()
        hours_since_last_fit = (current_time - self.last_garch_fit_time) / 3600
        
        # Check if we need to refit
        should_refit = (
            self.garch_params is None or
            hours_since_last_fit >= self.refit_interval_hours or
            len(returns) < self.min_observations
        )
        
        # Also refit if regime changed significantly
        if self.regime_detector and hasattr(self.regime_detector, 'regime_start_time'):
            if self.regime_detector.regime_start_time:
                hours_since_regime_change = (current_time - self.regime_detector.regime_start_time) / 3600
                if hours_since_regime_change < 2:  # Recent regime change
                    should_refit = True
        
        if should_refit:
            new_params = self.fit_garch_model(returns)
            if new_params:
                self.garch_params = new_params
                self.last_garch_fit_time = current_time
                return True
        
        return False
    
    def forecast_volatility(self, time_horizon_days: float, 
                          current_returns: List[float]) -> Optional[VolatilityForecast]:
        """
        Forecast volatility over specific time horizon
        
        Args:
            time_horizon_days: Forecast horizon in days
            current_returns: Recent return series for model fitting
            
        Returns:
            VolatilityForecast object or None if insufficient data
        """
        if len(current_returns) < self.min_observations:
            logging.debug(f"Insufficient data for forecasting: {len(current_returns)}")
            return None
        
        # Update GARCH model if needed
        self.update_garch_model(current_returns)
        
        if not self.garch_params:
            logging.warning("No GARCH parameters available for forecasting")
            return None
        
        # Get current conditional variance
        current_conditional_var = self._get_current_conditional_variance(current_returns)
        
        # GARCH volatility forecast
        garch_forecast = self._garch_volatility_forecast(
            current_conditional_var, time_horizon_days
        )
        
        # Get regime adjustment
        regime_adjustment = self._get_regime_adjustment(time_horizon_days)
        
        # Calculate components
        long_run_var = self.garch_params.long_run_variance()
        persistence = self.garch_params.persistence()
        
        # Current vol component (decays over time)
        decay_factor = persistence ** time_horizon_days
        current_vol_component = math.sqrt(current_conditional_var) * decay_factor
        
        # Mean reversion component (grows over time)
        mean_reversion_component = math.sqrt(long_run_var) * (1 - decay_factor)
        
        # Combined base forecast
        base_forecast = math.sqrt(
            current_conditional_var * decay_factor + 
            long_run_var * (1 - decay_factor)
        )
        
        # Apply regime adjustment
        adjusted_forecast = base_forecast * (1 + regime_adjustment)
        
        # Calculate forecast uncertainty (vol of vol)
        vol_of_vol = self._estimate_vol_of_vol(current_returns, time_horizon_days)
        
        # Confidence intervals (assume normal distribution of log vol)
        log_vol = math.log(adjusted_forecast)
        vol_std = vol_of_vol / adjusted_forecast  # Convert to relative terms
        
        ci_lower = math.exp(log_vol - 1.96 * vol_std)
        ci_upper = math.exp(log_vol + 1.96 * vol_std)
        
        # Forecast confidence based on data quality and horizon
        forecast_confidence = self._calculate_forecast_confidence(
            len(current_returns), time_horizon_days
        )
        
        return VolatilityForecast(
            time_horizon_days=time_horizon_days,
            expected_volatility=adjusted_forecast,
            volatility_confidence_interval=(ci_lower, ci_upper),
            current_vol_component=current_vol_component,
            mean_reversion_component=mean_reversion_component,
            regime_adjustment=regime_adjustment,
            vol_of_vol=vol_of_vol,
            skew_adjustment=0.0,  # Can be enhanced later
            forecast_confidence=forecast_confidence
        )
    
    def _get_current_conditional_variance(self, returns: List[float]) -> float:
        """Get current conditional variance from GARCH model"""
        if not self.garch_params or len(returns) == 0:
            return np.var(returns) if len(returns) > 1 else 0.0001
        
        # Use most recent conditional variance or calculate from last return
        if len(self.conditional_variance_history) > 0:
            last_var = self.conditional_variance_history[-1]
            last_return = returns[-1]
            
            # Update with most recent return
            current_var = (
                self.garch_params.omega + 
                self.garch_params.alpha * (last_return ** 2) + 
                self.garch_params.beta * last_var
            )
        else:
            current_var = self.garch_params.long_run_variance()
        
        return max(current_var, 1e-8)  # Avoid numerical issues
    
    def _garch_volatility_forecast(self, current_var: float, horizon_days: float) -> float:
        """Pure GARCH volatility forecast without regime adjustments"""
        if not self.garch_params:
            return math.sqrt(current_var * 252)  # Annualized fallback
        
        persistence = self.garch_params.persistence()
        long_run_var = self.garch_params.long_run_variance()
        
        # Multi-step ahead variance forecast
        # Var(t+h) = long_run_var + persistence^h * (current_var - long_run_var)
        forecast_var = long_run_var + (persistence ** horizon_days) * (current_var - long_run_var)
        
        # Annualize
        return math.sqrt(forecast_var * 252)
    
    def _get_regime_adjustment(self, horizon_days: float) -> float:
        """
        Get volatility adjustment based on current regime
        
        Args:
            horizon_days: Forecast horizon
            
        Returns:
            Adjustment factor (e.g., 0.1 = 10% increase, -0.1 = 10% decrease)
        """
        if not self.regime_detector:
            return 0.0
        
        regime_metrics = self.regime_detector.analyze_regime()
        if not regime_metrics:
            return 0.0
        
        regime = regime_metrics.regime
        vol_momentum = regime_metrics.vol_momentum
        
        # Base adjustments by regime
        regime_adjustments = {
            VolatilityRegime.LOW: 0.15,      # Expect vol expansion
            VolatilityRegime.NORMAL: 0.0,    # No adjustment
            VolatilityRegime.HIGH: -0.10,    # Expect some mean reversion
            VolatilityRegime.EXTREME: -0.20, # Strong mean reversion expected
            VolatilityRegime.CRISIS: -0.15   # Crisis vol may persist longer
        }
        
        base_adjustment = regime_adjustments.get(regime, 0.0)
        
        # Adjust based on volatility momentum
        momentum_adjustment = vol_momentum * 0.05  # Scale momentum effect
        
        # Time decay of regime effects (stronger for shorter horizons)
        time_decay = math.exp(-horizon_days / 30)  # 30-day half-life
        
        total_adjustment = (base_adjustment + momentum_adjustment) * time_decay
        
        # Cap adjustments
        return max(-0.3, min(0.3, total_adjustment))
    
    def _estimate_vol_of_vol(self, returns: List[float], horizon_days: float) -> float:
        """Estimate uncertainty in volatility forecast (volatility of volatility)"""
        if len(returns) < 50:
            return 0.1  # Default 10% vol of vol
        
        # Calculate rolling volatilities
        window_size = min(20, len(returns) // 5)
        rolling_vols = []
        
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            if len(window_returns) > 1:
                vol = statistics.stdev(window_returns) * math.sqrt(252)
                rolling_vols.append(vol)
        
        if len(rolling_vols) < 5:
            return 0.1
        
        # Vol of vol is standard deviation of rolling volatilities
        vol_of_vol = statistics.stdev(rolling_vols)
        
        # Scale by forecast horizon (longer horizons = more uncertainty)
        horizon_scaling = math.sqrt(horizon_days / 30)  # Scale relative to 30 days
        
        return vol_of_vol * horizon_scaling
    
    def _calculate_forecast_confidence(self, sample_size: int, horizon_days: float) -> float:
        """Calculate overall confidence in volatility forecast"""
        # Base confidence from sample size
        if sample_size >= 200:
            base_confidence = 0.85
        elif sample_size >= 100:
            base_confidence = 0.75
        elif sample_size >= 50:
            base_confidence = 0.65
        else:
            base_confidence = 0.50
        
        # Reduce confidence for longer horizons
        horizon_penalty = min(0.3, horizon_days / 90)  # Max 30% penalty for 90+ day forecasts
        
        # GARCH model quality adjustment
        model_quality = 0.0
        if self.garch_params and self.garch_params.convergence:
            if self.garch_params.is_stationary():
                model_quality = 0.1
        
        final_confidence = base_confidence - horizon_penalty + model_quality
        return max(0.3, min(0.95, final_confidence))
    
    def get_straddle_volatility_assessment(self, time_to_expiry_days: float, 
                                         current_returns: List[float]) -> Dict[str, any]:
        """
        Get volatility assessment specifically for straddle buying decisions
        
        Args:
            time_to_expiry_days: Days until option expiry
            current_returns: Recent return history
            
        Returns:
            Dictionary with straddle-specific volatility insights
        """
        forecast = self.forecast_volatility(time_to_expiry_days, current_returns)
        
        if not forecast:
            return {
                "status": "insufficient_data",
                "recommendation": "Need more price history for volatility forecasting"
            }
        
        # Current vs expected volatility
        current_vol = math.sqrt(self._get_current_conditional_variance(current_returns) * 252)
        expected_vol = forecast.expected_volatility
        vol_change_expected = (expected_vol - current_vol) / current_vol
        
        # Volatility regime context
        regime_context = ""
        if self.regime_detector:
            implications = self.regime_detector.get_regime_implications_for_straddles()
            regime_context = implications.get("implication", "")
        
        # Assessment
        if vol_change_expected > 0.15:  # Expect >15% vol increase
            status = "favorable_for_buying"
            recommendation = "Volatility expected to increase - favorable for straddle buying"
        elif vol_change_expected > 0.05:  # Expect 5-15% vol increase
            status = "moderately_favorable"
            recommendation = "Modest volatility increase expected - consider straddle buying"
        elif vol_change_expected > -0.05:  # Expect -5% to +5% vol change
            status = "neutral"
            recommendation = "Little volatility change expected - focus on relative pricing"
        elif vol_change_expected > -0.15:  # Expect 5-15% vol decrease
            status = "unfavorable"
            recommendation = "Volatility expected to decrease - risky for straddle buying"
        else:  # Expect >15% vol decrease
            status = "very_unfavorable"
            recommendation = "Significant volatility decrease expected - avoid straddle buying"
        
        return {
            "status": status,
            "recommendation": recommendation,
            "current_volatility": current_vol,
            "expected_volatility": expected_vol,
            "volatility_change_expected": vol_change_expected,
            "confidence_interval": forecast.volatility_confidence_interval,
            "forecast_confidence": forecast.forecast_confidence,
            "time_horizon_days": time_to_expiry_days,
            "regime_context": regime_context,
            "vol_of_vol": forecast.vol_of_vol,
            
            # Detailed breakdown
            "forecast_components": {
                "current_vol_component": forecast.current_vol_component,
                "mean_reversion_component": forecast.mean_reversion_component,
                "regime_adjustment": forecast.regime_adjustment
            }
        }
    
    def log_forecast_summary(self, forecast: VolatilityForecast):
        """Log detailed forecast summary"""
        logging.info("="*60)
        logging.info("FORWARD VOLATILITY FORECAST")
        logging.info("="*60)
        logging.info(f"Time Horizon: {forecast.time_horizon_days:.1f} days")
        logging.info(f"Expected Volatility: {forecast.expected_volatility:.1%}")
        logging.info(f"Confidence Interval: {forecast.volatility_confidence_interval[0]:.1%} - {forecast.volatility_confidence_interval[1]:.1%}")
        logging.info(f"Forecast Confidence: {forecast.forecast_confidence:.1%}")
        logging.info("")
        logging.info("FORECAST COMPONENTS:")
        logging.info(f"  Current Vol Component: {forecast.current_vol_component:.1%}")
        logging.info(f"  Mean Reversion Component: {forecast.mean_reversion_component:.1%}")
        logging.info(f"  Regime Adjustment: {forecast.regime_adjustment:+.1%}")
        logging.info(f"  Vol of Vol: {forecast.vol_of_vol:.1%}")
        logging.info("="*60)