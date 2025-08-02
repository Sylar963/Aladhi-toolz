#!/usr/bin/env python3
"""
Financial Mathematics Module
Provides accurate implementations of Black-Scholes and related financial formulas
"""

import math
from typing import Tuple, Optional
from scipy.stats import norm
import numpy as np

class TradingConstants:
    """Trading constants and configuration parameters"""
    TRADING_DAYS_PER_YEAR = 252
    HOURS_PER_TRADING_DAY = 24
    MINUTES_PER_HOUR = 60
    SECONDS_PER_MINUTE = 60
    
    # Statistical constants
    NORMAL_DISTRIBUTION_MAD_SD_RATIO = 0.7979  # Theoretical MAD/SD for normal distribution
    
    # Risk parameters
    DEFAULT_RISK_FREE_RATE = 0.05  # 5% annual risk-free rate
    
    # Confidence levels for statistical analysis
    CONFIDENCE_LEVELS = {
        "low": 0.5,
        "medium": 0.7, 
        "good": 0.85,
        "high": 0.95
    }
    
    # Minimum sample sizes by expiry characteristics
    MIN_SAMPLE_SIZES = {
        "very_short": 3,   # < 1 day
        "short": 5,        # 1-7 days
        "medium": 10,      # 7-30 days
        "long": 15         # > 30 days
    }
    
    # Volatility term structure adjustments
    TERM_STRUCTURE_ADJUSTMENTS = {
        "short": 1.25,     # Higher vol premium for gamma risk
        "medium": 1.0,     # Standard vol level
        "long": 0.95,      # Slight discount for mean reversion
        "very_long": 0.85  # Vol discount for long-term mean reversion
    }

class FinancialMath:
    """
    Accurate financial mathematics implementations
    """
    
    @staticmethod
    def black_scholes_call(spot: float, strike: float, time_to_expiry: float, 
                          risk_free_rate: float, volatility: float, 
                          dividend_yield: float = 0.0) -> float:
        """
        Calculate Black-Scholes call option price
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Implied volatility (annual)
            dividend_yield: Dividend yield (annual)
            
        Returns:
            Call option price
        """
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
            return 0.0
            
        # Calculate d1 and d2
        d1, d2 = FinancialMath._calculate_d1_d2(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
        )
        
        # Black-Scholes call formula
        call_price = (
            spot * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) -
            strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        )
        
        return max(call_price, 0.0)  # Ensure non-negative price
    
    @staticmethod
    def black_scholes_put(spot: float, strike: float, time_to_expiry: float,
                         risk_free_rate: float, volatility: float,
                         dividend_yield: float = 0.0) -> float:
        """
        Calculate Black-Scholes put option price
        
        Args:
            spot: Current underlying price
            strike: Strike price  
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Implied volatility (annual)
            dividend_yield: Dividend yield (annual)
            
        Returns:
            Put option price
        """
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
            return 0.0
            
        # Calculate d1 and d2
        d1, d2 = FinancialMath._calculate_d1_d2(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
        )
        
        # Black-Scholes put formula
        put_price = (
            strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
            spot * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
        )
        
        return max(put_price, 0.0)  # Ensure non-negative price
    
    @staticmethod
    def black_scholes_straddle(spot: float, strike: float, time_to_expiry: float,
                              risk_free_rate: float, volatility: float,
                              dividend_yield: float = 0.0) -> float:
        """
        Calculate Black-Scholes straddle price (call + put at same strike)
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Implied volatility (annual)
            dividend_yield: Dividend yield (annual)
            
        Returns:
            Straddle price (call + put)
        """
        call_price = FinancialMath.black_scholes_call(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
        )
        put_price = FinancialMath.black_scholes_put(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
        )
        
        return call_price + put_price
    
    @staticmethod
    def _calculate_d1_d2(spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float,
                        dividend_yield: float = 0.0) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula
        
        Returns:
            Tuple of (d1, d2)
        """
        vol_sqrt_t = volatility * math.sqrt(time_to_expiry)
        
        d1 = (
            math.log(spot / strike) + 
            (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
        ) / vol_sqrt_t
        
        d2 = d1 - vol_sqrt_t
        
        return d1, d2
    
    @staticmethod
    def calculate_greeks(spot: float, strike: float, time_to_expiry: float,
                        risk_free_rate: float, volatility: float,
                        dividend_yield: float = 0.0, option_type: str = "call") -> dict:
        """
        Calculate option Greeks (delta, gamma, theta, vega, rho)
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Implied volatility (annual)
            dividend_yield: Dividend yield (annual)
            option_type: "call" or "put"
            
        Returns:
            Dictionary containing Greeks
        """
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 
                'vega': 0.0, 'rho': 0.0
            }
        
        d1, d2 = FinancialMath._calculate_d1_d2(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
        )
        
        # Common terms
        sqrt_t = math.sqrt(time_to_expiry)
        exp_div_t = math.exp(-dividend_yield * time_to_expiry)
        exp_rf_t = math.exp(-risk_free_rate * time_to_expiry)
        phi_d1 = norm.pdf(d1)  # Standard normal PDF
        
        # Greeks calculations
        if option_type.lower() == "call":
            delta = exp_div_t * norm.cdf(d1)
            rho = strike * time_to_expiry * exp_rf_t * norm.cdf(d2)
        else:  # put
            delta = exp_div_t * (norm.cdf(d1) - 1)
            rho = -strike * time_to_expiry * exp_rf_t * norm.cdf(-d2)
        
        # Common Greeks (same for calls and puts)
        gamma = exp_div_t * phi_d1 / (spot * volatility * sqrt_t)
        
        theta = (
            -exp_div_t * spot * phi_d1 * volatility / (2 * sqrt_t) -
            risk_free_rate * strike * exp_rf_t * norm.cdf(d2 if option_type.lower() == "call" else -d2) +
            dividend_yield * spot * exp_div_t * norm.cdf(d1 if option_type.lower() == "call" else -d1)
        ) / TradingConstants.TRADING_DAYS_PER_YEAR  # Convert to daily theta
        
        vega = spot * exp_div_t * phi_d1 * sqrt_t / 100  # Per 1% vol change
        
        return {
            'delta': delta,
            'gamma': gamma, 
            'theta': theta,
            'vega': vega,
            'rho': rho / 100  # Per 1% rate change
        }
    
    @staticmethod
    def annualize_volatility(returns: list, frequency: str = "daily") -> float:
        """
        Calculate annualized volatility from returns
        
        Args:
            returns: List of periodic returns
            frequency: "daily", "hourly", "minute" etc.
            
        Returns:
            Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
            
        # Calculate standard deviation of returns
        std_dev = np.std(returns, ddof=1)  # Sample standard deviation
        
        # Annualization factors
        factors = {
            "daily": math.sqrt(TradingConstants.TRADING_DAYS_PER_YEAR),
            "hourly": math.sqrt(TradingConstants.TRADING_DAYS_PER_YEAR * TradingConstants.HOURS_PER_TRADING_DAY),
            "minute": math.sqrt(TradingConstants.TRADING_DAYS_PER_YEAR * TradingConstants.HOURS_PER_TRADING_DAY * TradingConstants.MINUTES_PER_HOUR)
        }
        
        annualization_factor = factors.get(frequency, math.sqrt(TradingConstants.TRADING_DAYS_PER_YEAR))
        
        return std_dev * annualization_factor
    
    @staticmethod
    def calculate_log_returns(prices: list) -> list:
        """
        Calculate logarithmic returns from price series
        
        Args:
            prices: List of prices
            
        Returns:
            List of log returns
        """
        if len(prices) < 2:
            return []
            
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                returns.append(math.log(prices[i] / prices[i-1]))
                
        return returns
    
    @staticmethod
    def implied_volatility_newton_raphson(market_price: float, spot: float, strike: float,
                                        time_to_expiry: float, risk_free_rate: float,
                                        option_type: str = "call", dividend_yield: float = 0.0,
                                        max_iterations: int = 100, tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Observed market price
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            option_type: "call" or "put"
            dividend_yield: Dividend yield
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if convergence fails
        """
        # Initial guess
        vol = 0.3  # 30% initial volatility guess
        
        for i in range(max_iterations):
            if option_type.lower() == "call":
                price = FinancialMath.black_scholes_call(
                    spot, strike, time_to_expiry, risk_free_rate, vol, dividend_yield
                )
            else:
                price = FinancialMath.black_scholes_put(
                    spot, strike, time_to_expiry, risk_free_rate, vol, dividend_yield
                )
            
            # Calculate vega (derivative of price w.r.t. volatility)
            greeks = FinancialMath.calculate_greeks(
                spot, strike, time_to_expiry, risk_free_rate, vol, dividend_yield, option_type
            )
            vega = greeks['vega'] * 100  # Convert back from per 1% to per 1
            
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            # Newton-Raphson update
            price_diff = price - market_price
            vol_new = vol - price_diff / vega
            
            # Ensure volatility stays positive
            vol_new = max(vol_new, 0.001)  # Minimum 0.1% volatility
            
            if abs(vol_new - vol) < tolerance:
                return vol_new
                
            vol = vol_new
            
        return None  # Convergence failed