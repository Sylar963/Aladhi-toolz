#!/usr/bin/env python3
"""
Volatility Surface Module
=========================

This module builds and maintains a dynamic volatility surface across strikes and expiries.
Captures volatility smile/skew effects that Black-Scholes ignores.

Key features:
- Real-time volatility surface construction from market prices
- Volatility smile/skew modeling
- Strike and time interpolation for missing data points
- Surface quality metrics and validation
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from financial_math import FinancialMath


@dataclass
class VolatilityPoint:
    """Single point on the volatility surface"""
    strike: float
    time_to_expiry: float
    implied_volatility: float
    bid_vol: Optional[float]
    ask_vol: Optional[float]
    timestamp: float
    confidence: float  # Quality of the vol estimate
    
    def moneyness(self, spot_price: float) -> float:
        """Calculate moneyness (K/S)"""
        return self.strike / spot_price if spot_price > 0 else 1.0
    
    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if data point is stale"""
        return time.time() - self.timestamp > max_age_seconds


@dataclass
class VolatilitySlice:
    """Volatility slice for a specific expiry"""
    expiry_timestamp: float
    time_to_expiry: float
    vol_points: List[VolatilityPoint]
    atm_vol: Optional[float]
    vol_smile_skew: float  # Negative = put skew, Positive = call skew
    
    def get_strikes(self) -> List[float]:
        """Get all strikes in this slice"""
        return [point.strike for point in self.vol_points]
    
    def get_vol_by_strike(self, strike: float, spot_price: float) -> Optional[float]:
        """Get volatility for specific strike (with interpolation)"""
        if not self.vol_points:
            return None
        
        # Find exact match first
        for point in self.vol_points:
            if abs(point.strike - strike) < 0.01:
                return point.implied_volatility
        
        # Interpolate if no exact match
        return self._interpolate_vol_by_strike(strike, spot_price)
    
    def _interpolate_vol_by_strike(self, target_strike: float, spot_price: float) -> Optional[float]:
        """Interpolate volatility for given strike"""
        if len(self.vol_points) < 2:
            return None
        
        # Sort by strike
        sorted_points = sorted(self.vol_points, key=lambda p: p.strike)
        strikes = [p.strike for p in sorted_points]
        vols = [p.implied_volatility for p in sorted_points]
        
        # Use linear interpolation (can be enhanced with splines)
        try:
            interpolated_vol = np.interp(target_strike, strikes, vols)
            return float(interpolated_vol)
        except:
            return None


class VolatilitySurface:
    """
    Dynamic volatility surface with real-time updates and smile/skew modeling
    """
    
    def __init__(self, spot_price: float):
        self.spot_price = spot_price
        self.last_update_time = 0.0
        
        # Surface data organized by expiry
        self.vol_slices: Dict[float, VolatilitySlice] = {}  # expiry_timestamp -> slice
        
        # Raw market data points
        self.market_vol_points = deque(maxlen=1000)
        
        # Surface interpolation cache
        self._interpolation_cache = {}
        self._cache_timestamp = 0.0
        self.cache_ttl_seconds = 30
        
        # Surface quality metrics
        self.surface_quality_score = 0.0
        self.coverage_score = 0.0
        self.staleness_score = 0.0
        
        logging.info(f"Initialized VolatilitySurface with spot=${spot_price:.0f}")
    
    def update_spot_price(self, new_spot_price: float):
        """Update underlying spot price"""
        if new_spot_price > 0:
            self.spot_price = new_spot_price
            # Clear cache since moneyness calculations will change
            self._interpolation_cache = {}
    
    def add_market_vol_point(self, strike: float, expiry_timestamp: float, 
                           implied_vol: float, bid_vol: Optional[float] = None,
                           ask_vol: Optional[float] = None, confidence: float = 1.0):
        """
        Add a market volatility observation
        
        Args:
            strike: Option strike price
            expiry_timestamp: Expiry timestamp
            implied_vol: Implied volatility
            bid_vol: Bid volatility (optional)
            ask_vol: Ask volatility (optional)  
            confidence: Quality score (0-1)
        """
        current_time = time.time()
        time_to_expiry = (expiry_timestamp - current_time) / (365.25 * 24 * 3600)
        
        if time_to_expiry <= 0:
            logging.debug(f"Skipping expired option: strike={strike}")
            return
        
        if implied_vol <= 0 or implied_vol > 10.0:  # Sanity checks
            logging.warning(f"Invalid implied vol: {implied_vol} for strike {strike}")
            return
        
        vol_point = VolatilityPoint(
            strike=strike,
            time_to_expiry=time_to_expiry,
            implied_volatility=implied_vol,
            bid_vol=bid_vol,
            ask_vol=ask_vol,
            timestamp=current_time,
            confidence=confidence
        )
        
        self.market_vol_points.append(vol_point)
        self._update_vol_slice(expiry_timestamp, vol_point)
        self.last_update_time = current_time
        
        # Clear interpolation cache
        self._interpolation_cache = {}
    
    def _update_vol_slice(self, expiry_timestamp: float, vol_point: VolatilityPoint):
        """Update volatility slice for specific expiry"""
        if expiry_timestamp not in self.vol_slices:
            self.vol_slices[expiry_timestamp] = VolatilitySlice(
                expiry_timestamp=expiry_timestamp,
                time_to_expiry=vol_point.time_to_expiry,
                vol_points=[],
                atm_vol=None,
                vol_smile_skew=0.0
            )
        
        vol_slice = self.vol_slices[expiry_timestamp]
        
        # Update or add vol point
        existing_point_index = None
        for i, existing_point in enumerate(vol_slice.vol_points):
            if abs(existing_point.strike - vol_point.strike) < 0.01:
                existing_point_index = i
                break
        
        if existing_point_index is not None:
            # Update existing point
            vol_slice.vol_points[existing_point_index] = vol_point
        else:
            # Add new point
            vol_slice.vol_points.append(vol_point)
        
        # Update slice metrics
        self._update_slice_metrics(vol_slice)
    
    def _update_slice_metrics(self, vol_slice: VolatilitySlice):
        """Update ATM vol and skew for a volatility slice"""
        if not vol_slice.vol_points:
            return
        
        # Find ATM volatility (closest to current spot)
        closest_point = min(vol_slice.vol_points, 
                          key=lambda p: abs(p.strike - self.spot_price))
        vol_slice.atm_vol = closest_point.implied_volatility
        
        # Calculate vol smile/skew if we have enough points
        if len(vol_slice.vol_points) >= 3:
            vol_slice.vol_smile_skew = self._calculate_vol_skew(vol_slice.vol_points)
    
    def _calculate_vol_skew(self, vol_points: List[VolatilityPoint]) -> float:
        """
        Calculate volatility skew (slope of vol smile)
        
        Returns:
            Skew measure: negative = put skew, positive = call skew
        """
        if len(vol_points) < 3:
            return 0.0
        
        # Sort by moneyness
        sorted_points = sorted(vol_points, key=lambda p: p.moneyness(self.spot_price))
        
        # Split into OTM puts and calls relative to ATM
        atm_moneyness = 1.0
        otm_puts = [p for p in sorted_points if p.moneyness(self.spot_price) < 0.95]
        otm_calls = [p for p in sorted_points if p.moneyness(self.spot_price) > 1.05]
        
        if not otm_puts or not otm_calls:
            return 0.0
        
        # Calculate average vol for OTM puts and calls
        import statistics
        put_vol = statistics.mean([p.implied_volatility for p in otm_puts])
        call_vol = statistics.mean([p.implied_volatility for p in otm_calls])
        
        # Skew = call vol - put Vol (negative means put skew)
        return call_vol - put_vol
    
    def get_implied_volatility(self, strike: float, time_to_expiry: float) -> Optional[float]:
        """
        Get implied volatility for specific strike and time to expiry
        
        Args:
            strike: Option strike
            time_to_expiry: Time to expiry in years
            
        Returns:
            Implied volatility or None if not available
        """
        cache_key = (strike, time_to_expiry)
        
        # Check cache first
        if (cache_key in self._interpolation_cache and 
            time.time() - self._cache_timestamp < self.cache_ttl_seconds):
            return self._interpolation_cache[cache_key]
        
        # Find closest expiry slice
        best_slice = self._find_closest_expiry_slice(time_to_expiry)
        if not best_slice:
            return None
        
        # Get vol from slice
        vol = best_slice.get_vol_by_strike(strike, self.spot_price)
        
        # If we don't have exact expiry, interpolate between expiries
        if vol is not None and abs(best_slice.time_to_expiry - time_to_expiry) > 0.01:
            vol = self._interpolate_across_expiries(strike, time_to_expiry)
        
        # Cache result
        if vol is not None:
            self._interpolation_cache[cache_key] = vol
            self._cache_timestamp = time.time()
        
        return vol
    
    def _find_closest_expiry_slice(self, time_to_expiry: float) -> Optional[VolatilitySlice]:
        """Find volatility slice closest to target time to expiry"""
        if not self.vol_slices:
            return None
        
        # Remove stale slices
        current_time = time.time()
        stale_expiries = []
        for expiry_ts, slice_obj in self.vol_slices.items():
            if expiry_ts <= current_time:
                stale_expiries.append(expiry_ts)
        
        for expiry_ts in stale_expiries:
            del self.vol_slices[expiry_ts]
        
        if not self.vol_slices:
            return None
        
        # Find closest by time to expiry
        best_slice = min(self.vol_slices.values(),
                        key=lambda s: abs(s.time_to_expiry - time_to_expiry))
        
        return best_slice
    
    def _interpolate_across_expiries(self, strike: float, time_to_expiry: float) -> Optional[float]:
        """Interpolate volatility across different expiries"""
        if len(self.vol_slices) < 2:
            return None
        
        # Get volatilities for this strike across all expiries
        expiry_vol_pairs = []
        for vol_slice in self.vol_slices.values():
            vol = vol_slice.get_vol_by_strike(strike, self.spot_price)
            if vol is not None:
                expiry_vol_pairs.append((vol_slice.time_to_expiry, vol))
        
        if len(expiry_vol_pairs) < 2:
            return None
        
        # Sort by time to expiry
        expiry_vol_pairs.sort(key=lambda x: x[0])
        
        expiries = [pair[0] for pair in expiry_vol_pairs]
        vols = [pair[1] for pair in expiry_vol_pairs]
        
        # Linear interpolation
        try:
            interpolated_vol = np.interp(time_to_expiry, expiries, vols)
            return float(interpolated_vol)
        except:
            return None
    
    def get_atm_volatility(self, time_to_expiry: float) -> Optional[float]:
        """Get ATM volatility for specific time to expiry"""
        return self.get_implied_volatility(self.spot_price, time_to_expiry)
    
    def get_vol_smile_data(self, time_to_expiry: float) -> Optional[Dict[str, any]]:
        """
        Get volatility smile data for specific expiry
        
        Returns:
            Dictionary with smile analysis or None if insufficient data
        """
        vol_slice = self._find_closest_expiry_slice(time_to_expiry)
        if not vol_slice or len(vol_slice.vol_points) < 3:
            return None
        
        # Sort points by moneyness
        sorted_points = sorted(vol_slice.vol_points, 
                             key=lambda p: p.moneyness(self.spot_price))
        
        moneyness_values = [p.moneyness(self.spot_price) for p in sorted_points]
        vol_values = [p.implied_volatility for p in sorted_points]
        
        # Calculate smile metrics
        atm_vol = vol_slice.atm_vol or 0.0
        vol_skew = vol_slice.vol_smile_skew
        
        # Find min vol (smile valley)
        min_vol_idx = vol_values.index(min(vol_values))
        min_vol_moneyness = moneyness_values[min_vol_idx]
        
        return {
            'time_to_expiry': vol_slice.time_to_expiry,
            'atm_volatility': atm_vol,
            'vol_skew': vol_skew,
            'min_vol_moneyness': min_vol_moneyness,
            'moneyness_range': (min(moneyness_values), max(moneyness_values)),
            'vol_range': (min(vol_values), max(vol_values)),
            'data_points': len(sorted_points),
            'moneyness_values': moneyness_values,
            'volatility_values': vol_values
        }
    
    def calculate_surface_quality_metrics(self) -> Dict[str, float]:
        """Calculate overall surface quality metrics"""
        current_time = time.time()
        
        if not self.market_vol_points:
            return {
                'surface_quality_score': 0.0,
                'coverage_score': 0.0,
                'staleness_score': 0.0,
                'data_points': 0
            }
        
        # Coverage score: how well we cover different strikes and expiries
        expiry_count = len(self.vol_slices)
        total_strikes = sum(len(slice_obj.vol_points) for slice_obj in self.vol_slices.values())
        coverage_score = min(1.0, (expiry_count * total_strikes) / 50)  # Normalize to 50 total points
        
        # Staleness score: how fresh is our data
        fresh_points = sum(1 for point in self.market_vol_points 
                          if not point.is_stale(300))  # 5 minutes
        staleness_score = fresh_points / len(self.market_vol_points) if self.market_vol_points else 0.0
        
        # Overall quality score
        surface_quality_score = (coverage_score * 0.6 + staleness_score * 0.4)
        
        # Update instance variables
        self.surface_quality_score = surface_quality_score
        self.coverage_score = coverage_score
        self.staleness_score = staleness_score
        
        return {
            'surface_quality_score': surface_quality_score,
            'coverage_score': coverage_score,
            'staleness_score': staleness_score,
            'data_points': len(self.market_vol_points),
            'expiry_slices': expiry_count,
            'total_strikes': total_strikes
        }
    
    def get_straddle_vol_analysis(self, strike: float, time_to_expiry: float) -> Dict[str, any]:
        """
        Get volatility analysis specifically for straddle at given strike/expiry
        
        Args:
            strike: Straddle strike
            time_to_expiry: Time to expiry in years
            
        Returns:
            Dictionary with straddle-specific vol analysis
        """
        # Get ATM vol and straddle vol
        atm_vol = self.get_atm_volatility(time_to_expiry)
        straddle_vol = self.get_implied_volatility(strike, time_to_expiry)
        
        if not atm_vol or not straddle_vol:
            return {
                'status': 'insufficient_data',
                'recommendation': 'Need more volatility surface data'
            }
        
        # Get vol smile data
        smile_data = self.get_vol_smile_data(time_to_expiry)
        
        # Analyze vol premium/discount relative to ATM
        vol_premium = (straddle_vol - atm_vol) / atm_vol
        
        # Moneyness analysis
        moneyness = strike / self.spot_price
        
        analysis = {
            'straddle_strike': strike,
            'straddle_volatility': straddle_vol,
            'atm_volatility': atm_vol,
            'vol_premium_vs_atm': vol_premium,
            'moneyness': moneyness,
            'time_to_expiry': time_to_expiry
        }
        
        if smile_data:
            analysis.update({
                'vol_skew': smile_data['vol_skew'],
                'min_vol_moneyness': smile_data['min_vol_moneyness'],
                'vol_range': smile_data['vol_range']
            })
            
            # Skew-based recommendations
            if smile_data['vol_skew'] < -0.02:  # Strong put skew
                if moneyness < 1.0:  # ITM put side
                    analysis['skew_comment'] = "Strike benefits from put skew - vol premium"
                else:
                    analysis['skew_comment'] = "Strike penalized by put skew - vol discount"
            elif smile_data['vol_skew'] > 0.02:  # Call skew (rare)
                if moneyness > 1.0:  # ITM call side
                    analysis['skew_comment'] = "Strike benefits from call skew - vol premium"
                else:
                    analysis['skew_comment'] = "Strike penalized by call skew - vol discount"
            else:
                analysis['skew_comment'] = "Minimal vol skew impact"
        
        return analysis
    
    def plot_volatility_smile(self, time_to_expiry: float, save_path: Optional[str] = None):
        """Plot volatility smile for specific expiry"""
        smile_data = self.get_vol_smile_data(time_to_expiry)
        
        if not smile_data:
            print("Insufficient data for volatility smile plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        moneyness = smile_data['moneyness_values']
        vols = smile_data['volatility_values']
        
        plt.plot(moneyness, vols, 'bo-', linewidth=2, markersize=6)
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
        
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Implied Volatility')
        plt.title(f'Volatility Smile - {time_to_expiry:.2f} years to expiry')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations
        atm_vol = smile_data['atm_volatility']
        vol_skew = smile_data['vol_skew']
        
        plt.text(0.02, 0.98, f'ATM Vol: {atm_vol:.1%}\nVol Skew: {vol_skew:+.1%}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def log_surface_summary(self):
        """Log comprehensive surface summary"""
        metrics = self.calculate_surface_quality_metrics()
        
        logging.info("="*60)
        logging.info("VOLATILITY SURFACE SUMMARY")
        logging.info("="*60)
        logging.info(f"Spot Price: ${self.spot_price:.0f}")
        logging.info(f"Surface Quality: {metrics['surface_quality_score']:.1%}")
        logging.info(f"Coverage Score: {metrics['coverage_score']:.1%}")
        logging.info(f"Staleness Score: {metrics['staleness_score']:.1%}")
        logging.info(f"Data Points: {metrics['data_points']}")
        logging.info(f"Expiry Slices: {metrics['expiry_slices']}")
        logging.info(f"Total Strikes: {metrics['total_strikes']}")
        
        if self.vol_slices:
            logging.info("\nEXPIRY BREAKDOWN:")
            for expiry_ts, vol_slice in sorted(self.vol_slices.items()):
                expiry_date = datetime.fromtimestamp(expiry_ts).strftime("%Y-%m-%d")
                atm_vol = vol_slice.atm_vol or 0.0
                skew = vol_slice.vol_smile_skew
                
                logging.info(f"  {expiry_date}: {len(vol_slice.vol_points)} strikes, "
                           f"ATM vol: {atm_vol:.1%}, Skew: {skew:+.1%}")
        
        logging.info("="*60)