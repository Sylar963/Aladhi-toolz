#!/usr/bin/env python3
"""
Hedge Cost Optimizer
====================

Optimizes delta hedging frequency and execution to minimize total cost:
- Balances hedge error vs transaction costs (Bennett's key insight)
- Dynamic rebalancing thresholds based on market conditions
- Execution optimization (timing, size, market impact)
- Portfolio-level hedging efficiency

Core Philosophy:
"The optimal hedge frequency balances the cost of tracking error against 
the cost of transactions. Use forecasted volatility to optimize this tradeoff."
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from financial_math import FinancialMath, TradingConstants
from optimal_delta_hedger import OptionParams, TransactionCostParams


class RebalancingTrigger(Enum):
    """Different triggers for rebalancing"""
    TIME_BASED = "time_based"           # Fixed time intervals
    THRESHOLD_BASED = "threshold_based" # Delta threshold
    VOLATILITY_BASED = "vol_based"      # Vol regime changes
    HYBRID = "hybrid"                   # Multiple triggers
    OPTIMAL = "optimal"                 # Cost-minimizing approach


@dataclass
class ExecutionParams:
    """Parameters for trade execution optimization"""
    max_position_size: float = 10.0     # Maximum position size
    max_trade_size: float = 2.0         # Maximum single trade size
    min_trade_size: float = 0.01        # Minimum trade size
    
    # Market impact model parameters
    temporary_impact_factor: float = 0.0001  # Temporary impact per unit
    permanent_impact_factor: float = 0.00005 # Permanent impact per unit
    impact_decay_half_life: float = 300      # Impact decay in seconds
    
    # Execution timing
    max_execution_time_seconds: float = 60   # Max time to complete trade
    slice_interval_seconds: float = 10       # Time between trade slices
    
    # Risk controls
    max_drawdown_threshold: float = 0.05     # 5% max drawdown
    position_limit_factor: float = 0.8       # Use 80% of available capital


@dataclass
class HedgingCostAnalysis:
    """Analysis of hedging costs and performance"""
    # Cost components
    transaction_costs: float            # Total transaction costs
    hedge_error_cost: float            # Cost from tracking error
    market_impact_cost: float          # Market impact costs
    opportunity_cost: float            # Cost from delays
    
    # Performance metrics
    total_cost: float                  # Sum of all costs
    cost_per_delta: float              # Cost per unit of delta hedged
    hedge_effectiveness: float         # How well hedge tracked target
    
    # Optimization metrics
    rebalancing_frequency: float       # Average rebalances per day
    average_trade_size: float          # Average trade size
    execution_efficiency: float        # Execution quality score
    
    # Risk metrics
    tracking_error: float              # Standard deviation of hedge error
    max_hedge_error: float             # Maximum hedge error observed
    cost_volatility: float             # Volatility of hedging costs


@dataclass
class OptimalRebalancingStrategy:
    """Optimal rebalancing strategy parameters"""
    trigger_type: RebalancingTrigger
    delta_threshold: float             # Rebalance when delta off by this much
    time_threshold_hours: float        # Maximum time between rebalances
    vol_change_threshold: float        # Vol change threshold for rebalancing
    
    # Dynamic adjustments
    vol_scaling_factor: float          # How thresholds scale with vol
    time_decay_factor: float           # How thresholds change near expiry
    cost_adjustment_factor: float      # How thresholds adjust for costs
    
    # Execution parameters
    trade_sizing_method: str           # "fixed", "proportional", "optimal"
    execution_style: str               # "aggressive", "passive", "optimal"


class HedgeCostOptimizer:
    """
    Optimizer for delta hedging costs and execution
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical data for optimization
        self.cost_history: List[HedgingCostAnalysis] = []
        self.rebalancing_history: List[Tuple[float, float, float]] = []  # time, delta_before, delta_after
        
        # Current optimization state
        self.current_strategy: Optional[OptimalRebalancingStrategy] = None
        self.last_optimization_time = 0.0
        self.optimization_interval_hours = 24  # Re-optimize daily
    
    def optimize_rebalancing_strategy(self, 
                                    option_params: OptionParams,
                                    forecast_volatility: float,
                                    transaction_costs: TransactionCostParams,
                                    execution_params: Optional[ExecutionParams] = None,
                                    historical_returns: Optional[List[float]] = None) -> OptimalRebalancingStrategy:
        """
        Optimize rebalancing strategy to minimize total hedging costs
        
        Args:
            option_params: Option parameters
            forecast_volatility: Expected volatility
            transaction_costs: Transaction cost parameters
            execution_params: Execution parameters
            historical_returns: Historical returns for calibration
            
        Returns:
            Optimal rebalancing strategy
        """
        if execution_params is None:
            execution_params = ExecutionParams()
        
        # Define cost function to minimize
        def total_cost_function(params) -> float:
            delta_threshold, time_threshold_hours = params
            
            # Estimate hedge error cost
            hedge_error_cost = self._estimate_hedge_error_cost(
                option_params, forecast_volatility, delta_threshold, time_threshold_hours
            )
            
            # Estimate transaction costs
            rebalancing_frequency = self._estimate_rebalancing_frequency(
                forecast_volatility, delta_threshold, time_threshold_hours
            )
            transaction_cost = self._estimate_transaction_costs(
                rebalancing_frequency, transaction_costs, option_params
            )
            
            # Estimate market impact costs
            market_impact_cost = self._estimate_market_impact_cost(
                rebalancing_frequency, execution_params, option_params
            )
            
            return hedge_error_cost + transaction_cost + market_impact_cost
        
        # Optimize parameters
        # Start with reasonable bounds
        bounds = [
            (0.005, 0.2),   # delta_threshold: 0.5% to 20%
            (0.1, 48.0)     # time_threshold: 6 minutes to 48 hours
        ]
        
        initial_guess = [
            transaction_costs.delta_threshold,
            transaction_costs.time_threshold_hours
        ]
        
        try:
            result = minimize(
                total_cost_function,
                initial_guess,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                optimal_delta_threshold, optimal_time_threshold = result.x
                self.logger.info(f"Optimization successful: delta_threshold={optimal_delta_threshold:.3f}, "
                               f"time_threshold={optimal_time_threshold:.1f}h")
            else:
                self.logger.warning(f"Optimization failed: {result.message}")
                optimal_delta_threshold = transaction_costs.delta_threshold
                optimal_time_threshold = transaction_costs.time_threshold_hours
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            optimal_delta_threshold = transaction_costs.delta_threshold
            optimal_time_threshold = transaction_costs.time_threshold_hours
        
        # Calculate dynamic adjustment factors
        vol_scaling = self._calculate_vol_scaling_factor(forecast_volatility)
        time_decay = self._calculate_time_decay_factor(option_params.time_to_expiration)
        cost_adjustment = self._calculate_cost_adjustment_factor(transaction_costs)
        
        # Determine optimal trigger type
        trigger_type = self._determine_optimal_trigger_type(
            option_params, forecast_volatility, transaction_costs
        )
        
        strategy = OptimalRebalancingStrategy(
            trigger_type=trigger_type,
            delta_threshold=optimal_delta_threshold,
            time_threshold_hours=optimal_time_threshold,
            vol_change_threshold=0.05,  # 5% vol change trigger
            vol_scaling_factor=vol_scaling,
            time_decay_factor=time_decay,
            cost_adjustment_factor=cost_adjustment,
            trade_sizing_method="optimal",
            execution_style="optimal"
        )
        
        self.current_strategy = strategy
        self.last_optimization_time = time.time()
        
        return strategy
    
    def _estimate_hedge_error_cost(self, 
                                 option_params: OptionParams,
                                 forecast_vol: float,
                                 delta_threshold: float,
                                 time_threshold_hours: float) -> float:
        """
        Estimate cost from hedge tracking error
        
        Uses option gamma to estimate P&L variance from discrete hedging
        """
        # Calculate option gamma
        greeks = FinancialMath.calculate_greeks(
            spot=option_params.current_spot_price,
            strike=option_params.strike_price,
            time_to_expiry=option_params.time_to_expiration,
            risk_free_rate=option_params.risk_free_rate,
            volatility=forecast_vol,
            dividend_yield=option_params.dividend_yield,
            option_type=option_params.option_type.value
        )
        
        gamma = greeks['gamma']
        
        # Estimate rebalancing frequency
        vol_annual = forecast_vol
        vol_daily = vol_annual / math.sqrt(252)
        
        # Expected time between rebalances
        expected_rebalance_interval_hours = min(
            time_threshold_hours,
            24 * delta_threshold / (vol_daily * math.sqrt(2/math.pi))  # Expected hitting time
        )
        
        expected_rebalance_interval_days = expected_rebalance_interval_hours / 24
        
        # Hedge error from discrete rebalancing (simplified model)
        # Error ≈ 0.5 * gamma * sigma^2 * S^2 * dt
        spot = option_params.current_spot_price
        dt = expected_rebalance_interval_days / 365.25  # Convert to years
        
        hedge_error_variance = 0.5 * gamma * (vol_annual * spot)**2 * dt
        hedge_error_cost = math.sqrt(hedge_error_variance) * abs(option_params.position_size)
        
        return hedge_error_cost
    
    def _estimate_rebalancing_frequency(self, 
                                      forecast_vol: float,
                                      delta_threshold: float,
                                      time_threshold_hours: float) -> float:
        """
        Estimate how often rebalancing will occur
        
        Returns: rebalances per day
        """
        # Model delta as a random walk driven by underlying price moves
        vol_daily = forecast_vol / math.sqrt(252)
        
        # Expected time to hit delta threshold (hitting time of Brownian motion)
        # This is a simplified approximation
        expected_hit_time_days = (delta_threshold / vol_daily)**2 * (math.pi / 2)
        
        # Also consider time-based rebalancing
        time_based_frequency = 24 / time_threshold_hours  # rebalances per day
        
        # Combine both triggers (whichever happens first)
        threshold_based_frequency = 1 / expected_hit_time_days if expected_hit_time_days > 0 else 0
        
        # Total frequency is approximately the sum (assuming independent events)
        total_frequency = threshold_based_frequency + time_based_frequency
        
        return min(total_frequency, 24)  # Cap at hourly rebalancing
    
    def _estimate_transaction_costs(self, 
                                  rebalancing_frequency: float,
                                  transaction_costs: TransactionCostParams,
                                  option_params: OptionParams) -> float:
        """Estimate total transaction costs per day"""
        # Average trade size (simplified assumption)
        avg_trade_size = abs(option_params.position_size) * 0.1  # 10% of position
        
        cost_per_trade = (
            transaction_costs.commission_per_trade + 
            transaction_costs.bid_ask_spread_cost
        ) * avg_trade_size * option_params.current_spot_price
        
        daily_transaction_cost = rebalancing_frequency * cost_per_trade
        
        return daily_transaction_cost
    
    def _estimate_market_impact_cost(self, 
                                   rebalancing_frequency: float,
                                   execution_params: ExecutionParams,
                                   option_params: OptionParams) -> float:
        """Estimate market impact costs"""
        # Average trade size
        avg_trade_size = abs(option_params.position_size) * 0.1
        
        # Market impact per trade
        impact_per_trade = (
            execution_params.temporary_impact_factor + 
            execution_params.permanent_impact_factor
        ) * avg_trade_size * option_params.current_spot_price
        
        daily_impact_cost = rebalancing_frequency * impact_per_trade
        
        return daily_impact_cost
    
    def _calculate_vol_scaling_factor(self, forecast_vol: float) -> float:
        """Calculate how thresholds should scale with volatility"""
        # Higher vol -> tighter thresholds (more frequent rebalancing)
        # Lower vol -> wider thresholds (less frequent rebalancing)
        
        base_vol = 0.5  # 50% reference volatility
        scaling_factor = math.sqrt(forecast_vol / base_vol)
        
        # Bound the scaling factor
        return max(0.5, min(2.0, scaling_factor))
    
    def _calculate_time_decay_factor(self, time_to_expiry: float) -> float:
        """Calculate how thresholds should adjust as expiry approaches"""
        # Shorter time to expiry -> tighter thresholds (gamma risk increases)
        
        if time_to_expiry > 0.25:  # > 3 months
            return 1.0
        elif time_to_expiry > 0.083:  # 1-3 months
            return 0.8
        elif time_to_expiry > 0.027:  # 1 week - 1 month
            return 0.6
        else:  # < 1 week
            return 0.4
    
    def _calculate_cost_adjustment_factor(self, transaction_costs: TransactionCostParams) -> float:
        """Calculate adjustment based on transaction cost levels"""
        # Higher costs -> wider thresholds
        
        total_cost_rate = (
            transaction_costs.commission_per_trade + 
            transaction_costs.bid_ask_spread_cost + 
            transaction_costs.market_impact_factor
        )
        
        # Scale relative to 0.1% total cost
        base_cost = 0.001
        cost_factor = total_cost_rate / base_cost
        
        return max(0.5, min(3.0, cost_factor))
    
    def _determine_optimal_trigger_type(self, 
                                      option_params: OptionParams,
                                      forecast_vol: float,
                                      transaction_costs: TransactionCostParams) -> RebalancingTrigger:
        """Determine the best trigger type based on option characteristics"""
        
        # For very short-dated options with high gamma, use hybrid approach
        if option_params.time_to_expiration < 0.027:  # < 1 week
            return RebalancingTrigger.HYBRID
        
        # For high volatility environments, use vol-based triggers
        if forecast_vol > 0.8:
            return RebalancingTrigger.VOLATILITY_BASED
        
        # For high transaction cost environments, use time-based
        total_cost_rate = (
            transaction_costs.commission_per_trade + 
            transaction_costs.bid_ask_spread_cost
        )
        if total_cost_rate > 0.002:  # > 0.2% total costs
            return RebalancingTrigger.TIME_BASED
        
        # Default to optimal threshold-based approach
        return RebalancingTrigger.THRESHOLD_BASED
    
    def calculate_optimal_trade_size(self, 
                                   target_delta_change: float,
                                   current_position: float,
                                   execution_params: ExecutionParams,
                                   market_conditions: Optional[Dict] = None) -> List[Tuple[float, float]]:
        """
        Calculate optimal trade sizing to minimize market impact
        
        Returns: List of (size, timing) pairs for execution
        """
        total_size = abs(target_delta_change)
        
        if total_size <= execution_params.min_trade_size:
            return [(target_delta_change, 0)]  # Execute immediately
        
        # If trade is small enough, execute in one go
        if total_size <= execution_params.max_trade_size:
            return [(target_delta_change, 0)]
        
        # For larger trades, slice into smaller pieces
        num_slices = math.ceil(total_size / execution_params.max_trade_size)
        slice_size = total_size / num_slices
        
        # Create execution schedule
        execution_schedule = []
        remaining_size = target_delta_change
        
        for i in range(num_slices):
            if abs(remaining_size) < execution_params.min_trade_size:
                break
            
            # Size of this slice
            current_slice_size = min(slice_size, abs(remaining_size))
            if remaining_size < 0:
                current_slice_size = -current_slice_size
            
            # Timing of this slice
            execution_time = i * execution_params.slice_interval_seconds
            
            execution_schedule.append((current_slice_size, execution_time))
            remaining_size -= current_slice_size
        
        return execution_schedule
    
    def should_rebalance(self, 
                        current_delta: float,
                        target_delta: float,
                        time_since_last_rebalance_hours: float,
                        current_vol: float,
                        strategy: Optional[OptimalRebalancingStrategy] = None) -> Tuple[bool, str]:
        """
        Determine if rebalancing should occur based on current strategy
        
        Returns: (should_rebalance, reason)
        """
        if strategy is None:
            strategy = self.current_strategy
        
        if strategy is None:
            # Default strategy
            delta_diff = abs(current_delta - target_delta)
            return delta_diff > 0.05, "default_threshold"
        
        delta_diff = abs(current_delta - target_delta)
        
        # Apply dynamic adjustments to thresholds
        adjusted_delta_threshold = (
            strategy.delta_threshold * 
            strategy.vol_scaling_factor * 
            strategy.time_decay_factor * 
            strategy.cost_adjustment_factor
        )
        
        adjusted_time_threshold = (
            strategy.time_threshold_hours * 
            strategy.cost_adjustment_factor
        )
        
        # Check different trigger conditions
        if strategy.trigger_type == RebalancingTrigger.THRESHOLD_BASED:
            if delta_diff > adjusted_delta_threshold:
                return True, f"delta_threshold_{delta_diff:.3f}>{adjusted_delta_threshold:.3f}"
        
        elif strategy.trigger_type == RebalancingTrigger.TIME_BASED:
            if time_since_last_rebalance_hours > adjusted_time_threshold:
                return True, f"time_threshold_{time_since_last_rebalance_hours:.1f}h>{adjusted_time_threshold:.1f}h"
        
        elif strategy.trigger_type == RebalancingTrigger.VOLATILITY_BASED:
            # This would need vol regime change detection
            # For now, use delta threshold with vol adjustment
            vol_adjusted_threshold = adjusted_delta_threshold * math.sqrt(current_vol / 0.5)
            if delta_diff > vol_adjusted_threshold:
                return True, f"vol_adjusted_threshold_{delta_diff:.3f}>{vol_adjusted_threshold:.3f}"
        
        elif strategy.trigger_type == RebalancingTrigger.HYBRID:
            # Multiple conditions
            if delta_diff > adjusted_delta_threshold:
                return True, f"hybrid_delta_{delta_diff:.3f}>{adjusted_delta_threshold:.3f}"
            if time_since_last_rebalance_hours > adjusted_time_threshold:
                return True, f"hybrid_time_{time_since_last_rebalance_hours:.1f}h>{adjusted_time_threshold:.1f}h"
        
        elif strategy.trigger_type == RebalancingTrigger.OPTIMAL:
            # Cost-based decision
            hedge_error_cost = delta_diff * 100  # Simplified
            transaction_cost = 5  # Simplified
            
            if hedge_error_cost > transaction_cost:
                return True, f"cost_optimal_{hedge_error_cost:.1f}>{transaction_cost:.1f}"
        
        return False, "no_trigger"
    
    def analyze_hedging_performance(self, 
                                  hedge_history: List[Tuple[float, float, float]],
                                  cost_data: Optional[List[float]] = None) -> HedgingCostAnalysis:
        """
        Analyze historical hedging performance
        
        Args:
            hedge_history: List of (timestamp, target_delta, actual_delta)
            cost_data: List of transaction costs
            
        Returns:
            HedgingCostAnalysis
        """
        if not hedge_history:
            return HedgingCostAnalysis(
                transaction_costs=0, hedge_error_cost=0, market_impact_cost=0,
                opportunity_cost=0, total_cost=0, cost_per_delta=0,
                hedge_effectiveness=0, rebalancing_frequency=0,
                average_trade_size=0, execution_efficiency=0,
                tracking_error=0, max_hedge_error=0, cost_volatility=0
            )
        
        # Calculate hedge errors
        hedge_errors = [abs(target - actual) for _, target, actual in hedge_history]
        
        # Performance metrics
        tracking_error = np.std(hedge_errors) if len(hedge_errors) > 1 else 0
        max_hedge_error = max(hedge_errors) if hedge_errors else 0
        average_hedge_error = np.mean(hedge_errors) if hedge_errors else 0
        
        # Calculate rebalancing frequency
        if len(hedge_history) > 1:
            time_diffs = [hedge_history[i][0] - hedge_history[i-1][0] 
                         for i in range(1, len(hedge_history))]
            avg_time_between_rebalances = np.mean(time_diffs) / 3600  # Convert to hours
            rebalancing_frequency = 24 / avg_time_between_rebalances if avg_time_between_rebalances > 0 else 0
        else:
            rebalancing_frequency = 0
        
        # Cost analysis
        if cost_data:
            transaction_costs = sum(cost_data)
            cost_volatility = np.std(cost_data) if len(cost_data) > 1 else 0
        else:
            transaction_costs = len(hedge_history) * 0.01  # Estimated $0.01 per rebalance
            cost_volatility = 0
        
        # Simplified cost estimates
        hedge_error_cost = average_hedge_error * 100  # Simplified
        market_impact_cost = transaction_costs * 0.5  # Assume 50% of transaction costs
        opportunity_cost = max_hedge_error * 50  # Simplified
        
        total_cost = transaction_costs + hedge_error_cost + market_impact_cost + opportunity_cost
        
        # Effectiveness metrics
        hedge_effectiveness = max(0, 1 - average_hedge_error / 0.1)  # Relative to 10% error
        cost_per_delta = total_cost / max(1, len(hedge_history))
        execution_efficiency = max(0, 1 - tracking_error / 0.05)  # Relative to 5% tracking error
        
        # Trade size analysis
        if len(hedge_history) > 1:
            delta_changes = [abs(hedge_history[i][1] - hedge_history[i-1][1]) 
                           for i in range(1, len(hedge_history))]
            average_trade_size = np.mean(delta_changes) if delta_changes else 0
        else:
            average_trade_size = 0
        
        return HedgingCostAnalysis(
            transaction_costs=transaction_costs,
            hedge_error_cost=hedge_error_cost,
            market_impact_cost=market_impact_cost,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost,
            cost_per_delta=cost_per_delta,
            hedge_effectiveness=hedge_effectiveness,
            rebalancing_frequency=rebalancing_frequency,
            average_trade_size=average_trade_size,
            execution_efficiency=execution_efficiency,
            tracking_error=tracking_error,
            max_hedge_error=max_hedge_error,
            cost_volatility=cost_volatility
        )
    
    def log_optimization_results(self, 
                               strategy: OptimalRebalancingStrategy,
                               cost_analysis: Optional[HedgingCostAnalysis] = None):
        """Log optimization results and strategy"""
        self.logger.info("="*60)
        self.logger.info("HEDGE COST OPTIMIZATION RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Optimal Strategy:")
        self.logger.info(f"  Trigger Type: {strategy.trigger_type.value}")
        self.logger.info(f"  Delta Threshold: {strategy.delta_threshold:.3f}")
        self.logger.info(f"  Time Threshold: {strategy.time_threshold_hours:.1f} hours")
        self.logger.info(f"  Vol Change Threshold: {strategy.vol_change_threshold:.3f}")
        self.logger.info("")
        self.logger.info("Dynamic Adjustments:")
        self.logger.info(f"  Vol Scaling Factor: {strategy.vol_scaling_factor:.2f}")
        self.logger.info(f"  Time Decay Factor: {strategy.time_decay_factor:.2f}")
        self.logger.info(f"  Cost Adjustment Factor: {strategy.cost_adjustment_factor:.2f}")
        
        if cost_analysis:
            self.logger.info("")
            self.logger.info("Performance Analysis:")
            self.logger.info(f"  Total Cost: ${cost_analysis.total_cost:.2f}")
            self.logger.info(f"  Hedge Effectiveness: {cost_analysis.hedge_effectiveness:.1%}")
            self.logger.info(f"  Tracking Error: {cost_analysis.tracking_error:.3f}")
            self.logger.info(f"  Rebalancing Frequency: {cost_analysis.rebalancing_frequency:.1f}/day")
        
        self.logger.info("="*60)