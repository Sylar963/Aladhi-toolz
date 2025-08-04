#!/usr/bin/env python3
"""
Delta Hedging Strategy - Main Orchestrator
==========================================

Main orchestrator that combines all delta hedging components:
- Integrates optimal delta calculation with Bennett's philosophy
- Manages real-time hedging execution and monitoring
- Handles portfolio-level risk management
- Provides comprehensive performance tracking and reporting

This is the primary interface for implementing optimal delta hedging
in a production trading environment.
"""

import asyncio
import csv
import json
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque

import numpy as np

# Import our custom modules
from optimal_delta_hedger import (
    OptimalDeltaHedger, OptionParams, MarketData, TransactionCostParams,
    OptimalDeltaResult, OptionType
)
from enhanced_delta_calculator import (
    EnhancedDeltaCalculator, DeltaCalculationMethod, 
    DeltaDecomposition, DeltaRiskMetrics
)
from hedge_cost_optimizer import (
    HedgeCostOptimizer, OptimalRebalancingStrategy, ExecutionParams,
    HedgingCostAnalysis, RebalancingTrigger
)
from forward_volatility import ForwardVolatilityEstimator
from volatility_surface import VolatilitySurface
from volatility_regime import VolatilityRegimeDetector
from financial_math import FinancialMath


@dataclass
class HedgePosition:
    """Current hedge position state"""
    instrument_name: str
    option_position: float          # Current option position
    hedge_position: float           # Current hedge position (e.g., perpetual)
    target_hedge_position: float    # Target hedge position
    hedge_error: float              # Current hedge error
    
    last_rebalance_time: float      # Timestamp of last rebalance
    cumulative_pnl: float           # Cumulative P&L from hedging
    transaction_costs: float        # Cumulative transaction costs
    
    # Option details
    option_params: OptionParams
    current_market_data: MarketData


@dataclass
class HedgingEvent:
    """Record of a hedging event"""
    timestamp: float
    event_type: str                 # "rebalance", "position_change", "pnl_update"
    instrument_name: str
    
    # Before/after states
    hedge_position_before: float
    hedge_position_after: float
    hedge_error_before: float
    hedge_error_after: float
    
    # Trade details
    trade_size: float
    trade_price: Optional[float]
    transaction_cost: float
    
    # Reasoning
    trigger_reason: str
    optimal_delta: float
    market_delta: float
    adjustment_factor: float
    
    # Context
    volatility_forecast: float
    volatility_regime: str
    rebalancing_strategy: str


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    total_delta_exposure: float
    total_gamma_exposure: float
    total_vega_exposure: float
    total_theta_exposure: float
    
    portfolio_var: float            # Value at Risk
    portfolio_tracking_error: float # Tracking error vs targets
    concentration_risk: float       # Risk from position concentration
    
    liquidity_risk_score: float     # Liquidity risk assessment
    model_risk_score: float         # Model risk assessment
    execution_risk_score: float     # Execution risk assessment


class DeltaHedgingStrategy:
    """
    Main delta hedging strategy orchestrator
    
    Implements Colin Bennett's optimal delta hedging philosophy with:
    - Forecasted volatility for hedge ratios
    - Cost-optimized rebalancing
    - Real-time risk monitoring
    - Comprehensive performance tracking
    """
    
    def __init__(self,
                 vol_estimator: Optional[ForwardVolatilityEstimator] = None,
                 vol_surface: Optional[VolatilitySurface] = None,
                 regime_detector: Optional[VolatilityRegimeDetector] = None,
                 enable_logging: bool = True,
                 log_file: str = "delta_hedging.log"):
        """
        Initialize the delta hedging strategy
        
        Args:
            vol_estimator: Forward volatility estimator
            vol_surface: Volatility surface
            regime_detector: Volatility regime detector
            enable_logging: Enable detailed logging
            log_file: Log file path
        """
        # Core components
        self.optimal_hedger = OptimalDeltaHedger(vol_estimator, vol_surface, regime_detector)
        self.enhanced_calculator = EnhancedDeltaCalculator(vol_surface)
        self.cost_optimizer = HedgeCostOptimizer()
        
        # Volatility components
        self.vol_estimator = vol_estimator
        self.vol_surface = vol_surface
        self.regime_detector = regime_detector
        
        # Position tracking
        self.hedge_positions: Dict[str, HedgePosition] = {}
        self.hedging_history: List[HedgingEvent] = []
        
        # Risk management
        self.risk_limits = {
            'max_delta_exposure': 50.0,
            'max_position_size': 10.0,
            'max_daily_loss': 1000.0,
            'max_drawdown': 0.10
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_pnl': 0.0,
            'total_transaction_costs': 0.0,
            'hedge_effectiveness': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Configuration
        self.enable_logging = enable_logging
        self.log_file = log_file
        
        # Data for optimization
        self.price_history = deque(maxlen=1000)
        self.return_history = deque(maxlen=1000)
        
        # Operational state
        self.is_running = False
        self.last_risk_check = 0.0
        self.last_optimization = 0.0
        
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized DeltaHedgingStrategy with Bennett philosophy")
    
    def add_position(self, 
                    instrument_name: str,
                    option_params: OptionParams,
                    initial_option_position: float,
                    initial_hedge_position: float = 0.0) -> bool:
        """
        Add a new position to hedge
        
        Args:
            instrument_name: Unique identifier for the position
            option_params: Option parameters
            initial_option_position: Initial option position size
            initial_hedge_position: Initial hedge position
            
        Returns:
            True if position added successfully
        """
        try:
            # Create market data placeholder (will be updated in real-time)
            market_data = MarketData(
                market_implied_volatility=0.5,  # Will be updated
                current_option_price=0.0,       # Will be updated
            )
            
            hedge_position = HedgePosition(
                instrument_name=instrument_name,
                option_position=initial_option_position,
                hedge_position=initial_hedge_position,
                target_hedge_position=0.0,  # Will be calculated
                hedge_error=0.0,
                last_rebalance_time=time.time(),
                cumulative_pnl=0.0,
                transaction_costs=0.0,
                option_params=option_params,
                current_market_data=market_data
            )
            
            self.hedge_positions[instrument_name] = hedge_position
            
            self.logger.info(f"Added position {instrument_name}: "
                           f"option_pos={initial_option_position}, "
                           f"hedge_pos={initial_hedge_position}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add position {instrument_name}: {e}")
            return False
    
    def update_market_data(self, 
                          instrument_name: str,
                          spot_price: float,
                          option_price: float,
                          implied_volatility: float,
                          bid_ask_data: Optional[Dict] = None) -> bool:
        """
        Update market data for a position
        
        Args:
            instrument_name: Position identifier
            spot_price: Current underlying price
            option_price: Current option price
            implied_volatility: Current implied volatility
            bid_ask_data: Optional bid/ask data
            
        Returns:
            True if updated successfully
        """
        if instrument_name not in self.hedge_positions:
            self.logger.warning(f"Position {instrument_name} not found for market data update")
            return False
        
        try:
            position = self.hedge_positions[instrument_name]
            
            # Update option parameters with current spot
            position.option_params.current_spot_price = spot_price
            
            # Update market data
            position.current_market_data.current_option_price = option_price
            position.current_market_data.market_implied_volatility = implied_volatility
            
            if bid_ask_data:
                position.current_market_data.bid_price = bid_ask_data.get('option_bid')
                position.current_market_data.ask_price = bid_ask_data.get('option_ask')
                position.current_market_data.underlying_bid = bid_ask_data.get('underlying_bid')
                position.current_market_data.underlying_ask = bid_ask_data.get('underlying_ask')
            
            # Update price history for volatility estimation
            self._update_price_history(instrument_name, spot_price)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update market data for {instrument_name}: {e}")
            return False
    
    def _update_price_history(self, instrument_name: str, price: float):
        """Update price history for volatility calculations"""
        current_time = time.time()
        
        if len(self.price_history) > 0:
            last_price = self.price_history[-1][1]
            if last_price > 0 and price > 0:
                log_return = math.log(price / last_price)
                self.return_history.append(log_return)
        
        self.price_history.append((current_time, price))
        
        # Update regime detector if available
        if self.regime_detector:
            self.regime_detector.add_price_observation(price, current_time)
    
    def calculate_optimal_hedge_positions(self, 
                                        transaction_costs: Optional[TransactionCostParams] = None) -> Dict[str, OptimalDeltaResult]:
        """
        Calculate optimal hedge positions for all instruments
        
        Args:
            transaction_costs: Transaction cost parameters
            
        Returns:
            Dictionary of optimal delta results by instrument
        """
        results = {}
        
        # Get forecasted volatility
        forecast_vol = self._get_forecast_volatility()
        
        for instrument_name, position in self.hedge_positions.items():
            try:
                # Calculate optimal delta using Bennett's method
                optimal_result = self.optimal_hedger.calculate_optimal_delta(
                    option_params=position.option_params,
                    market_data=position.current_market_data,
                    forecast_volatility=forecast_vol,
                    transaction_costs=transaction_costs,
                    historical_returns=list(self.return_history)
                )
                
                results[instrument_name] = optimal_result
                
                # Update target hedge position
                position.target_hedge_position = optimal_result.optimal_delta
                position.hedge_error = position.target_hedge_position - position.hedge_position
                
                self.logger.debug(f"{instrument_name}: optimal_delta={optimal_result.optimal_delta:.4f}, "
                                f"current_hedge={position.hedge_position:.4f}, "
                                f"error={position.hedge_error:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to calculate optimal delta for {instrument_name}: {e}")
                continue
        
        return results
    
    def _get_forecast_volatility(self) -> float:
        """Get forecasted volatility using available estimators"""
        if self.vol_estimator and len(self.return_history) >= 50:
            try:
                # Get average time to expiry across positions
                avg_time_to_expiry = np.mean([
                    pos.option_params.time_to_expiration 
                    for pos in self.hedge_positions.values()
                ])
                
                forecast = self.vol_estimator.forecast_volatility(
                    time_horizon_days=avg_time_to_expiry * 365.25,
                    current_returns=list(self.return_history)
                )
                
                if forecast:
                    return forecast.expected_volatility
                    
            except Exception as e:
                self.logger.warning(f"Vol forecasting failed: {e}")
        
        # Fallback to implied vol average
        if self.hedge_positions:
            implied_vols = [pos.current_market_data.market_implied_volatility 
                          for pos in self.hedge_positions.values()
                          if pos.current_market_data.market_implied_volatility > 0]
            if implied_vols:
                return np.mean(implied_vols)
        
        # Ultimate fallback
        return 0.5  # 50% default volatility
    
    def check_rebalancing_triggers(self, 
                                 transaction_costs: Optional[TransactionCostParams] = None) -> Dict[str, Tuple[bool, str]]:
        """
        Check which positions need rebalancing
        
        Args:
            transaction_costs: Transaction cost parameters
            
        Returns:
            Dictionary of (should_rebalance, reason) by instrument
        """
        rebalancing_decisions = {}
        
        # Get or optimize rebalancing strategy
        strategy = self._get_rebalancing_strategy(transaction_costs)
        
        current_vol = self._get_current_volatility()
        
        for instrument_name, position in self.hedge_positions.items():
            try:
                time_since_last = (time.time() - position.last_rebalance_time) / 3600  # hours
                
                should_rebalance, reason = self.cost_optimizer.should_rebalance(
                    current_delta=position.hedge_position,
                    target_delta=position.target_hedge_position,
                    time_since_last_rebalance_hours=time_since_last,
                    current_vol=current_vol,
                    strategy=strategy
                )
                
                rebalancing_decisions[instrument_name] = (should_rebalance, reason)
                
            except Exception as e:
                self.logger.error(f"Failed to check rebalancing for {instrument_name}: {e}")
                rebalancing_decisions[instrument_name] = (False, f"error: {e}")
        
        return rebalancing_decisions
    
    def _get_rebalancing_strategy(self, 
                                transaction_costs: Optional[TransactionCostParams]) -> OptimalRebalancingStrategy:
        """Get or optimize rebalancing strategy"""
        current_time = time.time()
        
        # Re-optimize strategy periodically
        if (self.cost_optimizer.current_strategy is None or 
            current_time - self.last_optimization > 24 * 3600):  # Daily re-optimization
            
            if self.hedge_positions and transaction_costs:
                # Use first position as representative for optimization
                first_position = next(iter(self.hedge_positions.values()))
                forecast_vol = self._get_forecast_volatility()
                
                try:
                    strategy = self.cost_optimizer.optimize_rebalancing_strategy(
                        option_params=first_position.option_params,
                        forecast_volatility=forecast_vol,
                        transaction_costs=transaction_costs,
                        historical_returns=list(self.return_history)
                    )
                    self.last_optimization = current_time
                    return strategy
                except Exception as e:
                    self.logger.warning(f"Strategy optimization failed: {e}")
        
        # Return current strategy or default
        if self.cost_optimizer.current_strategy:
            return self.cost_optimizer.current_strategy
        else:
            # Default strategy
            return OptimalRebalancingStrategy(
                trigger_type=RebalancingTrigger.THRESHOLD_BASED,
                delta_threshold=0.05,
                time_threshold_hours=24,
                vol_change_threshold=0.05,
                vol_scaling_factor=1.0,
                time_decay_factor=1.0,
                cost_adjustment_factor=1.0,
                trade_sizing_method="optimal",
                execution_style="optimal"
            )
    
    def _get_current_volatility(self) -> float:
        """Get current realized volatility estimate"""
        if len(self.return_history) >= 20:
            recent_returns = list(self.return_history)[-20:]  # Last 20 returns
            return FinancialMath.annualize_volatility(recent_returns, "hourly")
        return 0.5  # Default
    
    async def execute_rebalancing(self, 
                                instrument_name: str,
                                execution_params: Optional[ExecutionParams] = None,
                                dry_run: bool = False) -> bool:
        """
        Execute rebalancing for a specific instrument
        
        Args:
            instrument_name: Position to rebalance
            execution_params: Execution parameters
            dry_run: If True, simulate but don't execute
            
        Returns:
            True if rebalancing successful
        """
        if instrument_name not in self.hedge_positions:
            self.logger.error(f"Position {instrument_name} not found for rebalancing")
            return False
        
        position = self.hedge_positions[instrument_name]
        
        # Calculate trade size needed
        trade_size = position.target_hedge_position - position.hedge_position
        
        if abs(trade_size) < 0.001:  # Too small to trade
            return True
        
        try:
            if execution_params is None:
                execution_params = ExecutionParams()
            
            # Calculate optimal execution schedule
            execution_schedule = self.cost_optimizer.calculate_optimal_trade_size(
                target_delta_change=trade_size,
                current_position=position.hedge_position,
                execution_params=execution_params
            )
            
            if dry_run:
                self.logger.info(f"DRY RUN: Would execute {len(execution_schedule)} trades "
                               f"for {instrument_name}, total size: {trade_size:.4f}")
                return True
            
            # Execute trades according to schedule
            total_executed = 0.0
            total_cost = 0.0
            
            for trade_size_slice, delay_seconds in execution_schedule:
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                
                # Simulate trade execution (in real implementation, this would call exchange API)
                execution_price = position.option_params.current_spot_price  # Simplified
                execution_cost = abs(trade_size_slice) * 0.001  # 0.1% cost estimate
                
                total_executed += trade_size_slice
                total_cost += execution_cost
                
                self.logger.info(f"Executed trade: {instrument_name} size={trade_size_slice:.4f} "
                               f"price={execution_price:.2f} cost=${execution_cost:.2f}")
            
            # Update position
            position.hedge_position += total_executed
            position.transaction_costs += total_cost
            position.last_rebalance_time = time.time()
            position.hedge_error = position.target_hedge_position - position.hedge_position
            
            # Record hedging event
            self._record_hedging_event(
                instrument_name=instrument_name,
                event_type="rebalance",
                trade_size=total_executed,
                transaction_cost=total_cost,
                trigger_reason="manual_rebalance"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute rebalancing for {instrument_name}: {e}")
            return False
    
    def _record_hedging_event(self, 
                            instrument_name: str,
                            event_type: str,
                            trade_size: float = 0.0,
                            transaction_cost: float = 0.0,
                            trigger_reason: str = ""):
        """Record a hedging event for analysis"""
        if instrument_name not in self.hedge_positions:
            return
        
        position = self.hedge_positions[instrument_name]
        
        # Get current optimal delta info
        forecast_vol = self._get_forecast_volatility()
        try:
            optimal_result = self.optimal_hedger.calculate_optimal_delta(
                option_params=position.option_params,
                market_data=position.current_market_data,
                forecast_volatility=forecast_vol,
                historical_returns=list(self.return_history)
            )
            optimal_delta = optimal_result.optimal_delta
            market_delta = optimal_result.current_bsm_delta
            adjustment_factor = optimal_result.hedge_adjustment_factor
        except:
            optimal_delta = 0.0
            market_delta = 0.0
            adjustment_factor = 1.0
        
        event = HedgingEvent(
            timestamp=time.time(),
            event_type=event_type,
            instrument_name=instrument_name,
            hedge_position_before=position.hedge_position - trade_size,
            hedge_position_after=position.hedge_position,
            hedge_error_before=position.hedge_error + trade_size,
            hedge_error_after=position.hedge_error,
            trade_size=trade_size,
            trade_price=position.option_params.current_spot_price,
            transaction_cost=transaction_cost,
            trigger_reason=trigger_reason,
            optimal_delta=optimal_delta,
            market_delta=market_delta,
            adjustment_factor=adjustment_factor,
            volatility_forecast=forecast_vol,
            volatility_regime=self._get_current_regime(),
            rebalancing_strategy=self._get_current_strategy_name()
        )
        
        self.hedging_history.append(event)
        
        # Keep history manageable
        if len(self.hedging_history) > 10000:
            self.hedging_history = self.hedging_history[-5000:]
    
    def _get_current_regime(self) -> str:
        """Get current volatility regime"""
        if self.regime_detector:
            return self.regime_detector.current_regime.value
        return "unknown"
    
    def _get_current_strategy_name(self) -> str:
        """Get current rebalancing strategy name"""
        if self.cost_optimizer.current_strategy:
            return self.cost_optimizer.current_strategy.trigger_type.value
        return "default"
    
    def calculate_portfolio_risk_metrics(self) -> PortfolioRiskMetrics:
        """Calculate portfolio-level risk metrics"""
        if not self.hedge_positions:
            return PortfolioRiskMetrics(
                total_delta_exposure=0, total_gamma_exposure=0,
                total_vega_exposure=0, total_theta_exposure=0,
                portfolio_var=0, portfolio_tracking_error=0,
                concentration_risk=0, liquidity_risk_score=0,
                model_risk_score=0, execution_risk_score=0
            )
        
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        
        hedge_errors = []
        position_sizes = []
        
        for position in self.hedge_positions.values():
            try:
                # Calculate Greeks
                greeks = FinancialMath.calculate_greeks(
                    spot=position.option_params.current_spot_price,
                    strike=position.option_params.strike_price,
                    time_to_expiry=position.option_params.time_to_expiration,
                    risk_free_rate=position.option_params.risk_free_rate,
                    volatility=position.current_market_data.market_implied_volatility,
                    dividend_yield=position.option_params.dividend_yield,
                    option_type=position.option_params.option_type.value
                )
                
                pos_size = position.option_position
                total_delta += greeks['delta'] * pos_size
                total_gamma += greeks['gamma'] * pos_size
                total_vega += greeks['vega'] * pos_size
                total_theta += greeks['theta'] * pos_size
                
                hedge_errors.append(position.hedge_error)
                position_sizes.append(abs(pos_size))
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate Greeks for {position.instrument_name}: {e}")
                continue
        
        # Portfolio metrics
        portfolio_tracking_error = np.std(hedge_errors) if len(hedge_errors) > 1 else 0
        
        # Concentration risk (max position / total)
        if position_sizes:
            concentration_risk = max(position_sizes) / sum(position_sizes)
        else:
            concentration_risk = 0
        
        # Simplified VaR estimate (1% daily VaR)
        portfolio_var = abs(total_delta) * 0.02 * max(position_sizes) if position_sizes else 0
        
        # Risk scores (simplified)
        liquidity_risk_score = min(1.0, len(self.hedge_positions) / 10)  # More positions = more risk
        model_risk_score = concentration_risk  # Higher concentration = higher model risk
        execution_risk_score = portfolio_tracking_error * 10  # Higher tracking error = higher execution risk
        
        return PortfolioRiskMetrics(
            total_delta_exposure=total_delta,
            total_gamma_exposure=total_gamma,
            total_vega_exposure=total_vega,
            total_theta_exposure=total_theta,
            portfolio_var=portfolio_var,
            portfolio_tracking_error=portfolio_tracking_error,
            concentration_risk=concentration_risk,
            liquidity_risk_score=liquidity_risk_score,
            model_risk_score=model_risk_score,
            execution_risk_score=execution_risk_score
        )
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """Check if portfolio exceeds risk limits"""
        risk_metrics = self.calculate_portfolio_risk_metrics()
        
        violations = {}
        violations['delta_exposure'] = abs(risk_metrics.total_delta_exposure) > self.risk_limits['max_delta_exposure']
        violations['portfolio_var'] = risk_metrics.portfolio_var > self.risk_limits['max_daily_loss']
        violations['concentration'] = risk_metrics.concentration_risk > 0.5  # 50% max concentration
        
        # Check individual position limits
        for name, position in self.hedge_positions.items():
            violations[f'{name}_position_size'] = abs(position.option_position) > self.risk_limits['max_position_size']
        
        return violations
    
    async def run_hedging_loop(self,
                             transaction_costs: Optional[TransactionCostParams] = None,
                             execution_params: Optional[ExecutionParams] = None,
                             update_interval_seconds: float = 60.0,
                             dry_run: bool = False):
        """
        Main hedging loop for continuous operation
        
        Args:
            transaction_costs: Transaction cost parameters
            execution_params: Execution parameters
            update_interval_seconds: How often to check for rebalancing
            dry_run: If True, simulate but don't execute trades
        """
        self.is_running = True
        self.logger.info(f"Starting hedging loop with {update_interval_seconds}s intervals")
        
        try:
            while self.is_running:
                loop_start_time = time.time()
                
                # Calculate optimal hedge positions
                optimal_results = self.calculate_optimal_hedge_positions(transaction_costs)
                
                # Check rebalancing triggers
                rebalancing_decisions = self.check_rebalancing_triggers(transaction_costs)
                
                # Execute rebalancing for triggered positions
                for instrument_name, (should_rebalance, reason) in rebalancing_decisions.items():
                    if should_rebalance:
                        self.logger.info(f"Rebalancing {instrument_name}: {reason}")
                        await self.execute_rebalancing(
                            instrument_name, execution_params, dry_run
                        )
                
                # Check risk limits periodically
                if time.time() - self.last_risk_check > 300:  # Every 5 minutes
                    violations = self.check_risk_limits()
                    if any(violations.values()):
                        self.logger.warning(f"Risk limit violations: {violations}")
                    self.last_risk_check = time.time()
                
                # Log status
                self._log_portfolio_status()
                
                # Sleep until next iteration
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, update_interval_seconds - loop_duration)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"Error in hedging loop: {e}")
            raise
        finally:
            self.is_running = False
            self.logger.info("Hedging loop stopped")
    
    def stop_hedging_loop(self):
        """Stop the hedging loop"""
        self.is_running = False
        self.logger.info("Stopping hedging loop...")
    
    def _log_portfolio_status(self):
        """Log current portfolio status"""
        risk_metrics = self.calculate_portfolio_risk_metrics()
        
        self.logger.info("="*50)
        self.logger.info("PORTFOLIO STATUS")
        self.logger.info("="*50)
        self.logger.info(f"Total Delta: {risk_metrics.total_delta_exposure:.3f}")
        self.logger.info(f"Total Gamma: {risk_metrics.total_gamma_exposure:.3f}")
        self.logger.info(f"Tracking Error: {risk_metrics.portfolio_tracking_error:.3f}")
        self.logger.info(f"Portfolio VaR: ${risk_metrics.portfolio_var:.2f}")
        
        for name, position in self.hedge_positions.items():
            self.logger.info(f"{name}: hedge_pos={position.hedge_position:.3f}, "
                           f"target={position.target_hedge_position:.3f}, "
                           f"error={position.hedge_error:.3f}")
        
        self.logger.info("="*50)
    
    def export_performance_data(self, filename: str = "hedging_performance.csv"):
        """Export performance data to CSV"""
        try:
            with open(filename, 'w', newline='') as csvfile:
                if not self.hedging_history:
                    self.logger.warning("No hedging history to export")
                    return
                
                # Get field names from first event
                fieldnames = list(asdict(self.hedging_history[0]).keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for event in self.hedging_history:
                    writer.writerow(asdict(event))
                
                self.logger.info(f"Exported {len(self.hedging_history)} events to {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to export performance data: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get summary of hedging performance"""
        if not self.hedging_history:
            return {"status": "no_data"}
        
        # Analyze hedging history
        rebalance_events = [e for e in self.hedging_history if e.event_type == "rebalance"]
        
        if not rebalance_events:
            return {"status": "no_rebalances"}
        
        # Calculate performance metrics
        total_transaction_costs = sum(e.transaction_cost for e in rebalance_events)
        avg_trade_size = np.mean([abs(e.trade_size) for e in rebalance_events])
        
        hedge_errors_before = [abs(e.hedge_error_before) for e in rebalance_events]
        hedge_errors_after = [abs(e.hedge_error_after) for e in rebalance_events]
        
        avg_error_reduction = np.mean([b - a for b, a in zip(hedge_errors_before, hedge_errors_after)])
        
        # Rebalancing frequency
        if len(rebalance_events) > 1:
            time_span = rebalance_events[-1].timestamp - rebalance_events[0].timestamp
            rebalance_frequency = len(rebalance_events) / (time_span / 86400)  # per day
        else:
            rebalance_frequency = 0
        
        return {
            "status": "success",
            "total_rebalances": len(rebalance_events),
            "total_transaction_costs": total_transaction_costs,
            "average_trade_size": avg_trade_size,
            "average_error_reduction": avg_error_reduction,
            "rebalancing_frequency_per_day": rebalance_frequency,
            "hedge_effectiveness": max(0, avg_error_reduction / max(0.001, np.mean(hedge_errors_before))),
            "cost_per_rebalance": total_transaction_costs / len(rebalance_events) if rebalance_events else 0
        }