#!/usr/bin/env python3
"""
Volatility Model Backtesting Framework
=====================================

This module provides comprehensive backtesting for volatility models to validate:
- Forward volatility forecast accuracy
- Regime detection effectiveness
- Enhanced straddle pricing performance vs market

Critical for validating model performance before live trading.
"""

import math
import time
import statistics
from typing import Dict, List, Optional, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from financial_math import FinancialMath
from volatility_regime import VolatilityRegime, VolatilityRegimeDetector
from forward_volatility import ForwardVolatilityEstimator
from enhanced_straddle_pricing import EnhancedStraddlePricingModel


@dataclass
class BacktestTrade:
    """Individual backtest trade record"""
    entry_date: datetime
    exit_date: datetime
    
    # Trade parameters
    strike: float
    time_to_expiry_entry: float
    entry_spot_price: float
    exit_spot_price: float
    
    # Pricing at entry
    market_straddle_price: float
    enhanced_fair_value: float
    bs_fair_value: float
    
    # Volatility forecasts at entry
    implied_vol_entry: float
    forward_vol_forecast: float
    regime_forecast: VolatilityRegime
    
    # Realized outcomes
    realized_vol: float
    actual_regime: VolatilityRegime
    straddle_pnl: float
    
    # Performance metrics
    vol_forecast_error: float
    pricing_edge: float
    trade_success: bool


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    
    # Overall performance
    total_trades: int
    profitable_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Volatility forecasting accuracy
    vol_forecast_mae: float  # Mean Absolute Error
    vol_forecast_rmse: float # Root Mean Square Error
    vol_forecast_correlation: float
    
    # Regime detection accuracy
    regime_accuracy: float
    regime_precision_by_regime: Dict[VolatilityRegime, float]
    
    # Enhanced pricing performance
    enhanced_vs_bs_performance: float
    avg_pricing_edge: float
    
    # Risk metrics
    var_95: float  # Value at Risk
    expected_shortfall: float
    
    # Time-based analysis
    performance_by_expiry_bucket: Dict[str, float]
    performance_by_regime: Dict[VolatilityRegime, float]


class VolatilityBacktester:
    """
    Comprehensive backtesting framework for volatility models
    """
    
    def __init__(self):
        self.trades: List[BacktestTrade] = []
        self.price_history: List[Tuple[datetime, float]] = []
        
        # Model instances for backtesting
        self.regime_detector = VolatilityRegimeDetector()
        self.forward_vol_estimator = ForwardVolatilityEstimator(self.regime_detector)
        # Note: volatility_surface would need market data, so we'll simulate it
        
        # Backtesting parameters
        self.min_data_points = 100
        self.holding_period_days = 30  # Standard holding period
        self.transaction_costs = 0.02  # 2% of premium
        
        logging.info("Initialized VolatilityBacktester")
    
    def load_historical_data(self, price_data: List[Tuple[datetime, float]]):
        """
        Load historical price data for backtesting
        
        Args:
            price_data: List of (datetime, price) tuples
        """
        self.price_history = sorted(price_data, key=lambda x: x[0])
        logging.info(f"Loaded {len(self.price_history)} historical price points")
        
        # Feed data to regime detector
        for date, price in self.price_history:
            timestamp = date.timestamp()
            self.regime_detector.add_price_observation(price, timestamp)
        
        logging.info("Historical data loaded into regime detector")
    
    def run_backtest(self, start_date: datetime, end_date: datetime,
                    straddle_parameters: Dict) -> BacktestResults:
        """
        Run comprehensive backtest over specified period
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            straddle_parameters: Dictionary with straddle trading parameters
            
        Returns:
            BacktestResults object
        """
        if not self.price_history:
            raise ValueError("No historical data loaded")
        
        logging.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Filter price history to backtest period
        backtest_data = [
            (date, price) for date, price in self.price_history
            if start_date <= date <= end_date
        ]
        
        if len(backtest_data) < self.min_data_points:
            raise ValueError(f"Insufficient data points: {len(backtest_data)}")
        
        # Reset for clean backtest
        self.trades = []
        
        # Simulate straddle trading
        self._simulate_straddle_trading(backtest_data, straddle_parameters)
        
        # Analyze results
        results = self._analyze_backtest_results()
        
        logging.info(f"Backtest completed: {len(self.trades)} trades, "
                    f"Win rate: {results.win_rate:.1%}, Total PnL: {results.total_pnl:.2f}")
        
        return results
    
    def _simulate_straddle_trading(self, price_data: List[Tuple[datetime, float]],
                                 straddle_params: Dict):
        """Simulate straddle trading strategy"""
        
        entry_frequency_days = straddle_params.get('entry_frequency_days', 7)
        time_to_expiry_target = straddle_params.get('time_to_expiry_days', 30) / 365
        
        # Track open positions
        open_positions = []
        
        for i, (current_date, current_price) in enumerate(price_data):
            
            # Check for position exits first
            positions_to_close = []
            for pos_idx, (entry_date, entry_data) in enumerate(open_positions):
                days_held = (current_date - entry_date).days
                
                if days_held >= self.holding_period_days:
                    positions_to_close.append(pos_idx)
            
            # Close expired positions
            for pos_idx in reversed(positions_to_close):
                entry_date, entry_data = open_positions.pop(pos_idx)
                self._close_position(entry_date, entry_data, current_date, current_price)
            
            # Check for new position entries
            if self._should_enter_new_position(current_date, price_data, i):
                
                # Gather recent returns for model input
                recent_returns = self._get_recent_returns(price_data, i, lookback_days=60)
                
                if len(recent_returns) >= 50:  # Minimum for modeling
                    
                    # Simulate enhanced pricing analysis
                    enhanced_analysis = self._simulate_enhanced_pricing(
                        current_price, current_price, time_to_expiry_target, recent_returns
                    )
                    
                    if enhanced_analysis and self._should_take_trade(enhanced_analysis):
                        
                        # Record position entry
                        entry_data = {
                            'spot_price': current_price,
                            'strike': current_price,  # ATM straddle
                            'time_to_expiry': time_to_expiry_target,
                            'enhanced_analysis': enhanced_analysis,
                            'recent_returns': recent_returns.copy()
                        }
                        
                        open_positions.append((current_date, entry_data))
                        
                        logging.debug(f"Entered straddle position on {current_date.strftime('%Y-%m-%d')} "
                                    f"at ${current_price:.0f}")
        
        # Close any remaining open positions at end of backtest
        final_date, final_price = price_data[-1]
        for entry_date, entry_data in open_positions:
            self._close_position(entry_date, entry_data, final_date, final_price)
    
    def _should_enter_new_position(self, current_date: datetime, 
                                 price_data: List[Tuple[datetime, float]], 
                                 current_idx: int) -> bool:
        """Determine if we should enter a new position"""
        
        # Simple frequency-based entry (could be enhanced with signal-based entry)
        if current_idx < 30:  # Need sufficient history
            return False
        
        # Check if enough time has passed since last entry
        if self.trades:
            last_trade_date = self.trades[-1].entry_date
            days_since_last = (current_date - last_trade_date).days
            if days_since_last < 7:  # Minimum 7 days between entries
                return False
        
        return True
    
    def _get_recent_returns(self, price_data: List[Tuple[datetime, float]], 
                          current_idx: int, lookback_days: int = 60) -> List[float]:
        """Extract recent returns for model input"""
        
        if current_idx == 0:
            return []
        
        # Get data from last lookback_days
        cutoff_date = price_data[current_idx][0] - timedelta(days=lookback_days)
        
        recent_prices = []
        for i in range(max(0, current_idx - lookback_days * 24), current_idx + 1):  # Assume hourly data
            if i < len(price_data) and price_data[i][0] >= cutoff_date:
                recent_prices.append(price_data[i][1])
        
        # Calculate log returns
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0 and recent_prices[i] > 0:
                ret = math.log(recent_prices[i] / recent_prices[i-1])
                if abs(ret) < 0.2:  # Filter extreme moves
                    returns.append(ret)
        
        return returns
    
    def _simulate_enhanced_pricing(self, spot_price: float, strike_price: float,
                                 time_to_expiry: float, recent_returns: List[float]) -> Optional[Dict]:
        """Simulate enhanced pricing analysis"""
        
        # Calculate current implied volatility (simulate market straddle price)
        current_vol = FinancialMath.annualize_volatility(recent_returns, "daily")
        if current_vol == 0:
            return None
        
        # Simulate market straddle price with some noise
        noise_factor = 1 + np.random.normal(0, 0.05)  # 5% pricing noise
        market_straddle_price = FinancialMath.black_scholes_straddle(
            spot_price, strike_price, time_to_expiry, 0.05, current_vol
        ) * noise_factor
        
        # Get forward volatility forecast
        vol_forecast = self.forward_vol_estimator.forecast_volatility(
            time_to_expiry * 365, recent_returns
        )
        
        if not vol_forecast:
            return None
        
        # Get regime analysis
        regime_metrics = self.regime_detector.analyze_regime()
        
        # Calculate enhanced fair value (simplified)
        forward_vol = vol_forecast.expected_volatility
        enhanced_fair_value = FinancialMath.black_scholes_straddle(
            spot_price, strike_price, time_to_expiry, 0.05, forward_vol
        )
        
        return {
            'market_price': market_straddle_price,
            'enhanced_fair_value': enhanced_fair_value,
            'current_vol': current_vol,
            'forward_vol': forward_vol,
            'vol_forecast': vol_forecast,
            'regime_metrics': regime_metrics,
            'expected_vol_change': (forward_vol - current_vol) / current_vol
        }
    
    def _should_take_trade(self, enhanced_analysis: Dict) -> bool:
        """Determine if we should take the trade based on enhanced analysis"""
        
        market_price = enhanced_analysis['market_price']
        fair_value = enhanced_analysis['enhanced_fair_value']
        expected_vol_change = enhanced_analysis['expected_vol_change']
        
        # Simple trading rules
        price_attractiveness = (fair_value - market_price) / market_price
        
        # Take trade if:
        # 1. Enhanced model suggests straddle is underpriced by >5%
        # 2. Forward vol is expected to increase by >10%
        return (price_attractiveness > 0.05 and expected_vol_change > 0.10)
    
    def _close_position(self, entry_date: datetime, entry_data: Dict,
                       exit_date: datetime, exit_price: float):
        """Close position and record trade"""
        
        entry_spot = entry_data['spot_price']
        strike = entry_data['strike']
        entry_analysis = entry_data['enhanced_analysis']
        
        # Calculate realized volatility over holding period
        realized_returns = self._calculate_realized_returns_for_period(
            entry_date, exit_date, entry_spot, exit_price
        )
        
        realized_vol = FinancialMath.annualize_volatility(realized_returns, "daily") if realized_returns else 0
        
        # Calculate straddle P&L
        time_to_expiry_exit = max(0.01, entry_data['time_to_expiry'] - self.holding_period_days/365)
        
        # Entry cost (negative)
        entry_cost = -entry_analysis['market_price'] * (1 + self.transaction_costs)
        
        # Exit value (positive if ITM)
        intrinsic_value = max(0, abs(exit_price - strike))
        
        # Time value at exit (simplified)
        if time_to_expiry_exit > 0 and realized_vol > 0:
            time_value = FinancialMath.black_scholes_straddle(
                exit_price, strike, time_to_expiry_exit, 0.05, realized_vol
            ) - intrinsic_value
        else:
            time_value = 0
        
        exit_value = (intrinsic_value + time_value) * (1 - self.transaction_costs)
        total_pnl = entry_cost + exit_value
        
        # Determine current regime at exit (simplified)
        current_regime = self._get_regime_at_date(exit_date)
        
        # Create trade record
        trade = BacktestTrade(
            entry_date=entry_date,
            exit_date=exit_date,
            strike=strike,
            time_to_expiry_entry=entry_data['time_to_expiry'],
            entry_spot_price=entry_spot,
            exit_spot_price=exit_price,
            market_straddle_price=entry_analysis['market_price'],
            enhanced_fair_value=entry_analysis['enhanced_fair_value'],
            bs_fair_value=entry_analysis['market_price'],  # Simplified
            implied_vol_entry=entry_analysis['current_vol'],
            forward_vol_forecast=entry_analysis['forward_vol'],
            regime_forecast=entry_analysis['regime_metrics'].regime if entry_analysis['regime_metrics'] else VolatilityRegime.UNKNOWN,
            realized_vol=realized_vol,
            actual_regime=current_regime,
            straddle_pnl=total_pnl,
            vol_forecast_error=abs(realized_vol - entry_analysis['forward_vol']),
            pricing_edge=(entry_analysis['enhanced_fair_value'] - entry_analysis['market_price']),
            trade_success=(total_pnl > 0)
        )
        
        self.trades.append(trade)
    
    def _calculate_realized_returns_for_period(self, entry_date: datetime, 
                                             exit_date: datetime,
                                             entry_price: float, 
                                             exit_price: float) -> List[float]:
        """Calculate realized returns for the holding period"""
        
        # Find relevant price data
        period_data = [
            (date, price) for date, price in self.price_history
            if entry_date <= date <= exit_date
        ]
        
        if len(period_data) < 2:
            return []
        
        returns = []
        for i in range(1, len(period_data)):
            prev_price = period_data[i-1][1]
            curr_price = period_data[i][1]
            
            if prev_price > 0 and curr_price > 0:
                ret = math.log(curr_price / prev_price)
                if abs(ret) < 0.2:  # Filter extreme moves
                    returns.append(ret)
        
        return returns
    
    def _get_regime_at_date(self, date: datetime) -> VolatilityRegime:
        """Get volatility regime at specific date (simplified)"""
        
        # Find price data around this date
        target_timestamp = date.timestamp()
        
        # Look for data within 1 day
        nearby_data = [
            (d, p) for d, p in self.price_history
            if abs(d.timestamp() - target_timestamp) < 86400
        ]
        
        if not nearby_data:
            return VolatilityRegime.UNKNOWN
        
        # Simple regime classification based on recent volatility
        recent_prices = [p for d, p in nearby_data[-20:]]  # Last 20 observations
        
        if len(recent_prices) < 5:
            return VolatilityRegime.UNKNOWN
        
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                returns.append(math.log(recent_prices[i] / recent_prices[i-1]))
        
        if not returns:
            return VolatilityRegime.UNKNOWN
        
        vol = statistics.stdev(returns) * math.sqrt(252)
        
        # Simple regime classification
        if vol < 0.25:
            return VolatilityRegime.LOW
        elif vol < 0.50:
            return VolatilityRegime.NORMAL
        elif vol < 0.75:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _analyze_backtest_results(self) -> BacktestResults:
        """Analyze backtest results and generate comprehensive metrics"""
        
        if not self.trades:
            raise ValueError("No trades to analyze")
        
        # Basic performance metrics
        profitable_trades = sum(1 for trade in self.trades if trade.trade_success)
        win_rate = profitable_trades / len(self.trades)
        total_pnl = sum(trade.straddle_pnl for trade in self.trades)
        avg_pnl_per_trade = total_pnl / len(self.trades)
        
        # Risk metrics
        pnl_series = [trade.straddle_pnl for trade in self.trades]
        pnl_std = statistics.stdev(pnl_series) if len(pnl_series) > 1 else 0
        sharpe_ratio = avg_pnl_per_trade / pnl_std if pnl_std > 0 else 0
        
        # Drawdown calculation
        cumulative_pnl = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown)
        
        # Volatility forecasting accuracy
        vol_errors = [trade.vol_forecast_error for trade in self.trades if trade.vol_forecast_error > 0]
        vol_forecast_mae = statistics.mean(vol_errors) if vol_errors else 0
        vol_forecast_rmse = math.sqrt(statistics.mean([e**2 for e in vol_errors])) if vol_errors else 0
        
        # Correlation between forecasted and realized vol
        forecasted_vols = [trade.forward_vol_forecast for trade in self.trades]
        realized_vols = [trade.realized_vol for trade in self.trades]
        
        try:
            vol_forecast_correlation, _ = stats.pearsonr(forecasted_vols, realized_vols)
        except:
            vol_forecast_correlation = 0.0
        
        # Regime accuracy
        regime_predictions = [trade.regime_forecast for trade in self.trades]
        regime_actuals = [trade.actual_regime for trade in self.trades]
        
        correct_regime_predictions = sum(
            1 for pred, actual in zip(regime_predictions, regime_actuals) 
            if pred == actual and pred != VolatilityRegime.UNKNOWN
        )
        
        valid_regime_predictions = sum(
            1 for pred in regime_predictions 
            if pred != VolatilityRegime.UNKNOWN
        )
        
        regime_accuracy = (correct_regime_predictions / valid_regime_predictions 
                          if valid_regime_predictions > 0 else 0)
        
        # Performance by expiry bucket
        performance_by_expiry = defaultdict(list)
        for trade in self.trades:
            if trade.time_to_expiry_entry < 0.08:  # < 30 days
                bucket = "short"
            elif trade.time_to_expiry_entry < 0.25:  # 30-90 days
                bucket = "medium"
            else:
                bucket = "long"
            
            performance_by_expiry[bucket].append(trade.straddle_pnl)
        
        performance_by_expiry = {
            bucket: statistics.mean(pnls) 
            for bucket, pnls in performance_by_expiry.items()
        }
        
        # Performance by regime
        performance_by_regime = defaultdict(list)
        for trade in self.trades:
            performance_by_regime[trade.regime_forecast].append(trade.straddle_pnl)
        
        performance_by_regime = {
            regime: statistics.mean(pnls)
            for regime, pnls in performance_by_regime.items()
        }
        
        # Risk metrics
        var_95 = np.percentile(pnl_series, 5) if pnl_series else 0  # 5th percentile
        tail_losses = [pnl for pnl in pnl_series if pnl <= var_95]
        expected_shortfall = statistics.mean(tail_losses) if tail_losses else 0
        
        return BacktestResults(
            total_trades=len(self.trades),
            profitable_trades=profitable_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl_per_trade,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            vol_forecast_mae=vol_forecast_mae,
            vol_forecast_rmse=vol_forecast_rmse,
            vol_forecast_correlation=vol_forecast_correlation,
            regime_accuracy=regime_accuracy,
            regime_precision_by_regime={},  # Could be enhanced
            enhanced_vs_bs_performance=0,  # Could be enhanced
            avg_pricing_edge=statistics.mean([trade.pricing_edge for trade in self.trades]),
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            performance_by_expiry_bucket=performance_by_expiry,
            performance_by_regime=performance_by_regime
        )
    
    def generate_backtest_report(self, results: BacktestResults) -> str:
        """Generate comprehensive backtest report"""
        
        report_lines = [
            "="*80,
            "VOLATILITY MODEL BACKTEST RESULTS",
            "="*80,
            "",
            "OVERALL PERFORMANCE:",
            f"  Total Trades: {results.total_trades}",
            f"  Profitable Trades: {results.profitable_trades}",
            f"  Win Rate: {results.win_rate:.1%}",
            f"  Total P&L: {results.total_pnl:.2f}",
            f"  Average P&L per Trade: {results.avg_pnl_per_trade:.2f}",
            f"  Sharpe Ratio: {results.sharpe_ratio:.2f}",
            f"  Maximum Drawdown: {results.max_drawdown:.2f}",
            "",
            "VOLATILITY FORECASTING ACCURACY:",
            f"  Mean Absolute Error: {results.vol_forecast_mae:.2%}",
            f"  Root Mean Square Error: {results.vol_forecast_rmse:.2%}",
            f"  Forecast Correlation: {results.vol_forecast_correlation:.2f}",
            "",
            "REGIME DETECTION ACCURACY:",
            f"  Overall Accuracy: {results.regime_accuracy:.1%}",
            "",
            "RISK METRICS:",
            f"  95% VaR: {results.var_95:.2f}",
            f"  Expected Shortfall: {results.expected_shortfall:.2f}",
            "",
            "PERFORMANCE BY EXPIRY:",
        ]
        
        for expiry, perf in results.performance_by_expiry_bucket.items():
            report_lines.append(f"  {expiry.capitalize()}: {perf:.2f}")
        
        report_lines.extend([
            "",
            "PERFORMANCE BY REGIME:"
        ])
        
        for regime, perf in results.performance_by_regime.items():
            report_lines.append(f"  {regime.value}: {perf:.2f}")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)
    
    def plot_backtest_results(self, results: BacktestResults, save_path: Optional[str] = None):
        """Generate backtest visualization plots"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative P&L
        pnl_series = [trade.straddle_pnl for trade in self.trades]
        cumulative_pnl = np.cumsum(pnl_series)
        trade_dates = [trade.exit_date for trade in self.trades]
        
        ax1.plot(trade_dates, cumulative_pnl, linewidth=2, color='blue')
        ax1.set_title('Cumulative P&L')
        ax1.set_ylabel('Cumulative P&L')
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatility forecast accuracy
        forecasted = [trade.forward_vol_forecast for trade in self.trades]
        realized = [trade.realized_vol for trade in self.trades]
        
        ax2.scatter(forecasted, realized, alpha=0.6, color='red')
        ax2.plot([0, max(forecasted + realized)], [0, max(forecasted + realized)], 
                'k--', alpha=0.5, label='Perfect Forecast')
        ax2.set_xlabel('Forecasted Volatility')
        ax2.set_ylabel('Realized Volatility')
        ax2.set_title('Volatility Forecast Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. P&L distribution
        ax3.hist(pnl_series, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax3.set_xlabel('Trade P&L')
        ax3.set_ylabel('Frequency')
        ax3.set_title('P&L Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance by regime
        regime_performance = results.performance_by_regime
        regimes = list(regime_performance.keys())
        performance = list(regime_performance.values())
        
        colors = ['blue' if p > 0 else 'red' for p in performance]
        ax4.bar(range(len(regimes)), performance, color=colors, alpha=0.7)
        ax4.set_xticks(range(len(regimes)))
        ax4.set_xticklabels([r.value for r in regimes], rotation=45)
        ax4.set_ylabel('Average P&L')
        ax4.set_title('Performance by Volatility Regime')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def run_sample_backtest():
    """Run a sample backtest with simulated data"""
    
    # Generate sample price data
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(365*24)]  # 1 year of hourly data
    
    # Simulate price path with volatility clustering
    returns = np.random.normal(0, 0.02, len(dates))  # 2% hourly vol
    prices = [50000]  # Start at $50,000
    
    for ret in returns:
        prices.append(prices[-1] * math.exp(ret))
    
    price_data = list(zip(dates, prices[1:]))  # Align dates with prices
    
    # Run backtest
    backtester = VolatilityBacktester()
    backtester.load_historical_data(price_data)
    
    straddle_params = {
        'entry_frequency_days': 7,
        'time_to_expiry_days': 30
    }
    
    results = backtester.run_backtest(
        start_date=datetime(2023, 3, 1),  # Start after building history
        end_date=datetime(2023, 12, 1),
        straddle_parameters=straddle_params
    )
    
    # Generate report
    report = backtester.generate_backtest_report(results)
    print(report)
    
    # Generate plots
    backtester.plot_backtest_results(results)
    
    return results


if __name__ == "__main__":
    # Run sample backtest
    results = run_sample_backtest()