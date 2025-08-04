#!/usr/bin/env python3
"""
Demo: Optimal Delta Hedging with Colin Bennett Philosophy
==========================================================

This demo showcases the complete optimal delta hedging system implementing
Colin Bennett's key insights:

1. Use forecasted realized volatility (σ_R) for hedge ratios, not implied volatility (Σ)
2. Handle volatility smile/skew effects for accurate hedge ratios
3. Optimize rebalancing frequency considering transaction costs  
4. Model-aware hedging with regime adjustments

Usage:
    python demo_optimal_delta_hedging.py [--dry-run] [--config path/to/config.json]
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add path to core modules
sys.path.append(str(Path(__file__).parent.parent / "core"))

from delta_hedging_integration import create_integrated_system, IntegratedDeltaHedgingSystem
from optimal_delta_hedger import OptionParams, MarketData, OptionType
from delta_hedging_config import get_config_manager


class DeltaHedgingDemo:
    """Demo of the optimal delta hedging system"""
    
    def __init__(self, config_path=None, dry_run=True):
        self.config_path = config_path
        self.dry_run = dry_run
        self.system = None
        
        # Demo parameters
        self.btc_spot_price = 50000.0
        self.demo_positions = [
            {
                'name': 'BTC-CALL-50K-3M',
                'params': OptionParams(
                    current_spot_price=50000.0,
                    strike_price=50000.0,
                    time_to_expiration=0.25,  # 3 months
                    risk_free_rate=0.05,
                    dividend_yield=0.0,
                    option_type=OptionType.CALL,
                    position_size=1.0
                ),
                'market_data': MarketData(
                    market_implied_volatility=0.50,
                    current_option_price=2500.0,
                    bid_price=2450.0,
                    ask_price=2550.0
                )
            },
            {
                'name': 'BTC-PUT-45K-3M',
                'params': OptionParams(
                    current_spot_price=50000.0,
                    strike_price=45000.0,  # OTM put
                    time_to_expiration=0.25,
                    risk_free_rate=0.05,
                    dividend_yield=0.0,
                    option_type=OptionType.PUT,
                    position_size=2.0
                ),
                'market_data': MarketData(
                    market_implied_volatility=0.65,  # Higher vol for OTM put (skew)
                    current_option_price=800.0,
                    bid_price=790.0,
                    ask_price=810.0
                )
            },
            {
                'name': 'BTC-STRADDLE-50K-1M',
                'params': OptionParams(
                    current_spot_price=50000.0,
                    strike_price=50000.0,
                    time_to_expiration=0.083,  # 1 month
                    risk_free_rate=0.05,
                    dividend_yield=0.0,
                    option_type=OptionType.STRADDLE,
                    position_size=0.5
                ),
                'market_data': MarketData(
                    market_implied_volatility=0.80,  # High vol for short-dated straddle
                    current_option_price=3200.0,
                    bid_price=3150.0,
                    ask_price=3250.0
                )
            }
        ]
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def run_demo(self):
        """Run the complete demo"""
        print("🚀 OPTIMAL DELTA HEDGING DEMO")
        print("=" * 60)
        print("📚 Implementing Colin Bennett's Philosophy:")
        print("   • Use forecasted volatility (σ_R) for hedge ratios")
        print("   • Handle volatility skew for OTM options")
        print("   • Optimize transaction costs vs hedge error")
        print("   • Model-aware hedging with regime detection")
        print("=" * 60)
        
        try:
            # Step 1: Initialize the integrated system
            await self._initialize_system()
            
            # Step 2: Set up demo positions
            await self._setup_demo_positions()
            
            # Step 3: Add volatility surface data
            await self._setup_volatility_surface()
            
            # Step 4: Generate price history for forecasting
            await self._generate_price_history()
            
            # Step 5: Demonstrate optimal delta calculations
            await self._demonstrate_optimal_deltas()
            
            # Step 6: Show Bennett vs BSM comparison
            await self._compare_bennett_vs_bsm()
            
            # Step 7: Demonstrate cost optimization
            await self._demonstrate_cost_optimization()
            
            # Step 8: Run hedging simulation
            await self._run_hedging_simulation()
            
            # Step 9: Performance analysis
            await self._analyze_performance()
            
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise
        finally:
            if self.system:
                self.system.stop_hedging()
    
    async def _initialize_system(self):
        """Initialize the integrated delta hedging system"""
        print("\n📦 STEP 1: Initializing Integrated System")
        print("-" * 40)
        
        self.system = await create_integrated_system(
            config_path=self.config_path,
            spot_price=self.btc_spot_price,
            enable_database=False  # Skip database for demo
        )
        
        print("✅ System initialized with all components")
        
        # Show system status
        status = self.system.get_system_status()
        print(f"   • Components active: {sum(status['components'].values())}/8")
        print(f"   • Bennett philosophy: {status['config']['bennett_philosophy']}")
        print(f"   • Forecast vol enabled: {status['config']['use_forecast_vol']}")
        print(f"   • Delta method: {status['config']['delta_method']}")
    
    async def _setup_demo_positions(self):
        """Set up demo options positions"""
        print("\n📊 STEP 2: Setting Up Demo Positions")
        print("-" * 40)
        
        for i, position in enumerate(self.demo_positions, 1):
            # Add position to hedging strategy
            success = self.system.hedging_strategy.add_position(
                instrument_name=position['name'],
                option_params=position['params'],
                initial_option_position=position['params'].position_size
            )
            
            if success:
                # Update market data
                self.system.update_market_data(
                    instrument_name=position['name'],
                    spot_price=position['params'].current_spot_price,
                    option_price=position['market_data'].current_option_price,
                    implied_vol=position['market_data'].market_implied_volatility
                )
                
                print(f"   {i}. ✅ {position['name']}")
                print(f"      Type: {position['params'].option_type.value.title()}")
                print(f"      Strike: ${position['params'].strike_price:,.0f}")
                print(f"      Size: {position['params'].position_size}")
                print(f"      Implied Vol: {position['market_data'].market_implied_volatility:.1%}")
            else:
                print(f"   {i}. ❌ Failed to add {position['name']}")
    
    async def _setup_volatility_surface(self):
        """Set up volatility surface with smile/skew data"""
        print("\n📈 STEP 3: Building Volatility Surface")
        print("-" * 40)
        
        # Simulate volatility surface data with skew
        # (In practice, this would come from market data)
        
        strikes = [40000, 45000, 50000, 55000, 60000]
        base_vol = 0.50
        expiry_3m = datetime.now().timestamp() + 90 * 24 * 3600  # 3 months
        expiry_1m = datetime.now().timestamp() + 30 * 24 * 3600  # 1 month
        
        vol_points_added = 0
        
        for strike in strikes:
            # Create volatility skew (higher vol for lower strikes)
            moneyness = strike / self.btc_spot_price
            skew_adjustment = (1.0 - moneyness) * 0.3  # 30% skew effect
            
            for expiry, term_label in [(expiry_3m, "3M"), (expiry_1m, "1M")]:
                # Term structure effect (higher vol for shorter expiries)
                term_adjustment = 0.1 if expiry == expiry_1m else 0.0
                
                vol = base_vol + skew_adjustment + term_adjustment
                vol = max(0.15, min(1.5, vol))  # Reasonable bounds
                
                success = self.system.add_volatility_point(
                    strike=strike,
                    expiry_timestamp=expiry,
                    implied_vol=vol
                )
                
                if success:
                    vol_points_added += 1
        
        print(f"✅ Added {vol_points_added} volatility points")
        print("   • Skew: Higher vol for lower strikes (put skew)")
        print("   • Term structure: Higher vol for shorter expiries")
        
        # Show surface quality
        if self.system.vol_surface:
            metrics = self.system.vol_surface.calculate_surface_quality_metrics()
            print(f"   • Surface quality: {metrics['surface_quality_score']:.1%}")
    
    async def _generate_price_history(self):
        """Generate realistic price history for volatility forecasting"""
        print("\n📊 STEP 4: Generating Price History")
        print("-" * 40)
        
        # Simulate 100 days of hourly BTC price data
        import numpy as np
        
        current_price = self.btc_spot_price
        prices = [current_price]
        
        # Parameters for realistic BTC volatility clustering
        base_vol = 0.02  # 2% daily vol
        vol_persistence = 0.85
        current_vol = base_vol
        
        for hour in range(2400):  # 100 days * 24 hours
            # GARCH-like volatility updating
            current_vol = 0.0001 + 0.1 * (prices[-1]/prices[-2] - 1)**2 + vol_persistence * current_vol
            current_vol = min(0.05, max(0.005, current_vol))  # Reasonable bounds
            
            # Generate price move
            return_shock = np.random.normal(0, current_vol / np.sqrt(24))  # Hourly returns
            new_price = prices[-1] * np.exp(return_shock)
            prices.append(new_price)
        
        # Feed price history to system
        import time
        base_time = time.time() - len(prices) * 3600  # Start 100 days ago
        
        updates_sent = 0
        for i, price in enumerate(prices):
            timestamp = base_time + i * 3600
            
            # Update all positions with new spot price
            for position in self.demo_positions:
                success = self.system.update_market_data(
                    instrument_name=position['name'],
                    spot_price=price,
                    option_price=position['market_data'].current_option_price * (price / self.btc_spot_price),
                    implied_vol=position['market_data'].market_implied_volatility
                )
                if success:
                    updates_sent += 1
        
        print(f"✅ Generated {len(prices)} price points")
        print(f"   • Price range: ${min(prices):,.0f} - ${max(prices):,.0f}")
        print(f"   • Market data updates sent: {updates_sent}")
        
        # Update current spot price
        self.btc_spot_price = prices[-1]
    
    async def _demonstrate_optimal_deltas(self):
        """Demonstrate optimal delta calculations"""
        print("\n🎯 STEP 5: Optimal Delta Calculations")
        print("-" * 40)
        
        for position in self.demo_positions:
            print(f"\n📊 {position['name']}:")
            
            # Calculate optimal delta using Bennett's method
            optimal_result = self.system.get_optimal_delta(
                instrument_name=position['name'],
                option_params=position['params'],
                market_data=position['market_data']
            )
            
            if optimal_result:
                print(f"   🎯 Optimal Delta: {optimal_result['optimal_delta']:+.4f}")
                print(f"   📈 BSM Delta: {optimal_result['bsm_delta']:+.4f}")
                print(f"   ⚡ Adjustment Factor: {optimal_result['adjustment_factor']:.3f}x")
                print(f"   🔮 Forecast Vol: {optimal_result['forecast_volatility']:.1%}")
                print(f"   💹 Implied Vol: {optimal_result['implied_volatility']:.1%}")
                print(f"   🌡️  Vol Regime: {optimal_result['regime']}")
                print(f"   🎚️  Rebalance Threshold: {optimal_result['rebalance_threshold']:.3f}")
                
                # Interpretation
                if abs(optimal_result['adjustment_factor'] - 1.0) > 0.05:
                    print(f"   💡 Bennett insight: Significant adjustment needed!")
                else:
                    print(f"   💡 BSM delta approximately correct")
            else:
                print("   ❌ Failed to calculate optimal delta")
    
    async def _compare_bennett_vs_bsm(self):
        """Compare Bennett's approach vs standard BSM"""
        print("\n⚖️  STEP 6: Bennett vs BSM Comparison")
        print("-" * 40)
        
        comparison_table = []
        
        for position in self.demo_positions:
            result = self.system.get_optimal_delta(
                position['name'], position['params'], position['market_data']
            )
            
            if result:
                comparison_table.append({
                    'name': position['name'].split('-')[1],  # Extract option type
                    'bennett_delta': result['optimal_delta'],
                    'bsm_delta': result['bsm_delta'],
                    'difference': result['optimal_delta'] - result['bsm_delta'],
                    'adjustment_factor': result['adjustment_factor'],
                    'forecast_vol': result['forecast_volatility'],
                    'implied_vol': result['implied_volatility']
                })
        
        print("┌──────────────┬──────────┬──────────┬──────────┬───────────┐")
        print("│ Option       │ Bennett  │ BSM      │ Diff     │ Adj Factor│")
        print("├──────────────┼──────────┼──────────┼──────────┼───────────┤")
        
        for row in comparison_table:
            print(f"│ {row['name']:<12} │ {row['bennett_delta']:+8.4f} │ {row['bsm_delta']:+8.4f} │ {row['difference']:+8.4f} │ {row['adjustment_factor']:9.3f} │")
        
        print("└──────────────┴──────────┴──────────┴──────────┴───────────┘")
        
        # Analysis
        avg_adjustment = np.mean([abs(row['adjustment_factor'] - 1.0) for row in comparison_table])
        print(f"\n📊 Analysis:")
        print(f"   • Average adjustment magnitude: {avg_adjustment:.1%}")
        
        if avg_adjustment > 0.05:
            print("   💡 Significant differences found - Bennett's approach adds value!")
        else:
            print("   💡 Small differences - BSM approximately correct for these conditions")
    
    async def _demonstrate_cost_optimization(self):
        """Demonstrate transaction cost optimization"""
        print("\n💰 STEP 7: Transaction Cost Optimization")
        print("-" * 40)
        
        config_manager = get_config_manager(self.config_path)
        transaction_costs = config_manager.get_transaction_cost_params()
        
        print("Current cost parameters:")
        print(f"   • Commission per trade: {transaction_costs.commission_per_trade:.3%}")
        print(f"   • Bid-ask spread cost: {transaction_costs.bid_ask_spread_cost:.3%}")
        print(f"   • Delta threshold: {transaction_costs.delta_threshold:.3f}")
        print(f"   • Time threshold: {transaction_costs.time_threshold_hours:.1f}h")
        
        # Demonstrate rebalancing decisions
        print(f"\n🔄 Rebalancing Decisions:")
        
        current_vol = 0.6  # 60% current vol
        rebalancing_decisions = self.system.hedging_strategy.check_rebalancing_triggers(
            transaction_costs
        )
        
        for instrument, (should_rebalance, reason) in rebalancing_decisions.items():
            status = "🟢 REBALANCE" if should_rebalance else "🔴 HOLD"
            print(f"   • {instrument}: {status}")
            print(f"     Reason: {reason}")
        
        # Show cost-optimized strategy
        if self.system.cost_optimizer.current_strategy:
            strategy = self.system.cost_optimizer.current_strategy
            print(f"\n⚙️  Optimized Strategy:")
            print(f"   • Trigger type: {strategy.trigger_type.value}")
            print(f"   • Optimized delta threshold: {strategy.delta_threshold:.3f}")
            print(f"   • Optimized time threshold: {strategy.time_threshold_hours:.1f}h")
    
    async def _run_hedging_simulation(self):
        """Run a short hedging simulation"""
        print("\n🎮 STEP 8: Hedging Simulation")
        print("-" * 40)
        
        if self.dry_run:
            print("🧪 Running in DRY RUN mode (no actual trades)")
        
        # Run hedging for a few iterations
        print("Starting hedging loop for 3 iterations...")
        
        iteration_count = 0
        max_iterations = 3
        
        async def limited_hedging_loop():
            nonlocal iteration_count
            
            config_manager = get_config_manager(self.config_path)
            transaction_costs = config_manager.get_transaction_cost_params()
            execution_params = config_manager.get_execution_params()
            
            while iteration_count < max_iterations:
                iteration_count += 1
                print(f"\n🔄 Iteration {iteration_count}:")
                
                # Calculate optimal positions
                optimal_results = self.system.hedging_strategy.calculate_optimal_hedge_positions(
                    transaction_costs
                )
                
                # Check rebalancing
                rebalancing_decisions = self.system.hedging_strategy.check_rebalancing_triggers(
                    transaction_costs
                )
                
                # Execute rebalancing
                for instrument, (should_rebalance, reason) in rebalancing_decisions.items():
                    if should_rebalance:
                        print(f"   🎯 Rebalancing {instrument}: {reason}")
                        await self.system.hedging_strategy.execute_rebalancing(
                            instrument, execution_params, self.dry_run
                        )
                    else:
                        print(f"   ⏸️  Holding {instrument}: {reason}")
                
                # Portfolio status
                risk_metrics = self.system.hedging_strategy.calculate_portfolio_risk_metrics()
                print(f"   📊 Portfolio delta: {risk_metrics.total_delta_exposure:.3f}")
                print(f"   📊 Tracking error: {risk_metrics.portfolio_tracking_error:.3f}")
                
                # Simulate some time passing and price movement
                await asyncio.sleep(1)  # 1 second = simulated 1 hour
                
                # Simulate price movement
                import random
                price_change = random.uniform(-0.02, 0.02)  # ±2% move
                new_price = self.btc_spot_price * (1 + price_change)
                
                # Update market data
                for position in self.demo_positions:
                    self.system.update_market_data(
                        position['name'],
                        new_price,
                        position['market_data'].current_option_price * (new_price / self.btc_spot_price),
                        position['market_data'].market_implied_volatility
                    )
                
                self.btc_spot_price = new_price
                print(f"   📈 New BTC price: ${new_price:,.0f}")
        
        # Run the simulation
        await limited_hedging_loop()
        
        print("✅ Hedging simulation completed")
    
    async def _analyze_performance(self):
        """Analyze hedging performance"""
        print("\n📈 STEP 9: Performance Analysis")
        print("-" * 40)
        
        # Get performance summary
        performance = self.system.hedging_strategy.get_performance_summary()
        
        if performance.get('status') == 'success':
            print("📊 Hedging Performance:")
            print(f"   • Total rebalances: {performance['total_rebalances']}")
            print(f"   • Transaction costs: ${performance['total_transaction_costs']:.2f}")
            print(f"   • Average trade size: {performance['average_trade_size']:.4f}")
            print(f"   • Hedge effectiveness: {performance['hedge_effectiveness']:.1%}")
            print(f"   • Rebalancing frequency: {performance['rebalancing_frequency_per_day']:.1f}/day")
            print(f"   • Cost per rebalance: ${performance['cost_per_rebalance']:.2f}")
        else:
            print("📊 Performance data not available (insufficient trading history)")
        
        # Risk metrics
        risk_metrics = self.system.hedging_strategy.calculate_portfolio_risk_metrics()
        print(f"\n🛡️  Risk Metrics:")
        print(f"   • Total delta exposure: {risk_metrics.total_delta_exposure:.3f}")
        print(f"   • Total gamma exposure: {risk_metrics.total_gamma_exposure:.3f}")
        print(f"   • Total vega exposure: {risk_metrics.total_vega_exposure:.3f}")
        print(f"   • Portfolio VaR: ${risk_metrics.portfolio_var:.2f}")
        print(f"   • Concentration risk: {risk_metrics.concentration_risk:.1%}")
        
        # Export performance data
        export_success = self.system.export_performance_data("demo_performance.csv")
        if export_success:
            print(f"\n💾 Performance data exported to: demo_performance.csv")
    
    def print_final_summary(self):
        """Print final summary of Bennett's philosophy implementation"""
        print("\n" + "=" * 80)
        print("🧠 COLIN BENNETT'S DELTA HEDGING PHILOSOPHY - IMPLEMENTATION SUMMARY")
        print("=" * 80)
        print()
        print("✅ KEY INSIGHTS IMPLEMENTED:")
        print()
        print("1️⃣  FORECASTED vs IMPLIED VOLATILITY:")
        print("   💡 'Use your best estimate of future realized volatility (σ_R)'")
        print("   💡 'Market implied volatility (Σ) is just a quotation convention'")
        print("   ✅ System uses GARCH-forecasted volatility for hedge ratios")
        print("   ✅ Implied volatility only used for option pricing")
        print()
        print("2️⃣  VOLATILITY SMILE/SKEW HANDLING:")
        print("   💡 'BSM delta is often wrong for OTM options due to vol skew'")
        print("   ✅ Local volatility model adjusts deltas based on strike")
        print("   ✅ Skew adjustments incorporated for all option types")
        print()
        print("3️⃣  TRANSACTION COST OPTIMIZATION:")
        print("   💡 'Balance hedge error vs transaction costs optimally'")
        print("   ✅ Dynamic rebalancing thresholds based on costs")
        print("   ✅ Optimal trade sizing and execution timing")
        print()
        print("4️⃣  MODEL-DEPENDENT HEDGING:")
        print("   💡 'Hedge ratio is always model-dependent - choose consciously'")
        print("   ✅ Multiple delta calculation methods available")
        print("   ✅ Method selection based on market conditions")
        print()
        print("5️⃣  REGIME-AWARE ADJUSTMENTS:")
        print("   💡 'Adapt hedge strategy to current market regime'")
        print("   ✅ Volatility regime detection and classification")
        print("   ✅ Dynamic hedge adjustments based on regime")
        print()
        print("🎯 RESULT: A robust, cost-optimized delta hedging system that goes")
        print("   beyond naive BSM assumptions to deliver superior risk management.")
        print()
        print("=" * 80)


async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Optimal Delta Hedging Demo")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Run in dry-run mode (no actual trades)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--live", action="store_true", default=False,
                       help="Disable dry-run mode (enable actual trading)")
    
    args = parser.parse_args()
    
    # Determine dry-run mode
    dry_run = not args.live  # Default to dry-run unless --live specified
    
    demo = DeltaHedgingDemo(
        config_path=args.config,
        dry_run=dry_run
    )
    
    try:
        await demo.run_demo()
        demo.print_final_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        raise
    finally:
        print("\n👋 Demo finished. Thank you for exploring optimal delta hedging!")


if __name__ == "__main__":
    asyncio.run(main())