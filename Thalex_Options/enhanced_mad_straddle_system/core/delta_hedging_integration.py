#!/usr/bin/env python3
"""
Delta Hedging Integration Module
================================

Integration layer that connects the optimal delta hedging system with
existing Thalex_Options infrastructure components:

- Forward volatility estimation (GARCH models)
- Volatility surface construction
- Volatility regime detection  
- Historical data management
- Real-time market data feeds

This module provides a unified interface for initializing and coordinating
all components of the Bennett-philosophy delta hedging system.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import existing Thalex infrastructure
from forward_volatility import ForwardVolatilityEstimator, VolatilityForecast
from volatility_surface import VolatilitySurface, VolatilityPoint
from volatility_regime import VolatilityRegimeDetector, RegimeMetrics
from financial_math import FinancialMath
from supabase_data_manager import SupabaseDataManager, PriceDataPoint, OptionDataPoint

# Import our new delta hedging components
from optimal_delta_hedger import OptimalDeltaHedger, OptionParams, MarketData, OptionType
from enhanced_delta_calculator import EnhancedDeltaCalculator, DeltaCalculationMethod
from hedge_cost_optimizer import HedgeCostOptimizer
from delta_hedging_strategy import DeltaHedgingStrategy
from delta_hedging_config import DeltaHedgingConfigManager, DeltaHedgingConfig


class DeltaHedgingIntegrationError(Exception):
    """Exception raised by integration module"""
    pass


class IntegratedDeltaHedgingSystem:
    """
    Integrated delta hedging system that combines all components
    
    This is the main entry point for using the optimal delta hedging system
    with full integration to existing Thalex infrastructure.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enable_database: bool = True,
                 spot_price: float = 50000.0):  # Default BTC price
        """
        Initialize the integrated delta hedging system
        
        Args:
            config_path: Path to configuration file
            enable_database: Enable database integration
            spot_price: Current spot price for initialization
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_manager = DeltaHedgingConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize core infrastructure components
        self.data_manager: Optional[SupabaseDataManager] = None
        self.vol_estimator: Optional[ForwardVolatilityEstimator] = None
        self.vol_surface: Optional[VolatilitySurface] = None
        self.regime_detector: Optional[VolatilityRegimeDetector] = None
        
        # Initialize delta hedging components
        self.optimal_hedger: Optional[OptimalDeltaHedger] = None
        self.enhanced_calculator: Optional[EnhancedDeltaCalculator] = None
        self.cost_optimizer: Optional[HedgeCostOptimizer] = None
        self.hedging_strategy: Optional[DeltaHedgingStrategy] = None
        
        # System state
        self.is_initialized = False
        self.spot_price = spot_price
        self.enable_database = enable_database
        
        self.logger.info("Created IntegratedDeltaHedgingSystem")
    
    async def initialize(self) -> bool:
        """
        Initialize all components of the delta hedging system
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Integrated Delta Hedging System...")
            
            # Step 1: Initialize data management (if enabled)
            if self.enable_database:
                await self._initialize_data_manager()
            
            # Step 2: Initialize volatility infrastructure
            await self._initialize_volatility_components()
            
            # Step 3: Initialize delta hedging components
            self._initialize_delta_hedging_components()
            
            # Step 4: Connect components
            self._connect_components()
            
            # Step 5: Validate system
            validation_result = await self._validate_system()
            if not validation_result:
                raise DeltaHedgingIntegrationError("System validation failed")
            
            self.is_initialized = True
            self.logger.info("✅ Integrated Delta Hedging System initialized successfully")
            
            # Log system summary
            self._log_system_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Integrated Delta Hedging System: {e}")
            return False
    
    async def _initialize_data_manager(self):
        """Initialize database data manager"""
        try:
            self.data_manager = SupabaseDataManager()
            await self.data_manager.initialize()
            self.logger.info("✅ Data manager initialized")
        except Exception as e:
            self.logger.warning(f"⚠️  Data manager initialization failed: {e}")
            self.data_manager = None
    
    async def _initialize_volatility_components(self):
        """Initialize volatility-related components"""
        try:
            # Initialize volatility regime detector
            self.regime_detector = VolatilityRegimeDetector(
                lookback_days=self.config.min_forecast_data_points // 4  # Rough approximation
            )
            
            # Initialize volatility surface
            self.vol_surface = VolatilitySurface(spot_price=self.spot_price)
            
            # Initialize forward volatility estimator
            self.vol_estimator = ForwardVolatilityEstimator(
                regime_detector=self.regime_detector
            )
            
            self.logger.info("✅ Volatility components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize volatility components: {e}")
            raise
    
    def _initialize_delta_hedging_components(self):
        """Initialize delta hedging specific components"""
        try:
            # Initialize optimal delta hedger with volatility components
            self.optimal_hedger = OptimalDeltaHedger(
                vol_estimator=self.vol_estimator,
                vol_surface=self.vol_surface,
                regime_detector=self.regime_detector
            )
            
            # Initialize enhanced delta calculator
            self.enhanced_calculator = EnhancedDeltaCalculator(
                vol_surface=self.vol_surface
            )
            
            # Initialize cost optimizer
            self.cost_optimizer = HedgeCostOptimizer()
            
            # Initialize main hedging strategy
            self.hedging_strategy = DeltaHedgingStrategy(
                vol_estimator=self.vol_estimator,
                vol_surface=self.vol_surface,
                regime_detector=self.regime_detector,
                enable_logging=True
            )
            
            self.logger.info("✅ Delta hedging components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize delta hedging components: {e}")
            raise
    
    def _connect_components(self):
        """Connect components and configure cross-dependencies"""
        try:
            # Update hedging strategy risk limits from config
            if self.hedging_strategy:
                self.hedging_strategy.risk_limits = self.config_manager.get_risk_limits()
            
            self.logger.info("✅ Components connected")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect components: {e}")
            raise
    
    async def _validate_system(self) -> bool:
        """Validate that all components are working correctly"""
        try:
            validation_errors = []
            
            # Validate configuration
            config_errors = self.config_manager.validate_config()
            if config_errors:
                validation_errors.extend([f"Config: {err}" for err in config_errors])
            
            # Test volatility forecasting with dummy data
            if self.vol_estimator:
                dummy_returns = [0.01, -0.005, 0.02, -0.01, 0.005] * 20  # 100 returns
                try:
                    forecast = self.vol_estimator.forecast_volatility(30, dummy_returns)
                    if forecast is None:
                        validation_errors.append("Vol forecasting: No forecast generated")
                except Exception as e:
                    validation_errors.append(f"Vol forecasting: {e}")
            
            # Test optimal delta calculation with dummy data
            if self.optimal_hedger:
                try:
                    option_params = OptionParams(
                        current_spot_price=self.spot_price,
                        strike_price=self.spot_price,
                        time_to_expiration=0.25,  # 3 months
                        risk_free_rate=0.05,
                        dividend_yield=0.0,
                        option_type=OptionType.CALL
                    )
                    
                    market_data = MarketData(
                        market_implied_volatility=0.5,
                        current_option_price=2500.0
                    )
                    
                    result = self.optimal_hedger.calculate_optimal_delta(
                        option_params, market_data, 0.6  # 60% forecast vol
                    )
                    
                    if abs(result.optimal_delta) > 2.0:  # Sanity check
                        validation_errors.append(f"Optimal delta out of range: {result.optimal_delta}")
                        
                except Exception as e:
                    validation_errors.append(f"Optimal delta calculation: {e}")
            
            # Test hedging strategy
            if self.hedging_strategy:
                try:
                    # Add a test position
                    test_success = self.hedging_strategy.add_position(
                        "TEST_BTC_CALL",
                        option_params,
                        1.0  # 1 contract
                    )
                    if not test_success:
                        validation_errors.append("Hedging strategy: Failed to add test position")
                    else:
                        # Remove test position
                        del self.hedging_strategy.hedge_positions["TEST_BTC_CALL"]
                        
                except Exception as e:
                    validation_errors.append(f"Hedging strategy: {e}")
            
            if validation_errors:
                self.logger.error(f"❌ System validation failed: {validation_errors}")
                return False
            
            self.logger.info("✅ System validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System validation error: {e}")
            return False
    
    def _log_system_summary(self):
        """Log summary of initialized system"""
        self.logger.info("="*80)
        self.logger.info("INTEGRATED DELTA HEDGING SYSTEM - INITIALIZATION SUMMARY")
        self.logger.info("="*80)
        self.logger.info("🎯 PHILOSOPHY: Colin Bennett's Optimal Delta Hedging")
        self.logger.info("")
        self.logger.info("📊 CORE COMPONENTS:")
        self.logger.info(f"   ✅ Optimal Delta Hedger: {self.optimal_hedger is not None}")
        self.logger.info(f"   ✅ Enhanced Delta Calculator: {self.enhanced_calculator is not None}")
        self.logger.info(f"   ✅ Hedge Cost Optimizer: {self.cost_optimizer is not None}")
        self.logger.info(f"   ✅ Delta Hedging Strategy: {self.hedging_strategy is not None}")
        self.logger.info("")
        self.logger.info("📈 VOLATILITY INFRASTRUCTURE:")
        self.logger.info(f"   ✅ Forward Vol Estimator: {self.vol_estimator is not None}")
        self.logger.info(f"   ✅ Volatility Surface: {self.vol_surface is not None}")
        self.logger.info(f"   ✅ Regime Detector: {self.regime_detector is not None}")
        self.logger.info(f"   ✅ Data Manager: {self.data_manager is not None}")
        self.logger.info("")
        self.logger.info("⚙️  CONFIGURATION:")
        self.logger.info(f"   🔬 Use Forecast Vol for Hedging: {self.config.use_forecasted_vol_for_hedging}")
        self.logger.info(f"   📊 Primary Delta Method: {self.config.primary_delta_method.value}")
        self.logger.info(f"   🎯 Delta Threshold: {self.config.delta_threshold:.3f}")
        self.logger.info(f"   ⏱️  Time Threshold: {self.config.time_threshold_hours:.1f}h")
        self.logger.info(f"   💰 Commission per Trade: {self.config.commission_per_trade:.4f}")
        self.logger.info(f"   🛡️  Max Position Size: {self.config.max_position_size}")
        self.logger.info(f"   🚨 Max Daily Loss: ${self.config.max_daily_loss:.0f}")
        self.logger.info("")
        self.logger.info("🧠 BENNETT'S KEY INSIGHTS:")
        self.logger.info("   💡 Use forecasted σ_R, not market implied Σ for hedge ratios")
        self.logger.info("   💡 BSM delta often wrong for OTM options due to vol skew")
        self.logger.info("   💡 Balance hedge error vs transaction costs optimally")
        self.logger.info("   💡 Hedge ratio is always model-dependent - choose consciously")
        self.logger.info("="*80)
    
    def update_market_data(self, 
                          instrument_name: str,
                          spot_price: float,
                          option_price: float,
                          implied_vol: float,
                          timestamp: Optional[float] = None) -> bool:
        """
        Update market data across all components
        
        Args:
            instrument_name: Instrument identifier
            spot_price: Current spot price
            option_price: Current option price
            implied_vol: Implied volatility
            timestamp: Timestamp (current time if None)
            
        Returns:
            True if update successful
        """
        if not self.is_initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # Update spot price for vol surface
            if self.vol_surface:
                self.vol_surface.update_spot_price(spot_price)
            
            # Update regime detector with price data
            if self.regime_detector:
                self.regime_detector.add_price_observation(spot_price, timestamp)
            
            # Update hedging strategy
            if self.hedging_strategy:
                return self.hedging_strategy.update_market_data(
                    instrument_name, spot_price, option_price, implied_vol
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update market data: {e}")
            return False
    
    def add_volatility_point(self, 
                           strike: float,
                           expiry_timestamp: float,
                           implied_vol: float,
                           confidence: float = 1.0) -> bool:
        """
        Add volatility point to surface
        
        Args:
            strike: Strike price
            expiry_timestamp: Expiry timestamp
            implied_vol: Implied volatility
            confidence: Confidence in the data point
            
        Returns:
            True if added successfully
        """
        if not self.vol_surface:
            return False
        
        try:
            self.vol_surface.add_market_vol_point(
                strike=strike,
                expiry_timestamp=expiry_timestamp,
                implied_vol=implied_vol,
                confidence=confidence
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add volatility point: {e}")
            return False
    
    def get_optimal_delta(self, 
                         instrument_name: str,
                         option_params: OptionParams,
                         market_data: MarketData) -> Optional[Dict[str, Any]]:
        """
        Get optimal delta for an instrument
        
        Args:
            instrument_name: Instrument identifier
            option_params: Option parameters
            market_data: Market data
            
        Returns:
            Dictionary with optimal delta results
        """
        if not self.is_initialized or not self.optimal_hedger:
            return None
        
        try:
            # Get forecast volatility
            historical_returns = []
            if self.regime_detector and len(self.regime_detector.return_history) > 50:
                historical_returns = list(self.regime_detector.return_history)
            
            # Use reasonable forecast if no historical data
            forecast_vol = 0.6  # 60% default
            if self.vol_estimator and historical_returns:
                forecast = self.vol_estimator.forecast_volatility(
                    option_params.time_to_expiration * 365.25,
                    historical_returns
                )
                if forecast:
                    forecast_vol = forecast.expected_volatility
            
            # Calculate optimal delta
            result = self.optimal_hedger.calculate_optimal_delta(
                option_params=option_params,
                market_data=market_data,
                forecast_volatility=forecast_vol,
                transaction_costs=self.config_manager.get_transaction_cost_params(),
                historical_returns=historical_returns
            )
            
            return {
                'optimal_delta': result.optimal_delta,
                'bsm_delta': result.current_bsm_delta,
                'adjustment_factor': result.hedge_adjustment_factor,
                'forecast_volatility': result.forecast_volatility,
                'implied_volatility': result.market_implied_volatility,
                'regime': result.volatility_regime,
                'confidence': result.vol_forecast_confidence,
                'rebalance_threshold': result.recommended_rebalance_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal delta for {instrument_name}: {e}")
            return None
    
    async def start_hedging(self, dry_run: bool = False) -> bool:
        """
        Start the hedging system
        
        Args:
            dry_run: If True, simulate but don't execute trades
            
        Returns:
            True if started successfully
        """
        if not self.is_initialized:
            self.logger.error("System not initialized")
            return False
        
        if not self.hedging_strategy:
            self.logger.error("Hedging strategy not available")
            return False
        
        try:
            self.logger.info(f"🚀 Starting delta hedging system (dry_run={dry_run})")
            
            # Start the hedging loop
            await self.hedging_strategy.run_hedging_loop(
                transaction_costs=self.config_manager.get_transaction_cost_params(),
                execution_params=self.config_manager.get_execution_params(),
                update_interval_seconds=self.config.update_interval_seconds,
                dry_run=dry_run
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start hedging system: {e}")
            return False
    
    def stop_hedging(self):
        """Stop the hedging system"""
        if self.hedging_strategy:
            self.hedging_strategy.stop_hedging_loop()
            self.logger.info("🛑 Delta hedging system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'initialized': self.is_initialized,
            'components': {
                'optimal_hedger': self.optimal_hedger is not None,
                'enhanced_calculator': self.enhanced_calculator is not None,
                'cost_optimizer': self.cost_optimizer is not None,
                'hedging_strategy': self.hedging_strategy is not None,
                'vol_estimator': self.vol_estimator is not None,
                'vol_surface': self.vol_surface is not None,
                'regime_detector': self.regime_detector is not None,
                'data_manager': self.data_manager is not None
            },
            'config': {
                'bennett_philosophy': self.config.bennett_philosophy_enabled,
                'use_forecast_vol': self.config.use_forecasted_vol_for_hedging,
                'delta_method': self.config.primary_delta_method.value,
                'delta_threshold': self.config.delta_threshold,
                'dry_run': self.config.enable_dry_run_mode
            }
        }
        
        # Add hedging strategy status if available
        if self.hedging_strategy:
            status['hedging'] = {
                'running': self.hedging_strategy.is_running,
                'positions': len(self.hedging_strategy.hedge_positions),
                'total_pnl': self.hedging_strategy.performance_metrics.get('total_pnl', 0),
                'total_costs': self.hedging_strategy.performance_metrics.get('total_transaction_costs', 0)
            }
        
        # Add volatility regime info if available
        if self.regime_detector:
            status['regime'] = {
                'current_regime': self.regime_detector.current_regime.value,
                'data_points': len(self.regime_detector.return_history)
            }
        
        return status
    
    def export_performance_data(self, filename: Optional[str] = None) -> bool:
        """Export performance data"""
        if not self.hedging_strategy:
            return False
        
        if filename is None:
            filename = f"bennett_hedging_performance_{int(time.time())}.csv"
        
        self.hedging_strategy.export_performance_data(filename)
        return True


# Convenience function for easy initialization
async def create_integrated_system(config_path: Optional[str] = None,
                                 spot_price: float = 50000.0,
                                 enable_database: bool = True) -> IntegratedDeltaHedgingSystem:
    """
    Create and initialize an integrated delta hedging system
    
    Args:
        config_path: Path to configuration file
        spot_price: Current spot price
        enable_database: Enable database integration
        
    Returns:
        Initialized IntegratedDeltaHedgingSystem
    """
    system = IntegratedDeltaHedgingSystem(
        config_path=config_path,
        enable_database=enable_database,
        spot_price=spot_price
    )
    
    success = await system.initialize()
    if not success:
        raise DeltaHedgingIntegrationError("Failed to initialize integrated system")
    
    return system