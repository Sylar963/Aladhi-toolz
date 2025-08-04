#!/usr/bin/env python3
"""
Delta Hedging Configuration Manager
===================================

Configuration loader and validator for the optimal delta hedging system.
Handles loading, validation, and type conversion of configuration parameters.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from optimal_delta_hedger import TransactionCostParams
from hedge_cost_optimizer import ExecutionParams, RebalancingTrigger
from enhanced_delta_calculator import DeltaCalculationMethod


@dataclass
class DeltaHedgingConfig:
    """Complete configuration for delta hedging system"""
    
    # Core parameters
    use_forecasted_vol_for_hedging: bool = True
    use_implied_vol_for_pricing_only: bool = True
    primary_delta_method: DeltaCalculationMethod = DeltaCalculationMethod.LOCAL_VOLATILITY
    
    # Transaction cost parameters
    commission_per_trade: float = 0.0005
    bid_ask_spread_cost: float = 0.001
    market_impact_factor: float = 0.0001
    minimum_trade_size: float = 0.01
    
    # Rebalancing parameters
    rebalancing_frequency: str = "threshold"
    delta_threshold: float = 0.05
    time_threshold_hours: float = 24.0
    vol_change_threshold: float = 0.05
    
    # Risk management
    max_delta_exposure: float = 50.0
    max_position_size: float = 10.0
    max_daily_loss: float = 1000.0
    max_drawdown: float = 0.10
    
    # Execution parameters
    max_trade_size: float = 2.0
    execution_style: str = "optimal"
    temporary_impact_factor: float = 0.0001
    permanent_impact_factor: float = 0.00005
    
    # Volatility forecasting
    vol_forecast_confidence_threshold: float = 0.5
    min_forecast_data_points: int = 100
    garch_refit_interval_hours: float = 24.0
    
    # Operational
    update_interval_seconds: float = 60.0
    enable_dry_run_mode: bool = False
    log_level: str = "INFO"
    
    # Bennett philosophy flags
    bennett_philosophy_enabled: bool = True
    handle_vol_smile_skew: bool = True
    optimize_transaction_costs: bool = True


class DeltaHedgingConfigManager:
    """Configuration manager for delta hedging system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Default config path
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config" / "delta_hedging_config.json"
        
        self.config_path = Path(config_path)
        self.config: Optional[DeltaHedgingConfig] = None
        self.raw_config: Optional[Dict[str, Any]] = None
    
    def load_config(self) -> DeltaHedgingConfig:
        """
        Load configuration from JSON file
        
        Returns:
            DeltaHedgingConfig object
        """
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Config file not found: {self.config_path}")
                return self._create_default_config()
            
            with open(self.config_path, 'r') as f:
                self.raw_config = json.load(f)
            
            # Validate and convert configuration
            self.config = self._convert_raw_config(self.raw_config)
            
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._create_default_config()
    
    def _convert_raw_config(self, raw_config: Dict[str, Any]) -> DeltaHedgingConfig:
        """Convert raw JSON config to typed configuration object"""
        
        # Extract values with defaults
        vol_forecasting = raw_config.get("volatility_forecasting", {})
        delta_calc = raw_config.get("delta_calculation", {})
        transaction_costs = raw_config.get("transaction_costs", {})
        rebalancing = raw_config.get("rebalancing_strategy", {})
        execution = raw_config.get("execution_parameters", {})
        risk_mgmt = raw_config.get("risk_management", {})
        operational = raw_config.get("operational", {})
        performance = raw_config.get("performance_tracking", {})
        bennett_flags = raw_config.get("bennett_philosophy_flags", {})
        
        # Convert delta calculation method
        method_str = delta_calc.get("primary_method", "local_volatility")
        try:
            primary_method = DeltaCalculationMethod(method_str)
        except ValueError:
            self.logger.warning(f"Unknown delta method: {method_str}, using local_volatility")
            primary_method = DeltaCalculationMethod.LOCAL_VOLATILITY
        
        return DeltaHedgingConfig(
            # Volatility forecasting
            use_forecasted_vol_for_hedging=vol_forecasting.get("use_forecasted_vol_for_hedging", True),
            use_implied_vol_for_pricing_only=vol_forecasting.get("use_implied_vol_for_pricing_only", True),
            vol_forecast_confidence_threshold=vol_forecasting.get("vol_forecast_confidence_threshold", 0.5),
            min_forecast_data_points=vol_forecasting.get("min_forecast_data_points", 100),
            garch_refit_interval_hours=vol_forecasting.get("garch_refit_interval_hours", 24.0),
            
            # Delta calculation
            primary_delta_method=primary_method,
            handle_vol_smile_skew=delta_calc.get("enable_skew_adjustment", True),
            
            # Transaction costs
            commission_per_trade=transaction_costs.get("commission_per_trade", 0.0005),
            bid_ask_spread_cost=transaction_costs.get("bid_ask_spread_cost", 0.001),
            market_impact_factor=transaction_costs.get("market_impact_factor", 0.0001),
            minimum_trade_size=transaction_costs.get("minimum_trade_size", 0.01),
            
            # Rebalancing
            rebalancing_frequency=transaction_costs.get("rebalancing_frequency", "threshold"),
            delta_threshold=transaction_costs.get("delta_threshold", 0.05),
            time_threshold_hours=transaction_costs.get("time_threshold_hours", 24.0),
            vol_change_threshold=transaction_costs.get("vol_change_threshold", 0.05),
            
            # Risk management
            max_delta_exposure=risk_mgmt.get("max_delta_exposure", 50.0),
            max_position_size=risk_mgmt.get("max_position_size", 10.0),
            max_daily_loss=risk_mgmt.get("max_daily_loss", 1000.0),
            max_drawdown=risk_mgmt.get("max_drawdown", 0.10),
            
            # Execution
            max_trade_size=execution.get("max_trade_size", 2.0),
            execution_style=execution.get("execution_style", "optimal"),
            temporary_impact_factor=execution.get("temporary_impact_factor", 0.0001),
            permanent_impact_factor=execution.get("permanent_impact_factor", 0.00005),
            
            # Operational
            update_interval_seconds=operational.get("update_interval_seconds", 60.0),
            enable_dry_run_mode=operational.get("enable_dry_run_mode", False),
            log_level=performance.get("log_level", "INFO"),
            
            # Bennett philosophy
            bennett_philosophy_enabled=bennett_flags.get("use_forecast_vol_not_implied", True),
            optimize_transaction_costs=bennett_flags.get("optimize_transaction_costs", True),
        )
    
    def _create_default_config(self) -> DeltaHedgingConfig:
        """Create default configuration"""
        self.logger.info("Using default configuration")
        return DeltaHedgingConfig()
    
    def get_transaction_cost_params(self) -> TransactionCostParams:
        """Get TransactionCostParams from configuration"""
        if not self.config:
            self.config = self.load_config()
        
        # Convert rebalancing frequency string to enum
        freq_map = {
            "time": "time_based",
            "threshold": "threshold_based", 
            "hybrid": "hybrid"
        }
        rebalancing_freq = freq_map.get(self.config.rebalancing_frequency, "threshold_based")
        
        return TransactionCostParams(
            commission_per_trade=self.config.commission_per_trade,
            bid_ask_spread_cost=self.config.bid_ask_spread_cost,
            market_impact_factor=self.config.market_impact_factor,
            minimum_trade_size=self.config.minimum_trade_size,
            rebalancing_frequency=rebalancing_freq,
            delta_threshold=self.config.delta_threshold,
            time_threshold_hours=self.config.time_threshold_hours
        )
    
    def get_execution_params(self) -> ExecutionParams:
        """Get ExecutionParams from configuration"""
        if not self.config:
            self.config = self.load_config()
        
        return ExecutionParams(
            max_position_size=self.config.max_position_size,
            max_trade_size=self.config.max_trade_size,
            min_trade_size=self.config.minimum_trade_size,
            temporary_impact_factor=self.config.temporary_impact_factor,
            permanent_impact_factor=self.config.permanent_impact_factor,
            max_drawdown_threshold=self.config.max_drawdown
        )
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get risk limits from configuration"""
        if not self.config:
            self.config = self.load_config()
        
        return {
            'max_delta_exposure': self.config.max_delta_exposure,
            'max_position_size': self.config.max_position_size,
            'max_daily_loss': self.config.max_daily_loss,
            'max_drawdown': self.config.max_drawdown
        }
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration parameters
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.config:
            return ["Configuration not loaded"]
        
        # Validate ranges
        if not (0 < self.config.commission_per_trade < 0.1):
            errors.append("commission_per_trade must be between 0 and 0.1 (10%)")
        
        if not (0 < self.config.bid_ask_spread_cost < 0.1):
            errors.append("bid_ask_spread_cost must be between 0 and 0.1 (10%)")
        
        if not (0.001 <= self.config.delta_threshold <= 0.5):
            errors.append("delta_threshold must be between 0.001 and 0.5")
        
        if not (0.1 <= self.config.time_threshold_hours <= 168):
            errors.append("time_threshold_hours must be between 0.1 and 168 hours")
        
        if not (0.001 <= self.config.minimum_trade_size <= 10):
            errors.append("minimum_trade_size must be between 0.001 and 10")
        
        if not (1 <= self.config.max_position_size <= 1000):
            errors.append("max_position_size must be between 1 and 1000")
        
        if not (0.01 <= self.config.max_drawdown <= 0.5):
            errors.append("max_drawdown must be between 0.01 (1%) and 0.5 (50%)")
        
        if not (1 <= self.config.update_interval_seconds <= 3600):
            errors.append("update_interval_seconds must be between 1 and 3600 seconds")
        
        # Validate Bennett philosophy consistency
        if not self.config.bennett_philosophy_enabled:
            if self.config.use_forecasted_vol_for_hedging:
                errors.append("Inconsistent config: Bennett philosophy disabled but forecast vol enabled")
        
        # Validate logical consistency
        if self.config.max_trade_size < self.config.minimum_trade_size:
            errors.append("max_trade_size must be >= minimum_trade_size")
        
        if self.config.delta_threshold > 0.2 and self.config.optimize_transaction_costs:
            errors.append("Large delta_threshold (>20%) with cost optimization may be inefficient")
        
        return errors
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration parameters
        
        Args:
            updates: Dictionary of parameter updates
            
        Returns:
            True if update successful
        """
        try:
            if not self.raw_config:
                self.load_config()
            
            if not self.raw_config:
                return False
            
            # Apply updates to raw config
            for key, value in updates.items():
                if '.' in key:  # Nested key like "transaction_costs.delta_threshold"
                    section, param = key.split('.', 1)
                    if section not in self.raw_config:
                        self.raw_config[section] = {}
                    self.raw_config[section][param] = value
                else:
                    self.raw_config[key] = value
            
            # Reload configuration
            self.config = self._convert_raw_config(self.raw_config)
            
            # Validate updated config
            errors = self.validate_config()
            if errors:
                self.logger.error(f"Configuration validation errors after update: {errors}")
                return False
            
            self.logger.info(f"Updated configuration parameters: {list(updates.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """
        Save current configuration to file
        
        Args:
            path: Optional path to save to (defaults to original path)
            
        Returns:
            True if save successful
        """
        try:
            save_path = Path(path) if path else self.config_path
            
            if not self.raw_config:
                self.logger.error("No configuration to save")
                return False
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.raw_config, f, indent=2)
            
            self.logger.info(f"Saved configuration to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def log_config_summary(self):
        """Log summary of current configuration"""
        if not self.config:
            self.config = self.load_config()
        
        self.logger.info("="*60)
        self.logger.info("DELTA HEDGING CONFIGURATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Bennett Philosophy Enabled: {self.config.bennett_philosophy_enabled}")
        self.logger.info(f"Use Forecasted Vol for Hedging: {self.config.use_forecasted_vol_for_hedging}")
        self.logger.info(f"Primary Delta Method: {self.config.primary_delta_method.value}")
        self.logger.info(f"Delta Threshold: {self.config.delta_threshold:.3f}")
        self.logger.info(f"Time Threshold: {self.config.time_threshold_hours:.1f} hours")
        self.logger.info(f"Commission per Trade: {self.config.commission_per_trade:.4f}")
        self.logger.info(f"Max Position Size: {self.config.max_position_size:.1f}")
        self.logger.info(f"Max Daily Loss: ${self.config.max_daily_loss:.0f}")
        self.logger.info(f"Update Interval: {self.config.update_interval_seconds:.0f}s")
        self.logger.info(f"Dry Run Mode: {self.config.enable_dry_run_mode}")
        self.logger.info("="*60)


# Global configuration manager instance
_config_manager: Optional[DeltaHedgingConfigManager] = None

def get_config_manager(config_path: Optional[str] = None) -> DeltaHedgingConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = DeltaHedgingConfigManager(config_path)
    return _config_manager

def load_config(config_path: Optional[str] = None) -> DeltaHedgingConfig:
    """Convenience function to load configuration"""
    return get_config_manager(config_path).load_config()