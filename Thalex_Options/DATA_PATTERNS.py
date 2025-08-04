"""
Data Handling Patterns & Templates
==================================

This file contains proven patterns and templates for common data handling scenarios
in the Thalex Options project. Use these as starting points to avoid common bugs.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Union, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

import matplotlib.dates as mdates


# =============================================================================
# 1. DATA STRUCTURE PATTERNS
# =============================================================================

@dataclass
class OHLCBar:
    """Standard OHLC bar structure - single source of truth"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_matplotlib_tuple(self) -> tuple:
        """Convert to matplotlib-compatible format"""
        return (mdates.date2num(self.timestamp), self.open, self.high, self.low, self.close, self.volume)
    
    def is_valid(self) -> bool:
        """Validate OHLC data integrity"""
        return (
            self.high >= max(self.open, self.close) and
            self.low <= min(self.open, self.close) and
            self.volume >= 0
        )


@dataclass
class MarketDataPoint:
    """Generic market data structure with validation"""
    timestamp: float
    value: float
    metadata: Dict[str, Any]
    
    def is_stale(self, max_age_seconds: int = 60) -> bool:
        """Check if data is too old"""
        return time.time() - self.timestamp > max_age_seconds


class DataStatus(Enum):
    """Data quality status tracking"""
    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"
    MISSING = "missing"


# =============================================================================
# 2. PROTOCOL PATTERNS (Interface Definitions)
# =============================================================================

class DataSource(Protocol):
    """Protocol for data source implementations"""
    async def connect(self) -> bool: ...
    async def get_ticker_data(self, instrument: str) -> Optional[Dict]: ...
    async def disconnect(self) -> None: ...


class DataValidator(Protocol):
    """Protocol for data validation"""
    def validate(self, data: Any) -> bool: ...
    def get_errors(self) -> List[str]: ...


# =============================================================================
# 3. SAFE API WRAPPER PATTERNS
# =============================================================================

class SafeAPIWrapper:
    """Wrapper for external API calls with error handling and retry logic"""
    
    def __init__(self, api_client, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_client = api_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts = defaultdict(int)
    
    async def safe_call(self, method_name: str, **kwargs) -> Optional[Any]:
        """Make API call with error handling and retries"""
        for attempt in range(self.max_retries + 1):
            try:
                method = getattr(self.api_client, method_name)
                result = await method(**kwargs)
                
                # Reset error count on success
                self.error_counts[method_name] = 0
                return result
                
            except Exception as e:
                self.error_counts[method_name] += 1
                logging.error(f"API call failed: {method_name}, attempt {attempt + 1}, error: {e}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logging.error(f"Max retries exceeded for {method_name}")
                    return None


# =============================================================================
# 4. DATA VALIDATION PATTERNS
# =============================================================================

class OptionDataValidator:
    """Validator for options market data"""
    
    @staticmethod
    def validate_ticker_data(data: Dict) -> tuple[bool, List[str]]:
        """Validate ticker data structure and values"""
        errors = []
        
        # Required fields
        required_fields = ["instrument_name", "mark_price"]
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Value ranges
        if "mark_price" in data:
            price = data["mark_price"]
            if not isinstance(price, (int, float)) or price <= 0:
                errors.append(f"Invalid mark_price: {price}")
        
        if "iv" in data:
            iv = data["iv"]
            if not isinstance(iv, (int, float)) or not (0 <= iv <= 10):
                errors.append(f"Invalid IV (should be 0-10): {iv}")
        
        if "delta" in data:
            delta = data["delta"]
            if not isinstance(delta, (int, float)) or not (-1 <= delta <= 1):
                errors.append(f"Invalid delta (should be -1 to 1): {delta}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_ohlc_data(ohlc: OHLCBar) -> tuple[bool, List[str]]:
        """Validate OHLC bar data"""
        errors = []
        
        # OHLC relationships
        if ohlc.high < max(ohlc.open, ohlc.close):
            errors.append("High price is less than open/close")
        
        if ohlc.low > min(ohlc.open, ohlc.close):
            errors.append("Low price is greater than open/close")
        
        # Reasonable values
        if any(price <= 0 for price in [ohlc.open, ohlc.high, ohlc.low, ohlc.close]):
            errors.append("Invalid price values (must be > 0)")
        
        if ohlc.volume < 0:
            errors.append("Volume cannot be negative")
        
        return len(errors) == 0, errors


# =============================================================================
# 5. DATA COLLECTION PATTERNS
# =============================================================================

class RobustDataCollector:
    """Template for robust data collection with error handling"""
    
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self.data_cache = {}
        self.error_counts = defaultdict(int)
        self.last_successful_update = {}
        
        # Data storage
        self.ohlc_data = deque(maxlen=1000)
        self.volume_profile = defaultdict(float)
        
        # Quality tracking
        self.data_quality = defaultdict(lambda: DataStatus.MISSING)
    
    async def collect_data(self, instruments: List[str]) -> Dict[str, Any]:
        """Collect data for multiple instruments safely"""
        results = {}
        
        for instrument in instruments:
            try:
                data = await self._collect_single_instrument(instrument)
                if data:
                    results[instrument] = data
                    self.data_quality[instrument] = DataStatus.VALID
                    self.last_successful_update[instrument] = time.time()
                else:
                    self.data_quality[instrument] = DataStatus.INVALID
                    
            except Exception as e:
                logging.error(f"Failed to collect data for {instrument}: {e}")
                self.error_counts[instrument] += 1
                self.data_quality[instrument] = DataStatus.INVALID
        
        return results
    
    async def _collect_single_instrument(self, instrument: str) -> Optional[Dict]:
        """Collect data for single instrument with validation"""
        raw_data = await self.data_source.get_ticker_data(instrument)
        
        if not raw_data:
            return None
        
        # Validate data
        is_valid, errors = OptionDataValidator.validate_ticker_data(raw_data)
        if not is_valid:
            logging.warning(f"Invalid data for {instrument}: {errors}")
            return None
        
        return raw_data
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate data quality report"""
        total_instruments = len(self.data_quality)
        quality_counts = defaultdict(int)
        
        for status in self.data_quality.values():
            quality_counts[status.value] += 1
        
        return {
            "total_instruments": total_instruments,
            "quality_breakdown": dict(quality_counts),
            "error_counts": dict(self.error_counts),
            "health_score": quality_counts[DataStatus.VALID.value] / max(total_instruments, 1)
        }


# =============================================================================
# 6. OHLC BAR BUILDING PATTERN
# =============================================================================

class OHLCBarBuilder:
    """Robust OHLC bar building from tick data"""
    
    def __init__(self, bar_interval_minutes: int = 60):
        self.bar_interval_minutes = bar_interval_minutes
        self.current_bar = None
        self.completed_bars = deque(maxlen=1000)
        
    async def process_price_update(self, price: float, volume: float, timestamp: Optional[datetime] = None) -> Optional[OHLCBar]:
        """Process price update and return completed bar if any"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Validate input
        if price <= 0:
            logging.warning(f"Invalid price: {price}")
            return None
        
        # Initialize or update current bar
        completed_bar = None
        
        if self._should_start_new_bar(timestamp):
            # Complete current bar
            if self.current_bar and self.current_bar.is_valid():
                completed_bar = self.current_bar
                self.completed_bars.append(completed_bar)
            
            # Start new bar
            self.current_bar = OHLCBar(
                timestamp=timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume
            )
        else:
            # Update current bar
            if self.current_bar:
                self.current_bar.high = max(self.current_bar.high, price)
                self.current_bar.low = min(self.current_bar.low, price)
                self.current_bar.close = price
                self.current_bar.volume = max(self.current_bar.volume, volume)  # Use max for 24h volume
        
        return completed_bar
    
    def _should_start_new_bar(self, timestamp: datetime) -> bool:
        """Determine if we should start a new bar"""
        if not self.current_bar:
            return True
        
        time_diff = timestamp - self.current_bar.timestamp
        return time_diff.total_seconds() >= self.bar_interval_minutes * 60
    
    def get_recent_bars(self, count: int = 50) -> List[OHLCBar]:
        """Get recent completed bars"""
        return list(self.completed_bars)[-count:]


# =============================================================================
# 7. CONFIGURATION PATTERN
# =============================================================================

@dataclass
class TradingConfig:
    """Environment-specific configuration"""
    network_name: str
    underlying: str
    bar_interval_minutes: int
    max_data_age_seconds: int
    debug_mode: bool
    
    @classmethod
    def for_environment(cls, env: str = "test") -> 'TradingConfig':
        """Create config for specific environment"""
        configs = {
            "test": cls(
                network_name="test",
                underlying="BTCUSD",
                bar_interval_minutes=60,
                max_data_age_seconds=300,
                debug_mode=True
            ),
            "prod": cls(
                network_name="prod",
                underlying="BTCUSD", 
                bar_interval_minutes=15,
                max_data_age_seconds=60,
                debug_mode=False
            )
        }
        return configs.get(env, configs["test"])


# =============================================================================
# 8. PERFORMANCE MONITORING PATTERN
# =============================================================================

class PerformanceMonitor:
    """Monitor performance and data processing metrics"""
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.data_points_processed = defaultdict(int)
        self.start_time = time.time()
    
    def time_operation(self, operation_name: str):
        """Decorator to time operations"""
        def decorator(func: Callable) -> Callable:
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start
                    self.operation_times[operation_name].append(duration)
                    return result
                except Exception as e:
                    self.error_counts[operation_name] += 1
                    raise
            
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    self.operation_times[operation_name].append(duration)
                    return result
                except Exception as e:
                    self.error_counts[operation_name] += 1
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {
            "uptime_seconds": time.time() - self.start_time,
            "operations": {}
        }
        
        for op_name, times in self.operation_times.items():
            if times:
                report["operations"][op_name] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "max_time": max(times),
                    "min_time": min(times),
                    "errors": self.error_counts[op_name],
                    "success_rate": (len(times) - self.error_counts[op_name]) / len(times)
                }
        
        return report


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

async def example_usage():
    """Example of how to use these patterns"""
    
    # Configuration
    config = TradingConfig.for_environment("test")
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    # Data collection (you'd implement ThalexDataSource)
    # data_source = ThalexDataSource(config)
    # collector = RobustDataCollector(data_source)
    
    # OHLC bar building
    bar_builder = OHLCBarBuilder(config.bar_interval_minutes)
    
    # Example price processing
    @monitor.time_operation("process_price")
    async def process_price_safely(price: float, volume: float):
        try:
            # Validate input
            if price <= 0:
                raise ValueError(f"Invalid price: {price}")
            
            # Process price update
            completed_bar = await bar_builder.process_price_update(price, volume)
            
            if completed_bar:
                logging.info(f"Completed OHLC bar: {completed_bar}")
            
            return completed_bar
            
        except Exception as e:
            logging.error(f"Price processing failed: {e}")
            return None
    
    # Simulate some price updates
    for price in [50000, 50100, 49900, 50200]:
        await process_price_safely(price, 1000)
    
    # Generate reports
    print("Performance Report:")
    print(monitor.get_performance_report())


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())