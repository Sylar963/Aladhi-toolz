# Bug Registry & Data Handling Guidelines

This document tracks bugs encountered during development and establishes patterns for proper data handling to avoid repeating mistakes.

## 🐛 Bug Categories

### 1. **Data Structure & Type Issues**

#### BUG-001: matplotlib Incompatible Types
- **Issue**: `pd.Timedelta` passed to matplotlib `bar()` width parameter
- **Root Cause**: matplotlib expects numeric width, not pandas Timedelta objects
- **Fix**: Convert to fraction of day: `bar_width = 0.8 / 24`
- **Prevention**: Always check matplotlib documentation for expected parameter types
- **Pattern**: Use `matplotlib.dates.date2num()` for datetime conversions

```python
# ❌ Wrong
width=pd.Timedelta(hours=0.8)

# ✅ Correct  
bar_width = 0.8 / 24  # fraction of day
```

#### BUG-002: Duplicate Data Structures
- **Issue**: Multiple overlapping data structures (`volume_data`, `price_history`, `candlestick_data`)
- **Root Cause**: Incremental development without refactoring
- **Fix**: Consolidate into single `ohlc_data` deque with clear schema
- **Prevention**: Design data schema upfront, avoid ad-hoc additions
- **Pattern**: Single source of truth for each data type

```python
# ❌ Wrong - Multiple sources
self.volume_data = {}
self.price_history = []
self.candlestick_data = []

# ✅ Correct - Single source
self.ohlc_data = deque(maxlen=1000)  # (timestamp, O, H, L, C, volume)
```

### 2. **Import & Dependency Issues**

#### BUG-003: Import Path Errors
- **Issue**: `import thalex` failed, should be `import thalex_py.thalex as th`
- **Root Cause**: Package structure not understood
- **Fix**: Use correct import path from package documentation
- **Prevention**: Always verify import paths when setting up new packages
- **Pattern**: Check `__init__.py` and package structure first

#### BUG-004: Unused Import Accumulation
- **Issue**: Imports like `sys`, `numpy`, `Union`, `List` not actually used
- **Root Cause**: Copy-paste programming and incremental changes
- **Fix**: Regular import cleanup using IDE diagnostics
- **Prevention**: Use IDE warnings, run linters regularly
- **Pattern**: Import only what you use

### 3. **Data Flow & Processing Issues**

#### BUG-005: Synthetic vs Real Data Confusion
- **Issue**: Using generated fake data instead of actual Thalex feeds
- **Root Cause**: Testing code left in production logic
- **Fix**: Separate test data generation from production data processing
- **Prevention**: Clear separation between mock/test and production code paths
- **Pattern**: Use dependency injection for data sources

```python
# ❌ Wrong - Mixed synthetic and real data
def update_data(self):
    if not self.real_data_available:
        self.generate_fake_ohlc()  # Test code in production!
    else:
        self.process_real_data()

# ✅ Correct - Clear separation
class DataCollector:
    def __init__(self, data_source):
        self.data_source = data_source  # RealThalexSource or MockSource
```

#### BUG-006: Volume Data Aggregation Logic
- **Issue**: Adding volumes incorrectly, not using max or proper weighting
- **Root Cause**: Misunderstanding of how volume should be aggregated
- **Fix**: Use `max()` for 24h volume, weighted averages for options volume
- **Prevention**: Understand the domain before implementing calculations
- **Pattern**: Document calculation logic in code comments

```python
# ❌ Wrong - Adding volumes from same source
self.current_bar_data['volume'] += volume  

# ✅ Correct - Use max for 24h volume snapshots
self.current_bar_data['volume'] = max(self.current_bar_data['volume'], volume)
```

### 4. **WebSocket & API Issues**

#### BUG-007: Missing Error Handling in WebSocket Processing
- **Issue**: WebSocket message processing without try-catch blocks
- **Root Cause**: Happy path programming
- **Fix**: Wrap all message processing in comprehensive error handling
- **Prevention**: Always plan for network failures and malformed data
- **Pattern**: Graceful degradation and logging

```python
# ❌ Wrong - No error handling
async def process_ticker_update(self, params):
    data = params["data"]
    self.update_option(data["instrument_name"], data)

# ✅ Correct - Comprehensive error handling
async def process_ticker_update(self, params):
    try:
        data = params.get("data", {})
        instrument_name = data.get("instrument_name")
        if not instrument_name:
            logging.warning(f"Missing instrument_name in ticker data: {params}")
            return
        self.update_option(instrument_name, data)
    except Exception as e:
        logging.error(f"Failed to process ticker update: {e}, data: {params}")
```

## 📋 Data Handling Best Practices

### 1. **Data Schema Design**

```python
# Define clear data structures upfront
@dataclass
class OHLCBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Convert to matplotlib-compatible format"""
        return (mdates.date2num(self.timestamp), self.open, self.high, self.low, self.close, self.volume)
```

### 2. **Type Safety Patterns**

```python
from typing import Dict, List, Optional, Union, Protocol

class DataSource(Protocol):
    async def get_ticker_data(self, instrument: str) -> Optional[Dict]:
        ...

# Use type hints consistently
def process_options_data(self, options: Dict[str, OptionData]) -> List[StraddleRange]:
    ...
```

### 3. **Error Handling Patterns**

```python
# Pattern: Graceful degradation
async def safe_api_call(self, method_name: str, **kwargs) -> Optional[Dict]:
    try:
        result = await getattr(self.thalex, method_name)(**kwargs)
        return result
    except Exception as e:
        logging.error(f"API call failed: {method_name}, error: {e}")
        return None

# Pattern: Data validation
def validate_ticker_data(self, data: Dict) -> bool:
    required_fields = ["mark_price", "instrument_name"]
    for field in required_fields:
        if field not in data or data[field] is None:
            logging.warning(f"Invalid ticker data: missing {field}")
            return False
    return True
```

### 4. **Configuration Management**

```python
# Pattern: Environment-specific configs
@dataclass
class Config:
    network: th.Network
    underlying: str
    bar_interval_minutes: int
    debug_level: str
    
    @classmethod
    def from_env(cls, env: str = "test") -> 'Config':
        configs = {
            "test": cls(th.Network.TEST, "BTCUSD", 60, "DEBUG"),
            "prod": cls(th.Network.PROD, "BTCUSD", 15, "INFO")
        }
        return configs.get(env, configs["test"])
```

## 🚨 Common Anti-Patterns to Avoid

### 1. **Silent Failures**
```python
# ❌ Don't ignore errors silently
try:
    self.process_data()
except:
    pass  # BAD!

# ✅ Log and handle appropriately  
try:
    self.process_data()
except Exception as e:
    logging.error(f"Data processing failed: {e}")
    self.fallback_behavior()
```

### 2. **Magic Numbers**
```python
# ❌ Magic numbers
if price_diff > 50:  # What is 50?
    self.log_significant_move()

# ✅ Named constants
SIGNIFICANT_PRICE_MOVE_THRESHOLD = 50  # USD
if price_diff > SIGNIFICANT_PRICE_MOVE_THRESHOLD:
    self.log_significant_move()
```

### 3. **Mixed Concerns**
```python
# ❌ Data processing mixed with UI updates
def update_ohlc_bar(self, price, volume):
    # ... data processing ...
    plt.draw()  # UI update in data method!

# ✅ Separation of concerns
def update_ohlc_bar(self, price, volume):
    # ... data processing only ...
    self.notify_data_updated()  # Let UI decide when to update
```

## 🔍 Testing & Validation Patterns

### 1. **Data Validation**
```python
def validate_options_data(self, option: OptionData) -> bool:
    """Validate option data before processing"""
    checks = [
        ("strike", lambda x: x.strike > 0),
        ("expiry", lambda x: x.expiry_ts > time.time()),
        ("iv", lambda x: 0 <= x.iv <= 5.0),  # IV between 0-500%
        ("delta", lambda x: -1 <= x.delta <= 1)
    ]
    
    for field_name, check_func in checks:
        if not check_func(option):
            logging.warning(f"Invalid {field_name} for option {option.name}")
            return False
    return True
```

### 2. **Mock Data for Testing**
```python
class MockThalexDataCollector(ThalexDataCollector):
    """Test double for unit testing"""
    async def connect(self) -> bool:
        self.is_connected = True
        return True
        
    async def load_instruments(self):
        # Load test data from fixtures
        pass
```

## 📊 Performance Monitoring

### 1. **Data Processing Metrics**
```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.timings = defaultdict(list)
        self.error_counts = defaultdict(int)
    
    def time_operation(self, operation_name: str):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    self.timings[operation_name].append(time.time() - start)
                    return result
                except Exception as e:
                    self.error_counts[operation_name] += 1
                    raise
            return wrapper
        return decorator
```

## 🎯 Code Review Checklist

Before committing code, verify:

- [ ] All imports are actually used
- [ ] Error handling covers expected failure modes  
- [ ] Data types are compatible with external libraries (matplotlib, pandas)
- [ ] No duplicate data structures
- [ ] Clear separation between test and production code
- [ ] Type hints are accurate and helpful
- [ ] Constants are named and documented
- [ ] WebSocket/API calls have timeout and error handling
- [ ] Data validation checks are in place
- [ ] Performance implications considered for data structures

## 🔄 Refactoring Guidelines

When refactoring existing code:

1. **Identify** duplicate patterns and consolidate
2. **Extract** reusable components into separate classes/functions
3. **Document** any domain-specific logic or calculations
4. **Test** edge cases and error conditions
5. **Monitor** performance impact of changes
6. **Update** documentation and type hints

---

**Last Updated**: 2025-01-21  
**Next Review**: When adding new major features or after significant bugs