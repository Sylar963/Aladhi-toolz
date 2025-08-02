# Thalex Options Trading Scripts - Engineering Rules

This document provides comprehensive guidelines for creating new trading scripts in the Thalex Options repository. Follow these rules to ensure consistency, reliability, and maintainability across all scripts.

## State Management and Data Architecture

### Where State Lives

**Global Configuration (Script Level)**
```python
# Configuration constants at module top
CURRENT_POSITION_INSTRUMENTS = ['BTC-08AUG25-115000-P', ...]
NETWORK = th.Network.PROD  # or th.Network.TEST
UPDATE_INTERVAL = 5
CALL_ID_* = 1000, 1001, etc.  # Unique API call IDs
```

**Instance State (Main Class)**
```python
class TradingStrategy:
    def __init__(self):
        # Core WebSocket connection
        self.thalex: Optional[th.Thalex] = None
        self.login_success: bool = False
        
        # Data containers
        self.positions: Dict[str, PositionData] = {}
        self.order_books: Dict[str, OrderBookData] = {}
        
        # Tracking counters
        self.portfolio_updates_received: int = 0
        self.order_book_updates_received: int = 0
        self.ticker_updates_received: int = 0
        
        # API credentials
        self.key_id = keys.key_ids[NETWORK]
        self.private_key = keys.private_keys[NETWORK]
```

**Data Classes (Separate State Containers)**
```python
class OrderBookData:
    """Container for market data - keep pricing logic here"""
    
class PositionData:
    """Container for position info - keep position logic here"""
```

### Key Principles

1. **Separation of Concerns**: Keep data containers separate from business logic
2. **Single Source of Truth**: Each piece of data should live in exactly one place
3. **State Initialization**: Always initialize all attributes in `__init__`
4. **No Duplicate Storage**: Avoid storing the same data in multiple places

## Core Script Structure

### Required Script Pattern
```python
#!/usr/bin/env python3
"""
Script Name - Brief Description

Features:
- Feature 1
- Feature 2

Usage:
    python script_name.py

Configuration:
    1. Edit CONFIGURATION_VARIABLES
    2. Set API keys in keys.py
    3. Run script
"""

# Standard imports
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
import websockets

# Thalex imports
import thalex_py.thalex as th
import keys

# Configuration section
CONFIGURATION_VARIABLES = [...]
NETWORK = th.Network.PROD

# Data classes
class DataContainer:
    pass

# Main strategy class
class TradingStrategy:
    pass

# Helper functions
def helper_function():
    pass

# Setup functions
def setup_logging():
    pass

def validate_keys():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
```

## API Integration Standards

### Authentication Pattern
```python
async def connect_and_authenticate(self) -> bool:
    """Standard authentication flow"""
    try:
        self.thalex = th.Thalex(NETWORK)
        await self.thalex.connect()
        await self.thalex.login(self.key_id, self.private_key, id=CALL_ID_LOGIN)
        
        # Wait for login response with timeout
        max_wait = 10
        start_time = time.time()
        while not self.login_success and (time.time() - start_time) < max_wait:
            message_raw = await asyncio.wait_for(self.thalex.receive(), timeout=1.0)
            message = json.loads(message_raw)
            await self.handle_message(message)
        
        await self.thalex.set_cancel_on_disconnect(30)
        return self.login_success
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return False
```

### Market Data Subscription Pattern
```python
async def subscribe_to_market_data(self):
    """Standard market data subscription"""
    # 1. Get portfolio snapshot first
    await self.thalex.portfolio(id=CALL_ID_PORTFOLIO)
    
    # 2. Subscribe to real-time portfolio updates
    await self.thalex.private_subscribe(['account.portfolio'])
    
    # 3. Request ticker data for instruments
    for i, instrument in enumerate(instruments):
        call_id = CALL_ID_BOOK_BASE + i
        await self.thalex.ticker(instrument, id=call_id)
        await asyncio.sleep(0.1)  # Rate limiting
    
    # 4. Wait for initial data
    await asyncio.sleep(3)
```

### Message Handling Pattern
```python
async def handle_message(self, message: Dict[str, Any]):
    """Standard message handling"""
    try:
        if 'channel_name' in message:
            await self._handle_notification(message['channel_name'], message['notification'])
        elif 'result' in message:
            if message.get('id') is not None:
                await self._handle_api_result(message['id'], message['result'])
        elif 'error' in message:
            logging.error(f"API Error: {message['error']}")
            if message.get('id') == CALL_ID_LOGIN:
                self.login_success = False
    except Exception as e:
        logging.error(f"Error handling message: {e}")
```

## Data Handling Rules

### Position Data Management
```python
# CORRECT: Single data structure per concept
self.positions: Dict[str, PositionData] = {}

# WRONG: Multiple structures for same data
self.position_sizes = {}
self.position_marks = {}
self.position_updates = {}
```

### Order Book Data Management
```python
class OrderBookData:
    def __init__(self, instrument_name: str):
        self.instrument_name = instrument_name
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.mark_price: Optional[float] = None
        self.last_update: float = 0
    
    def get_closing_price(self, position_size: float) -> tuple[Optional[float], str]:
        """Logic for determining appropriate price based on position direction"""
        if position_size > 0:  # Long - sell at bid
            return (self.best_bid, "bid") if self.best_bid else (self.mark_price, "mark")
        elif position_size < 0:  # Short - buy at ask
            return (self.best_ask, "ask") if self.best_ask else (self.mark_price, "mark")
        return None, "no_data"
```

## Error Handling Standards

### Connection Management
```python
async def run(self):
    """Standard retry pattern with exponential backoff"""
    max_retries = 3
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            if not await self.connect_and_authenticate():
                raise Exception("Authentication failed")
            
            retry_count = 0  # Reset on success
            await self.subscribe_to_market_data()
            
            # Main message loop
            while True:
                try:
                    message_raw = await asyncio.wait_for(self.thalex.receive(), timeout=5.0)
                    message = json.loads(message_raw)
                    await self.handle_message(message)
                except asyncio.TimeoutError:
                    continue  # Expected for periodic updates
                except (websockets.ConnectionClosed, ConnectionResetError):
                    break  # Trigger reconnection
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                break
            logging.error(f"Error (attempt {retry_count}): {e}")
            await asyncio.sleep(5)
        finally:
            if self.thalex and self.thalex.connected():
                await self.thalex.disconnect()
```

### Graceful Error Handling
```python
# CORRECT: Continue operation on non-critical errors
try:
    result = await some_operation()
except Exception as e:
    logging.warning(f"Non-critical operation failed: {e}")
    continue  # Don't crash the entire script

# CORRECT: Stop on critical errors
try:
    await self.authenticate()
except Exception as e:
    logging.error(f"Critical error: {e}")
    raise  # Re-raise to stop execution
```

## Display and User Interface

### Console Display Pattern
```python
def display_analysis(self):
    """Standard display format"""
    clear_console()
    
    print("=" * 70)
    print("SCRIPT TITLE")
    print("=" * 70)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Network: {NETWORK.value}")
    print()
    
    # Connection status section
    print("CONNECTION STATUS:")
    print("-" * 40)
    print(f"WebSocket Connected: {'Yes' if self.thalex.connected() else 'No'}")
    print(f"Login Success: {'Yes' if self.login_success else 'No'}")
    print()
    
    # Data sections with clear headers
    print("DATA SECTION:")
    print("-" * 70)
    # Display data here
    
    print()
    print(f"Next update in {UPDATE_INTERVAL} seconds... | Press Ctrl+C to exit")

def clear_console():
    """Standard console clearing"""
    os.system('cls' if os.name == 'nt' else 'clear')
```

### Formatting Standards
```python
def format_currency(amount: float) -> str:
    """Standard currency formatting"""
    return f"${amount:,.2f}"

def format_position_size(size: float) -> str:
    """Standard position formatting"""
    if size > 0:
        return f"+{size:.4f}"
    elif size < 0:
        return f"{size:.4f}"
    else:
        return "0.0000"
```

## Logging and Debugging

### Logging Configuration
```python
def setup_logging():
    """Standard logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('script_name.log')
        ]
    )
```

### Logging Standards
```python
# CORRECT: Informational logging
logging.info("Portfolio update received with {len(notification)} positions")
logging.info(f"Updated position {instrument}: {position_size}")

# CORRECT: Error logging with context
logging.error(f"Failed to request ticker for {instrument}: {e}")
logging.warning(f"No closing price available for {instrument}")

# WRONG: Excessive debug logging
logging.debug(f"Message type: {msg_type}")  # Remove unless temporarily needed
```

## Code Quality Standards

### Anti-Patterns to Avoid
```python
# WRONG: Excessive debugging code
if DEBUG_MODE:
    logging.debug(f"Complex debug message with {variables}")

# WRONG: Magic numbers
await asyncio.sleep(2.5)  # What is this delay for?

# WRONG: Complex validation logic
def validate_everything_exhaustively():
    # 100 lines of validation code

# WRONG: Emoji and fancy formatting
print("✓ Connection successful! 🚀")
```

### Best Practices
```python
# CORRECT: Named constants
INITIAL_DATA_WAIT = 3  # seconds to wait for initial market data
RETRY_DELAY = 5       # seconds between connection retries

# CORRECT: Simple, clear logic
if not self.login_success:
    logging.error("Authentication failed")
    return False

# CORRECT: Minimal validation
if instrument not in self.positions:
    logging.warning(f"No position data for {instrument}")
    continue
```

### Import Management
```python
# CORRECT: Standard imports
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
import websockets

# Thalex specific
import thalex_py.thalex as th
import keys

# WRONG: Unused imports
import math  # Not used anywhere
import socket  # Not used
```

## Configuration Management

### Script Configuration
```python
# At top of script - easy to modify
INSTRUMENTS = ['BTC-08AUG25-115000-P', 'BTC-08AUG25-118000-P']
TARGET_EXPIRATION = '29AUG25'
UPDATE_INTERVAL = 5
NETWORK = th.Network.PROD

# API call IDs - unique per script
CALL_ID_LOGIN = 1000
CALL_ID_PORTFOLIO = 1001
CALL_ID_TICKER = 1003
CALL_ID_BOOK_BASE = 2000
```

### Key Validation
```python
def validate_keys():
    """Standard key validation"""
    try:
        key_id = keys.key_ids.get(NETWORK)
        private_key = keys.private_keys.get(NETWORK)
        
        if not key_id or not private_key:
            logging.error(f"Missing API credentials for network: {NETWORK}")
            return False
        return True
    except (AttributeError, ImportError) as e:
        logging.error(f"Error accessing keys.py: {e}")
        return False
```

## Performance Guidelines

### Rate Limiting
```python
# Always add delays between API requests
for instrument in instruments:
    await self.thalex.ticker(instrument, id=call_id)
    await asyncio.sleep(0.1)  # Prevent API rate limiting
```

### Memory Management
```python
# CORRECT: Reuse objects
self.order_books[instrument].update(new_data)

# WRONG: Create new objects constantly
self.order_books[instrument] = OrderBookData(new_data)
```

### Timeout Management
```python
# Always use timeouts for network operations
message_raw = await asyncio.wait_for(self.thalex.receive(), timeout=5.0)

# Wait for responses with reasonable timeouts
max_wait = 10  # seconds
start_time = time.time()
while not condition and (time.time() - start_time) < max_wait:
    # Wait for condition
```

## Testing and Validation

### Script Validation Checklist
- [ ] Script compiles without syntax errors: `python -m py_compile script.py`
- [ ] All imports are available and used
- [ ] All attributes are initialized in `__init__`
- [ ] Error handling doesn't crash on network issues
- [ ] Configuration is at the top and easy to modify
- [ ] Logging is informative but not excessive
- [ ] No emojis or unnecessary formatting
- [ ] Follows the established patterns from working scripts

### Network Configuration
```python
# ALWAYS default to testnet for safety
NETWORK = th.Network.TEST  # Change to PROD only when ready

# Document the network clearly
print(f"Network: {NETWORK.value}")
logging.info(f"Using network: {NETWORK}")
```

## Example Implementation

Reference the `roll_cost_analyzer.py` script as the gold standard implementation that follows all these rules. It demonstrates:

- Clean state management with separate data classes
- Proper API integration patterns
- Effective error handling and retry logic
- Clear, readable display output
- Minimal but sufficient logging
- Simple configuration management
- Robust connection handling

When creating new scripts, use `roll_cost_analyzer.py` as a template and follow these rules to ensure consistency and reliability across the codebase.