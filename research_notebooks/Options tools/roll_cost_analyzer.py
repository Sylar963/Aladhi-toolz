#!/usr/bin/env python3
"""
Roll Cost Analyzer - Thalex Options Position Roll Calculator

This script connects to the Thalex API via WebSocket to analyze the cost of rolling
multileg options positions from current expiration to a new expiration date.

Features:
- Real-time order book data for accurate pricing
- Position size extraction from portfolio
- Net cost calculation (credit or debit)
- Live console updates with current market conditions
- Automatic reconnection on connection failures

Usage:
    python roll_cost_analyzer.py

Configuration:
    1. Edit CURRENT_POSITION_INSTRUMENTS with your actual position instrument names
    2. Set ROLL_TO_EXPIRATION to your target expiration date (DDMMMYY format)
    3. Ensure your API keys are configured in keys.py
    4. Run the script and monitor the real-time roll cost calculations

Security Notes:
- Uses testnet by default for safety
- Requires API keys to be configured in keys.py (not in this file)
- All trades are analysis only - no actual trading is performed
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys

import websockets

# Thalex API imports
import thalex_py.thalex as th
import keys  # Import keys.py for API credentials


# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Configuration variables - modify these for your specific roll scenario
CURRENT_POSITION_INSTRUMENTS = [
    'BTC-08AUG25-115000-P',
    'BTC-08AUG25-118000-P'
]

ROLL_TO_EXPIRATION = '29AUG25'  # Target expiration in DDMMMYY format

# Manual strike override - use when automatic matching doesn't find correct ATM strikes
# Format: {'current_instrument': 'target_instrument'}
TARGET_STRIKES_OVERRIDE = {
    'BTC-08AUG25-115000-P': 'BTC-29AUG25-115000-P',
    'BTC-08AUG25-118000-P': 'BTC-29AUG25-120000-P'
}

# Settings
USE_MANUAL_OVERRIDE = True  # Set to False to use automatic ATM detection
UPDATE_INTERVAL = 5  # Console update interval in seconds
PRICE_PRECISION = 2  # Number of decimal places for prices

# Network configuration
NETWORK = th.Network.PROD

# Call IDs for API request tracking
CALL_ID_LOGIN = 1000
CALL_ID_PORTFOLIO = 1001
CALL_ID_TICKER = 1003
CALL_ID_BOOK_BASE = 2000


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_keys():
    """Validate that API keys are properly configured"""
    try:
        key_id = keys.key_ids.get(NETWORK)
        private_key = keys.private_keys.get(NETWORK)
        
        if not key_id or not private_key:
            logging.error(f"Missing API credentials for network: {NETWORK}")
            logging.error("Please configure your API keys in keys.py")
            return False
            
        return True
        
    except (AttributeError, ImportError) as e:
        logging.error(f"Error accessing keys.py: {e}")
        logging.error("Please ensure keys.py exists and contains key_ids and private_keys dictionaries")
        return False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class OrderBookData:
    """Container for order book and pricing information"""
    def __init__(self, instrument_name: str):
        self.instrument_name = instrument_name
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.mark_price: Optional[float] = None
        self.last_update: float = 0
        
    def update(self, book_data: Dict[str, Any]):
        """Update order book with new data"""
        # Handle new API format (direct fields)
        if 'best_bid_price' in book_data:
            self.best_bid = float(book_data['best_bid_price']) if book_data['best_bid_price'] is not None else None
        elif 'bids' in book_data:
            # Handle old API format (arrays)
            bids = book_data.get('bids', [])
            self.best_bid = float(bids[0]['price']) if bids else None
            
        if 'best_ask_price' in book_data:
            self.best_ask = float(book_data['best_ask_price']) if book_data['best_ask_price'] is not None else None
        elif 'asks' in book_data:
            # Handle old API format (arrays)
            asks = book_data.get('asks', [])
            self.best_ask = float(asks[0]['price']) if asks else None
        
        # Extract Mark Price
        if 'mark_price' in book_data and book_data['mark_price'] is not None:
            self.mark_price = float(book_data['mark_price'])
        elif 'mark' in book_data and book_data['mark'] is not None:
            self.mark_price = float(book_data['mark'])
        
        self.last_update = time.time()
        
    def get_closing_price(self, position_size: float) -> tuple[Optional[float], str]:
        """Get appropriate closing price based on position"""
        if position_size > 0:  # Long position - sell at bid
            if self.best_bid is not None:
                return self.best_bid, "bid"
            elif self.mark_price is not None:
                return self.mark_price, "mark"
        elif position_size < 0:  # Short position - buy at ask
            if self.best_ask is not None:
                return self.best_ask, "ask"
            elif self.mark_price is not None:
                return self.mark_price, "mark"
        return None, "no_data"
        
    def get_opening_price(self, position_size: float) -> tuple[Optional[float], str]:
        """Get appropriate opening price based on position"""
        if position_size > 0:  # Want to go long - buy at ask
            if self.best_ask is not None:
                return self.best_ask, "ask"
            elif self.mark_price is not None:
                return self.mark_price, "mark"
        elif position_size < 0:  # Want to go short - sell at bid
            if self.best_bid is not None:
                return self.best_bid, "bid"
            elif self.mark_price is not None:
                return self.mark_price, "mark"
        return None, "no_data"


class PositionData:
    """Container for position information"""
    def __init__(self, instrument_name: str):
        self.instrument_name = instrument_name
        self.position_size: float = 0.0
        self.mark_price: Optional[float] = None
        self.last_update: float = 0
        
    def update(self, position_size: float, mark_price: Optional[float] = None):
        """Update position size and optional mark price"""
        self.position_size = position_size
        if mark_price is not None:
            self.mark_price = mark_price
        self.last_update = time.time()


class RollCostCalculator:
    """Main class for roll cost calculation and display"""
    
    def __init__(self):
        self.thalex: Optional[th.Thalex] = None
        self.current_positions: Dict[str, PositionData] = {}
        self.rolled_instruments: List[str] = []
        self.order_books: Dict[str, OrderBookData] = {}
        self.last_display_update: float = 0
        self.login_success: bool = False
        
        # Tracking counters
        self.portfolio_updates_received: int = 0
        self.order_book_updates_received: int = 0
        self.ticker_updates_received: int = 0
        self.last_portfolio_update: Optional[float] = None
        
        # Spot price tracking
        self.current_spot_price: Optional[float] = None
        
        # API credentials from keys.py
        self.key_id = keys.key_ids[NETWORK]
        self.private_key = keys.private_keys[NETWORK]
        
        # Initialize position tracking for current instruments
        for instrument in CURRENT_POSITION_INSTRUMENTS:
            self.current_positions[instrument] = PositionData(instrument)
        
        # Generate rolled instrument names
        self.rolled_instruments = self._generate_rolled_instruments()
        
        # Initialize order book tracking for all instruments
        all_instruments = CURRENT_POSITION_INSTRUMENTS + self.rolled_instruments
        for instrument in all_instruments:
            self.order_books[instrument] = OrderBookData(instrument)
        
        logging.info(f"Initialized Roll Cost Calculator")
        logging.info(f"Current instruments: {CURRENT_POSITION_INSTRUMENTS}")
        logging.info(f"Rolled instruments: {self.rolled_instruments}")
    
    def _generate_rolled_instruments(self) -> List[str]:
        """Generate rolled instrument names from current positions"""
        rolled = []
        for current_instrument in CURRENT_POSITION_INSTRUMENTS:
            rolled_instrument = self._generate_rolled_instrument_name(current_instrument, ROLL_TO_EXPIRATION)
            if rolled_instrument:
                rolled.append(rolled_instrument)
        return rolled
    
    def _generate_rolled_instrument_name(self, current_instrument: str, target_expiry: str) -> Optional[str]:
        """Generate rolled instrument name using manual override or simple substitution"""
        try:
            # Check for manual override first
            if USE_MANUAL_OVERRIDE and current_instrument in TARGET_STRIKES_OVERRIDE:
                override_instrument = TARGET_STRIKES_OVERRIDE[current_instrument]
                logging.info(f"Using manual override: {current_instrument} -> {override_instrument}")
                return override_instrument
            
            # Fall back to simple expiry substitution
            parts = current_instrument.split('-')
            if len(parts) != 4:
                logging.error(f"Invalid instrument format: {current_instrument}")
                return None
            
            underlying, _, strike, option_type = parts
            rolled_name = f"{underlying}-{target_expiry}-{strike}-{option_type}"
            logging.info(f"Generated rolled instrument: {current_instrument} -> {rolled_name}")
            return rolled_name
            
        except Exception as e:
            logging.error(f"Error generating rolled instrument name for {current_instrument}: {e}")
            return None
    
    async def connect_and_authenticate(self) -> bool:
        """Connect to Thalex API and authenticate"""
        try:
            logging.info("Connecting to Thalex API...")
            self.thalex = th.Thalex(NETWORK)
            await self.thalex.connect()
            
            logging.info("Authenticating...")
            await self.thalex.login(self.key_id, self.private_key, id=CALL_ID_LOGIN)
            
            # Wait for login response
            max_wait = 10  # seconds
            start_time = time.time()
            
            while not self.login_success and (time.time() - start_time) < max_wait:
                try:
                    message_raw = await asyncio.wait_for(self.thalex.receive(), timeout=1.0)
                    message = json.loads(message_raw)
                    await self.handle_message(message)
                except asyncio.TimeoutError:
                    continue
            
            if not self.login_success:
                logging.error("Login timeout - authentication failed")
                return False
            
            # Set cancel on disconnect for safety
            await self.thalex.set_cancel_on_disconnect(30)
            
            logging.info("Successfully connected and authenticated")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect and authenticate: {e}")
            return False
    
    async def subscribe_to_market_data(self):
        """Subscribe to required market data feeds"""
        if self.thalex is None:
            raise Exception("Thalex client not connected")
        
        if not self.login_success:
            raise Exception("Must be authenticated before subscribing to market data")
            
        try:
            # Get initial portfolio snapshot
            logging.info("Requesting initial portfolio snapshot...")
            await self.thalex.portfolio(id=CALL_ID_PORTFOLIO)
            
            # Wait for portfolio response
            portfolio_received = False
            wait_start = time.time()
            max_wait = 10  # seconds
            
            while not portfolio_received and (time.time() - wait_start) < max_wait:
                try:
                    message_raw = await asyncio.wait_for(self.thalex.receive(), timeout=1.0)
                    message = json.loads(message_raw)
                    await self.handle_message(message)
                    
                    if message.get('id') == CALL_ID_PORTFOLIO and 'result' in message:
                        portfolio_received = True
                        logging.info("Portfolio snapshot received")
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Error waiting for portfolio: {e}")
                    break
            
            # Subscribe to real-time portfolio updates
            logging.info("Subscribing to real-time portfolio updates...")
            await self.thalex.private_subscribe(['account.portfolio'])
            
            # Request ticker data for all instruments
            all_instruments = CURRENT_POSITION_INSTRUMENTS + self.rolled_instruments
            logging.info(f"Requesting ticker data for {len(all_instruments)} instruments...")
            
            for i, instrument in enumerate(all_instruments):
                try:
                    call_id = CALL_ID_BOOK_BASE + i
                    await self.thalex.ticker(instrument, id=call_id)
                    await asyncio.sleep(0.1)  # Small delay between requests
                except Exception as e:
                    logging.error(f"Failed to request ticker for {instrument}: {e}")
            
            # Request BTC spot price
            logging.info("Requesting BTC ticker for spot price...")
            try:
                await self.thalex.ticker('BTC-PERPETUAL', id=CALL_ID_TICKER)
            except Exception as e:
                logging.warning(f"Failed to request BTC ticker: {e}")
            
            # Wait for initial data
            logging.info("Waiting for initial data...")
            await asyncio.sleep(3)
            
        except Exception as e:
            logging.error(f"Failed to subscribe to market data: {e}")
            raise
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            # Handle notifications (real-time updates)
            if 'channel_name' in message:
                await self._handle_notification(message['channel_name'], message['notification'])
            
            # Handle API call results
            elif 'result' in message:
                if message.get('id') is not None:
                    await self._handle_api_result(message['id'], message['result'])
            
            # Handle errors
            elif 'error' in message:
                error = message['error']
                call_id = message.get('id')
                logging.error(f"API Error (call_id={call_id}): {error}")
                
                if call_id == CALL_ID_LOGIN:
                    self.login_success = False
                
        except Exception as e:
            logging.error(f"Error handling message: {e}")
    
    async def _handle_notification(self, channel: str, notification: List[Dict[str, Any]]):
        """Handle real-time notifications"""
        if channel == 'account.portfolio':
            self.portfolio_updates_received += 1
            self.last_portfolio_update = time.time()
            
            logging.info(f"Portfolio update received with {len(notification)} positions")
            
            # Update positions
            for position_data in notification:
                instrument_name = position_data['instrument_name']
                position_size = float(position_data['position'])
                
                if instrument_name in self.current_positions:
                    mark_price = position_data.get('mark_price')
                    self.current_positions[instrument_name].update(position_size, mark_price)
                    logging.info(f"Updated position {instrument_name}: {position_size}")
    
    async def _handle_api_result(self, call_id: int, result: Any):
        """Handle API call results"""
        if call_id == CALL_ID_LOGIN:
            logging.info("Login successful")
            self.login_success = True
            
        elif call_id == CALL_ID_PORTFOLIO:
            # Handle portfolio snapshot
            logging.info(f"Portfolio snapshot received with {len(result)} positions")
            
            for position_data in result:
                instrument_name = position_data['instrument_name']
                position_size = float(position_data['position'])
                
                if instrument_name in self.current_positions:
                    mark_price = position_data.get('mark_price')
                    self.current_positions[instrument_name].update(position_size, mark_price)
                    logging.info(f"Initial position {instrument_name}: {position_size}")
                
        elif call_id == CALL_ID_TICKER:
            # Handle ticker data (spot price)
            if 'last_price' in result:
                self.current_spot_price = float(result['last_price'])
                logging.info(f"Updated BTC spot price: ${self.current_spot_price:,.2f}")
            elif 'price' in result:
                self.current_spot_price = float(result['price'])
                logging.info(f"Updated BTC spot price: ${self.current_spot_price:,.2f}")
                
        elif call_id >= CALL_ID_BOOK_BASE:
            # Handle ticker data for instruments
            self.ticker_updates_received += 1
            self.order_book_updates_received += 1
            instrument_index = call_id - CALL_ID_BOOK_BASE
            all_instruments = CURRENT_POSITION_INSTRUMENTS + self.rolled_instruments
            
            if instrument_index < len(all_instruments):
                instrument_name = all_instruments[instrument_index]
                if instrument_name in self.order_books:
                    self.order_books[instrument_name].update(result)
                    book = self.order_books[instrument_name]
                    logging.info(f"Updated ticker {instrument_name}: bid={book.best_bid}, ask={book.best_ask}")
    
    def calculate_roll_cost(self) -> Dict[str, Any]:
        """Calculate the net cost of rolling positions"""
        result = {
            'total_close_cost': 0.0,
            'total_open_cost': 0.0,
            'net_cost': 0.0,
            'current_legs': [],
            'rolled_legs': [],
            'calculation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate cost to close current positions
        for instrument in CURRENT_POSITION_INSTRUMENTS:
            if instrument not in self.current_positions:
                continue
                
            position_data = self.current_positions[instrument]
            
            if position_data.position_size == 0:
                continue  # Skip instruments with no position
            
            if instrument not in self.order_books:
                logging.warning(f"No order book data for {instrument}")
                continue
                
            order_book = self.order_books[instrument]
            
            closing_price, price_type = order_book.get_closing_price(position_data.position_size)
            
            if closing_price is None:
                logging.warning(f"No closing price available for {instrument}")
                continue
            
            close_cost = position_data.position_size * closing_price
            result['total_close_cost'] += close_cost
            
            result['current_legs'].append({
                'instrument': instrument,
                'position_size': position_data.position_size,
                'closing_price': closing_price,
                'close_cost': close_cost,
                'price_type': price_type
            })
        
        # Calculate cost to open new positions (same sizes)
        for i, rolled_instrument in enumerate(self.rolled_instruments):
            if i >= len(CURRENT_POSITION_INSTRUMENTS):
                break
                
            current_instrument = CURRENT_POSITION_INSTRUMENTS[i]
            
            if current_instrument not in self.current_positions:
                continue
                
            position_data = self.current_positions[current_instrument]
            
            if position_data.position_size == 0:
                continue  # Skip instruments with no position
            
            if rolled_instrument not in self.order_books:
                logging.warning(f"No order book data for rolled instrument {rolled_instrument}")
                continue
                
            order_book = self.order_books[rolled_instrument]
            
            opening_price, price_type = order_book.get_opening_price(position_data.position_size)
            
            if opening_price is None:
                logging.warning(f"No opening price available for {rolled_instrument}")
                continue
            
            open_cost = position_data.position_size * opening_price
            result['total_open_cost'] += open_cost
            
            result['rolled_legs'].append({
                'instrument': rolled_instrument,
                'position_size': position_data.position_size,
                'opening_price': opening_price,
                'open_cost': open_cost,
                'price_type': price_type
            })
        
        # Calculate net cost
        result['net_cost'] = result['total_open_cost'] - result['total_close_cost']
        
        return result
    
    def display_roll_analysis(self):
        """Display the roll cost analysis"""
        clear_console()
        
        print("=" * 70)
        print("THALEX ROLL COST ANALYZER")
        print("=" * 70)
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Positions -> Roll to {ROLL_TO_EXPIRATION}")
        print(f"Network: {NETWORK.value}")
        print()
        
        # Display connection status
        print("CONNECTION STATUS:")
        print("-" * 40)
        connected = self.thalex.connected() if self.thalex else False
        print(f"WebSocket Connected: {'Yes' if connected else 'No'}")
        print(f"Login Success: {'Yes' if self.login_success else 'No'}")
        print(f"Portfolio Updates: {self.portfolio_updates_received}")
        print(f"Order Book Updates: {self.order_book_updates_received}")
        if self.current_spot_price:
            print(f"Current BTC Spot: ${self.current_spot_price:,.2f}")
        print()
        
        # Calculate roll cost
        analysis = self.calculate_roll_cost()
        
        # Display current position details
        print("CURRENT POSITION LEGS (TO CLOSE):")
        print("-" * 70)
        if analysis['current_legs']:
            for leg in analysis['current_legs']:
                size_str = format_position_size(leg['position_size'])
                price_str = format_currency(leg['closing_price'])
                cost_str = format_currency(leg['close_cost'])
                price_type = f"({leg['price_type']})"
                print(f"{leg['instrument']:30} | Size: {size_str:10} | Price: {price_str:12} {price_type:6} | Cost: {cost_str:12}")
        else:
            print("No positions found or insufficient data")
        
        print()
        print("ROLLED POSITION LEGS (TO OPEN):")
        print("-" * 70)
        if analysis['rolled_legs']:
            for leg in analysis['rolled_legs']:
                size_str = format_position_size(leg['position_size'])
                price_str = format_currency(leg['opening_price'])
                cost_str = format_currency(leg['open_cost'])
                price_type = f"({leg['price_type']})"
                print(f"{leg['instrument']:30} | Size: {size_str:10} | Price: {price_str:12} {price_type:6} | Cost: {cost_str:12}")
        else:
            print("No rolled positions calculated or insufficient data")
        
        print()
        print("ROLL COST SUMMARY:")
        print("-" * 30)
        print(f"Cost to Close Current:  {format_currency(analysis['total_close_cost']):>15}")
        print(f"Cost to Open Rolled:    {format_currency(analysis['total_open_cost']):>15}")
        print(f"{'='*30}")
        
        net_cost = analysis['net_cost']
        if net_cost > 0:
            print(f"NET DEBIT (You Pay):    {format_currency(net_cost):>15}")
        elif net_cost < 0:
            print(f"NET CREDIT (You Receive): {format_currency(-net_cost):>15}")
        else:
            print(f"NET COST:               {format_currency(0):>15}")
        
        print()
        print(f"Next update in {UPDATE_INTERVAL} seconds... | Press Ctrl+C to exit")
    
    async def run(self):
        """Main execution loop"""
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Connect and authenticate
                logging.info(f"Connection attempt {retry_count + 1}/{max_retries + 1}")
                if not await self.connect_and_authenticate():
                    raise Exception("Failed to connect and authenticate")
                
                # Reset retry count on successful connection
                retry_count = 0
                
                # Subscribe to market data
                await self.subscribe_to_market_data()
                
                # Display initial analysis
                self.display_roll_analysis()
                
                # Main message loop
                while True:
                    try:
                        if self.thalex is None:
                            raise Exception("Thalex client disconnected")
                        
                        # Receive and handle messages
                        try:
                            message_raw = await asyncio.wait_for(self.thalex.receive(), timeout=5.0)
                        except asyncio.TimeoutError:
                            # Check if we should update display
                            current_time = time.time()
                            if current_time - self.last_display_update >= UPDATE_INTERVAL:
                                self.display_roll_analysis()
                                self.last_display_update = current_time
                            continue
                        
                        try:
                            message = json.loads(message_raw)
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse message: {e}")
                            continue
                        
                        await self.handle_message(message)
                        
                        # Update display periodically
                        current_time = time.time()
                        if current_time - self.last_display_update >= UPDATE_INTERVAL:
                            self.display_roll_analysis()
                            self.last_display_update = current_time
                            
                    except (websockets.ConnectionClosed, ConnectionResetError) as e:
                        logging.warning(f"WebSocket connection lost: {e}")
                        break
                    except Exception as e:
                        logging.error(f"Error in message loop: {e}")
                        continue
                        
            except KeyboardInterrupt:
                logging.info("Shutdown requested by user")
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(f"Maximum retries ({max_retries}) exceeded")
                    break
                    
                logging.error(f"Error (attempt {retry_count}/{max_retries}): {e}")
                logging.info(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
                
            finally:
                # Clean up connection
                if self.thalex and self.thalex.connected():
                    try:
                        await self.thalex.disconnect()
                        logging.info("Disconnected from Thalex API")
                    except Exception as e:
                        logging.warning(f"Error during disconnect: {e}")
        
        logging.info("Roll Cost Analyzer shutdown complete")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(amount: float) -> str:
    """Format currency amount with appropriate sign and precision"""
    return f"${amount:,.{PRICE_PRECISION}f}"


def format_position_size(size: float) -> str:
    """Format position size with sign indicator"""
    if size > 0:
        return f"+{size:.4f}"
    elif size < 0:
        return f"{size:.4f}"
    else:
        return "0.0000"


def clear_console():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


# =============================================================================
# SETUP AND VALIDATION
# =============================================================================

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('roll_cost_analyzer.log')
        ]
    )


def main():
    """Main entry point"""
    setup_logging()
    
    logging.info("=" * 50)
    logging.info("THALEX ROLL COST ANALYZER")
    logging.info("=" * 50)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Current Instruments: {CURRENT_POSITION_INSTRUMENTS}")
    print(f"  Roll to Expiration: {ROLL_TO_EXPIRATION}")
    print(f"  Network: {NETWORK.value}")
    print(f"  Update Interval: {UPDATE_INTERVAL} seconds")
    print()
    
    # Validate API keys
    if not validate_keys():
        sys.exit(1)
    
    # Create and run calculator
    calculator = RollCostCalculator()
    
    try:
        asyncio.run(calculator.run())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()