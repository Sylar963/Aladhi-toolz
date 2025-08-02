#!/usr/bin/env python3
"""
Interactive Straddle Analyzer - Simple user-driven expiration selection
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

# Configure matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import thalex_py.thalex as th
import keys

# Configuration
UNDERLYING = "BTCUSD"
NETWORK = th.Network.TEST
CID_LOGIN = 1000
CID_TICKER = 1001

class SimpleOption:
    """Simple option data"""
    def __init__(self, name: str, strike: float, option_type: str, expiry_ts: int):
        self.name = name
        self.strike = strike
        self.option_type = option_type
        self.expiry_ts = expiry_ts
        self.mark_price = 0.0
        
    def is_call(self) -> bool:
        return self.option_type == "call"
        
    def is_put(self) -> bool:
        return self.option_type == "put"

class ExpirationData:
    """Data for one expiration"""
    def __init__(self, expiry_ts: int, expiry_date: str):
        self.expiry_ts = expiry_ts
        self.expiry_date = expiry_date
        self.atm_call: Optional[SimpleOption] = None
        self.atm_put: Optional[SimpleOption] = None
        self.days_to_expiry = (expiry_ts - time.time()) / 86400
        
    def has_straddle(self) -> bool:
        return (self.atm_call is not None and self.atm_put is not None and 
                self.atm_call.mark_price > 0 and self.atm_put.mark_price > 0)
                
    def get_straddle_price(self) -> float:
        if self.has_straddle():
            return self.atm_call.mark_price + self.atm_put.mark_price
        return 0.0
        
    def get_breakeven_range(self) -> tuple:
        if self.has_straddle():
            straddle_price = self.get_straddle_price()
            # Use the call strike as reference (they should be close)
            strike = self.atm_call.strike
            return (strike - straddle_price, strike + straddle_price)
        return (0, 0)

class InteractiveStraddleAnalyzer:
    """Interactive straddle analyzer"""
    
    def __init__(self):
        self.thalex = th.Thalex(NETWORK)
        self.btc_price = 0.0
        self.login_success = False
        self.expirations: Dict[str, ExpirationData] = {}
        
        # API credentials
        self.key_id = keys.key_ids[NETWORK]
        self.private_key = keys.private_keys[NETWORK]
        
    async def connect_and_authenticate(self) -> bool:
        """Connect and authenticate"""
        try:
            logging.info("Connecting to Thalex API...")
            await self.thalex.connect()
            
            await self.thalex.login(self.key_id, self.private_key, id=CID_LOGIN)
            
            # Wait for login
            response = json.loads(await self.thalex.receive())
            if response.get('id') == CID_LOGIN and 'result' in response:
                self.login_success = True
                logging.info("Successfully authenticated")
                return True
            else:
                logging.error(f"Login failed: {response}")
                return False
                
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False
            
    async def get_btc_price(self):
        """Get current BTC price"""
        try:
            await self.thalex.ticker("BTC-PERPETUAL", id=CID_TICKER)
            response = json.loads(await self.thalex.receive())
            
            if response.get('id') == CID_TICKER and 'result' in response:
                self.btc_price = float(response['result']['mark_price'])
                print(f"Current BTC Price: ${self.btc_price:.2f}")
            else:
                print("Failed to get BTC price - invalid response")
        except Exception as e:
            print(f"Failed to get BTC price: {e}")
            self.btc_price = 0.0
            
    async def load_expiration_data(self):
        """Load 2 ATM options per expiration"""
        try:
            # Get instruments
            await self.thalex.instruments()
            response = json.loads(await self.thalex.receive())
            
            if 'result' not in response:
                print("Failed to load instruments - invalid response")
                return
                
            instruments = response["result"]
            
            # Group options by expiry
            options_by_expiry = {}
            current_time = time.time()
        
            for instrument in instruments:
                if (instrument["underlying"] == UNDERLYING and 
                    instrument["type"] == "option" and
                    instrument.get("expiration_timestamp", 0) > current_time):
                    
                    expiry_ts = instrument["expiration_timestamp"]
                    expiry_date = datetime.fromtimestamp(expiry_ts).strftime("%Y-%m-%d")
                    
                    if expiry_date not in options_by_expiry:
                        options_by_expiry[expiry_date] = []
                        
                    option = SimpleOption(
                        name=instrument["instrument_name"],
                        strike=instrument.get("strike_price", 0),
                        option_type=instrument.get("option_type", ""),
                        expiry_ts=expiry_ts
                    )
                    options_by_expiry[expiry_date].append(option)
            
            print(f"\nFound {len(options_by_expiry)} active expirations")
            
            # For each expiry, find the 2 closest ATM options
            for expiry_date, options in options_by_expiry.items():
                expiry_ts = options[0].expiry_ts
                expiry_data = ExpirationData(expiry_ts, expiry_date)
                
                # Find closest call and put to BTC price
                calls = [opt for opt in options if opt.is_call()]
                puts = [opt for opt in options if opt.is_put()]
                
                if calls and puts and self.btc_price > 0:
                    # Find strikes that have BOTH call and put (proper straddle)
                    call_strikes = set(call.strike for call in calls)
                    put_strikes = set(put.strike for put in puts)
                    available_strikes = call_strikes & put_strikes  # Intersection
                    
                    if available_strikes:
                        # Find ATM strike from available pairs
                        atm_strike = min(available_strikes, key=lambda x: abs(x - self.btc_price))
                        
                        # Get call and put at the same strike
                        atm_call = next(call for call in calls if call.strike == atm_strike)
                        atm_put = next(put for put in puts if put.strike == atm_strike)
                        
                        expiry_data.atm_call = atm_call
                        expiry_data.atm_put = atm_put
                    else:
                        # No matching strikes - skip this expiration
                        print(f"DEBUG: No matching call/put strikes for {expiry_date}")
                        continue
                    
                    # Request prices for these 2 options
                    await self._get_option_price(atm_call)
                    await self._get_option_price(atm_put)
                    
                    self.expirations[expiry_date] = expiry_data
                    
            print(f"Loaded data for {len(self.expirations)} expirations")
        
        except Exception as e:
            print(f"Failed to load expiration data: {e}")
            self.expirations = {}
        
    async def _get_option_price(self, option: SimpleOption):
        """Get price for a single option"""
        try:
            await self.thalex.ticker(option.name, id=CID_TICKER + 1)
            response = json.loads(await self.thalex.receive())
            
            if response.get('id') == CID_TICKER + 1 and 'result' in response:
                option.mark_price = response['result'].get('mark_price', 0.0)
                
        except Exception as e:
            logging.error(f"Failed to get price for {option.name}: {e}")
            
    def show_available_expirations(self):
        """Show available expirations to user"""
        print("\n" + "="*70)
        print("AVAILABLE EXPIRATION DATES")
        print("="*70)
        print(f"{'#':<3} {'Date':<12} {'Days':<6} {'Call Strike':<12} {'Put Strike':<12} {'Straddle':<10} {'Status'}")
        print("-"*70)
        
        valid_expirations = []
        for i, (date, exp_data) in enumerate(sorted(self.expirations.items()), 1):
            call_strike = f"${exp_data.atm_call.strike:.0f}" if exp_data.atm_call else "N/A"
            put_strike = f"${exp_data.atm_put.strike:.0f}" if exp_data.atm_put else "N/A"
            
            if exp_data.has_straddle():
                straddle_price = f"${exp_data.get_straddle_price():.2f}"
                status = "✓ Ready"
                valid_expirations.append((i, date, exp_data))
            else:
                straddle_price = "N/A"
                status = "✗ No Data"
                
            print(f"{i:<3} {date:<12} {exp_data.days_to_expiry:<6.1f} {call_strike:<12} {put_strike:<12} {straddle_price:<10} {status}")
            
        return valid_expirations
        
    def plot_straddle_chart(self, expiry_data: ExpirationData):
        """Plot straddle chart for selected expiration"""
        try:
            print(f"DEBUG: Plotting chart for {expiry_data.expiry_date}")
            
            if not expiry_data.has_straddle():
                print("Cannot plot: No complete straddle data")
                print(f"DEBUG: Call exists: {expiry_data.atm_call is not None}")
                print(f"DEBUG: Put exists: {expiry_data.atm_put is not None}")
                if expiry_data.atm_call:
                    print(f"DEBUG: Call price: {expiry_data.atm_call.mark_price}")
                if expiry_data.atm_put:
                    print(f"DEBUG: Put price: {expiry_data.atm_put.mark_price}")
                return
            
            print("DEBUG: Setting up chart...")
            
            # Setup chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_title(f"BTC Straddle Breakeven Ranges - {expiry_data.expiry_date}", fontsize=14, pad=20)
            ax.set_ylabel("Price ($)")
            ax.grid(True, alpha=0.3)
            
            print("DEBUG: Getting data...")
            
            # Get data
            call = expiry_data.atm_call
            put = expiry_data.atm_put
            straddle_price = expiry_data.get_straddle_price()
            lower_breakeven, upper_breakeven = expiry_data.get_breakeven_range()
            
            print(f"DEBUG: Straddle price: ${straddle_price:.2f}")
            print(f"DEBUG: Breakeven range: ${lower_breakeven:.0f} - ${upper_breakeven:.0f}")
            
            # Plot current BTC price
            ax.axhline(y=self.btc_price, color='blue', linewidth=2, 
                      label=f'BTC Price: ${self.btc_price:.0f}')
            
            # Plot breakeven lines
            ax.axhline(y=upper_breakeven, color='red', linestyle='--', 
                      linewidth=1.5, label=f'Upper Breakeven: ${upper_breakeven:.0f}')
            ax.axhline(y=lower_breakeven, color='red', linestyle='--', 
                      linewidth=1.5, label=f'Lower Breakeven: ${lower_breakeven:.0f}')
            
            # Plot strike prices
            ax.axhline(y=call.strike, color='green', linestyle=':', 
                      linewidth=1, alpha=0.7, label=f'Call Strike: ${call.strike:.0f}')
            ax.axhline(y=put.strike, color='orange', linestyle=':', 
                      linewidth=1, alpha=0.7, label=f'Put Strike: ${put.strike:.0f}')
            
            # Add info box
            info_text = (
                f"Call: {call.name} = ${call.mark_price:.2f}\n"
                f"Put: {put.name} = ${put.mark_price:.2f}\n"
                f"Straddle Price: ${straddle_price:.2f}\n"
                f"Range Width: ${upper_breakeven - lower_breakeven:.0f}\n"
                f"Days to Expiry: {expiry_data.days_to_expiry:.1f}"
            )
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            # Set y-axis limits around the range
            padding = (upper_breakeven - lower_breakeven) * 0.1
            ax.set_ylim(lower_breakeven - padding, upper_breakeven + padding)
            
            ax.legend(loc='upper right')
            print("DEBUG: Displaying chart...")
            plt.tight_layout()
            plt.show()
            print("DEBUG: Chart displayed successfully!")
            
        except Exception as e:
            print(f"ERROR plotting chart: {e}")
            import traceback
            traceback.print_exc()
        
    async def run_interactive_session(self):
        """Run interactive session"""
        print("Starting Interactive Straddle Analyzer...")
        
        # Connect and get basic data
        if not await self.connect_and_authenticate():
            print("Failed to connect")
            return
            
        await self.get_btc_price()
        await self.load_expiration_data()
        
        # Interactive loop
        while True:
            valid_expirations = self.show_available_expirations()
            
            if not valid_expirations:
                print("\nNo valid expirations found with complete data")
                break
                
            print("\nOptions:")
            print("  Enter expiration number (1-{}) to view chart".format(len(valid_expirations)))
            print("  'refresh' to reload data")
            print("  'quit' to exit")
            
            choice = input("\nYour choice: ").strip()
            
            if choice.lower() == 'quit':
                break
            elif choice.lower() == 'refresh':
                print("\nRefreshing data...")
                
                # Clean disconnect old connection
                if hasattr(self.thalex, 'disconnect'):
                    try:
                        await self.thalex.disconnect()
                    except:
                        pass  # Ignore if already dead
                
                # Create fresh connection and reload all data
                if await self.connect_and_authenticate():
                    await self.get_btc_price()
                    await self.load_expiration_data()
                    print("Data refreshed successfully!")
                else:
                    print("Failed to refresh - connection error")
                continue
            else:
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(valid_expirations):
                        _, date, exp_data = valid_expirations[choice_num - 1]
                        print(f"\nShowing chart for {date}...")
                        self.plot_straddle_chart(exp_data)
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        # Cleanup
        if hasattr(self.thalex, 'disconnect'):
            await self.thalex.disconnect()

async def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    analyzer = InteractiveStraddleAnalyzer()
    try:
        await analyzer.run_interactive_session()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())