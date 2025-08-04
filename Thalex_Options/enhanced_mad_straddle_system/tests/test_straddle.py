#!/usr/bin/env python3
"""
Quick test of the simplified straddle implementation
"""
import asyncio
import sys

# Import our new simplified straddle analyzer
from StraddleRanges import StraddleApp

async def quick_test():
    """Run a quick test of straddle data collection"""
    app = StraddleApp()
    
    try:
        # Connect and get initial data
        if not await app.analyzer.connect_and_authenticate():
            print("Failed to connect")
            return
            
        print("Connected successfully!")
        
        # Load instruments
        await app.analyzer.load_instruments_and_request_tickers()
        print(f"Loaded {len(app.analyzer.options_data)} options")
        
        # Wait for data
        await asyncio.sleep(3)
        
        # Check BTC price
        print(f"BTC Price: ${app.analyzer.perpetual_price:.2f}")
        
        # Find straddle
        straddle = app.analyzer.find_atm_straddle()
        if straddle:
            print(f"Found ATM straddle:")
            print(f"  Strike: ${straddle.atm_strike:.0f}")
            print(f"  Call Price: ${straddle.call_price:.2f}")
            print(f"  Put Price: ${straddle.put_price:.2f}")
            print(f"  Straddle Price: ${straddle.straddle_price:.2f}")
            print(f"  Upper Breakeven: ${straddle.upper_breakeven:.0f}")
            print(f"  Lower Breakeven: ${straddle.lower_breakeven:.0f}")
            print(f"  Range Width: ${straddle.range_width:.0f}")
            print(f"  Days to Expiry: {straddle.expiry_days:.1f}")
        else:
            print("No straddle data found")
            
        # Show some option data
        options_with_prices = [opt for opt in app.analyzer.options_data.values() if opt.mark_price > 0]
        print(f"Options with prices: {len(options_with_prices)}")
        
        if options_with_prices:
            # Show first 5 options
            for i, opt in enumerate(options_with_prices[:5]):
                print(f"  {opt.name}: ${opt.mark_price:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(app.analyzer.thalex, 'disconnect'):
            await app.analyzer.thalex.disconnect()

if __name__ == "__main__":
    asyncio.run(quick_test())