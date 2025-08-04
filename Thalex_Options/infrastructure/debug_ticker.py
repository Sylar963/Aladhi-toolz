#!/usr/bin/env python3
"""
Debug ticker responses to see what data we're getting
"""
import asyncio
import json
import logging

import thalex_py.thalex as th
import keys

# Configuration
NETWORK = th.Network.TEST
CID_LOGIN = 1000
CID_TICKER = 1001

async def debug_ticker_response():
    """Debug what ticker responses look like"""
    thalex = th.Thalex(NETWORK)
    
    try:
        # Connect and authenticate
        await thalex.connect()
        await thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK], id=CID_LOGIN)
        
        # Wait for login
        login_response = json.loads(await thalex.receive())
        print(f"Login response: {login_response}")
        
        # Get instruments
        await thalex.instruments()
        instruments_response = json.loads(await thalex.receive())
        
        instruments = instruments_response["result"]
        btc_options = [inst for inst in instruments 
                      if inst["underlying"] == "BTCUSD" and inst["type"] == "option"]
        
        print(f"Found {len(btc_options)} BTC options")
        
        # Test BTC perpetual ticker first
        print("\n=== Testing BTC Perpetual ===")
        await thalex.ticker("BTC-PERPETUAL", id=CID_TICKER)
        btc_response = json.loads(await thalex.receive())
        print(f"BTC ticker response: {json.dumps(btc_response, indent=2)}")
        
        # Test a few option tickers
        print("\n=== Testing Option Tickers ===")
        for i, option in enumerate(btc_options[:3]):  # Just first 3 options
            option_name = option["instrument_name"]
            print(f"\nTesting option: {option_name}")
            
            await thalex.ticker(option_name, id=CID_TICKER + i + 1)
            option_response = json.loads(await thalex.receive())
            print(f"Option ticker response: {json.dumps(option_response, indent=2)}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(thalex, 'disconnect'):
            await thalex.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_ticker_response())