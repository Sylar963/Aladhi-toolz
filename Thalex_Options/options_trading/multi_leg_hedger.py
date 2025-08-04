import asyncio
import csv
import json
import logging
import socket
import time
from typing import Optional, Dict, List
import datetime

import websockets
import keys
import thalex as th

# Configuration
PERPETUAL = "BTC-PERPETUAL"
PERP_TICKER = f"ticker.{PERPETUAL}.1000ms"
ORDER_DELAY = 0.1

# We will be trading on testnet
NETWORK = th.Network.TEST
KEY_ID = keys.key_ids[NETWORK]
PRIV_KEY = keys.private_keys[NETWORK]

# Call IDs for matching responses
CID_PORTFOLIO = 1001
CID_INSERT = 1002
CID_CANCEL = 1003

class MultiLegHedger:
    def __init__(self, thalex):
        self.perp_position = 0
        self.options_positions = {}
        self.mark_prices = {}
        self.order_book = {}
        self.active_orders = {}
        self.thalex = thalex
        self.last_order_ts = 0
        
        # Configuration
        self.HEDGE_RATIO = 0.5  # How much of perp position to hedge with options
        self.STRIKES_TO_QUOTE = 3  # Number of strikes above/below current price
        self.MAX_POSITION = 5  # Maximum allowed position
        self.QUOTE_SIZE = 0.1
        self.SPREAD_PCT = 0.02
        
    def calculate_strikes_to_quote(self) -> List[str]:
        """Calculate which strikes to quote based on current price"""
        if PERPETUAL not in self.mark_prices:
            return []
            
        current_price = self.mark_prices[PERPETUAL]
        strike_increment = 1000  # Assuming $1000 strike increments
        
        strikes = []
        base_strike = round(current_price / strike_increment) * strike_increment
        
        for i in range(-self.STRIKES_TO_QUOTE, self.STRIKES_TO_QUOTE + 1):
            strike = base_strike + (i * strike_increment)
            # Format the option instrument name - adjust expiry as needed
            option_name = f"BTC-25OCT24-{strike}-C"
            strikes.append(option_name)
            
        return strikes

    def calculate_quote_size(self, strike: str) -> float:
        """Calculate appropriate size for each quote"""
        base_size = abs(self.perp_position) * self.HEDGE_RATIO / (2 * self.STRIKES_TO_QUOTE + 1)
        return min(base_size, self.QUOTE_SIZE, self.MAX_POSITION - abs(self.perp_position))

    async def manage_quotes(self):
        """Manage all option quotes"""
        if abs(self.perp_position) < 0.1:
            # Cancel all quotes if position is too small
            await self.cancel_existing_quotes()
            return
            
        strikes = self.calculate_strikes_to_quote()
        
        for strike in strikes:
            if self.perp_position > 0:
                # If long perp, sell calls
                direction = th.Direction.SELL
            else:
                # If short perp, buy calls
                direction = th.Direction.BUY
                
            size = self.calculate_quote_size(strike)
            if size > 0:
                await self.place_option_quote(strike, direction, size)

    async def place_option_quote(self, instrument: str, direction: th.Direction, size: float):
        """Place a new quote for an option"""
        if self.last_order_ts + ORDER_DELAY > time.time():
            return
            
        # Calculate price based on mark and spread
        if instrument not in self.mark_prices:
            return
            
        base_price = self.mark_prices[instrument]
        price = base_price * (1 + (self.SPREAD_PCT if direction == th.Direction.SELL else -self.SPREAD_PCT))
        
        await self.thalex.insert(
            id=CID_INSERT,
            direction=direction,
            instrument_name=instrument,
            order_type=th.OrderType.LIMIT,
            amount=size,
            price=price
        )
        self.last_order_ts = time.time()

    async def cancel_existing_quotes(self):
        """Cancel all existing option quotes"""
        if self.active_orders:
            for order_id in list(self.active_orders.keys()):
                await self.thalex.cancel(id=CID_CANCEL, order_id=order_id)

    def notification(self, channel: str, notification: dict):
        """Handle websocket notifications"""
        if "ticker" in channel:
            instrument = channel.split(".")[1]
            self.mark_prices[instrument] = notification["mark_price"]

    def result(self, msg_id: int, result: dict):
        """Handle API call results"""
        if msg_id is None:
            return
            
        if msg_id == CID_PORTFOLIO:
            # Update positions from portfolio
            for pos in result:
                if pos["instrument_name"] == PERPETUAL:
                    self.perp_position = pos["position"]
                else:
                    self.options_positions[pos["instrument_name"]] = pos["position"]
                    
        elif msg_id == CID_INSERT:
            # Track new orders
            if "order_id" in result:
                self.active_orders[result["order_id"]] = result
                
        elif msg_id == CID_CANCEL:
            # Remove cancelled orders
            if "order_id" in result:
                self.active_orders.pop(result["order_id"], None)

    def flush_data(self, trades, error=None):
        """Log trade data to CSV"""
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open("multi_leg_hedger.csv", "a", newline="") as f:
            w = csv.writer(f)
            data = {
                "timestamp": time_now,
                "perp_position": round(self.perp_position, 3),
                "options_positions": json.dumps(self.options_positions),
                "error": error
            }
            if f.tell() == 0:
                w.writerow(data.keys())
            w.writerow(data.values())

    async def manage_positions(self):
        """Main loop for managing positions and quotes"""
        await self.thalex.connect()
        await self.thalex.login(KEY_ID, PRIV_KEY)
        await self.thalex.set_cancel_on_disconnect(5)
        
        # Subscribe to market data
        instruments = [PERP_TICKER]
        instruments.extend([f"ticker.{strike}.1000ms" for strike in self.calculate_strikes_to_quote()])
        await self.thalex.public_subscribe(instruments)
        
        # Get initial portfolio
        await self.thalex.portfolio(CID_PORTFOLIO)
        
        while True:
            msg = json.loads(await self.thalex.receive())
            error = msg.get("error")
            if error is not None:
                logging.error(msg)
                self.flush_data(error=error["message"], trades=[])
                continue
                
            channel = msg.get("channel_name")
            if channel is not None:
                self.notification(channel, msg["notification"])
            else:
                self.result(msg.get("id"), msg["result"])
                
            await self.manage_quotes()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    run = True
    while run:
        thalex = th.Thalex(NETWORK)
        hedger = MultiLegHedger(thalex)
        task = asyncio.create_task(hedger.manage_positions())
        try:
            await task
        except (websockets.ConnectionClosed, socket.gaierror):
            logging.exception(f"Lost connection. Reconnecting...")
            time.sleep(0.1)
        except asyncio.CancelledError:
            logging.info("Signal received. Stopping...")
            run = False
        if thalex.connected():
            await thalex.disconnect()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main()) 
