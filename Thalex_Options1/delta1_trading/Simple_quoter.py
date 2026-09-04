import asyncio
import json
import logging
import socket
import sys
import time
from typing import Optional, Deque
from collections import deque
import websockets

import thalex
from thalex.thalex import Direction
import keys  # Rename _keys.py to keys.py and add your keys. There are instructions how to create keys in that file.

NETWORK = thalex.Network.TEST
ORDER_LABEL = "simple_quoter"
INSTRUMENT = "BTC-PERPETUAL"
PRICE_TICK = 1
SIZE_TICK = 0.001
HALF_SPREAD = .5
AMEND_THRESHOLD = 5
SIZE = 0.1
MAX_POSITION = 0.3
QUOTE_ID = {Direction.BUY: 1001, Direction.SELL: 1002}

# New parameters for TAR calculation
price_history: Deque[float] = deque(maxlen=14)  # Store last 14 prices for TAR calculation

def round_to_tick(value):
    return PRICE_TICK * round(value / PRICE_TICK)

def round_size(size):
    return SIZE_TICK * round(size / SIZE_TICK)

def calculate_true_average_range():
    if len(price_history) < 2:
        return HALF_SPREAD  # Default to HALF_SPREAD if not enough data
    return max(price_history) - min(price_history)

class Quoter:
    def __init__(self, tlx: thalex.Thalex):
        self.tlx = tlx
        self.mark: Optional[float] = None
        self.quotes: dict[thalex.Direction, Optional[dict]] = {
            Direction.BUY: {},
            Direction.SELL: {},
        }
        self.position: Optional[float] = None

    async def adjust_order(self, side, price, amount):
        confirmed = self.quotes[side]
        assert confirmed is not None
        is_open = (confirmed.get("status") or "") in [
            "open",
            "partially_filled",
        ]
        if is_open:
            if amount == 0 or abs(confirmed["price"] - price) > AMEND_THRESHOLD:
                logging.info(f"Amending {side} to {amount} @ {price}")
                await self.tlx.amend(
                    amount=amount,
                    price=price,
                    client_order_id=QUOTE_ID[side],
                    id=QUOTE_ID[side],
                )
        elif amount > 0:
            logging.info(f"Inserting {side}: {amount} @ {price}")
            await self.tlx.insert(
                amount=amount,
                price=price,
                direction=side,
                instrument_name=INSTRUMENT,
                client_order_id=QUOTE_ID[side],
                id=QUOTE_ID[side],
                label=ORDER_LABEL,
            )
            self.quotes[side] = {"status": "open", "price": price}

    async def update_quotes(self, new_mark):
        up = self.mark is None or new_mark > self.mark
        self.mark = new_mark
        price_history.append(new_mark)  # Update price history for TAR calculation
        tar = calculate_true_average_range()  # Calculate TAR

        if self.position is None:
            return

        bid_price = round_to_tick(new_mark - (tar / 10_000 * new_mark))
        bid_size = round_size(max(min(SIZE, MAX_POSITION - self.position), 0))
        ask_price = round_to_tick(new_mark + (tar / 10_000 * new_mark))
        ask_size = round_size(max(min(SIZE, MAX_POSITION + self.position), 0))

        if up:
            await self.adjust_order(Direction.SELL, price=ask_price, amount=ask_size)
            await self.adjust_order(Direction.BUY, price=bid_price, amount=bid_size)
        else:
            await self.adjust_order(Direction.BUY, price=bid_price, amount=bid_size)
            await self.adjust_order(Direction.SELL, price=ask_price, amount=ask_size)

    async def handle_notification(self, channel: str, notification):
        logging.debug(f"notification in channel {channel} {notification}")
        if channel == "session.orders":
            for order in notification:
                self.quotes[Direction(order["direction"])] = order
        elif channel.startswith("lwt"):
            await self.update_quotes(new_mark=notification["m"])
        elif channel == "account.portfolio":
            await self.tlx.cancel_session()
            self.quotes = {Direction.BUY: {}, Direction.SELL: {}}
            try:
                self.position = next(
                    p for p in notification if p["instrument_name"] == INSTRUMENT
                )["position"]
            except StopIteration:
                self.position = self.position or 0
            logging.info(f"Portfolio updated - {INSTRUMENT} position: {self.position}")

    async def quote(self):
        await self.tlx.connect()
        await self.tlx.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK])
        await self.tlx.set_cancel_on_disconnect(timeout_secs=6)
        await self.tlx.public_subscribe([f"lwt.{INSTRUMENT}.1000ms"])
        await self.tlx.private_subscribe(["session.orders", "account.portfolio"])
        while True:
            msg = await self.tlx.receive()
            msg = json.loads(msg)
            if "channel_name" in msg:
                await self.handle_notification(msg["channel_name"], msg["notification"])
            elif "result" in msg:
                logging.debug(msg)
            else:
                logging.error(msg)
                await self.tlx.cancel_session()
                self.quotes = {Direction.BUY: {}, Direction.SELL: {}}


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("log.txt", mode="a"),
        ],
    )
    run = True
    while run:
        tlx = thalex.Thalex(network=NETWORK)
        quoter = Quoter(tlx)
        task = asyncio.create_task(quoter.quote())
        try:
            await task
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            logging.error(f"Lost connection ({e}). Reconnecting...")
            time.sleep(0.1)
        except asyncio.CancelledError:
            logging.info("Quoting cancelled")
            run = False
        except:
            logging.exception("There was an unexpected error:")
            run = False
        if tlx.connected():
            await tlx.cancel_session()
            await tlx.disconnect()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())

