import asyncio
import json
import logging
import socket
import time
from typing import Union, Dict, Optional, List, Any
import enum
import websockets
from threading import Lock
import numpy as np
from collections import deque

import thalex as th
import keys  # Rename _keys.py to keys.py and add your keys. There are instructions how to create keys in that file.

# Market Configuration
MARKET_CONFIG = {
    "underlying": "BTCUSD",
    "network": th.Network.TEST,
    "label": "P",
}

# Order Book Configuration
ORDERBOOK_CONFIG = {
    "spread": 0.5,          # Base spread in ticks
    "amend_threshold": 25,  # Minimum price change to amend orders
    "bid_step": 25,        # Price step between bid levels
    "ask_step": 25,        # Price step between ask levels
    "bid_sizes": [0.2, 0.8],  # Size for each bid level
    "ask_sizes": [0.2, 0.8],  # Size for each ask level
}

# Risk Management Configuration
RISK_LIMITS = {
    "max_position": 1.0,     # Maximum position size
    "max_notional": 50000,   # Maximum notional exposure (USD)
    "stop_loss_pct": 0.06,   # Stop loss percentage
    "base_take_profit_pct": 0.02,  # Base take profit target
    "max_take_profit_pct": 0.05,   # Maximum take profit
    "min_take_profit_pct": 0.01,   # Minimum take profit
    "rebalance_threshold": 0.8,  # Rebalance trigger
    "take_profit_pct": 0.03,  # Default take profit percentage
    "trailing_stop_activation": 0.015,  # Activate trailing stop at 1.5% profit
    "trailing_stop_distance": 0.01,    # Trailing stop follows at 1% distance
}

# Advanced Trading Parameters
TRADING_PARAMS = {
    "trailing_stop": {
        "activation": 0.015,  # Activate at 1.5% profit
        "distance": 0.01,     # 1% trailing distance
    },
    "position_management": {
        
        "gamma": 0.1,               # Risk aversion (Avellaneda-Stoikov)
        "inventory_weight": 0.5,    # Inventory skew factor
    },
    "volatility": {
        "window": 100,        # Volatility calculation window
        "min_samples": 20,    # Minimum samples for vol calc
        "scaling": 1.0,       # Volatility scaling factor
    }
}

# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    "zscore": {
        "window": 100,
        "threshold": 2.0
    },
    "atr": {
        "period": 14,
        "multiplier": 1.0
    }
}

# API Call IDs
CALL_IDS = {
    "instruments": 0,
    "instrument": 1,
    "subscribe": 2,
    "login": 3,
    "cancel_session": 4,
    "set_cod": 5
}

# Performance Configuration
PERFORMANCE_METRICS = {
    "successful_trades": 0,
    "average_fill_price": 0.0,
    "total_trades": 0,
    "win_loss_ratio": 0.0,
    "expected_value": 0.0,
    "window_size": 100,        # Window for calculating metrics
    "min_trades": 10,         # Minimum trades for metrics calculation
    "target_profit": 0.02,    # Target profit percentage
    "max_drawdown": 0.05,     # Maximum allowable drawdown
    "max_position_time": 3600  # Maximum position holding time in seconds
}

# Keep original variable names for code compatibility
UNDERLYING = MARKET_CONFIG["underlying"]
NETWORK = MARKET_CONFIG["network"]
LABEL = MARKET_CONFIG["label"]
SPREAD = ORDERBOOK_CONFIG["spread"]
BID_STEP = ORDERBOOK_CONFIG["bid_step"]
ASK_STEP = ORDERBOOK_CONFIG["ask_step"]
BID_SIZES = ORDERBOOK_CONFIG["bid_sizes"]
ASK_SIZES = ORDERBOOK_CONFIG["ask_sizes"]
AMEND_THRESHOLD = ORDERBOOK_CONFIG["amend_threshold"]
POSITION_LIMITS = RISK_LIMITS

# Add metrics lock
metrics_lock = Lock()

# Call IDs for Thalex API
CALL_ID_INSTRUMENTS = 0
CALL_ID_INSTRUMENT = 1
CALL_ID_SUBSCRIBE = 2
CALL_ID_LOGIN = 3
CALL_ID_CANCEL_SESSION = 4
CALL_ID_SET_COD = 5

# Order status enumeration
class OrderStatus(enum.Enum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    CANCELLED_PARTIALLY_FILLED = "cancelled_partially_filled"
    FILLED = "filled"

# Order data structure
class Order:
    def __init__(self, oid: int, price: float, amount: float, status: Optional[OrderStatus] = None):
        self.id: int = oid
        self.price: float = price
        self.amount: float = amount
        self.status: Optional[OrderStatus] = status

    def is_open(self):
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

# Ticker data structure
class Ticker:
    def __init__(self, data: Dict):
        self.mark_price: float = data["mark_price"]
        self.best_bid: Optional[float] = data.get("best_bid_price")
        self.best_ask: Optional[float] = data.get("best_ask_price")
        self.index: float = data["index"]
        self.mark_ts: float = data["mark_timestamp"]
        self.funding_rate: float = data["funding_rate"]

# Perpetual Quoter class
class PerpQuoter:
    def __init__(self, thalex: th.Thalex):
        self.thalex = thalex
        self.ticker: Optional[Ticker] = None
        self.index: Optional[float] = None
        self.quote_cv = asyncio.Condition()
        self.portfolio: Dict[str, float] = {}
        self.orders: List[List[Order]] = [[], []]  # bids, asks
        self.client_order_id: int = 100
        self.tick: Optional[float] = None
        self.perp_name: Optional[str] = None
        
        # Position tracking
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_rebalance = time.time()
        self.position_start_time = 0.0
        
        # Risk management
        self.highest_profit = 0.0
        self.trailing_stop_active = False
        self.entry_prices = {}
        
        # Market making parameters
        self.gamma = TRADING_PARAMS["position_management"]["gamma"]
        self.k = 1.5  # Order flow intensity
        self.sigma = 0.0  # Market volatility (dynamic)
        self.T = 1.0  # Time horizon
        
        # Price tracking and caching
        self.price_window = deque(maxlen=TRADING_PARAMS["volatility"]["window"])
        self.price_history = deque(maxlen=TRADING_PARAMS["volatility"]["window"])
        self.last_atr = 0.0
        self.last_atr_time = 0
        self.last_zscore = 0.0
        self.last_zscore_time = 0
        
        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "average_fill_price": 0.0,
            "successful_trades": 0,
            "win_loss_ratio": 0.0,
            "expected_value": 0.0
        }
        
        # Order tracking
        self.order_cache = {}  # client_order_id -> Order mapping
        self.last_order_update = {}  # client_order_id -> timestamp
        self.max_orders = 50  # Maximum number of open orders
        
        # Rate limiting
        self.last_quote_time = 0
        self.last_order_time = 0
        self.min_order_interval = 0.1  # Minimum time between orders in seconds
        
        # Alert management
        self.alert_counts = {}
        self.alert_cooldown = 50  # 5 minutes
        self.last_alert_time = {}
        self.quoting_enabled = True

    def round_to_tick(self, value):
        return self.tick * round(value / self.tick)

    def calculate_zscore(self) -> float:
        """Calculate Z-score for current price with caching"""
        current_time = time.time()
        if current_time - self.last_zscore_time < 1.0:  # Cache for 1 second
            return self.last_zscore
            
        if len(self.price_history) < 20:
            self.last_zscore = 0
            self.last_zscore_time = current_time
            return 0
            
        prices = np.array(self.price_history)
        self.last_zscore = (self.ticker.mark_price - np.mean(prices)) / np.std(prices)
        self.last_zscore_time = current_time
        return self.last_zscore

    def calculate_atr(self) -> float:
        """Calculate ATR with caching"""
        current_time = time.time()
        if current_time - self.last_atr_time < 1.0:  # Cache for 1 second
            return self.last_atr
            
        if len(self.price_history) < 2:
            self.last_atr = 0.0
            self.last_atr_time = current_time
            return 0.0
        
        # Convert deque to numpy array for calculations
        prices = np.array(list(self.price_history))
        # Calculate differences between consecutive prices
        high_low = np.abs(np.diff(prices))
        # Calculate and cache ATR
        self.last_atr = np.mean(high_low)
        self.last_atr_time = current_time
        return self.last_atr

    async def can_place_order(self) -> bool:
        """Check if we can place a new order based on rate limits and order count"""
        current_time = time.time()
        
        # Check rate limiting
        if current_time - self.last_order_time < self.min_order_interval:
            return False
            
        # Check maximum order count
        open_orders = len([o for orders in self.orders for o in orders if o.is_open()])
        if open_orders >= self.max_orders:
            logging.warning(f"Maximum order count reached: {open_orders}")
            return False
            
        # Update last order time
        self.last_order_time = current_time
        return True

    async def check_risk_limits(self) -> bool:
        """Check if current position is within risk limits with alert management"""
        current_time = time.time()
        
        if abs(self.position_size) >= POSITION_LIMITS["max_position"]:
            alert_key = "position_size"
            if self.should_alert(alert_key, current_time):
                logging.warning(f"Position size {self.position_size} exceeds limit")
                await self.handle_risk_breach()
            return False
            
        if self.ticker:
            notional = abs(self.position_size * self.ticker.mark_price)
            if notional >= POSITION_LIMITS["max_notional"]:
                alert_key = "notional"
                if self.should_alert(alert_key, current_time):
                    logging.warning(f"Notional value {notional} exceeds limit")
                    await self.handle_risk_breach()
                return False
                
        return True

    def should_alert(self, alert_key: str, current_time: float) -> bool:
        """Determine if alert should be shown based on cooldown"""
        if alert_key not in self.alert_counts:
            self.alert_counts[alert_key] = 0
            self.last_alert_time[alert_key] = 0

        if current_time - self.last_alert_time[alert_key] > self.alert_cooldown:
            self.alert_counts[alert_key] = 0

        if self.alert_counts[alert_key] < 2:
            self.alert_counts[alert_key] += 1
            self.last_alert_time[alert_key] = current_time
            return True
        return False

    async def handle_risk_breach(self):
        """Handle risk limit breach"""
        self.quoting_enabled = False
        zscore = self.calculate_zscore()
        atr = self.calculate_atr()
        
        # Market condition based take profit strategy
        if abs(zscore) > 2:  # Extreme market condition
            take_profit_pct = POSITION_LIMITS["take_profit_pct"] * 1.5
        else:
            take_profit_pct = POSITION_LIMITS["take_profit_pct"]
            
        # Adjust take profit based on volatility
        if atr > 0:
            take_profit_pct *= (1 + atr/100)
        
        await self.smart_position_reduction(take_profit_pct)

    async def smart_position_reduction(self, take_profit_pct: float):
        """Reduce position size based on market conditions"""
        if self.position_size == 0:
            return

        reduction_size = self.position_size * 0.25  # Reduce 25% at a time
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        
        price_adjustment = (1 + take_profit_pct) if self.position_size > 0 else (1 - take_profit_pct)
        target_price = self.entry_price * price_adjustment
        
        logging.info(f"Starting smart position reduction: {reduction_size} @ {target_price}")
        
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=abs(reduction_size),
            price=target_price,
            client_order_id=self.client_order_id,
            id=self.client_order_id
        )
        self.client_order_id += 1

    async def manage_position(self):
        """Monitor and manage open positions"""
        if self.position_size == 0 or not self.ticker or self.entry_price == 0:
            return

        try:
            # Calculate PnL
            pnl_pct = (self.ticker.mark_price - self.entry_price) / self.entry_price
            if self.position_size < 0:
                pnl_pct = -pnl_pct
        except ZeroDivisionError:
            logging.error("Entry price is zero, skipping position management")
            return

        # Check stop loss
        if pnl_pct <= -POSITION_LIMITS["stop_loss_pct"]:
            await self.close_position("Stop loss triggered")
            
        # Check take profit    
        elif pnl_pct >= POSITION_LIMITS["take_profit_pct"]:
            await self.close_position("Take profit triggered")
            
        # Check rebalance
        elif abs(self.position_size) >= POSITION_LIMITS["max_position"] * POSITION_LIMITS["rebalance_threshold"]:
            await self.rebalance_position()

    async def close_position(self, reason: str):
        """Close entire position"""
        if self.position_size == 0:
            return
            
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        logging.info(f"Closing position: {reason}")
        
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=abs(self.position_size),
            price=self.ticker.mark_price,
            client_order_id=self.client_order_id,
            id=self.client_order_id
        )
        self.client_order_id += 1

    async def rebalance_position(self):
        """Rebalance position back to target size"""
        target = POSITION_LIMITS["max_position"] * 0.5
        reduce_by = abs(self.position_size) - target
        
        if reduce_by <= 0:
            return
            
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        logging.info(f"Rebalancing position by {reduce_by}")
        
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=reduce_by,
            price=self.ticker.mark_price,
            client_order_id=self.client_order_id,
            id=self.client_order_id
        )
        self.client_order_id += 1

    async def make_quotes(self) -> List[List[th.SideQuote]]:
        """Generate quotes using Avellaneda-Stoikov model"""
        if not await self.check_risk_limits():
            return [[], []]
            
        mid_price = self.ticker.mark_price
        optimal_spread = self.calculate_optimal_spread()
        inventory_skew = self.calculate_inventory_skew()
        
        # Adjust base quotes by inventory position
        base_bid = mid_price - optimal_spread/2 - inventory_skew
        base_ask = mid_price + optimal_spread/2 - inventory_skew
        
        # Generate layered quotes
        bids = [
            th.SideQuote(
                price=self.round_to_tick(base_bid - BID_STEP * lvl * self.tick),
                amount=self.calculate_level_size(BID_SIZES[lvl], "bid", lvl)
            )
            for lvl, amt in enumerate(BID_SIZES)
        ]
        
        asks = [
            th.SideQuote(
                price=self.round_to_tick(base_ask + ASK_STEP * lvl * self.tick),
                amount=self.calculate_level_size(ASK_SIZES[lvl], "ask", lvl)
            )
            for lvl, amt in enumerate(ASK_SIZES)
        ]
        
        return [bids, asks]

    async def quote_task(self):
        """Main quoting loop with sanity checks"""
        while True:
            async with self.quote_cv:
                await self.quote_cv.wait()
            if not await self.sanity_check():
                logging.error("Sanity check failed, skipping quote")
                continue
            if self.ticker and self.index:
                await self.manage_take_profit()
                quotes = await self.make_quotes()
                await self.adjust_quotes(quotes)

    async def adjust_quotes(self, desired: List[List[th.SideQuote]]):
        """Adjust quotes with rate limiting and validation"""
        if not await self.can_place_order():
            logging.debug("Rate limit or order count exceeded, skipping quote adjustment")
            return
            
        for side_i, side in enumerate([th.Direction.BUY, th.Direction.SELL]):
            orders = self.orders[side_i]
            quotes = desired[side_i]
            
            # Cancel excess orders
            for i in range(len(quotes), len(orders)):
                if orders[i].is_open():
                    await self.thalex.cancel(client_order_id=orders[i].id, id=orders[i].id)
                    self.cleanup_cancelled_order(orders[i])
            
            # Process each quote level
            for q_lvl, q in enumerate(quotes):
                # Validate quote size
                if q.a <= 0 or q.p <= 0:
                    logging.error(f"Invalid quote: price={q.p}, amount={q.a}")
                    continue
                    
                # Place new order or update existing
                if len(orders) <= q_lvl or (orders[q_lvl].status is not None and not orders[q_lvl].is_open()):
                    if not await self.can_place_order():
                        logging.debug("Rate limit reached during quote placement")
                        return
                        
                    client_order_id = self.client_order_id
                    self.client_order_id += 1
                    
                    # Create new order
                    new_order = Order(client_order_id, q.p, q.a)
                    if len(orders) <= q_lvl:
                        orders.append(new_order)
                    else:
                        orders[q_lvl] = new_order
                        
                    # Place order and update cache
                    await self.thalex.insert(
                        direction=side,
                        instrument_name=self.perp_name,
                        amount=q.a,
                        price=q.p,
                        post_only=True,
                        label=LABEL,
                        client_order_id=client_order_id,
                        id=client_order_id
                    )
                    self.order_cache[client_order_id] = new_order
                    self.last_order_update[client_order_id] = time.time()
                    
                elif abs(orders[q_lvl].price - q.p) > AMEND_THRESHOLD * self.tick:
                    # Amend existing order
                    await self.thalex.amend(
                        amount=q.a,
                        price=q.p,
                        client_order_id=orders[q_lvl].id,
                        id=orders[q_lvl].id
                    )
                    # Update cache
                    orders[q_lvl].price = q.p
                    orders[q_lvl].amount = q.a
                    self.order_cache[orders[q_lvl].id] = orders[q_lvl]
                    self.last_order_update[orders[q_lvl].id] = time.time()

    async def await_instruments(self):
        await self.thalex.instruments(CALL_ID_INSTRUMENTS)
        msg = await self.thalex.receive()
        msg = json.loads(msg)
        assert msg["id"] == CALL_ID_INSTRUMENTS
        for i in msg["result"]:
            if i["type"] == "perpetual" and i["underlying"] == UNDERLYING:
                self.tick = i["tick_size"]
                self.perp_name = i["instrument_name"]
                return
        assert False  # Perp not found

    async def listen_task(self):
        await self.thalex.connect()
        await self.await_instruments()
        await self.thalex.login(keys.key_ids[NETWORK], keys.private_keys[NETWORK], id=CALL_ID_LOGIN)
        await self.thalex.set_cancel_on_disconnect(6, id=CALL_ID_SET_COD)
        await self.thalex.private_subscribe(["session.orders", "account.portfolio", "account.trade_history"], id=CALL_ID_SUBSCRIBE)
        await self.thalex.public_subscribe([f"ticker.{self.perp_name}.raw", f"price_index.{UNDERLYING}"], id=CALL_ID_SUBSCRIBE)
        
        while True:
            await self.manage_position()
            msg = await self.thalex.receive()
            msg = json.loads(msg)
            if "channel_name" in msg:
                await self.notification(msg["channel_name"], msg["notification"])
            elif "result" in msg:
                await self.result_callback(msg["result"], msg.get("id"))  # Use class method
            else:
                await self.error_callback(msg["error"], msg.get("id"))

    async def notification(self, channel: str, notification: Union[Dict, List[Dict]]):
        """Handle incoming notifications from different channels.
        
        Args:
            channel: The notification channel name
            notification: The notification payload
        """
        try:
            if not isinstance(notification, (dict, list)):
                logging.error(f"Invalid notification format: {type(notification)}")
                return
                
            if channel.startswith("ticker."):
                await self.ticker_callback(notification)
            elif channel.startswith("price_index."):
                await self.index_callback(notification["price"])
            elif channel == "session.orders":
                await self.orders_callback(notification)
            elif channel == "account.portfolio":
                self.portfolio_callback(notification)
            elif channel == "account.trade_history":
                await self.trades_callback(notification)
            else:
                logging.error(f"Notification for unknown channel: {channel}")
        except Exception as e:
            logging.error(f"Error processing notification: {str(e)}")

    async def ticker_callback(self, ticker: Dict[str, Any]):
        """Handle ticker updates."""
        try:
            self.ticker = Ticker(ticker)
            self.price_history.append(self.ticker.mark_price)
            async with self.quote_cv:
                self.quote_cv.notify()
        except Exception as e:
            logging.error(f"Error in ticker callback: {str(e)}")

    async def index_callback(self, index: float):
        """Handle index price updates."""
        try:
            self.index = float(index)  # Validate type
            async with self.quote_cv:
                self.quote_cv.notify()
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid index value: {str(e)}")

    async def trades_callback(self, trades: List[Dict]):
        """Process trades with validation and proper position tracking"""
        for t in trades:
            if t.get("label") == LABEL:
                amount = float(t["amount"])
                price = float(t["price"])
                
                if amount == 0 or price <= 0:
                    logging.error(f"Invalid trade data: amount={amount}, price={price}")
                    continue

                if t["direction"] == "sell":
                    amount = -amount
                
                old_position = self.position_size
                old_entry = self.entry_price
                
                # Update position and entry price
                if self.position_size == 0:
                    self.entry_price = price
                else:
                    # Calculate new entry price based on weighted average
                    total_value = (old_position * old_entry) + (amount * price)
                    self.entry_price = total_value / (old_position + amount)
                
                self.position_size += amount
                
                # Update metrics
                with metrics_lock:
                    self.update_performance_metrics(t)
                
                # Verify position update
                if abs(self.position_size) > POSITION_LIMITS["max_position"] * 1.1:
                    logging.error(f"Position limit breach after trade: {self.position_size}")
                    await self.emergency_close()

    async def order_error(self, error, oid):
        """Handle order errors with proper cleanup"""
        logging.error(f"Error with order({oid}): {error}")
        try:
            # Check order cache first
            if oid in self.order_cache:
                order = self.order_cache[oid]
                if order.is_open():
                    await self.thalex.cancel(client_order_id=oid, id=oid)
                self.cleanup_cancelled_order(order)
                return
                
            # Fallback to searching in order lists
            for side in [0, 1]:
                for i, o in enumerate(self.orders[side]):
                    if o.id == oid:
                        if o.is_open():
                            await self.thalex.cancel(client_order_id=oid, id=oid)
                        self.cleanup_cancelled_order(o)
                        return
                        
        except Exception as e:
            logging.error(f"Error handling order error: {str(e)}")

    async def error_callback(self, error, cid=None):
        """Handle API errors with context"""
        try:
            if cid > 99:
                await self.order_error(error, cid)
            else:
                logging.error(f"{cid=}: error: {error}")
                if cid == CALL_ID_LOGIN:
                    logging.critical("Login failed, cannot continue")
                    raise RuntimeError("Login failed")
                elif cid == CALL_ID_SET_COD:
                    logging.warning("Failed to set cancel on disconnect")
        except Exception as e:
            logging.error(f"Error in error_callback: {str(e)}")

    def update_order(self, order):
        """Update order in both cache and order lists"""
        if not order:
            return False
            
        # Update cache
        self.order_cache[order.id] = order
        self.last_order_update[order.id] = time.time()
        
        # Update order lists
        for side in [0, 1]:
            for i, have in enumerate(self.orders[side]):
                if have.id == order.id:
                    self.orders[side][i] = order
                    return True
                    
        return False

    async def orders_callback(self, orders: List[Dict]):
        """Process order updates with validation"""
        try:
            current_time = time.time()
            
            # Clean up expired orders from cache
            expired_orders = [
                oid for oid, last_time in self.last_order_update.items()
                if current_time - last_time > 300  # 5 minutes
            ]
            for oid in expired_orders:
                if oid in self.order_cache:
                    self.cleanup_cancelled_order(self.order_cache[oid])
            
            # Process new updates
            for o in orders:
                try:
                    # Create and validate order
                    order = self.order_from_data(o)
                    if not order:
                        continue
                        
                    # Update order tracking
                    if not self.update_order(order):
                        logging.warning(f"Order not found in tracking: {o}")
                        
                    # Handle filled orders
                    if order.status == OrderStatus.FILLED:
                        await self.process_filled_order(order)
                    elif order.status == OrderStatus.PARTIALLY_FILLED:
                        await self.process_partial_fill(order)
                    elif order.status in [OrderStatus.CANCELLED, OrderStatus.CANCELLED_PARTIALLY_FILLED]:
                        self.cleanup_cancelled_order(order)
                        
                except Exception as e:
                    logging.error(f"Error processing order update: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error in orders_callback: {str(e)}")

    def portfolio_callback(self, portfolio: List[Dict]):
        """Update portfolio positions with validation"""
        try:
            for position in portfolio:
                instrument = position.get("instrument_name")
                pos_size = position.get("position")
                
                if not instrument or pos_size is None:
                    logging.error(f"Invalid portfolio data: {position}")
                    continue
                    
                try:
                    pos_size = float(pos_size)
                    self.portfolio[instrument] = pos_size
                    
                    # Verify position consistency
                    if instrument == self.perp_name:
                        if abs(self.position_size - pos_size) > 1e-6:
                            logging.warning(
                                f"Position mismatch: internal={self.position_size}, "
                                f"portfolio={pos_size}"
                            )
                            self.position_size = pos_size
                            
                except ValueError as e:
                    logging.error(f"Error parsing position size: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error in portfolio_callback: {str(e)}")

    def order_from_data(self, data: Dict) -> Order:
        """Create Order object from API data with validation"""
        try:
            client_order_id = data["client_order_id"]
            price = float(data["price"])
            amount = float(data["amount"])
            status = OrderStatus(data["status"])
            
            if price <= 0 or amount <= 0:
                logging.error(f"Invalid order data: price={price}, amount={amount}")
                return None
                
            order = Order(client_order_id, price, amount, status)
            
            # Update order cache
            self.order_cache[client_order_id] = order
            self.last_order_update[client_order_id] = time.time()
            
            return order
        except (KeyError, ValueError) as e:
            logging.error(f"Error parsing order data: {str(e)}")
            return None

    async def result_callback(self, result, cid=None):
        """Handle API call results with proper error handling"""
        try:
            if cid == CALL_ID_INSTRUMENT:
                logging.debug(f"Instrument result: {result}")
            elif cid == CALL_ID_SUBSCRIBE:
                logging.info(f"Subscription confirmed: {result}")
            elif cid == CALL_ID_LOGIN:
                logging.info("Login successful")
            elif cid == CALL_ID_SET_COD:
                logging.debug("Cancel on disconnect set")
            elif cid > 99:
                # Handle order results
                if "error" in result:
                    await self.order_error(result["error"], cid)
                    # Clean up failed order
                    if cid in self.order_cache:
                        self.cleanup_cancelled_order(self.order_cache[cid])
                else:
                    logging.debug(f"Order {cid} result: {result}")
                    # Update order status if available
                    if "status" in result:
                        order = self.order_cache.get(cid)
                        if order:
                            order.status = OrderStatus(result["status"])
                            self.last_order_update[cid] = time.time()
            else:
                logging.debug(f"Result {cid}: {result}")
        except Exception as e:
            logging.error(f"Error in result_callback: {str(e)}")
            if cid > 99:
                # Ensure cleanup on error
                await self.order_error(str(e), cid)

    # When receiving new prices
    def update_price(self, new_price):
        self.price_history.append(float(new_price))

    async def handle_risk_breach(self):
        """Handle risk limit breach by reducing position"""
        if self.position_size == 0:
            return
            
        # Calculate reduction size (50% of current position)
        reduction_size = -self.position_size * 0.5
        
        # Calculate target price based on market conditions
        base_price = self.ticker.mark_price if self.ticker else 0
        if base_price <= 0:
            logging.error("Invalid market price for risk breach handling")
            return
            
        # Add spread based on reduction direction
        spread_adjustment = SPREAD * self.tick
        if self.position_size > 0:
            target_price = base_price - spread_adjustment  # Selling, so slightly below market
        else:
            target_price = base_price + spread_adjustment  # Buying, so slightly above market
            
        # Align price to tick size
        aligned_price = self.round_to_tick(target_price)
        
        logging.info(f"Starting smart position reduction: {reduction_size} @ {aligned_price}")
        
        # Place the reduction order
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=abs(reduction_size),
            price=aligned_price,
            client_order_id=self.client_order_id,
            id=self.client_order_id
        )
        self.client_order_id += 1

    async def check_position_limits(self) -> bool:
        """Check if position needs rebalancing"""
        abs_position = abs(self.position_size)
        rebalance_size = POSITION_LIMITS["max_position"] * POSITION_LIMITS["rebalance_threshold"]
        
        if abs_position >= rebalance_size:
            zscore = self.calculate_zscore()
            atr = self.calculate_atr()
            
            # Adjust rebalancing based on market conditions
            if abs(zscore) > 2.0:  # High volatility
                await self.aggressive_rebalance()
            else:
                await self.gradual_rebalance()
            return False
        return True

    async def gradual_rebalance(self):
        """Gradually reduce position size"""
        target_size = POSITION_LIMITS["max_position"] * 0.5
        current_size = abs(self.position_size)
        reduction_size = (current_size - target_size) * 0.25  # 25% reduction steps
        
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        price = self.round_to_tick(self.get_rebalance_price())
        
        logging.info(f"Gradual rebalance: {reduction_size} @ {price}")
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=reduction_size,
            price=price,
            client_order_id=self.client_order_id,
            id=self.client_order_id
        )
        self.client_order_id += 1

    async def aggressive_rebalance(self):
        """Quickly reduce position in volatile conditions"""
        target_size = POSITION_LIMITS["max_position"] * 0.3  # More aggressive reduction
        current_size = abs(self.position_size)
        reduction_size = (current_size - target_size) * 0.5  # 50% reduction steps
        
        direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
        price = self.round_to_tick(self.get_rebalance_price())
        
        logging.info(f"Aggressive rebalance: {reduction_size} @ {price}")
        await self.thalex.insert(
            direction=direction,
            instrument_name=self.perp_name,
            amount=reduction_size,
            price=price,
            client_order_id=self.client_order_id,
            id=self.client_order_id
        )
        self.client_order_id += 1

    def get_rebalance_price(self) -> float:
        """Calculate rebalance price based on market conditions"""
        base_price = self.ticker.mark_price
        zscore = self.calculate_zscore()
        atr = self.calculate_atr()
        
        # Adjust price based on market conditions
        if self.position_size > 0:
            price_adjustment = max(0.1, min(0.5, abs(zscore) * 0.1)) * atr
            return base_price - price_adjustment
        else:
            price_adjustment = max(0.1, min(0.5, abs(zscore) * 0.1)) * atr
            return base_price + price_adjustment

    def calculate_dynamic_take_profit(self) -> float:
        """Calculate take profit based on market conditions"""
        base_tp = POSITION_LIMITS["base_take_profit_pct"]
        
        # Adjust based on volatility (ATR)
        atr = self.calculate_atr()
        volatility_scalar = min(2.0, max(0.5, atr / self.ticker.mark_price))
        
        # Adjust based on trend strength (Z-score)
        zscore = abs(self.calculate_zscore())
        trend_scalar = min(1.5, max(0.7, zscore / 2))
        
        # Calculate final take profit
        dynamic_tp = base_tp * volatility_scalar * trend_scalar
        
        # Clamp to min/max bounds
        return min(
            POSITION_LIMITS["max_take_profit_pct"],
            max(POSITION_LIMITS["min_take_profit_pct"], dynamic_tp)
        )

    async def manage_take_profit(self):
        """Handle take profit and trailing stops"""
        if self.position_size == 0:
            self.trailing_stop_active = False
            self.highest_profit = 0.0
            return
            
        current_pnl_pct = self.calculate_position_pnl()
        
        # Update highest profit for trailing stop
        if current_pnl_pct > self.highest_profit:
            self.highest_profit = current_pnl_pct
            
        # Check if trailing stop should activate
        if current_pnl_pct >= POSITION_LIMITS["trailing_stop_activation"]:
            self.trailing_stop_active = True
            
        if self.trailing_stop_active:
            trailing_stop_level = self.highest_profit - POSITION_LIMITS["trailing_stop_distance"]
            if current_pnl_pct < trailing_stop_level:
                await self.close_position("Trailing stop hit")
                return
                
        # Dynamic take profit check
        dynamic_tp = self.calculate_dynamic_take_profit()
        if current_pnl_pct >= dynamic_tp:
            await self.close_position("Dynamic take profit hit")

    def calculate_position_pnl(self) -> float:
        """Calculate current position PnL percentage"""
        if self.position_size == 0 or not self.ticker or self.entry_price <= 0:
            return 0.0
            
        direction = 1 if self.position_size > 0 else -1
        try:
            return direction * (self.ticker.mark_price - self.entry_price) / self.entry_price
        except ZeroDivisionError:
            logging.error("Entry price is zero or invalid, returning 0 PnL")
            return 0.0

    def __init__(self, thalex: th.Thalex):
        self.thalex = thalex
        self.ticker: Optional[Ticker] = None
        self.index: Optional[float] = None
        self.quote_cv = asyncio.Condition()
        self.portfolio: Dict[str, float] = {}
        self.orders: List[List[Order]] = [[], []]  # bids, asks
        self.client_order_id: int = 100
        self.tick: Optional[float] = None
        self.perp_name: Optional[str] = None
        
        # Position tracking
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_rebalance = time.time()
        self.position_start_time = 0.0
        
        # Risk management
        self.highest_profit = 0.0
        self.trailing_stop_active = False
        self.entry_prices = {}
        
        # Market making parameters
        self.gamma = TRADING_PARAMS["position_management"]["gamma"]
        self.k = 1.5  # Order flow intensity
        self.sigma = 0.0  # Market volatility (dynamic)
        self.T = 1.0  # Time horizon
        
        # Price tracking and volatility metrics
        self.price_window = deque(maxlen=TRADING_PARAMS["volatility"]["window"])
        self.price_history = deque(maxlen=TRADING_PARAMS["volatility"]["window"])
        self.last_atr = 0.0
        self.last_atr_time = 0.0
        self.last_zscore = 0.0
        self.last_zscore_time = 0.0
        
        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "average_fill_price": 0.0,
            "successful_trades": 0,
            "win_loss_ratio": 0.0,
            "expected_value": 0.0
        }
        
        # Order tracking
        self.order_cache = {}  # client_order_id -> Order mapping
        self.last_order_update = {}  # client_order_id -> timestamp
        
        # Rate limiting
        self.last_quote_time = 0
        self.last_order_time = 0
        self.min_order_interval = 0.1  # Minimum time between orders in seconds
        self.max_orders = 50  # Maximum number of open orders
        
        # Alert management
        self.alert_counts = {}
        self.alert_cooldown = 50  # 5 minutes
        self.last_alert_time = {}
        self.quoting_enabled = True

    async def handle_order(self, order_id: int):
        """Handle order updates and maintain order cache"""
        try:
            order = self.order_cache.get(order_id)
            if order is None:
                logging.warning(f"Order {order_id} not found in cache")
                return
                
            # Check order age
            current_time = time.time()
            last_update = self.last_order_update.get(order_id, 0)
            if current_time - last_update > 300:  # 5 minutes
                logging.warning(f"Order {order_id} cache expired")
                await self.thalex.cancel(client_order_id=order_id, id=order_id)
                return
                
            # Process order based on status
            if order.status == OrderStatus.FILLED:
                await self.process_filled_order(order)
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                await self.process_partial_fill(order)
            elif order.status == OrderStatus.CANCELLED:
                self.cleanup_cancelled_order(order)
                
        except Exception as e:
            logging.error(f"Error handling order {order_id}: {str(e)}")
            
    def cleanup_cancelled_order(self, order: Order):
        """Clean up cancelled order from cache"""
        if order.id in self.order_cache:
            del self.order_cache[order.id]
        if order.id in self.last_order_update:
            del self.last_order_update[order.id]
            
    async def process_filled_order(self, order: Order):
        """Process a fully filled order"""
        # Update position tracking
        if self.position_size == 0:
            self.position_start_time = time.time()
        
        # Check position holding time
        if self.position_size != 0:
            holding_time = time.time() - self.position_start_time
            if holding_time > PERFORMANCE_METRICS["max_position_time"]:
                logging.warning(f"Position held for {holding_time}s, exceeding limit")
                await self.close_position("Maximum holding time exceeded")
                
    async def process_partial_fill(self, order: Order):
        """Process a partially filled order"""
        # Update order cache
        self.order_cache[order.id] = order
        self.last_order_update[order.id] = time.time()

    def calculate_optimal_spread(self) -> float:
        """Calculate optimal spread using Avellaneda-Stoikov formula"""
        if len(self.price_window) < 2:
            return SPREAD * self.tick
            
        # Calculate volatility
        self.sigma = np.std(np.diff(np.array(self.price_window)))
        
        # Optimal spread = γσ²(T-t) + 2/γ log(1 + γ/k)
        spread = (self.gamma * self.sigma**2 * self.T + 
                 2/self.gamma * np.log(1 + self.gamma/self.k))
        return max(SPREAD * self.tick, spread)
        
    def calculate_inventory_skew(self) -> float:
        """Calculate inventory-based price skew"""
        q = self.position_size / POSITION_LIMITS["max_position"]
        return self.gamma * self.sigma**2 * self.T * q

    def calculate_level_size(self, base_size: float, side: str, level: int) -> float:
        """Calculate size for each quote level based on inventory"""
        inventory_ratio = abs(self.position_size) / POSITION_LIMITS["max_position"]
        
        # Reduce size when inventory grows in that direction
        if (side == "bid" and self.position_size > 0) or \
           (side == "ask" and self.position_size < 0):
            size_multiplier = 1 - inventory_ratio
        else:
            size_multiplier = 1 + inventory_ratio * 0.5
            
        return base_size * size_multiplier * (1 - level * 0.1)  # Reduce size in outer levels

    async def update_model_parameters(self):
        """Update model parameters based on market conditions"""
        # Update volatility estimate
        if self.ticker and self.ticker.mark_price:
            self.price_window.append(self.ticker.mark_price)
            
        # Adjust risk aversion based on PnL
        pnl_pct = self.calculate_position_pnl()
        self.gamma = max(0.05, min(0.3, 0.1 + abs(pnl_pct) * 0.5))
        
        # Update order flow intensity based on recent trades
        # Implementation depends on available market data

    async def sanity_check(self):
        """Run sanity checks before operations"""
        try:
            # Configuration validation
            if not ConfigValidator.validate_config():
                raise ValueError("Invalid configuration parameters")

            # Position consistency check
            portfolio_position = self.portfolio.get(self.perp_name, 0)
            if abs(self.position_size - portfolio_position) > 1e-6:
                logging.error(f"Position mismatch: internal={self.position_size}, portfolio={portfolio_position}")
                self.position_size = portfolio_position

            # Order book consistency
            for side in [0, 1]:
                for order in self.orders[side]:
                    if order.is_open() and order.price <= 0:
                        logging.error(f"Invalid order price: {order.price}")
                        await self.thalex.cancel(client_order_id=order.id, id=order.id)

            # Risk limits validation
            if self.ticker:
                notional = abs(self.position_size * self.ticker.mark_price)
                if notional > POSITION_LIMITS["max_notional"] * 1.1:  # 10% buffer
                    logging.error(f"Critical notional value exceeded: {notional}")
                    await self.emergency_close()

        except Exception as e:
            logging.error(f"Sanity check failed: {str(e)}")
            return False
        return True

    async def emergency_close(self):
        """Emergency position closure"""
        try:
            if self.position_size != 0:
                direction = th.Direction.SELL if self.position_size > 0 else th.Direction.BUY
                await self.thalex.insert(
                    direction=direction,
                    instrument_name=self.perp_name,
                    amount=abs(self.position_size),
                    price=self.ticker.mark_price,
                    client_order_id=self.client_order_id,
                    id=self.client_order_id
                )
                self.client_order_id += 1
                logging.warning("Emergency position closure initiated")
        except Exception as e:
            logging.error(f"Emergency close failed: {str(e)}")

    def check_sanity(self) -> bool:
        try:
            if "total_trades" not in self.performance_metrics:
                logging.error("Sanity check failed: 'total_trades' is not defined")
                return False
            if self.performance_metrics["total_trades"] > 0:
                avg_price = self.performance_metrics["average_fill_price"]
                if avg_price <= 0:
                    logging.error("Invalid average fill price")
                    self.performance_metrics["average_fill_price"] = 0
            # ... rest of your sanity checks
            return True
        except Exception as e:
            logging.error(f"Sanity check failed: {str(e)}")
            return False
            
    def update_performance_metrics(self, trade: Dict) -> None:
        """Update trading performance metrics with thread safety.
        
        Args:
            trade: Dictionary containing trade information
        """
        with metrics_lock:
            try:
                # Update trade count
                self.performance_metrics["total_trades"] += 1
                
                # Calculate trade PnL
                price = float(trade["price"])
                amount = float(trade["amount"])
                direction = 1 if trade["direction"] == "buy" else -1
                
                # Use entry price for PnL calculation
                pnl = direction * (self.entry_price - price) * amount
                
                # Update average fill price with weighted average
                prev_avg = self.performance_metrics["average_fill_price"]
                prev_count = self.performance_metrics["total_trades"] - 1
                self.performance_metrics["average_fill_price"] = (
                    (prev_avg * prev_count + price) / 
                    self.performance_metrics["total_trades"]
                )
                
                # Update success metrics
                if pnl > 0:
                    self.performance_metrics["successful_trades"] += 1
                
                # Update win/loss ratio
                total = self.performance_metrics["total_trades"]
                wins = self.performance_metrics["successful_trades"]
                self.performance_metrics["win_loss_ratio"] = wins / total if total > 0 else 0.0
                
                # Update expected value (average PnL)
                prev_ev = self.performance_metrics["expected_value"]
                self.performance_metrics["expected_value"] = (
                    (prev_ev * prev_count + pnl) / total
                )
                
            except Exception as e:
                logging.error(f"Error updating performance metrics: {str(e)}")

class ConfigValidator:
    @staticmethod
    def validate_config():
        """Validate configuration parameters"""
        checks = [
            SPREAD > 0,
            len(BID_SIZES) == len(ASK_SIZES),
            POSITION_LIMITS["max_position"] > 0,
            POSITION_LIMITS["max_notional"] > 0,
            POSITION_LIMITS["stop_loss_pct"] > 0,
            POSITION_LIMITS["base_take_profit_pct"] > POSITION_LIMITS["min_take_profit_pct"],
            POSITION_LIMITS["max_take_profit_pct"] > POSITION_LIMITS["base_take_profit_pct"],
            POSITION_LIMITS["rebalance_threshold"] > 0 and POSITION_LIMITS["rebalance_threshold"] < 1
        ]
        return all(checks)

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    import tracemalloc
    tracemalloc.start()
    
    run = True
    while run:
        try:
            thalex = th.Thalex(network=NETWORK)
            perp_quoter = PerpQuoter(thalex)
            tasks = [
                asyncio.create_task(perp_quoter.listen_task()),
                asyncio.create_task(perp_quoter.quote_task()),
            ]
            
            logging.info(f"Starting on {NETWORK} {UNDERLYING=}")
            await asyncio.gather(*tasks)
            
        except (websockets.ConnectionClosed, socket.gaierror) as e:
            logging.error(f"Connection error ({e}). Reconnecting...")
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            run = False
            logging.info("Shutting down...")
        except Exception as e:
            logging.exception("Unexpected error:")
            await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    async def shutdown(thalex, tasks):
        """Graceful shutdown handler"""
        if thalex.connected:  # Remove await since it's a property
            await thalex.cancel_session(id=CALL_ID_CANCEL_SESSION)
            await thalex.disconnect()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_quoter():
        while True:
            thalex = th.Thalex(network=NETWORK)
            perp_quoter = PerpQuoter(thalex)
            tasks = [
                asyncio.create_task(perp_quoter.listen_task()),
                asyncio.create_task(perp_quoter.quote_task()),
            ]
            
            try:
                logging.info(f"Starting on {NETWORK} {UNDERLYING=}")
                await asyncio.gather(*tasks)
            except (websockets.ConnectionClosed, socket.gaierror) as e:
                logging.error(f"Connection error ({e}). Reconnecting...")
                await shutdown(thalex, tasks)
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                await shutdown(thalex, tasks)
                break
            except Exception as e:
                logging.exception("Unexpected error:")
                await shutdown(thalex, tasks)
                await asyncio.sleep(1)

    try:
        asyncio.run(run_quoter())
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")


