import asyncio
import csv
import json
import logging
import socket
import time
from typing import Optional, Dict
import datetime
import math
import os

import websockets
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

import keys
import thalex as th

# Default configuration
DEFAULT_CONFIG = {
    # Instruments
    "option_call": "BTC-28MAR25-80000-C",
    "option_put": "BTC-28MAR25-80000-P",
    "perpetual": "BTC-PERPETUAL",
    
    # Trading parameters
    "quote_size": 0.1,
    "spread_pct": 0.02,
    "min_hedge_threshold": 0.01,
    "delta_hedge_threshold": 0.05,
    "gamma_hedge_threshold": 0.001,
    "max_position_size": 1.0,
    "min_trade_size": 0.01,
    
    # Risk parameters
    "pnl_drawdown_threshold": 0.05,
    "usd_loss_threshold": 100,
    "aggressive_hedge_threshold": 0.03,
    
    # Volatility parameters
    "vol_window": 20,
    "min_volatility": 0.05,
    "max_volatility": 2.00,
    "default_volatility": 0.50,
    
    # Timing parameters
    "order_delay": 0.1,
    "stale_mark_threshold": 5.0,
    "min_hedge_interval": 1.0,
    "portfolio_refresh_interval": 5.0,
    "parity_check_interval": 10.0,
    "pnl_calculation_interval": 1.0,
    "dashboard_refresh": 0.5,
    
    # Network settings
    "network": "test",  # "test" or "main"
    
    # Logging
    "log_level": "INFO",
    "log_file": "option_hedger.log"
}

# Load configuration from file if it exists
def load_config():
    config = DEFAULT_CONFIG.copy()
    config_file = os.path.join(os.path.dirname(__file__), "option_hedger_config.json")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                if file_config:
                    config.update(file_config)
            logging.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logging.error(f"Error loading configuration from {config_file}: {e}")
    else:
        logging.info(f"No configuration file found at {config_file}, using defaults")
        # Create default config file for future use
        try:
            with open(config_file, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            logging.info(f"Created default configuration file at {config_file}")
        except Exception as e:
            logging.error(f"Error creating default configuration file: {e}")
            
    return config

# Load configuration
CONFIG = load_config()

# Set up instruments based on configuration
OPTION = CONFIG["option_call"]
OPTION_PUT = CONFIG["option_put"]
OPTION_TICKER = f"ticker.{OPTION}.1000ms"
OPTION_PUT_TICKER = f"ticker.{OPTION_PUT}.1000ms"
PERPETUAL = CONFIG["perpetual"]
PERP_TICKER = f"ticker.{PERPETUAL}.1000ms"
ORDER_DELAY = CONFIG["order_delay"]

# Set up network based on configuration
NETWORK = th.Network.TEST if CONFIG["network"].lower() == "test" else th.Network.MAIN
KEY_ID = keys.key_ids[NETWORK]
PRIV_KEY = keys.private_keys[NETWORK]

# Call IDs for matching responses
CID_PORTFOLIO = 1001
CID_INSERT = 1002

class OptionGreeks:
    def __init__(self):
        self.delta: float = 0
        self.gamma: float = 0
        self.vega: float = 0
        self.theta: float = 0

    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes Greeks"""
        # Apply a minimum volatility to prevent numerical issues
        MIN_VOLATILITY = 0.05  # 5% minimum volatility 
        sigma = max(sigma, MIN_VOLATILITY)
        
        # Handle extreme moneyness cases
        EXTREME_ITM_THRESHOLD = 5.0  # Deep ITM when spot/strike > 5 for calls
        EXTREME_OTM_THRESHOLD = 0.2  # Deep OTM when spot/strike < 0.2 for calls
        
        # For extremely deep ITM/OTM options, use approximations but never exactly 0 or 1
        moneyness = S/K
        
        if option_type.lower() == 'call' and moneyness > EXTREME_ITM_THRESHOLD:
            delta = 0.9999  # Cap at 0.9999 instead of 1.0
            gamma = 0.0001  # Small but non-zero gamma
            vega = 0.0001   # Small but non-zero vega
            theta = -r*K*math.exp(-r*T)  # Time value decay
            
            logging.warning(f"Deep ITM call detected (S/K={moneyness:.2f}). Using approximated Greeks.")
            return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
        
        if option_type.lower() == 'call' and moneyness < EXTREME_OTM_THRESHOLD:
            delta = 0.0001  # Floor at 0.0001 instead of 0
            gamma = 0.0001  # Small but non-zero gamma
            vega = 0.0001   # Small but non-zero vega
            theta = -0.0001 # Small time decay
            
            logging.warning(f"Deep OTM call detected (S/K={moneyness:.2f}). Using approximated Greeks.")
            return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
        
        # For normal cases, use standard Black-Scholes
        try:
            sqrt_t = math.sqrt(T)
            d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt_t)
            d2 = d1 - sigma*sqrt_t
            
            # Standard normal PDF and CDF
            npdf = lambda x: math.exp(-x*x/2) / math.sqrt(2*math.pi)
            ncdf = lambda x: (1 + math.erf(x/math.sqrt(2))) / 2
            
            N_d1 = ncdf(d1)
            N_d2 = ncdf(d2)
            
            if option_type.lower() == 'call':
                # Cap delta at 0.9999 to prevent exact 1.0
                delta = min(N_d1, 0.9999)
                theta = (-S*npdf(d1)*sigma/(2*sqrt_t) - r*K*math.exp(-r*T)*N_d2)
            else:
                # Floor put delta at -0.9999 to prevent exact -1.0
                delta = max(N_d1 - 1, -0.9999)
                theta = (-S*npdf(d1)*sigma/(2*sqrt_t) + r*K*math.exp(-r*T)*(1-N_d2))
                
            gamma = npdf(d1)/(S*sigma*sqrt_t)
            vega = S*sqrt_t*npdf(d1)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta
            }
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logging.error(f"Error in BS calculation: {e}, using fallback values. Inputs: S={S}, K={K}, T={T}, sigma={sigma}")
            # Provide fallback values for numerical errors
            if option_type.lower() == 'call':
                delta = 0.5 if S > K else 0.1
            else:
                delta = -0.5 if S < K else -0.1
            return {'delta': delta, 'gamma': 0.01, 'vega': 0.01, 'theta': -0.01}

class OptionPosition:
    def __init__(self, instrument_name: str):
        self.instrument_name = instrument_name
        self.position: float = 0
        self.greeks = OptionGreeks()
        self.mark_price: Optional[float] = None
        self.strike: float = float(instrument_name.split('-')[2])
        self.is_call = instrument_name.split('-')[3] == 'C'
        self.implied_vol: Optional[float] = None
        self.expected_position: Optional[float] = None  # Track expected position after trades
        
        # P&L tracking
        self.avg_entry_price: Optional[float] = None
        self.realized_pnl: float = 0.0
        self.trades_history = []
        
    def update_position(self, trade_size: float, trade_price: float, direction: str):
        """Update position and track P&L when a trade occurs"""
        old_position = self.position
        
        # Calculate trade direction multiplier (buy = +1, sell = -1)
        dir_mult = 1 if direction.lower() == 'buy' else -1
        
        # Calculate trade size with direction
        signed_trade_size = dir_mult * abs(trade_size)
        
        # If closing or reducing position, calculate realized P&L
        if (old_position > 0 and signed_trade_size < 0) or (old_position < 0 and signed_trade_size > 0):
            # Determine how much of the position is being closed
            close_size = min(abs(old_position), abs(signed_trade_size))
            close_dir = -1 if old_position > 0 else 1
            
            # Calculate P&L on the closed portion
            if self.avg_entry_price is not None:
                pnl_per_contract = (trade_price - self.avg_entry_price) * close_dir
                realized_pnl = pnl_per_contract * close_size
                self.realized_pnl += realized_pnl
        
        # Update position
        new_position = old_position + signed_trade_size
        
        # Update average entry price for new positions
        if abs(new_position) > abs(old_position):
            # If increasing position or changing direction
            if self.avg_entry_price is None or old_position * new_position <= 0:
                # New position or direction change
                self.avg_entry_price = trade_price
            else:
                # Adding to existing position, update average price
                old_value = abs(old_position) * self.avg_entry_price
                new_value = abs(signed_trade_size) * trade_price
                self.avg_entry_price = (old_value + new_value) / abs(new_position)
        
        # If position is completely closed, reset avg entry price
        if new_position == 0:
            self.avg_entry_price = None
            
        # Store the trade
        self.trades_history.append({
            'timestamp': time.time(),
            'direction': direction,
            'size': trade_size,
            'price': trade_price,
            'old_position': old_position,
            'new_position': new_position,
            'realized_pnl': self.realized_pnl
        })
        
        # Update position
        self.position = new_position
        
    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L based on current mark price"""
        if self.position == 0 or self.mark_price is None or self.avg_entry_price is None:
            return 0.0
            
        # Calculate unrealized P&L
        if self.position > 0:
            # Long position
            return (self.mark_price - self.avg_entry_price) * self.position
        else:
            # Short position
            return (self.avg_entry_price - self.mark_price) * abs(self.position)
            
    def calculate_total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)"""
        return self.realized_pnl + self.calculate_unrealized_pnl()
        
    def update_greeks(self, spot_price: float, vol: float):
        """Update all Greeks for the position"""
        # Store the volatility used
        self.implied_vol = vol
        
        # Calculate time to expiry more precisely
        expiry_str = self.instrument_name.split('-')[1]
        expiry_date = datetime.datetime.strptime(expiry_str, "%d%b%y")
        now = datetime.datetime.now()
        
        # Include partial days in the calculation
        days_to_expiry = (expiry_date - now).total_seconds() / (24 * 3600)
        T = max(days_to_expiry / 365.0, 0.0001)  # Avoid division by zero
        
        # Log the inputs to Greeks calculation for verification
        logging.debug(f"Greeks Calculation Inputs:")
        logging.debug(f"Spot: {spot_price:.2f}")
        logging.debug(f"Strike: {self.strike:.2f}")
        logging.debug(f"Time to Expiry: {T:.4f} years ({days_to_expiry:.1f} days)")
        logging.debug(f"Volatility: {vol:.2%}")
        
        r = 0.0  # Risk-free rate
        greeks = OptionGreeks.black_scholes_greeks(
            spot_price, 
            self.strike, 
            T, 
            r, 
            vol,
            'call' if self.is_call else 'put'
        )
        
        # Store previous values for change monitoring
        old_delta = self.greeks.delta
        
        self.greeks.delta = greeks['delta']
        self.greeks.gamma = greeks['gamma']
        self.greeks.vega = greeks['vega']
        self.greeks.theta = greeks['theta']
        
        # Log significant changes in Greeks
        if old_delta is not None:
            delta_change = self.greeks.delta - old_delta
            if abs(delta_change) > 0.1:  # 10% delta change threshold
                logging.info(f"Significant delta change: {delta_change:.4f} "
                            f"({old_delta:.4f} -> {self.greeks.delta:.4f})")

class OptionInventoryHedger:
    def __init__(self, thalex):
        self.perp_position: Optional[float] = None
        self.mark_perp: Optional[float] = None
        self.options_positions: Dict[str, OptionPosition] = {}
        self.thalex: th.Thalex = thalex
        self.last_order_ts: float = 0
        
        # Configuration from config file
        self.QUOTE_SIZE = CONFIG["quote_size"]
        self.SPREAD_PCT = CONFIG["spread_pct"]
        self.MIN_HEDGE_THRESHOLD = CONFIG["min_hedge_threshold"]
        self.DELTA_HEDGE_THRESHOLD = CONFIG["delta_hedge_threshold"]
        
        # Risk management parameters
        self.GAMMA_HEDGE_THRESHOLD = CONFIG["gamma_hedge_threshold"]
        self.MAX_POSITION_SIZE = CONFIG["max_position_size"]
        self.VOL_WINDOW = CONFIG["vol_window"]
        self.price_history = []
        
        # Add logger configuration
        self.logger = logging.getLogger('OptionHedger')
        self.logger.setLevel(getattr(logging, CONFIG["log_level"]))
        # Add file handler
        fh = logging.FileHandler(CONFIG["log_file"])
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
        # Initialize option positions for both call and put
        self.options_positions[OPTION] = OptionPosition(OPTION)
        self.options_positions[OPTION_PUT] = OptionPosition(OPTION_PUT)
        
        # Add timestamp tracking
        self.last_mark_update: float = 0
        self.last_hedge_time: float = 0
        self.STALE_MARK_THRESHOLD = CONFIG["stale_mark_threshold"]
        self.MIN_HEDGE_INTERVAL = CONFIG["min_hedge_interval"]
        
        # Add position tracking
        self.expected_perp_position: Optional[float] = None
        self.last_position_update: float = 0
        self.MAX_POSITION_DRIFT = 0.01  # Maximum allowed drift of 0.01 BTC
        
        # Add fee tracking
        self.total_fees = 0
        self.MIN_TRADE_SIZE = CONFIG["min_trade_size"]
        
        # Add volatility safeguards
        self.MIN_VOLATILITY = CONFIG["min_volatility"]
        self.MAX_VOLATILITY = CONFIG["max_volatility"]
        self.DEFAULT_VOLATILITY = CONFIG["default_volatility"]
        
        # Improve tracking
        self.volatility_history = []
        
        # Add dashboard components
        self.console = Console()
        self.layout = Layout()
        self.last_dashboard_update = 0
        self.DASHBOARD_REFRESH = CONFIG["dashboard_refresh"]
        
        # Add transaction log for detailed tracking
        self.transaction_log = []
        self.DRIFT_HISTORY_LENGTH = 10
        self.drift_history = []
        
        self.last_portfolio_refresh = 0
        self.PORTFOLIO_REFRESH_INTERVAL = CONFIG["portfolio_refresh_interval"]
        
        # Add put-call parity tracking
        self.parity_check_time = 0
        self.PARITY_CHECK_INTERVAL = CONFIG["parity_check_interval"]
        
        # Add P&L tracking
        self.initial_portfolio_value = None
        self.current_portfolio_value = None
        self.portfolio_values_history = []
        self.trade_pnl_history = []
        self.max_portfolio_value = 0
        self.min_portfolio_value = float('inf')
        
        # P&L-based hedging thresholds
        self.PNL_DRAWDOWN_THRESHOLD = CONFIG["pnl_drawdown_threshold"]
        self.USD_LOSS_THRESHOLD = CONFIG["usd_loss_threshold"]
        self.AGGRESSIVE_HEDGE_THRESHOLD = CONFIG["aggressive_hedge_threshold"]
        self.last_pnl_calculation = 0
        self.PNL_CALCULATION_INTERVAL = CONFIG["pnl_calculation_interval"]
        
        # Log configuration
        self.logger.info("Initialized OptionInventoryHedger with configuration:")
        for key, value in CONFIG.items():
            self.logger.info(f"  {key}: {value}")
        
        # Store CID values for easier reference
        self.CID_PORTFOLIO = CID_PORTFOLIO
        self.CID_INSERT = CID_INSERT

    def calculate_total_delta(self) -> float:
        """Calculate total delta exposure across all positions"""
        total_delta = 0.0
        
        # Add perpetual position delta (1:1)
        if self.perp_position is not None:
            total_delta += self.perp_position
            
        # Add options deltas
        for option in self.options_positions.values():
            if option.position and option.greeks.delta:
                total_delta += option.position * option.greeks.delta
                
        return total_delta

    def calculate_historical_vol(self) -> float:
        """Calculate historical volatility from price history with safeguards"""
        # Return default if not enough data
        if len(self.price_history) < 5:  # Require at least 5 price points for better estimation
            self.logger.warning(f"Insufficient price history for vol calculation. Using default: {self.DEFAULT_VOLATILITY:.2%}")
            return self.DEFAULT_VOLATILITY
            
        try:
            # Calculate log returns
            returns = [math.log(p2/p1) for p1, p2 in zip(self.price_history[:-1], self.price_history[1:])]
            
            # Calculate standard deviation of returns
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return)**2 for r in returns) / len(returns)
            
            # Annualize - assuming the data points are 1 second apart
            vol = math.sqrt(variance * 365 * 24 * 60 * 60)
            
            # Apply floor and ceiling
            vol = max(min(vol, self.MAX_VOLATILITY), self.MIN_VOLATILITY)
            
            # Keep volatility history for monitoring
            self.volatility_history.append(vol)
            if len(self.volatility_history) > 20:
                self.volatility_history.pop(0)
                
            # Log significant changes
            if len(self.volatility_history) > 1:
                vol_change = (vol / self.volatility_history[-2] - 1) * 100
                if abs(vol_change) > 20:  # Log if vol changes by more than 20%
                    self.logger.warning(f"Significant volatility change: {vol_change:.1f}% to {vol:.2%}")
                    
            return vol
        except (ValueError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating volatility: {e}. Using default: {self.DEFAULT_VOLATILITY:.2%}")
            return self.DEFAULT_VOLATILITY

    def calculate_total_gamma(self) -> float:
        """Calculate total gamma exposure"""
        total_gamma = 0.0
        for option in self.options_positions.values():
            if option.position and option.greeks.gamma:
                total_gamma += option.position * option.greeks.gamma
        return total_gamma

    async def manage_gamma(self):
        """Manage gamma exposure by adjusting hedge positions"""
        total_gamma = self.calculate_total_gamma()
        
        self.logger.info(f"Current gamma exposure: {total_gamma:.6f}")
        
        if abs(total_gamma) < self.GAMMA_HEDGE_THRESHOLD:
            self.logger.debug("Gamma exposure within threshold, no hedging needed")
            return
            
        # Calculate gamma hedge size
        spot_move = self.mark_perp * 0.01  # 1% move
        gamma_exposure = total_gamma * spot_move * spot_move
        hedge_size = min(abs(gamma_exposure), self.QUOTE_SIZE)
        
        direction = th.Direction.SELL if total_gamma > 0 else th.Direction.BUY
        self.logger.info(f"Placing gamma hedge order: {direction} {hedge_size} contracts")
        
        await self.thalex.insert(
            id=self.CID_INSERT,
            direction=direction,
            instrument_name=PERPETUAL,
            order_type=th.OrderType.LIMIT,
            amount=hedge_size,
            price=self.mark_perp
        )

    def calculate_portfolio_value(self) -> float:
        """Calculate the total portfolio value in USD"""
        if self.mark_perp is None:
            return 0.0
            
        total_value = 0.0
        
        # Add perpetual value
        if self.perp_position is not None:
            total_value += self.perp_position * self.mark_perp
            
        # Add options values
        for option_name, option in self.options_positions.items():
            if option.position and option.mark_price:
                total_value += option.position * option.mark_price
                
        return total_value
        
    def calculate_portfolio_pnl(self) -> Dict[str, float]:
        """Calculate the total P&L of the portfolio, broken down by instrument"""
        result = {
            "total_realized": 0.0,
            "total_unrealized": 0.0,
            "total_pnl": 0.0,
            "instruments": {}
        }
        
        # Calculate P&L for perpetual
        perp_pnl = 0.0
        if hasattr(self, 'perp_avg_entry_price') and self.perp_avg_entry_price is not None and self.perp_position is not None and self.mark_perp is not None:
            if self.perp_position > 0:
                # Long position
                perp_pnl = (self.mark_perp - self.perp_avg_entry_price) * self.perp_position
            elif self.perp_position < 0:
                # Short position
                perp_pnl = (self.perp_avg_entry_price - self.mark_perp) * abs(self.perp_position)
                
        result["instruments"][PERPETUAL] = {
            "realized": getattr(self, 'perp_realized_pnl', 0.0),
            "unrealized": perp_pnl,
            "total": getattr(self, 'perp_realized_pnl', 0.0) + perp_pnl
        }
        
        # Add options P&L
        for option_name, option in self.options_positions.items():
            realized_pnl = option.realized_pnl
            unrealized_pnl = option.calculate_unrealized_pnl()
            total_pnl = option.calculate_total_pnl()
            
            result["instruments"][option_name] = {
                "realized": realized_pnl,
                "unrealized": unrealized_pnl,
                "total": total_pnl
            }
            
            # Add to totals
            result["total_realized"] += realized_pnl
            result["total_unrealized"] += unrealized_pnl
            result["total_pnl"] += total_pnl
            
        return result
        
    def update_pnl_metrics(self):
        """Update P&L metrics and history"""
        current_time = time.time()
        
        # Skip if we've updated recently
        if current_time - self.last_pnl_calculation < self.PNL_CALCULATION_INTERVAL:
            return
            
        # Calculate current portfolio value and P&L
        portfolio_value = self.calculate_portfolio_value()
        portfolio_pnl = self.calculate_portfolio_pnl()
        
        self.current_portfolio_value = portfolio_value
        self.current_portfolio_pnl = portfolio_pnl["total_pnl"]
        
        # Initialize initial value if not set
        if self.initial_portfolio_value is None:
            self.initial_portfolio_value = portfolio_value
            if portfolio_value > 0:  # Ensure we don't initialize with zero
                self.logger.info(f"Initial portfolio value: ${portfolio_value:.2f}")
            else:
                self.logger.warning("Initial portfolio value is zero or negative, using 1.0 as default")
                self.initial_portfolio_value = 1.0  # Use a safe default to avoid division by zero
        
        # Update max/min values
        if portfolio_value > self.max_portfolio_value:
            old_max = self.max_portfolio_value
            self.max_portfolio_value = portfolio_value
            if old_max > 0:  # Don't log the first initialization
                self.logger.info(f"New portfolio value high: ${portfolio_value:.2f} (previous: ${old_max:.2f})")
                
        if portfolio_value < self.min_portfolio_value:
            old_min = self.min_portfolio_value
            self.min_portfolio_value = portfolio_value
            if old_min < float('inf'):  # Don't log the first initialization
                self.logger.info(f"New portfolio value low: ${portfolio_value:.2f} (previous: ${old_min:.2f})")
        
        # Calculate drawdown
        drawdown = self.calculate_drawdown()
        
        # Store history
        self.portfolio_values_history.append({
            'timestamp': current_time,
            'value': portfolio_value,
            'pnl': portfolio_pnl["total_pnl"],
            'realized_pnl': portfolio_pnl["total_realized"],
            'unrealized_pnl': portfolio_pnl["total_unrealized"],
            'drawdown': drawdown
        })
        
        # Trim history if needed
        if len(self.portfolio_values_history) > 1000:
            self.portfolio_values_history = self.portfolio_values_history[-1000:]
            
        # Log significant drawdowns
        if drawdown > self.PNL_DRAWDOWN_THRESHOLD:
            self.logger.warning(f"Significant drawdown detected: {drawdown:.2%} from peak (${self.max_portfolio_value:.2f} → ${portfolio_value:.2f})")
            
        # Calculate percentage P&L safely (avoid division by zero)
        if self.initial_portfolio_value > 0:
            pct_pnl = (portfolio_pnl["total_pnl"] / self.initial_portfolio_value)
            pct_pnl_str = f"({pct_pnl:.2%})"
        else:
            pct_pnl_str = "(N/A)"
        
        # Log P&L status periodically (every 10 seconds)
        if int(current_time) % 10 == 0 and int(current_time) != int(self.last_pnl_calculation):
            self.logger.info(f"P&L Status: ${portfolio_pnl['total_pnl']:.2f} {pct_pnl_str}, Realized: ${portfolio_pnl['total_realized']:.2f}, Unrealized: ${portfolio_pnl['total_unrealized']:.2f}, Drawdown: {drawdown:.2%}")
            
        self.last_pnl_calculation = current_time

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak portfolio value"""
        if self.max_portfolio_value <= 0:
            return 0.0
            
        current_value = self.current_portfolio_value or 0
        return (self.max_portfolio_value - current_value) / self.max_portfolio_value

    def get_dynamic_hedge_threshold(self) -> float:
        """Get a dynamic hedge threshold based on P&L and drawdown"""
        # Default threshold
        threshold = self.DELTA_HEDGE_THRESHOLD
        
        # If we don't have P&L data yet, use default
        if self.current_portfolio_value is None or self.max_portfolio_value == 0:
            return threshold
            
        # Calculate drawdown
        drawdown = self.calculate_drawdown()
        
        # If in significant drawdown, use more aggressive threshold
        if drawdown > self.PNL_DRAWDOWN_THRESHOLD:
            threshold = self.AGGRESSIVE_HEDGE_THRESHOLD
            self.logger.info(f"Using aggressive hedge threshold ({threshold:.2%}) due to drawdown of {drawdown:.2%}")
            
        # Calculate absolute P&L
        absolute_pnl = self.current_portfolio_value - self.initial_portfolio_value
        
        # If losing more than USD threshold, use more aggressive threshold
        if absolute_pnl < -self.USD_LOSS_THRESHOLD:
            threshold = self.AGGRESSIVE_HEDGE_THRESHOLD
            self.logger.info(f"Using aggressive hedge threshold ({threshold:.2%}) due to USD loss of ${-absolute_pnl:.2f}")
            
        return threshold

    def log_debug_info(self):
        """Log detailed debug information for troubleshooting"""
        self.logger.info("=== DEBUG INFORMATION ===")
        
        # Check mark price
        self.logger.info(f"Mark Price: {self.mark_perp or 'None'}")
        self.logger.info(f"Mark Price Age: {time.time() - self.last_mark_update:.1f}s")
        self.logger.info(f"Mark Price Stale: {'Yes' if time.time() - self.last_mark_update > self.STALE_MARK_THRESHOLD else 'No'}")
        
        # Check positions and deltas
        self.logger.info(f"Perpetual Position: {self.perp_position or 'None'}")
        total_delta = self.calculate_total_delta()
        self.logger.info(f"Total Delta: {total_delta:.4f}")
        
        # Log individual option positions and deltas
        self.logger.info("Option Positions:")
        for name, option in self.options_positions.items():
            self.logger.info(f"  {name}: Position={option.position or 0:.4f}, "
                           f"Delta={option.greeks.delta:.4f}, "
                           f"Position Delta={(option.position or 0) * option.greeks.delta:.4f}, "
                           f"Mark Price={option.mark_price or 'None'}")
        
        # Check volatility
        hist_vol = self.calculate_historical_vol()
        self.logger.info(f"Historical Volatility: {hist_vol:.2%}")
        self.logger.info(f"Volatility at MIN: {'Yes' if abs(hist_vol - self.MIN_VOLATILITY) < 0.0001 else 'No'}")
        self.logger.info(f"Volatility at MAX: {'Yes' if abs(hist_vol - self.MAX_VOLATILITY) < 0.0001 else 'No'}")
        
        # Check P&L metrics
        if self.current_portfolio_value is not None:
            if self.initial_portfolio_value is not None and self.initial_portfolio_value > 0:
                absolute_pnl = self.current_portfolio_value - self.initial_portfolio_value
                pct_pnl = absolute_pnl / self.initial_portfolio_value
                self.logger.info(f"Portfolio Value: ${self.current_portfolio_value:.2f}")
                self.logger.info(f"Absolute P&L: ${absolute_pnl:.2f} ({pct_pnl:.2%})")
            else:
                self.logger.info(f"Portfolio Value: ${self.current_portfolio_value:.2f}")
                self.logger.info(f"Absolute P&L: N/A (no valid initial value)")
                
            drawdown = self.calculate_drawdown()
            self.logger.info(f"Drawdown: {drawdown:.2%}")
            self.logger.info(f"Dynamic Hedge Threshold: {self.get_dynamic_hedge_threshold():.2%}")
        
        # Check put-call parity data
        call_option = self.options_positions.get(OPTION)
        put_option = self.options_positions.get(OPTION_PUT)
        self.logger.info("Put-Call Parity Data:")
        self.logger.info(f"  Call Option: {OPTION}, Mark Price: {call_option.mark_price if call_option else 'None'}")
        self.logger.info(f"  Put Option: {OPTION_PUT}, Mark Price: {put_option.mark_price if put_option else 'None'}")
        self.logger.info(f"  Parity Status: {'Ready for calculation' if (call_option and put_option and call_option.mark_price is not None and put_option.mark_price is not None) else 'Insufficient data'}")
        
        # Check subscription status
        self.logger.info(f"Subscribed to PERP_TICKER: {PERP_TICKER}")
        self.logger.info(f"Subscribed to OPTION_TICKER: {OPTION_TICKER}")
        self.logger.info(f"Subscribed to OPTION_PUT_TICKER: {OPTION_PUT_TICKER}")
        
        self.logger.info("=== END DEBUG INFO ===")

    async def quote_options(self):
        # Check if we need to refresh portfolio
        current_time = time.time()
        if current_time - self.last_portfolio_refresh > self.PORTFOLIO_REFRESH_INTERVAL:
            await self.thalex.portfolio(CID_PORTFOLIO)
            self.last_portfolio_refresh = current_time
            return

        # Update P&L metrics
        self.update_pnl_metrics()

        # Check for stale mark price
        if current_time - self.last_mark_update > self.STALE_MARK_THRESHOLD:
            self.logger.warning(f"Mark price is stale! Last update: {current_time - self.last_mark_update:.1f}s ago")
            return

        # Prevent over-hedging from queued messages
        if current_time - self.last_hedge_time < self.MIN_HEDGE_INTERVAL:
            self.logger.debug("Skipping hedge due to minimum interval")
            return

        # Log current state with more detail
        self.logger.info("=== Portfolio Status Update ===")
        self.logger.info(f"Mark Price: {self.mark_perp:.2f}")
        self.logger.info(f"Perpetual Position: {'None' if self.perp_position is None else f'{self.perp_position:.4f}'}")
        
        # Log detailed options state
        for option_name, option in self.options_positions.items():
            self.logger.info(f"\nOption {option_name}:")
            self.logger.info(f"Position: {option.position:.4f}")
            self.logger.info(f"Strike: {option.strike}")
            self.logger.info(f"Implied Vol: {option.implied_vol:.2%}" if option.implied_vol else "Implied Vol: None")
            self.logger.info(f"Greeks: Delta={option.greeks.delta:.4f}, "
                           f"Gamma={option.greeks.gamma:.6f}, Vega={option.greeks.vega:.4f}, "
                           f"Theta={option.greeks.theta:.4f}")

        # Calculate and log total delta with threshold check
        total_delta = self.calculate_total_delta()
        self.logger.info(f"\nRisk Metrics:")
        self.logger.info(f"Total Portfolio Delta: {total_delta:.4f}")
        
        # Get dynamic hedge threshold based on P&L
        dynamic_threshold = self.get_dynamic_hedge_threshold()
        self.logger.info(f"Delta Hedge Threshold: {dynamic_threshold:.2%}")
        self.logger.info(f"Delta Hedge Required: {abs(total_delta) > dynamic_threshold}")
        
        # Check put-call parity if it's time
        if current_time - self.parity_check_time > self.PARITY_CHECK_INTERVAL:
            self.check_put_call_parity()
            self.parity_check_time = current_time
        
        # Position drift check
        if self.expected_perp_position is not None:
            drift = abs((self.perp_position or 0) - self.expected_perp_position)
            if drift > self.MAX_POSITION_DRIFT:
                self.logger.error(f"Position drift detected! Expected: {self.expected_perp_position:.4f}, "
                                f"Actual: {self.perp_position:.4f}, Drift: {drift:.4f}")
                # Force portfolio refresh
                await self.thalex.portfolio(CID_PORTFOLIO)
                return

        # Log debug info every 30 seconds
        if int(current_time) % 30 == 0 and int(current_time) != int(self.last_hedge_time):
            self.log_debug_info()

        # Modify hedging logic to avoid small trades
        if abs(total_delta) > dynamic_threshold:
            hedge_size = min(abs(total_delta), self.QUOTE_SIZE)
            if hedge_size < self.MIN_TRADE_SIZE:
                self.logger.info(f"Skipping hedge - size too small: {hedge_size:.4f}")
                return

            # Determine if we're long or short inventory
            is_long_inventory = total_delta > 0
            
            # Choose which option to use for hedging based on inventory position
            # When long inventory (positive delta), sell calls
            # When short inventory (negative delta), sell puts
            option_to_hedge = OPTION if is_long_inventory else OPTION_PUT
            
            # Always SELL options as a hedge (selling calls when long, selling puts when short)
            direction = th.Direction.SELL
            
            self.logger.info(f"\nHedge Planning:")
            self.logger.info(f"Inventory Position: {'LONG' if is_long_inventory else 'SHORT'}")
            self.logger.info(f"Hedging with: {option_to_hedge} ({is_long_inventory and 'CALL' or 'PUT'})")
            self.logger.info(f"Hedge Direction: {direction}")
            self.logger.info(f"Hedge Size: {hedge_size:.4f}")
            
            # Position limit check with detailed logging
            option_position = self.options_positions[option_to_hedge]
            potential_position = (option_position.position or 0) - hedge_size  # Always selling
            position_limit_ok = abs(potential_position) <= self.MAX_POSITION_SIZE
            
            self.logger.info(f"Potential Option Position After Hedge: {potential_position:.4f}")
            self.logger.info(f"Position Limit Check: {'PASS' if position_limit_ok else 'FAIL'}")
            
            if position_limit_ok:
                # Calculate appropriate price for the option
                hist_vol = self.calculate_historical_vol()
                
                # Get current option mark price or calculate theoretical price
                option_mark_price = option_position.mark_price
                if option_mark_price is None:
                    # If no mark price, calculate theoretical price
                    # For simplicity, use a basic approximation
                    S = self.mark_perp
                    K = option_position.strike
                    T = 0.25  # Approximate time to expiry in years
                    r = 0.0   # Risk-free rate
                    
                    greeks = OptionGreeks.black_scholes_greeks(
                        S, K, T, r, hist_vol,
                        'call' if option_position.is_call else 'put'
                    )
                    
                    # Very basic approximation of option price
                    if option_position.is_call:
                        option_mark_price = max(0, S - K) + S * greeks['delta'] * 0.1
                    else:
                        option_mark_price = max(0, K - S) + S * abs(greeks['delta']) * 0.1
                
                # Add a spread to the selling price
                hedge_price = option_mark_price * (1 - self.SPREAD_PCT)  # Selling at a slight discount
                
                self.logger.info(f"Executing Option Hedge: {direction} {hedge_size:.4f} {option_to_hedge} at {hedge_price:.2f}")
                
                await self.thalex.insert(
                    id=self.CID_INSERT,
                    direction=direction,
                    instrument_name=option_to_hedge,
                    order_type=th.OrderType.LIMIT,
                    amount=hedge_size,
                    price=hedge_price
                )
            else:
                self.logger.warning(f"Position limit reached for {option_to_hedge}. Hedge cancelled. "
                                  f"Current: {option_position.position:.4f}, "
                                  f"Attempted: {hedge_size:.4f}, "
                                  f"Limit: {self.MAX_POSITION_SIZE}")
            
            self.last_hedge_time = current_time
            # Update expected position for the option
            option_position = self.options_positions[option_to_hedge]
            option_position.expected_position = (option_position.position or 0) - hedge_size  # Always selling
        
        await self.manage_gamma()

    def check_put_call_parity(self):
        """Check put-call parity to ensure option prices are consistent"""
        # Get call and put options with the same strike and expiry
        call_option = self.options_positions.get(OPTION)
        put_option = self.options_positions.get(OPTION_PUT)
        
        if not call_option or not put_option or call_option.strike != put_option.strike:
            self.logger.warning("Cannot check put-call parity: options with matching strikes not found")
            return
            
        if call_option.mark_price is None or put_option.mark_price is None:
            self.logger.warning("Cannot check put-call parity: missing mark prices")
            return
            
        # Calculate expiry time in years
        expiry_str = call_option.instrument_name.split('-')[1]
        expiry_date = datetime.datetime.strptime(expiry_str, "%d%b%y")
        now = datetime.datetime.now()
        days_to_expiry = (expiry_date - now).total_seconds() / (24 * 3600)
        T = max(days_to_expiry / 365.0, 0.0001)
        
        # Put-call parity formula: C - P = S - K*e^(-rT)
        # For simplicity, assume r = 0, so C - P = S - K
        S = self.mark_perp
        K = call_option.strike
        C = call_option.mark_price
        P = put_option.mark_price
        
        # Calculate parity difference
        parity_diff = (C - P) - (S - K)
        parity_pct = abs(parity_diff) / S * 100
        
        self.logger.info(f"Put-Call Parity Check:")
        self.logger.info(f"Call Price: {C:.2f}, Put Price: {P:.2f}")
        self.logger.info(f"Spot: {S:.2f}, Strike: {K:.2f}")
        self.logger.info(f"Parity Difference: {parity_diff:.2f} ({parity_pct:.2f}%)")
        
        # Log warning if parity violation is significant
        if parity_pct > 1.0:  # More than 1% difference
            self.logger.warning(f"Significant put-call parity violation: {parity_pct:.2f}%")
            
        return parity_diff

    def create_dashboard(self) -> Layout:
        """Create Rich layout for monitoring dashboard"""
        # Price Panel
        price_table = Table.grid(padding=(0, 4))
        price_table.add_row("[bold cyan]Mark Price[/]", f"{self.mark_perp or 0:.2f}")
        price_table.add_row("[yellow]24h Change[/]", f"{(self.price_history[-1]/self.price_history[0]-1)*100:.2f}%" 
                            if len(self.price_history) > 1 else "N/A")
        
        # Positions Panel
        pos_table = Table.grid(padding=(0, 2))
        pos_table.add_row("[bold]Instrument[/]", "[bold]Size[/]", "[bold]Avg Entry[/]", "[bold]Mark[/]", "[bold]P&L[/]")
        
        # Handle None values safely for perpetual position
        perp_pos = self.perp_position or 0
        mark_price = self.mark_perp or 0
        perp_avg_entry = getattr(self, 'perp_avg_entry_price', None)
        perp_pnl = 0.0
        
        if perp_avg_entry is not None and perp_pos != 0:
            if perp_pos > 0:
                perp_pnl = (mark_price - perp_avg_entry) * perp_pos
            else:
                perp_pnl = (perp_avg_entry - mark_price) * abs(perp_pos)
                
        perp_avg_entry_str = f"{perp_avg_entry:.2f}" if perp_avg_entry is not None else "N/A"
        perp_pnl_color = "green" if perp_pnl >= 0 else "red"
        pos_table.add_row("PERP", f"{perp_pos:.4f}", perp_avg_entry_str, f"{mark_price:.2f}", f"[{perp_pnl_color}]{perp_pnl:.2f}")
        
        # Add options positions
        for opt in self.options_positions.values():
            opt_pos = opt.position or 0
            opt_price = opt.mark_price or 0
            opt_type = "CALL" if opt.is_call else "PUT"
            opt_avg_entry = opt.avg_entry_price
            opt_pnl = opt.calculate_total_pnl()
            
            opt_avg_entry_str = f"{opt_avg_entry:.2f}" if opt_avg_entry is not None else "N/A"
            opt_pnl_color = "green" if opt_pnl >= 0 else "red"
            
            pos_table.add_row(
                f"{opt.instrument_name} ({opt_type})", 
                f"{opt_pos:.4f}", 
                opt_avg_entry_str,
                f"{opt_price:.2f}", 
                f"[{opt_pnl_color}]{opt_pnl:.2f}"
            )

        # Greeks Panel
        greeks_table = Table.grid(padding=(0, 2))
        total_delta = self.calculate_total_delta()
        total_gamma = self.calculate_total_gamma()
        greeks_table.add_row("[bold]Delta[/]", f"[{'red' if total_delta < 0 else 'green'}]{total_delta:.4f}")
        greeks_table.add_row("[bold]Gamma[/]", f"{total_gamma:.6f}")
        greeks_table.add_row("[bold]Vega[/]", f"{sum(op.greeks.vega for op in self.options_positions.values()):.2f}")
        greeks_table.add_row("[bold]Theta[/]", f"{sum(op.greeks.theta for op in self.options_positions.values()):.2f}")

        # Risk Panel
        risk_table = Table.grid(padding=(0, 2))
        hist_vol = self.calculate_historical_vol()
        risk_table.add_row("[bold]Volatility[/]", f"[magenta]{hist_vol:.2%}")
        risk_table.add_row("[bold]Position Limit[/]", f"{self.MAX_POSITION_SIZE:.2f}")
        risk_table.add_row("[bold]Fees Paid[/]", f"{self.total_fees:.6f}")
        
        # Put-Call Parity Panel
        parity_table = Table.grid(padding=(0, 2))
        call_option = self.options_positions.get(OPTION)
        put_option = self.options_positions.get(OPTION_PUT)
        
        if (call_option and put_option and 
            call_option.mark_price is not None and 
            put_option.mark_price is not None):
            
            # Calculate parity difference
            S = self.mark_perp
            K = call_option.strike
            C = call_option.mark_price
            P = put_option.mark_price
            
            parity_diff = (C - P) - (S - K)
            parity_pct = abs(parity_diff) / S * 100
            
            parity_color = "green" if parity_pct < 0.5 else "yellow" if parity_pct < 1.0 else "red"
            
            parity_table.add_row("[bold]Call Price[/]", f"{C:.2f}")
            parity_table.add_row("[bold]Put Price[/]", f"{P:.2f}")
            parity_table.add_row("[bold]Parity Diff[/]", f"[{parity_color}]{parity_diff:.2f} ({parity_pct:.2f}%)")
        else:
            parity_table.add_row("[bold]Parity Status[/]", "[yellow]Insufficient data")
            if call_option and put_option:
                parity_table.add_row("[bold]Call Mark Price[/]", f"{call_option.mark_price or 'None'}")
                parity_table.add_row("[bold]Put Mark Price[/]", f"{put_option.mark_price or 'None'}")
            
        # P&L Panel
        pnl_table = Table.grid(padding=(0, 2))
        if hasattr(self, 'current_portfolio_pnl'):
            portfolio_pnl = self.calculate_portfolio_pnl()
            total_pnl = portfolio_pnl["total_pnl"]
            realized_pnl = portfolio_pnl["total_realized"]
            unrealized_pnl = portfolio_pnl["total_unrealized"]
            
            # Calculate percentage P&L safely
            if self.initial_portfolio_value and self.initial_portfolio_value > 0:
                pct_pnl = (total_pnl / self.initial_portfolio_value) * 100
            else:
                pct_pnl = 0.0
                
            drawdown = self.calculate_drawdown() * 100  # Convert to percentage
            
            pnl_color = "green" if total_pnl >= 0 else "red"
            realized_color = "green" if realized_pnl >= 0 else "red"
            unrealized_color = "green" if unrealized_pnl >= 0 else "red"
            dd_color = "green" if drawdown < 1 else "yellow" if drawdown < 5 else "red"
            
            pnl_table.add_row("[bold]Portfolio Value[/]", f"${self.current_portfolio_value:.2f}")
            pnl_table.add_row("[bold]Total P&L[/]", f"[{pnl_color}]${total_pnl:.2f} ({pct_pnl:.2f}%)")
            pnl_table.add_row("[bold]Realized P&L[/]", f"[{realized_color}]${realized_pnl:.2f}")
            pnl_table.add_row("[bold]Unrealized P&L[/]", f"[{unrealized_color}]${unrealized_pnl:.2f}")
            pnl_table.add_row("[bold]Drawdown[/]", f"[{dd_color}]{drawdown:.2f}%")
            pnl_table.add_row("[bold]Hedge Threshold[/]", f"{self.get_dynamic_hedge_threshold():.2%}")
        else:
            pnl_table.add_row("[bold]P&L Status[/]", "[yellow]Initializing...")

        # Create columns with the new P&L panel
        return Layout(Columns([
            Panel(price_table, title="Prices", border_style="cyan"),
            Panel(pos_table, title="Positions", border_style="blue"),
            Panel(greeks_table, title="Greeks Exposure", border_style="green"),
            Panel(risk_table, title="Risk Metrics", border_style="magenta"),
            Panel(parity_table, title="Put-Call Parity", border_style="yellow"),
            Panel(pnl_table, title="P&L Metrics", border_style="red")
        ]))

    async def manage_positions(self):
        """Main method to manage positions and handle WebSocket communication"""
        max_reconnect_attempts = 10
        reconnect_attempt = 0
        reconnect_delay = 1.0  # Start with 1 second delay
        
        while reconnect_attempt <= max_reconnect_attempts:
            try:
                # Connect to the exchange
                self.logger.info(f"Connecting to Thalex exchange (attempt {reconnect_attempt + 1}/{max_reconnect_attempts + 1})")
                await self.thalex.connect()
                
                # Reset reconnect counters on successful connection
                reconnect_attempt = 0
                reconnect_delay = 1.0
                
                # Login and set up
                self.logger.info("Logging in to Thalex exchange")
                await self.thalex.login(KEY_ID, PRIV_KEY)
                await self.thalex.set_cancel_on_disconnect(5)
                
                # Subscribe to both call and put option tickers
                self.logger.info(f"Subscribing to: {PERP_TICKER}, {OPTION_TICKER}, {OPTION_PUT_TICKER}")
                await self.thalex.public_subscribe([PERP_TICKER, OPTION_TICKER, OPTION_PUT_TICKER])
                
                # Log the subscription
                self.logger.info("Subscriptions complete, requesting portfolio")
                await self.thalex.portfolio(self.CID_PORTFOLIO)
                
                # Wait for initial market data
                self.logger.info("Waiting for initial market data...")
                initial_data_timeout = 10  # seconds
                start_time = time.time()
                
                # Wait for perpetual mark price
                while self.mark_perp is None and time.time() - start_time < initial_data_timeout:
                    await asyncio.sleep(0.1)
                
                if self.mark_perp is None:
                    self.logger.warning(f"No perpetual mark price received after {initial_data_timeout}s. Using default value.")
                    self.mark_perp = 50000.0  # Default BTC price as fallback
                
                # Initialize option mark prices with theoretical values if needed
                self.logger.info("Initializing option mark prices with theoretical values")
                self.initialize_option_mark_prices()
                
                # Verify option mark prices are set
                missing_prices = [name for name, opt in self.options_positions.items() if opt.mark_price is None]
                if missing_prices:
                    self.logger.warning(f"Still missing mark prices for: {missing_prices}")
                    # Try to set them again with more aggressive defaults
                    for name in missing_prices:
                        option = self.options_positions[name]
                        # Set a very basic default price
                        if option.is_call:
                            option.mark_price = max(0, self.mark_perp - option.strike) + 100  # Intrinsic + time value
                        else:
                            option.mark_price = max(0, option.strike - self.mark_perp) + 100  # Intrinsic + time value
                        self.logger.info(f"Forced initialization of {name} with fallback price: {option.mark_price}")
                
                # Check put-call parity after forced initialization
                self.check_put_call_parity()
                
                # Main loop with dashboard
                with Live(console=self.console, refresh_per_second=4) as live:
                    while True:
                        try:
                            # Receive message with timeout
                            msg = json.loads(await asyncio.wait_for(self.thalex.receive(), timeout=30))
                            
                            # Process message
                            error = msg.get("error")
                            if error is not None:
                                self.logger.error(f"Error from exchange: {error}")
                                self.flush_data(error=error["message"], trades=[])
                                continue
                                
                            channel = msg.get("channel_name")
                            # If the message is not an error, then it's either a notification for a subscription
                            # or a result of an api call.
                            if channel is not None:
                                self.notification(channel, msg["notification"])
                            else:
                                self.result(msg.get("id"), msg["result"])
                                
                            await self.quote_options()
                            
                            # Update dashboard if needed
                            if time.time() - self.last_dashboard_update > self.DASHBOARD_REFRESH:
                                live.update(self.create_dashboard())
                                self.last_dashboard_update = time.time()
                                
                        except asyncio.TimeoutError:
                            # No message received within timeout, check connection
                            self.logger.warning("No message received for 30 seconds, checking connection...")
                            # Ping the server or perform other connection check
                            # For now, just continue and let the next receive attempt handle any issues
                            continue
                            
                        except (websockets.ConnectionClosed, ConnectionResetError) as e:
                            self.logger.error(f"WebSocket connection closed: {e}")
                            # Break out of the inner loop to trigger reconnection
                            break
                            
                        except Exception as e:
                            self.logger.error(f"Error in main loop: {e}", exc_info=True)
                            # Continue processing in case of other errors
                            continue
                
            except (websockets.ConnectionClosed, ConnectionRefusedError, socket.gaierror) as e:
                # Connection issues
                reconnect_attempt += 1
                self.logger.error(f"Connection error (attempt {reconnect_attempt}/{max_reconnect_attempts + 1}): {e}")
                
                if reconnect_attempt > max_reconnect_attempts:
                    self.logger.critical(f"Maximum reconnection attempts ({max_reconnect_attempts}) reached. Exiting.")
                    break
                    
                # Exponential backoff for reconnection
                self.logger.info(f"Reconnecting in {reconnect_delay:.1f} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Cap at 60 seconds
                
            except Exception as e:
                # Other unexpected errors
                self.logger.critical(f"Unexpected error: {e}", exc_info=True)
                reconnect_attempt += 1
                
                if reconnect_attempt > max_reconnect_attempts:
                    self.logger.critical(f"Maximum reconnection attempts ({max_reconnect_attempts}) reached. Exiting.")
                    break
                    
                # Exponential backoff for reconnection
                self.logger.info(f"Reconnecting in {reconnect_delay:.1f} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Cap at 60 seconds
                
            finally:
                # Clean up connection if needed
                try:
                    if self.thalex.connected():
                        self.logger.info("Disconnecting from exchange...")
                        await self.thalex.disconnect()
                except Exception as e:
                    self.logger.error(f"Error during disconnect: {e}")
                    
        self.logger.critical("Exiting manage_positions after maximum reconnection attempts")

    def initialize_option_mark_prices(self):
        """Initialize option mark prices with theoretical values if market data is not available"""
        if self.mark_perp is None:
            self.logger.warning("Cannot initialize option prices: perpetual mark price is not available")
            return
            
        hist_vol = self.calculate_historical_vol()
        self.logger.info(f"Initializing option mark prices with volatility: {hist_vol:.2%}")
        
        for option_name, option in self.options_positions.items():
            if option.mark_price is None:
                # Calculate theoretical price
                S = self.mark_perp
                K = option.strike
                
                # Calculate time to expiry
                expiry_str = option.instrument_name.split('-')[1]
                try:
                    expiry_date = datetime.datetime.strptime(expiry_str, "%d%b%y")
                    now = datetime.datetime.now()
                    days_to_expiry = (expiry_date - now).total_seconds() / (24 * 3600)
                    T = max(days_to_expiry / 365.0, 0.0001)  # Avoid division by zero
                except Exception as e:
                    self.logger.error(f"Error calculating expiry: {e}")
                    T = 0.25  # Default to 3 months
                
                r = 0.0  # Risk-free rate
                
                # Get option greeks
                greeks = OptionGreeks.black_scholes_greeks(
                    S, K, T, r, hist_vol,
                    'call' if option.is_call else 'put'
                )
                
                # Calculate theoretical price using Black-Scholes
                if option.is_call:
                    # For call options
                    d1 = (math.log(S/K) + (r + hist_vol**2/2)*T) / (hist_vol*math.sqrt(T))
                    d2 = d1 - hist_vol*math.sqrt(T)
                    
                    # Standard normal CDF
                    N = lambda x: (1 + math.erf(x/math.sqrt(2))) / 2
                    
                    theo_price = S * N(d1) - K * math.exp(-r*T) * N(d2)
                else:
                    # For put options
                    d1 = (math.log(S/K) + (r + hist_vol**2/2)*T) / (hist_vol*math.sqrt(T))
                    d2 = d1 - hist_vol*math.sqrt(T)
                    
                    # Standard normal CDF
                    N = lambda x: (1 + math.erf(x/math.sqrt(2))) / 2
                    
                    theo_price = K * math.exp(-r*T) * N(-d2) - S * N(-d1)
                
                # Ensure price is not negative
                theo_price = max(0.0, theo_price)
                
                # Set theoretical price
                option.mark_price = theo_price
                self.logger.info(f"Initialized {option_name} with theoretical price: {theo_price:.2f}")
                
                # Update option greeks
                option.update_greeks(S, hist_vol)
                
        # Check if we can now calculate put-call parity
        call_option = self.options_positions.get(OPTION)
        put_option = self.options_positions.get(OPTION_PUT)
        if (call_option and put_option and 
            call_option.mark_price is not None and 
            put_option.mark_price is not None):
            self.logger.info("Both call and put prices are now available for put-call parity")
            # Force a parity check
            self.check_put_call_parity()
            
        # Log the status of all option mark prices
        self.logger.info("Option mark price status after initialization:")
        for name, option in self.options_positions.items():
            self.logger.info(f"  {name}: Mark Price = {option.mark_price or 'None'}")
            
        # Verify put-call parity
        if call_option and put_option and call_option.mark_price and put_option.mark_price:
            # Calculate parity
            S = self.mark_perp
            K = call_option.strike
            C = call_option.mark_price
            P = put_option.mark_price
            
            parity_diff = (C - P) - (S - K)
            parity_pct = abs(parity_diff) / S * 100
            
            self.logger.info(f"Put-Call Parity Check after initialization:")
            self.logger.info(f"  C - P = {C - P:.2f}, S - K = {S - K:.2f}")
            self.logger.info(f"  Parity Difference: {parity_diff:.2f} ({parity_pct:.2f}%)")
        else:
            self.logger.warning("Still unable to calculate put-call parity after initialization")

    def flush_data(self, trades, error=None):
        # This is the format of the data we want for parsing later to calculate total pnl
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open("inventory_hedger.csv", "a", newline="") as f:
            w = csv.writer(f)
            data = {
                "timestamp": time_now,
                "direction": "",
                "ticker_UL": PERPETUAL,
                "amount": "",
                "price": "",
                "mark_perpetual": round(self.mark_perp, 3) if self.mark_perp else None,
                "perp_position": round(self.perp_position, 3) if self.perp_position else None,
                "ticker_option_call": OPTION,
                "ticker_option_put": OPTION_PUT,  # Add put option to the data
                "error": error,
            }
            if f.tell() == 0:
                w.writerow(data.keys())
            for trade in trades:
                data.update(
                    {
                        "direction": trade.get("direction"),
                        "amount": trade.get("amount"),
                        "price": trade.get("price"),
                    }
                )
                w.writerow(data.values())

    def notification(self, channel, notification):
        if channel == PERP_TICKER:
            self.last_mark_update = time.time()
            old_mark = self.mark_perp
            self.mark_perp = notification["mark_price"]
            
            if old_mark is not None:
                price_change = (self.mark_perp - old_mark) / old_mark * 100
                self.logger.info(f"Price changed by {price_change:.2f}% to {self.mark_perp}")
            
            # Update price history for vol calculation
            self.price_history.append(self.mark_perp)
            if len(self.price_history) > self.VOL_WINDOW:
                self.price_history.pop(0)
            
            # Calculate and log historical vol
            hist_vol = self.calculate_historical_vol()
            self.logger.info(f"Current historical volatility: {hist_vol:.2%}")
            
            # Log warning if volatility is at minimum floor (indicating potential issue)
            if abs(hist_vol - self.MIN_VOLATILITY) < 0.0001:
                self.logger.warning("Volatility is at minimum floor - potential data issue!")
            
            # Update options greeks with enhanced logging
            for option_name, option in self.options_positions.items():
                old_delta = option.greeks.delta
                option.update_greeks(self.mark_perp, hist_vol)
                
                # Log significant delta changes with more detail
                if old_delta:
                    delta_diff = option.greeks.delta - old_delta
                    if abs(delta_diff) > 0.05:
                        self.logger.warning(
                            f"Significant delta change in {option_name}: {old_delta:.4f} → {option.greeks.delta:.4f} "
                            f"(change: {delta_diff:.4f}). "
                            f"S/K ratio: {self.mark_perp/option.strike:.2f}, "
                            f"Vol: {hist_vol:.2%}"
                        )
                
                # Extra warning for extreme deltas
                if option.greeks.delta > 0.99 or option.greeks.delta < 0.01:
                    moneyness = self.mark_perp / option.strike
                    self.logger.warning(
                        f"Extreme delta ({option.greeks.delta:.4f}) for {option_name}. "
                        f"S/K: {moneyness:.2f}, Vol: {hist_vol:.2%}"
                    )
        elif channel == OPTION_TICKER or channel == OPTION_PUT_TICKER:
            # Handle option ticker updates
            instrument_name = notification.get("instrument_name")
            if instrument_name in self.options_positions:
                option = self.options_positions[instrument_name]
                old_mark = option.mark_price
                option.mark_price = notification.get("mark_price")
                
                # Log more detailed information about the option price update
                self.logger.info(f"Updated {instrument_name} mark price from {old_mark} to {option.mark_price}")
                
                # If this is the first time we're getting a mark price, log it specially
                if old_mark is None and option.mark_price is not None:
                    self.logger.info(f"Received first mark price for {instrument_name}: {option.mark_price}")
                    
                # Check if we now have both option prices for put-call parity
                call_option = self.options_positions.get(OPTION)
                put_option = self.options_positions.get(OPTION_PUT)
                if (call_option and put_option and 
                    call_option.mark_price is not None and 
                    put_option.mark_price is not None):
                    self.logger.info("Both call and put prices are now available for put-call parity")
                    # Force a parity check
                    self.check_put_call_parity()
            else:
                self.logger.warning(f"Received update for unknown instrument: {instrument_name}")
        else:
            self.logger.info(f"Received notification for channel: {channel}")

    def result(self, msg_id, result):
        if msg_id is None:
            return
        if msg_id == self.CID_INSERT:
            # Track fees
            total_fill_amount = 0
            total_fill_value = 0
            
            for fill in result.get("fills", []):
                amount = fill["amount"]
                price = fill["price"]
                fee = fill.get("fee", 0)
                
                total_fill_amount += amount
                total_fill_value += amount * price
                self.total_fees += fee

            # Log trade analytics
            if total_fill_amount > 0:
                avg_price = total_fill_value / total_fill_amount
                self.logger.info(f"Trade Analytics:")
                self.logger.info(f"Average Fill Price: {avg_price:.2f}")
                self.logger.info(f"Total Fees: {self.total_fees:.6f}")
                self.logger.info(f"Fill Amount: {total_fill_amount:.4f}")

            fills = result["fills"]
            direction = result["direction"]
            side_sign = 1 if direction == "buy" else -1
            trades = [
                {"direction": direction, "price": f["price"], "amount": f["amount"]}
                for f in fills
            ]
            
            # Log trade execution details
            for trade in trades:
                self.logger.info(f"Executed trade: {trade['direction']} {trade['amount']} "
                               f"contracts at {trade['price']}")
            
            # Get the instrument being traded
            instrument_name = result.get("instrument_name", "")
            
            # Update position based on instrument type
            if instrument_name == PERPETUAL:
                old_position = self.perp_position
                total_fill_amount = 0
                
                # Initialize perpetual P&L tracking if not exists
                if not hasattr(self, 'perp_avg_entry_price'):
                    self.perp_avg_entry_price = None
                    self.perp_realized_pnl = 0.0
                    self.perp_trades_history = []
                
                # Calculate average fill price
                avg_fill_price = sum(f["price"] * f["amount"] for f in fills) / sum(f["amount"] for f in fills) if fills else 0
                
                # Update position and P&L
                for fill in fills:
                    if self.perp_position is None:
                        self.perp_position = 0
                    
                    fill_amount = side_sign * fill["amount"]
                    fill_price = fill["price"]
                    
                    # If closing or reducing position, calculate realized P&L
                    if (old_position > 0 and fill_amount < 0) or (old_position < 0 and fill_amount > 0):
                        # Determine how much of the position is being closed
                        close_size = min(abs(old_position), abs(fill_amount))
                        close_dir = -1 if old_position > 0 else 1
                        
                        # Calculate P&L on the closed portion
                        if self.perp_avg_entry_price is not None:
                            pnl_per_contract = (fill_price - self.perp_avg_entry_price) * close_dir
                            realized_pnl = pnl_per_contract * close_size
                            self.perp_realized_pnl += realized_pnl
                            self.logger.info(f"Realized P&L: ${realized_pnl:.2f} (price diff: {pnl_per_contract:.2f} × size: {close_size})")
                    
                    # Update position
                    self.perp_position += fill_amount
                    total_fill_amount += fill_amount
                    
                    # Update average entry price for new positions
                    new_position = old_position + fill_amount
                    if abs(new_position) > abs(old_position):
                        # If increasing position or changing direction
                        if self.perp_avg_entry_price is None or old_position * new_position <= 0:
                            # New position or direction change
                            self.perp_avg_entry_price = fill_price
                            self.logger.info(f"New perp entry price: {fill_price:.2f}")
                        else:
                            # Adding to existing position, update average price
                            old_value = abs(old_position) * self.perp_avg_entry_price
                            new_value = abs(fill_amount) * fill_price
                            self.perp_avg_entry_price = (old_value + new_value) / abs(new_position)
                            self.logger.info(f"Updated perp avg entry price: {self.perp_avg_entry_price:.2f}")
                    
                    # If position is completely closed, reset avg entry price
                    if self.perp_position == 0:
                        self.perp_avg_entry_price = None
                        self.logger.info("Position closed, reset entry price")
                    
                    # Store the trade
                    self.perp_trades_history.append({
                        'timestamp': time.time(),
                        'direction': direction,
                        'size': fill["amount"],
                        'price': fill_price,
                        'old_position': old_position,
                        'new_position': self.perp_position,
                        'realized_pnl': self.perp_realized_pnl
                    })
                
                old_pos_str = "None" if old_position is None else f"{old_position:.4f}"
                self.logger.info(f"Total perpetual position change: {total_fill_amount:+.4f}, From {old_pos_str} to {self.perp_position:.4f}")
                self.logger.info(f"Current perp avg entry price: {self.perp_avg_entry_price}")
            
            elif instrument_name in self.options_positions:
                # Handle option position updates
                option = self.options_positions[instrument_name]
                old_position = option.position
                
                # Calculate average fill price
                avg_fill_price = sum(f["price"] * f["amount"] for f in fills) / sum(f["amount"] for f in fills) if fills else 0
                
                for fill in fills:
                    fill_amount = fill["amount"]
                    fill_price = fill["price"]
                    
                    # Update position with P&L tracking
                    option.update_position(fill_amount, fill_price, direction)
                
                self.logger.info(f"Total {instrument_name} position change: {option.position - old_position:+.4f}, From {old_position:.4f} to {option.position:.4f}")
                self.logger.info(f"Current {instrument_name} avg entry price: {option.avg_entry_price}")
                self.logger.info(f"Realized P&L for {instrument_name}: ${option.realized_pnl:.2f}")
                
                # Update expected position
                option.expected_position = option.position
            
            self.flush_data(trades=trades)
            
            # Record expected position change with timestamp and trade ID
            trade_id = result.get("order_id", "unknown")
            self.transaction_log.append({
                "timestamp": time.time(),
                "trade_id": trade_id,
                "instrument": instrument_name,
                "old_position": old_position,
                "expected_new_position": self.perp_position if instrument_name == PERPETUAL else self.options_positions[instrument_name].position if instrument_name in self.options_positions else None,
                "fills": fills,
                "direction": direction
            })
            
            # Log the 5 most recent transactions when drift is detected
            if hasattr(self, 'last_drift_detected') and self.last_drift_detected:
                self.logger.info("Recent transactions:")
                for tx in self.transaction_log[-5:]:
                    self.logger.info(f"  {datetime.datetime.fromtimestamp(tx['timestamp']).strftime('%H:%M:%S')} - "
                                   f"ID: {tx['trade_id']}, {tx['direction']} - "
                                   f"Changed from {tx['old_position']} to {tx['expected_new_position']}")
                self.last_drift_detected = False
            
        elif msg_id == self.CID_PORTFOLIO:
            self.last_position_update = time.time()
            self.logger.info("Updating portfolio positions")
            # Update both perpetual and options positions
            for pos in result:
                instrument = pos["instrument_name"]
                if instrument == PERPETUAL:
                    old_pos = self.perp_position
                    self.perp_position = pos["position"]
                    self.logger.info(f"Updated perpetual position from {old_pos} to {self.perp_position}")
                elif instrument in self.options_positions:
                    old_pos = self.options_positions[instrument].position
                    self.options_positions[instrument].position = pos["position"]
                    self.options_positions[instrument].expected_position = pos["position"]  # Reset expected position
                    self.logger.info(f"Updated {instrument} position from {old_pos} to {pos['position']}")
            
            # Track position drift for perpetual
            if self.expected_perp_position is not None and self.perp_position is not None:
                drift = self.perp_position - self.expected_perp_position
                self.drift_history.append(drift)
                if len(self.drift_history) > self.DRIFT_HISTORY_LENGTH:
                    self.drift_history.pop(0)
                
                # Analyze drift patterns
                if len(self.drift_history) >= 3:
                    consistent_direction = all(d > 0 for d in self.drift_history) or all(d < 0 for d in self.drift_history)
                    if consistent_direction:
                        self.logger.warning(f"Consistent drift pattern detected in {self.DRIFT_HISTORY_LENGTH} recent updates. "
                                          f"Check for systematic issues. Recent drifts: {self.drift_history}")
                
                # Reset expected position to actual position
                self.expected_perp_position = self.perp_position
                self.last_drift_detected = abs(drift) > self.MAX_POSITION_DRIFT

    def validate_hedge_requirements(self, total_delta: float) -> bool:
        """
        Validate if hedging is required and possible
        Returns: bool indicating if hedging should proceed
        """
        # Check if delta is near threshold for testing
        near_threshold = abs(abs(total_delta) - self.DELTA_HEDGE_THRESHOLD) < 0.001
        if near_threshold:
            self.logger.warning(f"Delta {total_delta:.4f} is very close to "
                              f"threshold {self.DELTA_HEDGE_THRESHOLD}")
        
        # Verify position limits
        hedge_size = min(abs(total_delta), self.QUOTE_SIZE)
        potential_position = (self.perp_position or 0) + hedge_size
        within_limits = abs(potential_position) <= self.MAX_POSITION_SIZE
        
        self.logger.info(f"Hedge Validation:")
        self.logger.info(f"Delta: {total_delta:.4f}")
        self.logger.info(f"Above Threshold: {abs(total_delta) > self.DELTA_HEDGE_THRESHOLD}")
        self.logger.info(f"Within Position Limits: {within_limits}")
        
        return abs(total_delta) > self.DELTA_HEDGE_THRESHOLD and within_limits


async def main():
    # Set up logging
    log_level = getattr(logging, CONFIG["log_level"])
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    
    logging.info(f"Starting option inventory hedger with network: {CONFIG['network']}")
    logging.info(f"Call option: {OPTION}, Put option: {OPTION_PUT}, Perpetual: {PERPETUAL}")
    
    run = True
    reconnect_delay = 0.1  # Starting delay in seconds
    max_reconnect_delay = 30  # Maximum delay in seconds
    
    while run:
        thalex = th.Thalex(NETWORK)
        hedger = OptionInventoryHedger(thalex)
        task = asyncio.create_task(hedger.manage_positions())
        try:
            await task
        except (websockets.ConnectionClosed, socket.gaierror):
            # Implement exponential backoff for reconnection attempts
            logging.exception(f"Lost connection. Reconnecting in {reconnect_delay:.1f}s...")
            time.sleep(reconnect_delay)
            # Increase delay for next attempt, capped at the maximum
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
        except asyncio.CancelledError:
            logging.info("Signal received. Stopping...")
            run = False
        except Exception as e:
            # Catch other unexpected exceptions
            logging.exception(f"Unexpected error: {e}. Reconnecting in {reconnect_delay:.1f}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            
        # Reset delay after successful connection
        if task.done() and not task.cancelled() and not task.exception():
            reconnect_delay = 0.1
            
        if thalex.connected():
            await thalex.disconnect()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main()) 
