#!/usr/bin/env python3
"""
Background Data Collection Service
=================================

This service runs continuously in the background to collect and store:
- Real-time BTC prices
- Options market data
- Volatility surface updates
- Regime detection data

This ensures the database is always populated with fresh data for 
the enhanced MAD straddle analyzer.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import signal
import sys

import thalex_py.thalex as th
import keys
from supabase_config import SupabaseManager, create_supabase_config
from financial_math import FinancialMath, TradingConstants


class BackgroundDataService:
    """
    Background service for continuous data collection
    """
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval  # seconds
        self.supabase_manager: Optional[SupabaseManager] = None
        self.thalex_client: Optional[th.Thalex] = None
        self.logger = logging.getLogger(__name__)
        
        # Service state
        self.running = False
        self.last_price_update = None
        self.last_options_update = None
        self.last_cleanup = None
        
        # Data collection statistics
        self.stats = {
            'prices_collected': 0,
            'options_collected': 0,
            'surface_points_added': 0,
            'regime_updates': 0,
            'errors': 0,
            'uptime_start': None
        }
        
        # Error handling
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        self.reconnect_delay = 60  # seconds
        
    async def initialize(self) -> bool:
        """Initialize the background service"""
        try:
            # Initialize Supabase
            supabase_config = create_supabase_config()
            self.supabase_manager = SupabaseManager(supabase_config)
            
            if not await self.supabase_manager.initialize():
                self.logger.error("Failed to initialize Supabase manager")
                return False
            
            # Initialize Thalex client
            self.thalex_client = th.Thalex(
                network=th.Network.TEST,  # Use testnet for safety
                key_id=keys.key_ids[th.Network.TEST],
                private_key=keys.private_keys[th.Network.TEST]
            )
            
            self.stats['uptime_start'] = datetime.now(timezone.utc)
            self.logger.info("Background data service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize background service: {e}")
            return False
    
    async def start(self):
        """Start the background data collection service"""
        try:
            if not await self.initialize():
                self.logger.error("Service initialization failed")
                return
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.running = True
            self.logger.info("Starting background data collection service...")
            self.logger.info(f"Collection interval: {self.collection_interval} seconds")
            
            # Main service loop
            while self.running:
                try:
                    await self._collection_cycle()
                    self.consecutive_errors = 0  # Reset error counter on success
                    
                except Exception as e:
                    self.consecutive_errors += 1
                    self.stats['errors'] += 1
                    self.logger.error(f"Collection cycle error ({self.consecutive_errors}/{self.max_consecutive_errors}): {e}")
                    
                    # If too many consecutive errors, try to reconnect
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self.logger.warning("Too many consecutive errors, attempting reconnection...")
                        await self._handle_reconnection()
                
                # Wait for next collection cycle
                if self.running:
                    await asyncio.sleep(self.collection_interval)
            
            self.logger.info("Background data service stopped")
            
        except Exception as e:
            self.logger.error(f"Service error: {e}")
        finally:
            await self._cleanup()
    
    async def _collection_cycle(self):
        """Execute one complete data collection cycle"""
        try:
            # Ensure Thalex connection
            if not await self._ensure_thalex_connection():
                raise Exception("Could not establish Thalex connection")
            
            # Collect BTC price
            await self._collect_btc_price()
            
            # Collect options data (less frequent - every 5 minutes)
            now = datetime.now(timezone.utc)
            if (not self.last_options_update or 
                (now - self.last_options_update).total_seconds() >= 300):
                await self._collect_options_data()
                self.last_options_update = now
            
            # Perform database cleanup (daily)
            if (not self.last_cleanup or 
                (now - self.last_cleanup).total_seconds() >= 86400):  # 24 hours
                await self._perform_database_cleanup()
                self.last_cleanup = now
            
            # Log periodic status
            if self.stats['prices_collected'] % 100 == 0:
                self._log_service_status()
                
        except Exception as e:
            self.logger.error(f"Collection cycle failed: {e}")
            raise
    
    async def _ensure_thalex_connection(self) -> bool:
        """Ensure Thalex WebSocket connection is active"""
        try:
            if not self.thalex_client:
                return False
            
            # Check if already connected
            if hasattr(self.thalex_client, 'ws') and self.thalex_client.ws:
                # Test connection with ping or simple request
                try:
                    await self.thalex_client.get_index("BTCUSD")
                    return True
                except:
                    # Connection is stale, disconnect and reconnect
                    pass
            
            # Connect and authenticate
            await self.thalex_client.connect()
            await self.thalex_client.login()
            
            self.logger.info("Thalex connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to establish Thalex connection: {e}")
            return False
    
    async def _collect_btc_price(self):
        """Collect current BTC price and store in database"""
        try:
            if not self.thalex_client or not self.supabase_manager:
                return
            
            # Get index price
            index_data = await self.thalex_client.get_index("BTCUSD")
            if not index_data or 'index_price' not in index_data:
                raise Exception("Could not get BTC index price")
            
            price = float(index_data['index_price'])
            timestamp = datetime.now(timezone.utc)
            
            # Store in database
            success = await self.supabase_manager.insert_btc_price(
                timestamp, price, source='background_service'
            )
            
            if success:
                self.stats['prices_collected'] += 1
                self.last_price_update = timestamp
                
                # Log periodic price updates
                if self.stats['prices_collected'] % 50 == 0:
                    self.logger.info(f"Collected price update: ${price:.2f} ({self.stats['prices_collected']} total)")
            else:
                raise Exception("Failed to store price in database")
                
        except Exception as e:
            self.logger.error(f"Error collecting BTC price: {e}")
            raise
    
    async def _collect_options_data(self):
        """Collect options market data and update volatility surface"""
        try:
            if not self.thalex_client or not self.supabase_manager:
                return
            
            self.logger.info("Collecting options market data...")
            
            # Get current BTC price for calculations
            btc_price = await self._get_current_btc_price()
            if not btc_price:
                raise Exception("Could not get current BTC price")
            
            # Get all BTCUSD instruments
            instruments = await self.thalex_client.get_instruments("BTCUSD")
            
            # Filter for options and collect data
            options_collected = 0
            surface_points = 0
            
            for instrument in instruments:
                if instrument.get('instrument_type') == 'option':
                    try:
                        # Get market data for this option
                        option_data = await self._get_option_market_data(instrument, btc_price)
                        
                        if option_data:
                            # Store option data
                            if await self.supabase_manager.insert_option_data(option_data):
                                options_collected += 1
                            
                            # Add to volatility surface if implied vol is available
                            if option_data.get('implied_volatility'):
                                surface_data = self._create_surface_data_point(option_data, btc_price)
                                if await self.supabase_manager.insert_volatility_surface_point(surface_data):
                                    surface_points += 1
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.debug(f"Error processing option {instrument.get('instrument_name', 'unknown')}: {e}")
                        continue
            
            self.stats['options_collected'] += options_collected
            self.stats['surface_points_added'] += surface_points
            
            self.logger.info(f"Collected {options_collected} options, added {surface_points} surface points")
            
        except Exception as e:
            self.logger.error(f"Error collecting options data: {e}")
            raise
    
    async def _get_current_btc_price(self) -> Optional[float]:
        """Get current BTC price from Thalex"""
        try:
            if not self.thalex_client:
                return None
            
            index_data = await self.thalex_client.get_index("BTCUSD")
            if index_data and 'index_price' in index_data:
                return float(index_data['index_price'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting BTC price: {e}")
            return None
    
    async def _get_option_market_data(self, instrument: Dict[str, Any], btc_price: float) -> Optional[Dict[str, Any]]:
        """Get market data for a specific option"""
        try:
            instrument_name = instrument['instrument_name']
            
            # Get orderbook
            orderbook = await self.thalex_client.get_orderbook(instrument_name)
            if not orderbook:
                return None
            
            # Parse instrument details
            parts = instrument_name.split('-')
            if len(parts) != 4:
                return None
            
            expiry_str = parts[1]
            strike_str = parts[2]
            option_type = 'call' if parts[3] == 'C' else 'put'
            
            # Parse expiry date
            expiry_date = self._parse_expiry_date(expiry_str)
            if not expiry_date:
                return None
            
            # Calculate days to expiry
            now = datetime.now(timezone.utc)
            days_to_expiry = (expiry_date - now).days
            
            if days_to_expiry < 0:  # Skip expired options
                return None
            
            # Get prices
            mark_price = orderbook.get('mark_price', 0)
            best_bid = orderbook.get('best_bid_price', 0) if orderbook.get('best_bid_qty', 0) > 0 else None
            best_ask = orderbook.get('best_ask_price', 0) if orderbook.get('best_ask_qty', 0) > 0 else None
            
            # Calculate implied volatility
            implied_vol = None
            if mark_price > 0 and days_to_expiry > 0:
                implied_vol = FinancialMath.implied_volatility_newton_raphson(
                    mark_price, btc_price, float(strike_str), 
                    days_to_expiry / 365.0, TradingConstants.DEFAULT_RISK_FREE_RATE, option_type
                )
            
            return {
                'timestamp': now,
                'expiry_date': expiry_date.date(),
                'expiry_timestamp': int(expiry_date.timestamp()),
                'strike_price': float(strike_str),
                'option_type': option_type,
                'mark_price': float(mark_price),
                'bid_price': float(best_bid) if best_bid else None,
                'ask_price': float(best_ask) if best_ask else None,
                'implied_volatility': implied_vol,
                'volume': 0,  # Would need trade data
                'open_interest': 0,  # Would need OI data
                'underlying_price': btc_price,
                'days_to_expiry': days_to_expiry
            }
            
        except Exception as e:
            self.logger.debug(f"Error getting option data for {instrument.get('instrument_name', 'unknown')}: {e}")
            return None
    
    def _parse_expiry_date(self, expiry_str: str) -> Optional[datetime]:
        """Parse Thalex expiry date format (DDMMMYY)"""
        try:
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            if len(expiry_str) != 7:
                return None
            
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5]
            year = int(expiry_str[5:7]) + 2000
            
            month = month_map.get(month_str)
            if not month:
                return None
            
            return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)
            
        except Exception:
            return None
    
    def _create_surface_data_point(self, option_data: Dict[str, Any], btc_price: float) -> Dict[str, Any]:
        """Create volatility surface data point"""
        strike_price = option_data['strike_price']
        moneyness = strike_price / btc_price
        
        return {
            'timestamp': option_data['timestamp'],
            'expiry_date': option_data['expiry_date'],
            'strike_price': strike_price,
            'implied_volatility': option_data['implied_volatility'],
            'confidence_level': 0.8,  # Default confidence
            'underlying_price': btc_price,
            'days_to_expiry': option_data['days_to_expiry'],
            'moneyness': moneyness,
            'data_source': 'background_service'
        }
    
    async def _perform_database_cleanup(self):
        """Perform daily database maintenance"""
        try:
            if not self.supabase_manager:
                return
            
            self.logger.info("Performing database cleanup...")
            
            # Clean up old data (keep 30 days)
            await self.supabase_manager.cleanup_old_data(days_to_keep=30)
            
            self.logger.info("Database cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Database cleanup error: {e}")
    
    async def _handle_reconnection(self):
        """Handle reconnection after consecutive errors"""
        try:
            self.logger.info("Attempting service reconnection...")
            
            # Disconnect from Thalex
            if self.thalex_client:
                try:
                    await self.thalex_client.disconnect()
                except:
                    pass
            
            # Wait before reconnecting
            await asyncio.sleep(self.reconnect_delay)
            
            # Reinitialize
            if await self.initialize():
                self.consecutive_errors = 0
                self.logger.info("Service reconnection successful")
            else:
                self.logger.error("Service reconnection failed")
                
        except Exception as e:
            self.logger.error(f"Reconnection error: {e}")
    
    def _log_service_status(self):
        """Log periodic service status"""
        if not self.stats['uptime_start']:
            return
        
        uptime = datetime.now(timezone.utc) - self.stats['uptime_start']
        uptime_hours = uptime.total_seconds() / 3600
        
        self.logger.info("=== Background Data Service Status ===")
        self.logger.info(f"Uptime: {uptime_hours:.1f} hours")
        self.logger.info(f"Prices collected: {self.stats['prices_collected']:,}")
        self.logger.info(f"Options collected: {self.stats['options_collected']:,}")
        self.logger.info(f"Surface points: {self.stats['surface_points_added']:,}")
        self.logger.info(f"Total errors: {self.stats['errors']}")
        
        if self.last_price_update:
            age = (datetime.now(timezone.utc) - self.last_price_update).total_seconds()
            self.logger.info(f"Last price update: {age:.0f} seconds ago")
        
        self.logger.info("=====================================")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _cleanup(self):
        """Clean up service resources"""
        try:
            if self.thalex_client:
                await self.thalex_client.disconnect()
            
            if self.supabase_manager:
                await self.supabase_manager.close()
                
            self.logger.info("Service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        """Stop the service"""
        self.running = False


async def main():
    """Main entry point for background data service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Starting Background Data Collection Service...")
    print("This service will continuously collect BTC prices and options data")
    print("Press Ctrl+C to stop the service gracefully")
    
    service = BackgroundDataService(collection_interval=30)  # 30 second intervals
    
    try:
        await service.start()
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"Service error: {e}")
        logging.error(f"Service error: {e}", exc_info=True)
    finally:
        print("Background data service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())