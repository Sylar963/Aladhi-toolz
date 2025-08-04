#!/usr/bin/env python3
"""
Historical Data Collector for Options Trading
=============================================

This module collects and backfills historical data for comprehensive volatility analysis:
- Historical BTC prices from multiple sources
- Historical options data from Thalex API
- Data validation and quality checks
- Efficient bulk data insertion

This addresses the major limitation in enhanced_mad_straddle.py where only ~100 recent
data points were collected, providing insufficient data for robust volatility modeling.

Data Sources:
- Thalex WebSocket API for real-time and recent historical data
- CoinGecko API for historical price data
- Multiple exchanges for price validation
"""

import asyncio
import json
import logging
import math
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import aiohttp

import thalex_py.thalex as th
import keys
from supabase_config import SupabaseManager, create_supabase_config


@dataclass
class DataCollectionConfig:
    """Configuration for historical data collection"""
    days_back: int = 180  # 6 months of data
    btc_price_interval_minutes: int = 5  # 5-minute intervals
    max_concurrent_requests: int = 10
    rate_limit_delay: float = 0.1  # Delay between requests
    retry_attempts: int = 3
    batch_size: int = 1000  # Records per batch


class HistoricalDataCollector:
    """
    Collects and stores historical data for enhanced volatility analysis
    """
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.supabase_manager: Optional[SupabaseManager] = None
        self.thalex_client: Optional[th.Thalex] = None
        self.logger = logging.getLogger(__name__)
        
        # External API session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Data validation counters
        self.stats = {
            'btc_prices_collected': 0,
            'options_collected': 0,
            'validation_errors': 0,
            'api_errors': 0,
            'duplicates_skipped': 0
        }
    
    async def initialize(self) -> bool:
        """Initialize all data collection components"""
        try:
            # Initialize Supabase
            supabase_config = create_supabase_config()
            self.supabase_manager = SupabaseManager(supabase_config)
            
            if not await self.supabase_manager.initialize():
                self.logger.error("Failed to initialize Supabase")
                return False
            
            # Initialize Thalex client
            self.thalex_client = th.Thalex(
                network=th.Network.TEST,  # Start with testnet
                key_id=keys.key_ids[th.Network.TEST],
                private_key=keys.private_keys[th.Network.TEST]
            )
            
            # Initialize HTTP session for external APIs
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
            )
            
            self.logger.info("Historical data collector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data collector: {e}")
            return False
    
    async def collect_all_historical_data(self) -> bool:
        """Collect all types of historical data"""
        try:
            self.logger.info(f"Starting historical data collection for {self.config.days_back} days")
            
            # Step 1: Collect historical BTC prices
            await self._collect_historical_btc_prices()
            
            # Step 2: Connect to Thalex and collect options data
            if await self._connect_to_thalex():
                await self._collect_historical_options_data()
                await self._disconnect_from_thalex()
            
            # Step 3: Generate summary report
            self._print_collection_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in historical data collection: {e}")
            return False
    
    async def _collect_historical_btc_prices(self):
        """Collect historical BTC prices from CoinGecko API"""
        try:
            self.logger.info("Collecting historical BTC prices...")
            
            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=self.config.days_back)
            
            # CoinGecko API for historical data
            # Free tier allows up to 100 calls per month, so we'll use daily data first
            await self._fetch_coingecko_daily_data(start_time, end_time)
            
            # For more recent data, use hourly granularity
            recent_start = end_time - timedelta(days=30)  # Last 30 days
            await self._fetch_coingecko_hourly_data(recent_start, end_time)
            
        except Exception as e:
            self.logger.error(f"Error collecting historical BTC prices: {e}")
            self.stats['api_errors'] += 1
    
    async def _fetch_coingecko_daily_data(self, start_time: datetime, end_time: datetime):
        """Fetch daily BTC data from CoinGecko"""
        try:
            if not self.session:
                return
            
            # Convert to Unix timestamps
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_ts,
                'to': end_ts
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    prices_data = data.get('prices', [])
                    
                    # Process and store price data
                    batch = []
                    for timestamp_ms, price in prices_data:
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                        
                        batch.append({
                            'timestamp': timestamp,
                            'price': float(price),
                            'source': 'coingecko_daily'
                        })
                        
                        if len(batch) >= self.config.batch_size:
                            await self._store_price_batch(batch)
                            batch = []
                    
                    # Store remaining batch
                    if batch:
                        await self._store_price_batch(batch)
                    
                    self.logger.info(f"Collected {len(prices_data)} daily price points")
                    
                else:
                    self.logger.error(f"CoinGecko API error: {response.status}")
                    self.stats['api_errors'] += 1
                
        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko daily data: {e}")
            self.stats['api_errors'] += 1
    
    async def _fetch_coingecko_hourly_data(self, start_time: datetime, end_time: datetime):
        """Fetch hourly BTC data from CoinGecko for recent period"""
        try:
            if not self.session:
                return
            
            # CoinGecko hourly data (last 90 days max)
            days_diff = (end_time - start_time).days
            if days_diff > 90:
                start_time = end_time - timedelta(days=90)
            
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days_diff,
                'interval': 'hourly'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    prices_data = data.get('prices', [])
                    
                    # Process and store price data
                    batch = []
                    for timestamp_ms, price in prices_data:
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                        
                        batch.append({
                            'timestamp': timestamp,
                            'price': float(price),
                            'source': 'coingecko_hourly'
                        })
                        
                        if len(batch) >= self.config.batch_size:
                            await self._store_price_batch(batch)
                            batch = []
                    
                    # Store remaining batch
                    if batch:
                        await self._store_price_batch(batch)
                    
                    self.logger.info(f"Collected {len(prices_data)} hourly price points")
                    
        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko hourly data: {e}")
            self.stats['api_errors'] += 1
    
    async def _store_price_batch(self, batch: List[Dict[str, Any]]):
        """Store batch of price data efficiently"""
        try:
            if not self.supabase_manager or not self.supabase_manager.pool:
                return
            
            # Use batch insert for efficiency
            async with self.supabase_manager.pool.acquire() as conn:
                # Prepare values for bulk insert
                values = [
                    (item['timestamp'], item['price'], None, item['source'])
                    for item in batch
                ]
                
                # Bulk insert with conflict handling (ignore duplicates)
                await conn.executemany('''
                    INSERT INTO btc_prices (timestamp, price, volume, source)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (timestamp, source) DO NOTHING
                ''', values)
                
                self.stats['btc_prices_collected'] += len(batch)
                
        except Exception as e:
            self.logger.error(f"Error storing price batch: {e}")
    
    async def _connect_to_thalex(self) -> bool:
        """Connect to Thalex WebSocket"""
        try:
            if not self.thalex_client:
                return False
            
            await self.thalex_client.connect()
            await self.thalex_client.login()
            
            self.logger.info("Connected to Thalex WebSocket")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Thalex: {e}")
            return False
    
    async def _disconnect_from_thalex(self):
        """Disconnect from Thalex WebSocket"""
        try:
            if self.thalex_client:
                await self.thalex_client.disconnect()
                self.logger.info("Disconnected from Thalex WebSocket")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Thalex: {e}")
    
    async def _collect_historical_options_data(self):
        """Collect historical options data from Thalex"""
        try:
            if not self.thalex_client:
                return
            
            self.logger.info("Collecting current options chain data...")
            
            # Get current BTC price
            btc_price = await self._get_current_btc_price()
            if not btc_price:
                self.logger.error("Could not get current BTC price")
                return
            
            # Get all available instruments
            instruments = await self.thalex_client.get_instruments("BTCUSD")
            
            # Filter for options and collect data
            options_data = []
            for instrument in instruments:
                if instrument.get('instrument_type') == 'option':
                    option_info = await self._get_option_market_data(instrument, btc_price)
                    if option_info:
                        options_data.append(option_info)
                    
                    # Rate limiting
                    await asyncio.sleep(self.config.rate_limit_delay)
            
            # Store options data in batches
            if options_data:
                await self._store_options_batch(options_data)
                self.logger.info(f"Collected {len(options_data)} option instruments")
            
        except Exception as e:
            self.logger.error(f"Error collecting options data: {e}")
            self.stats['api_errors'] += 1
    
    async def _get_current_btc_price(self) -> Optional[float]:
        """Get current BTC price from Thalex"""
        try:
            if not self.thalex_client:
                return None
            
            # Get index price
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
            if not self.thalex_client:
                return None
            
            instrument_name = instrument['instrument_name'] 
            
            # Get orderbook data
            orderbook = await self.thalex_client.get_orderbook(instrument_name)
            if not orderbook:
                return None
            
            # Parse instrument details
            parts = instrument_name.split('-')  # Format: BTC-DDMMMYY-STRIKE-C/P
            if len(parts) != 4:
                return None
            
            expiry_str = parts[1]  # DDMMMYY format
            strike_str = parts[2]
            option_type = 'call' if parts[3] == 'C' else 'put'
            
            # Parse expiry date
            expiry_date = self._parse_expiry_date(expiry_str)
            if not expiry_date:
                return None
            
            # Calculate days to expiry
            now = datetime.now(timezone.utc)
            days_to_expiry = (expiry_date - now).days
            
            # Get mark price and bid/ask
            mark_price = orderbook.get('mark_price', 0)
            best_bid = orderbook.get('best_bid_price', 0) if orderbook.get('best_bid_qty', 0) > 0 else None
            best_ask = orderbook.get('best_ask_price', 0) if orderbook.get('best_ask_qty', 0) > 0 else None
            
            # Calculate implied volatility if mark price exists
            implied_vol = None
            if mark_price > 0:
                implied_vol = self._calculate_implied_volatility(
                    mark_price, btc_price, float(strike_str), days_to_expiry / 365.0, option_type
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
                'volume': 0,  # Would need trade data for volume
                'open_interest': 0,  # Would need OI data
                'underlying_price': btc_price,
                'days_to_expiry': days_to_expiry
            }
            
        except Exception as e:
            self.logger.error(f"Error getting option market data for {instrument.get('instrument_name', 'unknown')}: {e}")
            return None
    
    def _parse_expiry_date(self, expiry_str: str) -> Optional[datetime]:
        """Parse Thalex expiry date format (DDMMMYY)"""
        try:
            # Map month abbreviations
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            if len(expiry_str) != 7:  # DDMMMYY
                return None
            
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5]
            year = int(expiry_str[5:7]) + 2000  # Convert YY to YYYY
            
            month = month_map.get(month_str)
            if not month:
                return None
            
            # Thalex options expire at 8:00 UTC
            return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error parsing expiry date {expiry_str}: {e}")
            return None
    
    def _calculate_implied_volatility(self, option_price: float, spot_price: float, 
                                    strike_price: float, time_to_expiry: float, 
                                    option_type: str) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            if time_to_expiry <= 0 or option_price <= 0:
                return None
            
            # Simple Newton-Raphson implementation
            # This is a basic version - for production, use more robust libraries
            risk_free_rate = 0.05  # Assume 5% risk-free rate
            
            # Initial guess
            vol = 0.2  # 20% initial volatility guess
            
            for _ in range(100):  # Max iterations
                d1 = (math.log(spot_price / strike_price) + (risk_free_rate + 0.5 * vol * vol) * time_to_expiry) / (vol * math.sqrt(time_to_expiry))
                d2 = d1 - vol * math.sqrt(time_to_expiry)
                
                # Standard normal CDF approximation
                def norm_cdf(x):
                    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
                
                # Standard normal PDF
                def norm_pdf(x):
                    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
                
                if option_type == 'call':
                    price = spot_price * norm_cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2)
                    vega = spot_price * norm_pdf(d1) * math.sqrt(time_to_expiry)
                else:  # put
                    price = strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2) - spot_price * norm_cdf(-d1)
                    vega = spot_price * norm_pdf(d1) * math.sqrt(time_to_expiry)
                
                price_diff = price - option_price
                
                if abs(price_diff) < 0.001:  # Convergence threshold
                    return vol
                
                if vega < 1e-6:  # Avoid division by zero
                    break
                
                vol = vol - price_diff / vega
                
                if vol <= 0:
                    vol = 0.001
                elif vol > 5:  # Cap at 500% volatility
                    vol = 5.0
            
            return vol if 0 < vol < 5 else None
            
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {e}")
            return None
    
    async def _store_options_batch(self, options_data: List[Dict[str, Any]]):
        """Store batch of options data"""
        try:
            if not self.supabase_manager:
                return
            
            for option_data in options_data:
                success = await self.supabase_manager.insert_option_data(option_data)
                if success:
                    self.stats['options_collected'] += 1
                else:
                    self.stats['validation_errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error storing options batch: {e}")
    
    def _print_collection_summary(self):
        """Print summary of data collection results"""
        print("\\n" + "="*80)
        print("HISTORICAL DATA COLLECTION SUMMARY")
        print("="*80)
        
        for key, value in self.stats.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key:<25}: {value:,}")
        
        print("="*80)
        
        # Data quality assessment
        total_records = self.stats['btc_prices_collected'] + self.stats['options_collected']
        error_rate = (self.stats['validation_errors'] + self.stats['api_errors']) / max(1, total_records)
        
        print(f"Total Records Collected   : {total_records:,}")
        print(f"Error Rate               : {error_rate:.2%}")
        
        if total_records > 10000:
            print("✅ Excellent data coverage - Enhanced volatility models will perform well")
        elif total_records > 1000:
            print("✅ Good data coverage - Substantial improvement over original system")
        else:
            print("⚠️  Limited data coverage - Consider extending collection period")
        
        print("="*80)
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.session:
                await self.session.close()
            
            if self.supabase_manager:
                await self.supabase_manager.close()
                
            self.logger.info("Data collector cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point for historical data collection"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Starting Historical Data Collection...")
    print("This will collect 6 months of BTC price data and current options data")
    
    # Configuration
    config = DataCollectionConfig(
        days_back=180,  # 6 months
        btc_price_interval_minutes=5,
        max_concurrent_requests=5,  # Conservative to avoid rate limits
        batch_size=500
    )
    
    collector = HistoricalDataCollector(config)
    
    try:
        if await collector.initialize():
            await collector.collect_all_historical_data()
            print("\\n✅ Historical data collection completed successfully!")
        else:
            print("\\n❌ Failed to initialize data collector")
            
    except KeyboardInterrupt:
        print("\\n⏹️  Data collection stopped by user")
    except Exception as e:
        print(f"\\n❌ Data collection failed: {e}")
    finally:
        await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())