#!/usr/bin/env python3
"""
Supabase Configuration and Database Schema Management
=====================================================

This module handles the connection to Supabase and manages the database schema
for storing comprehensive options trading data including:
- Historical BTC prices with high frequency
- Options chains and implied volatilities
- Volatility surface data
- Regime detection history
- Risk metrics and analytics

Database Schema:
---------------
1. btc_prices: High-frequency BTC price data
2. option_chains: Complete options market data
3. volatility_surfaces: Implied volatility surface points
4. regime_history: Volatility regime detection results
5. risk_metrics: Calculated risk and analytics data
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from supabase import create_client, Client
import asyncpg


@dataclass
class SupabaseConfig:
    """Supabase configuration parameters"""
    url: str
    key: str
    connection_pool_size: int = 10
    connection_timeout: int = 30
    query_timeout: int = 60


class SupabaseManager:
    """
    Manages Supabase database connections and operations for options trading data
    """
    
    def __init__(self, config: SupabaseConfig):
        self.config = config
        self.client: Optional[Client] = None
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> bool:
        """Initialize Supabase connection and create tables if needed"""
        try:
            # Initialize Supabase client
            self.client = create_client(self.config.url, self.config.key)
            
            # Create asyncpg connection pool for high-performance operations
            self.pool = await asyncpg.create_pool(
                dsn=self._get_postgres_dsn(),
                min_size=2,
                max_size=self.config.connection_pool_size,
                command_timeout=self.config.query_timeout
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            self.logger.info("Supabase connection initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase: {e}")
            return False
    
    def _get_postgres_dsn(self) -> str:
        """Convert Supabase URL to PostgreSQL DSN"""
        # Extract connection details from Supabase URL
        # Format: https://PROJECT_ID.supabase.co
        project_id = self.config.url.replace('https://', '').split('.')[0]
        
        # Supabase PostgreSQL connection details
        return f"postgresql://postgres:{self.config.key}@db.{project_id}.supabase.co:5432/postgres"
    
    async def _create_tables(self):
        """Create database tables for options trading data"""
        
        # Table creation SQL statements
        tables = {
            'btc_prices': '''
                CREATE TABLE IF NOT EXISTS btc_prices (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    price DECIMAL(15,2) NOT NULL,
                    volume DECIMAL(20,8),
                    source VARCHAR(50) NOT NULL DEFAULT 'thalex',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_btc_prices_timestamp 
                ON btc_prices(timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_btc_prices_source_timestamp 
                ON btc_prices(source, timestamp DESC);
            ''',
            
            'option_chains': '''
                CREATE TABLE IF NOT EXISTS option_chains (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    expiry_date DATE NOT NULL,
                    expiry_timestamp BIGINT NOT NULL,
                    strike_price DECIMAL(15,2) NOT NULL,
                    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
                    mark_price DECIMAL(15,8) NOT NULL,
                    bid_price DECIMAL(15,8),
                    ask_price DECIMAL(15,8),
                    implied_volatility DECIMAL(8,6),
                    volume DECIMAL(20,8) DEFAULT 0,
                    open_interest DECIMAL(20,8) DEFAULT 0,
                    underlying_price DECIMAL(15,2) NOT NULL,
                    days_to_expiry INTEGER NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_option_chains_expiry_strike 
                ON option_chains(expiry_date, strike_price, option_type);
                
                CREATE INDEX IF NOT EXISTS idx_option_chains_timestamp 
                ON option_chains(timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_option_chains_expiry_timestamp 
                ON option_chains(expiry_timestamp);
            ''',
            
            'volatility_surfaces': '''
                CREATE TABLE IF NOT EXISTS volatility_surfaces (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    expiry_date DATE NOT NULL,
                    strike_price DECIMAL(15,2) NOT NULL,
                    implied_volatility DECIMAL(8,6) NOT NULL,
                    confidence_level DECIMAL(4,3) DEFAULT 0.8,
                    underlying_price DECIMAL(15,2) NOT NULL,
                    days_to_expiry INTEGER NOT NULL,
                    moneyness DECIMAL(8,6) NOT NULL,
                    data_source VARCHAR(50) DEFAULT 'calculated',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_vol_surface_expiry_strike 
                ON volatility_surfaces(expiry_date, strike_price);
                
                CREATE INDEX IF NOT EXISTS idx_vol_surface_timestamp 
                ON volatility_surfaces(timestamp DESC);
            ''',
            
            'regime_history': '''
                CREATE TABLE IF NOT EXISTS regime_history (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    regime VARCHAR(20) NOT NULL,
                    confidence_level DECIMAL(4,3) NOT NULL,
                    current_vol DECIMAL(8,6) NOT NULL,
                    short_term_vol DECIMAL(8,6) NOT NULL,
                    medium_term_vol DECIMAL(8,6) NOT NULL,
                    long_term_vol DECIMAL(8,6) NOT NULL,
                    vol_momentum DECIMAL(8,6) NOT NULL,
                    vol_mean_reversion DECIMAL(8,6) NOT NULL,
                    vol_percentile DECIMAL(5,2) NOT NULL,
                    vol_price_correlation DECIMAL(6,4) NOT NULL,
                    time_since_regime_change DECIMAL(10,2) NOT NULL,
                    underlying_price DECIMAL(15,2) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_regime_history_timestamp 
                ON regime_history(timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_regime_history_regime 
                ON regime_history(regime, timestamp DESC);
            ''',
            
            'risk_metrics': '''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    expiry_date DATE,
                    strike_price DECIMAL(15,2),
                    metric_value DECIMAL(15,8) NOT NULL,
                    metadata JSONB,
                    underlying_price DECIMAL(15,2) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_risk_metrics_type_timestamp 
                ON risk_metrics(metric_type, timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_risk_metrics_expiry 
                ON risk_metrics(expiry_date, metric_type);
            '''
        }
        
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            for table_name, sql in tables.items():
                try:
                    await conn.execute(sql)
                    self.logger.info(f"Successfully created/verified table: {table_name}")
                except Exception as e:
                    self.logger.error(f"Error creating table {table_name}: {e}")
                    raise
    
    async def insert_btc_price(self, timestamp: datetime, price: float, 
                              volume: Optional[float] = None, source: str = 'thalex') -> bool:
        """Insert BTC price data"""
        try:
            if not self.pool:
                return False
                
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO btc_prices (timestamp, price, volume, source)
                    VALUES ($1, $2, $3, $4)
                ''', timestamp, price, volume, source)
            return True
        except Exception as e:
            self.logger.error(f"Error inserting BTC price: {e}")
            return False
    
    async def insert_option_data(self, option_data: Dict[str, Any]) -> bool:
        """Insert options chain data"""
        try:
            if not self.pool:
                return False
                
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO option_chains (
                        timestamp, expiry_date, expiry_timestamp, strike_price, 
                        option_type, mark_price, bid_price, ask_price, 
                        implied_volatility, volume, open_interest, 
                        underlying_price, days_to_expiry
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ''', 
                option_data['timestamp'],
                option_data['expiry_date'],
                option_data['expiry_timestamp'],
                option_data['strike_price'],
                option_data['option_type'],
                option_data['mark_price'],
                option_data.get('bid_price'),
                option_data.get('ask_price'),
                option_data.get('implied_volatility'),
                option_data.get('volume', 0),
                option_data.get('open_interest', 0),
                option_data['underlying_price'],
                option_data['days_to_expiry']
                )
            return True
        except Exception as e:
            self.logger.error(f"Error inserting option data: {e}")
            return False
    
    async def get_historical_prices(self, hours_back: int = 24, 
                                   source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical BTC prices"""
        try:
            if not self.pool:
                return []
            
            query = '''
                SELECT timestamp, price, volume, source
                FROM btc_prices
                WHERE timestamp >= NOW() - INTERVAL '%d hours'
            ''' % hours_back
            
            if source:
                query += f" AND source = '{source}'"
            
            query += " ORDER BY timestamp ASC"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error fetching historical prices: {e}")
            return []
    
    async def get_recent_options_data(self, hours_back: int = 1) -> List[Dict[str, Any]]:
        """Get recent options chain data"""
        try:
            if not self.pool:
                return []
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM option_chains
                    WHERE timestamp >= NOW() - INTERVAL '%d hours'
                    ORDER BY timestamp DESC, expiry_date ASC, strike_price ASC
                ''' % hours_back)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error fetching options data: {e}")
            return []
    
    async def insert_volatility_surface_point(self, surface_data: Dict[str, Any]) -> bool:
        """Insert volatility surface data point"""
        try:
            if not self.pool:
                return False
                
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO volatility_surfaces (
                        timestamp, expiry_date, strike_price, implied_volatility,
                        confidence_level, underlying_price, days_to_expiry,
                        moneyness, data_source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''',
                surface_data['timestamp'],
                surface_data['expiry_date'],
                surface_data['strike_price'],
                surface_data['implied_volatility'],
                surface_data.get('confidence_level', 0.8),
                surface_data['underlying_price'],
                surface_data['days_to_expiry'],
                surface_data['moneyness'],
                surface_data.get('data_source', 'calculated')
                )
            return True
        except Exception as e:
            self.logger.error(f"Error inserting volatility surface data: {e}")
            return False
    
    async def insert_regime_data(self, regime_data: Dict[str, Any]) -> bool:
        """Insert volatility regime data"""
        try:
            if not self.pool:
                return False
                
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO regime_history (
                        timestamp, regime, confidence_level, current_vol,
                        short_term_vol, medium_term_vol, long_term_vol,
                        vol_momentum, vol_mean_reversion, vol_percentile,
                        vol_price_correlation, time_since_regime_change,
                        underlying_price
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ''',
                regime_data['timestamp'],
                regime_data['regime'],
                regime_data['confidence_level'],
                regime_data['current_vol'],
                regime_data['short_term_vol'],
                regime_data['medium_term_vol'],
                regime_data['long_term_vol'],
                regime_data['vol_momentum'],
                regime_data['vol_mean_reversion'],
                regime_data['vol_percentile'],
                regime_data['vol_price_correlation'],
                regime_data['time_since_regime_change'],
                regime_data['underlying_price']
                )
            return True
        except Exception as e:
            self.logger.error(f"Error inserting regime data: {e}")
            return False
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage database size"""
        try:
            if not self.pool:
                return
            
            cutoff_date = f"NOW() - INTERVAL '{days_to_keep} days'"
            
            tables_to_clean = [
                'btc_prices',
                'option_chains', 
                'volatility_surfaces',
                'regime_history',
                'risk_metrics'
            ]
            
            async with self.pool.acquire() as conn:
                for table in tables_to_clean:
                    result = await conn.execute(f'''
                        DELETE FROM {table} 
                        WHERE created_at < {cutoff_date}
                    ''')
                    self.logger.info(f"Cleaned {result} old records from {table}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning old data: {e}")
    
    async def close(self):
        """Close database connections"""
        try:
            if self.pool:
                await self.pool.close()
                self.logger.info("Database pool closed")
        except Exception as e:
            self.logger.error(f"Error closing database pool: {e}")


def create_supabase_config() -> SupabaseConfig:
    """Create Supabase configuration from environment variables"""
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for server-side operations
    
    if not url or not key:
        raise ValueError(
            "Supabase configuration missing. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
        )
    
    return SupabaseConfig(
        url=url,
        key=key,
        connection_pool_size=int(os.getenv('SUPABASE_POOL_SIZE', '10')),
        connection_timeout=int(os.getenv('SUPABASE_TIMEOUT', '30')),
        query_timeout=int(os.getenv('SUPABASE_QUERY_TIMEOUT', '60'))
    )


# Example usage and testing
async def test_supabase_connection():
    """Test Supabase connection and basic operations"""
    try:
        config = create_supabase_config()
        manager = SupabaseManager(config)
        
        if await manager.initialize():
            print("✅ Supabase connection successful")
            
            # Test inserting a BTC price
            test_time = datetime.now(timezone.utc)
            success = await manager.insert_btc_price(test_time, 45000.0, source='test')
            
            if success:
                print("✅ Test data insertion successful")
                
                # Test retrieving data
                prices = await manager.get_historical_prices(hours_back=1, source='test')
                print(f"✅ Retrieved {len(prices)} price records")
            else:
                print("❌ Test data insertion failed")
        else:
            print("❌ Supabase connection failed")
            
        await manager.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    # Run connection test
    asyncio.run(test_supabase_connection())