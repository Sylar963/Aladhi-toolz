#!/usr/bin/env python3
"""
Enhanced Options Analysis System Setup & Test
=============================================

This script sets up and tests the complete enhanced system that addresses
the data limitations in the original enhanced_mad_straddle.py.

System Components:
- Supabase database with comprehensive schema
- Historical data collection (6+ months)
- Real-time data management with caching
- Enhanced volatility analysis with substantial data
- Background data collection service

Usage:
1. Set environment variables: SUPABASE_URL and SUPABASE_SERVICE_KEY
2. Run: python setup_enhanced_system.py
"""

import asyncio
import os
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from supabase_config import SupabaseManager, create_supabase_config
from supabase_data_manager import SupabaseDataManager, DataAnalysisUtils
from historical_data_collector import HistoricalDataCollector, DataCollectionConfig
from background_data_service import BackgroundDataService


class SystemSetupValidator:
    """
    Complete system setup and validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results: Dict[str, Any] = {}
        
    async def run_complete_setup_and_test(self):
        """Run complete system setup and validation"""
        print("="*80)
        print("ENHANCED OPTIONS ANALYSIS SYSTEM SETUP & TEST")
        print("="*80)
        print()
        
        # Test sequence
        tests = [
            ("Environment Check", self._test_environment),
            ("Database Connection", self._test_database_connection),
            ("Database Schema", self._test_database_schema),
            ("Historical Data Collection", self._test_historical_data_collection),
            ("Data Manager", self._test_data_manager),
            ("Background Service", self._test_background_service),
            ("System Integration", self._test_system_integration)
        ]
        
        for test_name, test_func in tests:
            print(f"Running {test_name}...")
            try:
                result = await test_func()
                self.test_results[test_name] = {'status': 'PASS', 'details': result}
                print(f"✅ {test_name}: PASSED")
                if result:
                    print(f"   {result}")
            except Exception as e:
                self.test_results[test_name] = {'status': 'FAIL', 'error': str(e)}
                print(f"❌ {test_name}: FAILED - {e}")
            print()
        
        # Final report
        self._print_final_report()
    
    async def _test_environment(self) -> str:
        """Test environment configuration"""
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise Exception(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Check if keys.py exists
        try:
            import keys
            if not hasattr(keys, 'key_ids') or not hasattr(keys, 'private_keys'):
                raise Exception("keys.py missing required attributes: key_ids, private_keys")
        except ImportError:
            raise Exception("keys.py not found - required for Thalex API access")
        
        return "All required environment variables and API keys configured"
    
    async def _test_database_connection(self) -> str:
        """Test Supabase database connection"""
        try:
            config = create_supabase_config()
            manager = SupabaseManager(config)
            
            success = await manager.initialize()
            if not success:
                raise Exception("Failed to initialize Supabase connection")
            
            # Test a simple operation
            await manager.insert_btc_price(datetime.now(timezone.utc), 50000.0, source='test')
            
            await manager.close()
            return "Database connection and basic operations working"
            
        except Exception as e:
            raise Exception(f"Database connection failed: {e}")
    
    async def _test_database_schema(self) -> str:
        """Test database schema creation and structure"""
        try:
            config = create_supabase_config()
            manager = SupabaseManager(config)
            
            await manager.initialize()
            
            # Test all table operations
            test_time = datetime.now(timezone.utc)
            
            # Test btc_prices table
            success = await manager.insert_btc_price(test_time, 50000.0, 100.0, 'schema_test')
            if not success:
                raise Exception("btc_prices table operation failed")
            
            # Test option_chains table
            option_data = {
                'timestamp': test_time,
                'expiry_date': test_time.date(),
                'expiry_timestamp': int(test_time.timestamp()),
                'strike_price': 50000.0,
                'option_type': 'call',
                'mark_price': 1000.0,
                'implied_volatility': 0.5,
                'underlying_price': 50000.0,
                'days_to_expiry': 30
            }
            success = await manager.insert_option_data(option_data)
            if not success:
                raise Exception("option_chains table operation failed")
            
            # Test volatility_surfaces table
            surface_data = {
                'timestamp': test_time,
                'expiry_date': test_time.date(),
                'strike_price': 50000.0,
                'implied_volatility': 0.5,
                'underlying_price': 50000.0,
                'days_to_expiry': 30,
                'moneyness': 1.0
            }
            success = await manager.insert_volatility_surface_point(surface_data)
            if not success:
                raise Exception("volatility_surfaces table operation failed")
            
            await manager.close()
            return "All database tables created and operational"
            
        except Exception as e:
            raise Exception(f"Database schema test failed: {e}")
    
    async def _test_historical_data_collection(self) -> str:
        """Test historical data collection capabilities"""
        try:
            # Test with a small sample to avoid long execution times
            config = DataCollectionConfig(
                days_back=7,  # Just 1 week for testing
                btc_price_interval_minutes=60,  # Hourly data
                max_concurrent_requests=3,
                batch_size=100
            )
            
            collector = HistoricalDataCollector(config)
            
            if not await collector.initialize():
                raise Exception("Historical data collector initialization failed")
            
            # Test external API access (just CoinGecko daily data)
            await collector._fetch_coingecko_daily_data(
                datetime.now(timezone.utc) - timedelta(days=7),
                datetime.now(timezone.utc)
            )
            
            await collector.cleanup()
            
            return f"Historical data collection tested - {collector.stats['btc_prices_collected']} prices collected"
            
        except Exception as e:
            raise Exception(f"Historical data collection test failed: {e}")
    
    async def _test_data_manager(self) -> str:
        """Test the enhanced data manager"""
        try:
            manager = SupabaseDataManager(cache_size=1000)
            
            if not await manager.initialize():
                raise Exception("Data manager initialization failed")
            
            # Test price history retrieval
            prices = await manager.get_price_history_optimized(hours_back=24)
            
            # Test volatility analysis data
            vol_data = await manager.get_returns_for_volatility_analysis(hours_back=48)
            
            # Test data validation
            if len(prices) > 0:
                is_valid, msg = DataAnalysisUtils.validate_price_data([p.price for p in prices])
                if not is_valid:
                    self.logger.warning(f"Price data validation warning: {msg}")
            
            if len(vol_data.returns) > 0:
                is_valid, msg = DataAnalysisUtils.validate_returns_data(vol_data.returns)
                if not is_valid:
                    self.logger.warning(f"Returns data validation warning: {msg}")
            
            # Test cache statistics
            stats = manager.get_cache_statistics()
            
            await manager.cleanup()
            
            return f"Data manager operational - {len(prices)} prices, {len(vol_data.returns)} returns, cache hit rate: {stats['cache_hit_rate']:.1%}"
            
        except Exception as e:
            raise Exception(f"Data manager test failed: {e}")
    
    async def _test_background_service(self) -> str:
        """Test background data collection service"""
        try:
            service = BackgroundDataService(collection_interval=5)  # 5 second test interval
            
            if not await service.initialize():
                raise Exception("Background service initialization failed")
            
            # Test one collection cycle
            await service._collection_cycle()
            
            await service._cleanup()
            
            stats = service.stats
            return f"Background service tested - {stats['prices_collected']} prices, {stats['options_collected']} options collected"
            
        except Exception as e:
            raise Exception(f"Background service test failed: {e}")
    
    async def _test_system_integration(self) -> str:
        """Test the complete integrated system"""
        try:
            # This tests the key integration points without running the full UI
            from enhanced_mad_straddle import EnhancedMADStraddleAnalyzer
            
            analyzer = EnhancedMADStraddleAnalyzer()
            
            # Test data manager initialization
            if not await analyzer.data_manager.initialize():
                raise Exception("Enhanced analyzer data manager initialization failed")
            
            # Test comprehensive data loading
            await analyzer._initialize_comprehensive_data()
            
            if not analyzer.volatility_data:
                raise Exception("Failed to load comprehensive volatility data")
            
            # Test data quality
            returns_count = len(analyzer.volatility_data.returns)
            if returns_count < 100:
                raise Exception(f"Insufficient volatility data: {returns_count} returns")
            
            # Clean up
            await analyzer.data_manager.cleanup()
            
            return f"System integration successful - loaded {returns_count} returns for analysis"
            
        except Exception as e:
            raise Exception(f"System integration test failed: {e}")
    
    def _print_final_report(self):
        """Print final test report"""
        print("="*80)
        print("FINAL TEST REPORT")
        print("="*80)
        
        passed = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print()
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! 🎉")
            print()
            print("Your enhanced options analysis system is ready!")
            print()
            print("Next steps:")
            print("1. Run historical_data_collector.py to populate your database with historical data")
            print("2. Start background_data_service.py for continuous data collection")
            print("3. Run enhanced_mad_straddle.py to use the enhanced analyzer")
            print()
            print("Key improvements over original system:")
            print("• 100x more historical data (months vs minutes)")
            print("• Persistent storage across restarts")
            print("• Advanced volatility modeling with comprehensive data")
            print("• Real-time background data collection")
            print("• Intelligent caching and performance optimization")
            
        else:
            print("❌ SOME TESTS FAILED")
            print()
            print("Failed tests:")
            for test_name, result in self.test_results.items():
                if result['status'] == 'FAIL':
                    print(f"• {test_name}: {result['error']}")
            
            print("\nPlease fix the failing tests before proceeding.")
        
        print("="*80)


async def main():
    """Run the complete system setup and test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    validator = SystemSetupValidator()
    await validator.run_complete_setup_and_test()


if __name__ == "__main__":
    asyncio.run(main())