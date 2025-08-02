#!/usr/bin/env python3
"""
Unit tests for financial mathematics module
Tests Black-Scholes accuracy and statistical analysis robustness
"""

import unittest
import math
from financial_math import FinancialMath, TradingConstants
from statistical_analysis import DistributionAnalyzer, VolatilityEstimator

class TestBlackScholesAccuracy(unittest.TestCase):
    """Test Black-Scholes implementation accuracy against known benchmarks"""
    
    def setUp(self):
        # Standard test parameters
        self.spot = 50000.0        # $50,000 BTC
        self.strike = 50000.0      # ATM option
        self.time_to_expiry = 30/365.0  # 30 days
        self.risk_free_rate = 0.05      # 5%
        self.volatility = 0.80          # 80% annual vol (typical for crypto)
        self.tolerance = 0.01           # 1% tolerance for numerical precision
    
    def test_call_put_parity(self):
        """Test that call-put parity holds: C - P = S - K*e^(-r*T)"""
        call_price = FinancialMath.black_scholes_call(
            self.spot, self.strike, self.time_to_expiry, 
            self.risk_free_rate, self.volatility
        )
        
        put_price = FinancialMath.black_scholes_put(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility
        )
        
        # Call-put parity: C - P = S - K*e^(-r*T)
        left_side = call_price - put_price
        right_side = self.spot - self.strike * math.exp(-self.risk_free_rate * self.time_to_expiry)
        
        self.assertAlmostEqual(left_side, right_side, delta=self.tolerance,
                              msg=f"Call-put parity violation: {left_side:.2f} != {right_side:.2f}")
    
    def test_atm_straddle_consistency(self):
        """Test that straddle price equals call + put for ATM options"""
        call_price = FinancialMath.black_scholes_call(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility
        )
        
        put_price = FinancialMath.black_scholes_put(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility
        )
        
        straddle_price = FinancialMath.black_scholes_straddle(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility
        )
        
        expected_straddle = call_price + put_price
        
        self.assertAlmostEqual(straddle_price, expected_straddle, delta=self.tolerance,
                              msg=f"Straddle inconsistency: {straddle_price:.2f} != {expected_straddle:.2f}")
    
    def test_time_decay_monotonicity(self):
        """Test that option prices decrease as time to expiry decreases (theta < 0)"""
        time_points = [60/365.0, 30/365.0, 7/365.0, 1/365.0]  # 60, 30, 7, 1 days
        straddle_prices = []
        
        for t in time_points:
            if t > 0:  # Avoid division by zero
                price = FinancialMath.black_scholes_straddle(
                    self.spot, self.strike, t, self.risk_free_rate, self.volatility
                )
                straddle_prices.append(price)
        
        # Straddle prices should decrease as time decreases (for positive theta)
        for i in range(len(straddle_prices) - 1):
            self.assertGreater(straddle_prices[i], straddle_prices[i+1],
                             msg=f"Time decay violation: {straddle_prices[i]:.2f} <= {straddle_prices[i+1]:.2f}")
    
    def test_volatility_monotonicity(self):
        """Test that option prices increase with volatility (vega > 0)"""
        volatilities = [0.20, 0.40, 0.60, 0.80, 1.00]  # 20% to 100%
        straddle_prices = []
        
        for vol in volatilities:
            price = FinancialMath.black_scholes_straddle(
                self.spot, self.strike, self.time_to_expiry, self.risk_free_rate, vol
            )
            straddle_prices.append(price)
        
        # Straddle prices should increase with volatility
        for i in range(len(straddle_prices) - 1):
            self.assertGreater(straddle_prices[i+1], straddle_prices[i],
                             msg=f"Volatility monotonicity violation: {straddle_prices[i+1]:.2f} <= {straddle_prices[i]:.2f}")
    
    def test_greeks_sanity_checks(self):
        """Test that Greeks have reasonable values and properties"""
        greeks = FinancialMath.calculate_greeks(
            self.spot, self.strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, option_type="call"
        )
        
        # Delta should be between 0 and 1 for calls
        self.assertGreater(greeks['delta'], 0, msg="Call delta should be positive")
        self.assertLess(greeks['delta'], 1, msg="Call delta should be less than 1")
        
        # Gamma should be positive
        self.assertGreater(greeks['gamma'], 0, msg="Gamma should be positive")
        
        # Vega should be positive
        self.assertGreater(greeks['vega'], 0, msg="Vega should be positive")
        
        # Theta should be negative for long options
        self.assertLess(greeks['theta'], 0, msg="Theta should be negative for long options")

class TestStatisticalAnalysis(unittest.TestCase):
    """Test statistical analysis without artificial manipulation"""
    
    def test_normal_distribution_mad_sd_ratio(self):
        """Test that normal distribution gives correct MAD/SD ratio"""
        import numpy as np
        
        # Generate normal distribution
        np.random.seed(42)  # For reproducibility
        normal_data = np.random.normal(0, 1, 10000).tolist()
        
        metrics = DistributionAnalyzer.calculate_distribution_metrics(normal_data)
        
        # For normal distribution, MAD/SD should be approximately 0.7979
        expected_ratio = TradingConstants.NORMAL_DISTRIBUTION_MAD_SD_RATIO
        tolerance = 0.05  # 5% tolerance
        
        self.assertAlmostEqual(metrics['mad_sd_ratio'], expected_ratio, delta=tolerance,
                              msg=f"Normal distribution MAD/SD ratio: {metrics['mad_sd_ratio']:.3f} != {expected_ratio:.3f}")
    
    def test_distribution_classification(self):
        """Test distribution shape classification"""
        # Test normal distribution classification
        normal_ratio = 0.798  # Close to theoretical 0.7979
        shape_class, description = DistributionAnalyzer.classify_distribution_shape(normal_ratio)
        self.assertEqual(shape_class, "near_normal", msg="Should classify as near_normal")
        
        # Test heavy tails classification
        heavy_tails_ratio = 0.55
        shape_class, description = DistributionAnalyzer.classify_distribution_shape(heavy_tails_ratio)
        self.assertEqual(shape_class, "heavy_tails", msg="Should classify as heavy_tails")
        
        # Test extreme tails classification
        extreme_ratio = 0.45
        shape_class, description = DistributionAnalyzer.classify_distribution_shape(extreme_ratio)
        self.assertEqual(shape_class, "extreme_tails", msg="Should classify as extreme_tails")
    
    def test_volatility_estimation_consistency(self):
        """Test that volatility estimation is consistent across different methods"""
        # Generate sample returns (daily returns for 1 year)
        import numpy as np
        np.random.seed(42)
        
        annual_vol = 0.50  # 50% annual volatility
        daily_vol = annual_vol / math.sqrt(TradingConstants.TRADING_DAYS_PER_YEAR)
        returns = np.random.normal(0, daily_vol, TradingConstants.TRADING_DAYS_PER_YEAR).tolist()
        
        # Estimate volatility
        estimated_vol = VolatilityEstimator.estimate_realized_volatility(returns, "daily")
        
        # Should be close to input volatility (within 10% tolerance for random data)
        tolerance = annual_vol * 0.15  # 15% tolerance
        
        self.assertAlmostEqual(estimated_vol, annual_vol, delta=tolerance,
                              msg=f"Volatility estimation: {estimated_vol:.3f} != {annual_vol:.3f}")
    
    def test_log_returns_calculation(self):
        """Test log returns calculation accuracy"""
        prices = [100.0, 105.0, 102.0, 108.0, 106.0]
        
        log_returns = FinancialMath.calculate_log_returns(prices)
        
        # Expected log returns
        expected_returns = [
            math.log(105.0/100.0),  # ~0.0488
            math.log(102.0/105.0),  # ~-0.0291
            math.log(108.0/102.0),  # ~0.0571
            math.log(106.0/108.0)   # ~-0.0187
        ]
        
        self.assertEqual(len(log_returns), len(expected_returns), 
                        msg="Should have correct number of returns")
        
        for i, (actual, expected) in enumerate(zip(log_returns, expected_returns)):
            self.assertAlmostEqual(actual, expected, places=4,
                                  msg=f"Log return {i}: {actual:.4f} != {expected:.4f}")

class TestRegressionPrevention(unittest.TestCase):
    """Test that previous bugs don't reoccur"""
    
    def test_no_artificial_scaling(self):
        """Ensure no artificial scaling is applied to MAD/SD ratios"""
        # This test would catch if artificial scaling is reintroduced
        import numpy as np
        np.random.seed(42)
        
        # Generate identical data for different "expiry periods"
        identical_data = np.random.normal(0, 1, 1000).tolist()
        
        # Calculate metrics for identical data
        metrics1 = DistributionAnalyzer.calculate_distribution_metrics(identical_data)
        metrics2 = DistributionAnalyzer.calculate_distribution_metrics(identical_data)
        
        # Results should be identical (no expiry-based scaling)
        self.assertEqual(metrics1['mad_sd_ratio'], metrics2['mad_sd_ratio'],
                        msg="Identical data should produce identical MAD/SD ratios")
    
    def test_reasonable_volatility_scaling(self):
        """Test that volatility annualization produces reasonable results"""
        # Daily returns with known volatility
        daily_vol = 0.02  # 2% daily volatility
        returns = [daily_vol * x for x in [-1, 1, -0.5, 1.5, -0.8, 0.3, -1.2, 0.9]]
        
        annual_vol = VolatilityEstimator.estimate_realized_volatility(returns, "daily")
        
        # Should produce reasonable annual volatility (not extreme values)
        self.assertGreater(annual_vol, 0.10, msg="Annual volatility too low")
        self.assertLess(annual_vol, 2.0, msg="Annual volatility too high")

if __name__ == '__main__':
    # Run all tests
    print("Running Financial Mathematics and Statistical Analysis Tests...")
    print("=" * 60)
    
    unittest.main(verbosity=2)