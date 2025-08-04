#!/usr/bin/env python3
"""
Test enhanced pricing system with actual market scenario
"""

import logging
import time
from MADStraddle import ExpirySpecificMADAnalyzer

# Configure logging to see the enhanced output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_enhanced_pricing():
    print("=== TESTING ENHANCED PRICING SYSTEM ===")
    print()
    
    # Create test scenario matching your image data
    expiry_timestamp = int(time.time() + (1.6 * 24 * 3600))  # 1.6 days from now
    days_to_expiry = 1.6
    
    # Create analyzer
    analyzer = ExpirySpecificMADAnalyzer(expiry_timestamp, days_to_expiry)
    
    # Add some sample price points to simulate market data collection
    print("📊 Simulating price data collection...")
    base_price = 118000.0
    for i in range(30):  # Add 30 price points
        # Simulate some price volatility
        import random
        random.seed(42 + i)  # Deterministic for testing
        price_change = random.uniform(-0.005, 0.005)  # ±0.5% changes
        current_price = base_price * (1 + price_change)
        current_time = time.time() + (i * 60)  # 1 minute intervals
        
        analyzer.add_price_point(current_price, current_time)
    
    print("✅ Added 30 price data points")
    print()
    
    # Test parameters from your scenario  
    market_straddle_price = 2341.48
    spot_price = 118000.0
    strike_price = 118000.0  # ATM straddle
    
    print("🧮 ANALYZING STRADDLE EFFICIENCY...")
    print(f"Market Data: Spot=${spot_price:.0f}, Strike=${strike_price:.0f}, Straddle=${market_straddle_price:.2f}")
    print(f"Time to Expiry: {days_to_expiry:.1f} days")
    print()
    
    # Run the enhanced analysis
    analysis = analyzer.analyze_straddle_efficiency(
        market_straddle_price, spot_price, strike_price
    )
    
    if analysis:
        print()
        print("=== ENHANCED ANALYSIS RESULTS ===")
        print(f"MAD/SD Ratio: {analysis.mad_sd_ratio:.3f}")
        print(f"Distribution: {analysis.get_distribution_assessment()}")
        print()
        print("💰 PRICING COMPARISON:")
        print(f"Market Price:           ${analysis.actual_straddle:.2f}")
        print(f"BS Estimated Vol Price: ${analysis.bs_theoretical:.2f}")
        print(f"BS Implied Vol Price:   ${analysis.bs_theoretical_implied:.2f}")
        print(f"MAD Enhanced Price:     ${analysis.theoretical_straddle:.2f}")
        print() 
        print("📈 VOLATILITY ANALYSIS:")
        print(f"Estimated Volatility:   {analysis.estimated_vol:.1%}")
        print(f"Implied Volatility:     {analysis.bs_implied_vol:.1%}")
        print(f"Vol Ratio (Impl/Est):   {analysis.bs_implied_vol/analysis.estimated_vol:.1f}x")
        print()
        print("⚖️  EFFICIENCY RATIOS:")
        print(f"Market/BS Estimated:    {analysis.bs_efficiency_ratio:.2f}")
        print(f"Market/BS Implied:      {analysis.bs_implied_efficiency_ratio:.2f}")
        print(f"Market/MAD Enhanced:    {analysis.efficiency_ratio:.2f}")
        print()
        print("🎯 TRADING ASSESSMENT:")
        print(f"Straddle Assessment: {analysis.get_straddle_assessment()}")
        
        # Check if our fixes worked
        print()
        print("=== FIX VALIDATION ===")
        
        # Test 1: Reasonable pricing ratios
        if 0.5 <= analysis.bs_efficiency_ratio <= 2.0:
            print("✅ Strike price fix working - BS ratios now reasonable")
        else:
            print(f"❌ Still extreme BS ratio: {analysis.bs_efficiency_ratio:.1f}x")
        
        # Test 2: Implied volatility makes sense
        if 0.8 <= analysis.bs_implied_efficiency_ratio <= 1.2:
            print("✅ Implied volatility calculation working - near perfect match")
        else:
            print(f"⚠️  Implied vol ratio: {analysis.bs_implied_efficiency_ratio:.2f} (should be ~1.0)")
            
        # Test 3: Volatility estimation improved
        if analysis.estimated_vol >= 0.20:  # At least 20% for crypto
            print(f"✅ Volatility floor working - estimated vol: {analysis.estimated_vol:.1%}")
        else:
            print(f"❌ Vol still too low: {analysis.estimated_vol:.1%}")
        
        # Test 4: Near-expiry adjustments
        if days_to_expiry < 7 and analysis.estimated_vol > 0.30:
            print("✅ Near-expiry volatility adjustments applied")
        
    else:
        print("❌ Analysis failed - insufficient data")

if __name__ == "__main__":
    test_enhanced_pricing()