#!/usr/bin/env python3
"""
Test script to validate pricing fixes
"""

from financial_math import FinancialMath, TradingConstants

# Test parameters from your image
spot_price = 118000.0  # Assuming BTC is close to strike based on option prices
strike_price = 118000.0  # From your image
actual_straddle_price = 2341.48
days_to_expiry = 1.6
time_to_expiry = days_to_expiry / 365.0

print(f"=== PRICING FIX VALIDATION ===")
print(f"Spot: ${spot_price:.0f}")
print(f"Strike: ${strike_price:.0f}")
print(f"Time to Expiry: {days_to_expiry:.1f} days ({time_to_expiry:.4f} years)")
print(f"Market Straddle Price: ${actual_straddle_price:.2f}")
print()

# Test different volatility scenarios
test_volatilities = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # 50% to 500%

print("Testing different volatility scenarios:")
print("Vol%     BS_Price    Ratio    Comments")
print("-" * 50)

risk_free_rate = TradingConstants.DEFAULT_RISK_FREE_RATE

for vol in test_volatilities:
    bs_price = FinancialMath.black_scholes_straddle(
        spot=spot_price,
        strike=strike_price,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=vol
    )
    
    ratio = actual_straddle_price / bs_price if bs_price > 0 else float('inf')
    
    if 0.8 <= ratio <= 1.2:
        comment = "✅ REASONABLE"
    elif 0.5 <= ratio <= 2.0:
        comment = "⚠️  ACCEPTABLE"
    else:
        comment = "❌ TOO EXTREME"
    
    print(f"{vol:.0%}      ${bs_price:>8.2f}    {ratio:>5.1f}x   {comment}")

print()

# Find implied volatility that would match market price
print("=== IMPLIED VOLATILITY CALCULATION ===")

# Use Newton-Raphson to find implied vol
market_price = actual_straddle_price
tolerance = 1.0  # $1 tolerance
max_iterations = 100

vol = 2.0  # Start with 200% vol guess
for i in range(max_iterations):
    bs_price = FinancialMath.black_scholes_straddle(
        spot_price, strike_price, time_to_expiry, risk_free_rate, vol
    )
    
    # Calculate vega for derivative
    call_greeks = FinancialMath.calculate_greeks(
        spot_price, strike_price, time_to_expiry, risk_free_rate, vol, option_type="call"
    )
    put_greeks = FinancialMath.calculate_greeks(
        spot_price, strike_price, time_to_expiry, risk_free_rate, vol, option_type="put"
    )
    
    straddle_vega = (call_greeks['vega'] + put_greeks['vega']) * 100
    
    if abs(straddle_vega) < 1e-10:
        break
        
    price_diff = bs_price - market_price
    vol_new = vol - price_diff / straddle_vega
    vol_new = max(0.1, min(vol_new, 10.0))  # Keep reasonable bounds
    
    if abs(vol_new - vol) < 0.001 or abs(price_diff) < tolerance:
        break
        
    vol = vol_new

implied_vol = vol
bs_at_implied_vol = FinancialMath.black_scholes_straddle(
    spot_price, strike_price, time_to_expiry, risk_free_rate, implied_vol
)

print(f"Implied Volatility: {implied_vol:.1%}")
print(f"BS Price at Implied Vol: ${bs_at_implied_vol:.2f}")
print(f"Market Price: ${market_price:.2f}")
print(f"Difference: ${abs(bs_at_implied_vol - market_price):.2f}")

# Assessment
if implied_vol > 3.0:
    assessment = "EXTREME volatility - likely near-expiry premium"
elif implied_vol > 1.5:
    assessment = "HIGH volatility - significant risk premium" 
elif implied_vol > 0.8:
    assessment = "ELEVATED volatility - moderate risk premium"
else:
    assessment = "NORMAL volatility range"

print(f"Assessment: {assessment}")
print()

print("=== CONCLUSION ===")
if implied_vol > 2.0:
    print(f"✅ The {implied_vol:.0%} implied volatility explains the high straddle price")
    print("   This is typical for crypto options with <2 days to expiry")
    print("   Our pricing model should use near-expiry volatility adjustments")
else:
    print(f"❌ Even at {implied_vol:.0%} implied vol, significant mispricing remains")
    print("   Need to investigate other factors (bid-ask spreads, liquidity, etc.)")