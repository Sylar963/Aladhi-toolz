#!/usr/bin/env python3
"""
Demo Script for Enhanced Straddle Analysis
==========================================

This script demonstrates how to use the enhanced volatility models
for straddle analysis that goes beyond Black-Scholes assumptions.

Run this script to see the enhanced analysis in action.
"""

import asyncio
import logging
from enhanced_mad_straddle import EnhancedMADStraddleAnalyzer

# Configure logging to see detailed analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    """
    Main demo function showing enhanced straddle analysis
    """
    print("="*80)
    print("ENHANCED STRADDLE ANALYSIS DEMO")
    print("Enhanced Volatility Models + MAD Analysis")
    print("="*80)
    print()
    
    print("This enhanced system addresses your core question:")
    print("'Are straddles truly underpriced when accounting for volatility dynamics?'")
    print()
    
    print("KEY IMPROVEMENTS OVER BLACK-SCHOLES:")
    print("✓ Forward volatility expectations (not constant vol)")
    print("✓ Volatility regime detection (low/normal/high/extreme)")
    print("✓ GARCH modeling for vol clustering and mean reversion")
    print("✓ Volatility surface with smile/skew effects")
    print("✓ Combined with existing MAD tail-risk analysis")
    print()
    
    print("WHAT THIS SOLVES:")
    print("• Black-Scholes assumes constant volatility - WRONG for crypto")
    print("• When buying straddles, you need vol to EXPAND")
    print("• Enhanced models predict when vol expansion is likely")
    print("• Regime detection warns when high vol may mean-revert")
    print("• Forward vol forecasts tell you if current low vol will persist")
    print()
    
    print("Starting enhanced analyzer...")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedMADStraddleAnalyzer()
    
    try:
        # Run enhanced interactive session
        await analyzer.run_enhanced_interactive_session()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        logging.error(f"Demo error: {e}", exc_info=True)
    finally:
        print("\nDemo completed")

def show_model_explanation():
    """Show explanation of the enhanced models"""
    
    print("\n" + "="*80)
    print("ENHANCED VOLATILITY MODEL EXPLANATION")
    print("="*80)
    
    explanations = [
        {
            "Model": "Volatility Regime Detection",
            "Purpose": "Classify market into Low/Normal/High/Extreme vol regimes",
            "For Straddles": "Low vol regime = expansion likely (good for buying)",
            "Black-Scholes Miss": "Ignores current vol environment entirely"
        },
        {
            "Model": "Forward Volatility (GARCH)",
            "Purpose": "Forecast volatility over option lifetime using clustering",
            "For Straddles": "Predicts if current vol will expand or contract",
            "Black-Scholes Miss": "Assumes volatility stays constant (wrong!)"
        },
        {
            "Model": "Volatility Surface",
            "Purpose": "Map vol across strikes and expiries (smile/skew)",
            "For Straddles": "Shows if ATM straddles are rich/cheap vs wings",
            "Black-Scholes Miss": "Uses single vol for all strikes"
        },
        {
            "Model": "Enhanced Straddle Pricing",
            "Purpose": "Combine all models for dynamic fair value",
            "For Straddles": "True fair value accounting for vol dynamics",
            "Black-Scholes Miss": "Static pricing with unrealistic assumptions"
        }
    ]
    
    for i, exp in enumerate(explanations, 1):
        print(f"\n{i}. {exp['Model']}")
        print(f"   Purpose: {exp['Purpose']}")
        print(f"   For Straddles: {exp['For Straddles']}")
        print(f"   Black-Scholes Miss: {exp['Black-Scholes Miss']}")
    
    print("\n" + "="*80)
    print("INTEGRATION WITH MAD ANALYSIS")
    print("="*80)
    
    print("\nYour existing MAD analysis is preserved and enhanced:")
    print("• MAD/SD ratio still detects tail risk (heavy tails increase option value)")
    print("• Enhanced models add volatility dynamics on top")
    print("• Final recommendation combines both approaches")
    print("• Result: More accurate assessment of true straddle value")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Enhanced Volatility Models for Straddle Trading")
    print("Addressing the limitations of Black-Scholes constant volatility")
    
    choice = input("\nWould you like to:\n1. See model explanation\n2. Run live demo\n3. Both\n\nChoice (1/2/3): ")
    
    if choice in ['1', '3']:
        show_model_explanation()
    
    if choice in ['2', '3']:
        print("\n" + "="*50)
        print("STARTING LIVE DEMO")
        print("="*50)
        print("This will connect to Thalex and analyze real straddle opportunities...")
        print("Make sure you have valid API keys in keys.py")
        
        proceed = input("\nProceed with live demo? (y/n): ")
        if proceed.lower() == 'y':
            asyncio.run(main())
        else:
            print("Demo cancelled")
    
    print("\nThank you for using the Enhanced Straddle Analysis System!")
    print("This system helps answer: 'When is volatility likely to expand for straddle buying?'")