# Options Trading Strategies

Non-linear derivative trading strategies for options and complex multi-leg positions.

## Scripts:

### Delta Hedging
- **Option_delta_replicator.py** - Basic delta hedging using perpetuals
- **Option_delta_replicatorv2.py** - Enhanced delta replicator with improved features

### Market Making & Inventory Management
- **Basev2.py** - Advanced market maker with position limits and risk controls
- **option_inventory_hedger.py** - Inventory management for options positions
- **multi_leg_hedger.py** - Multi-instrument hedging strategies

### Straddle Analysis
- **MADStraddle.py** - Original MAD-based straddle analysis
- **StraddlePositionCalculator.py** - Interactive straddle position tools

## Characteristics:
- **Non-linear payoff** - Delta varies with underlying price
- **Time decay (Theta)** - Options lose value over time
- **Volatility exposure** - Sensitive to implied volatility changes
- **Complex Greeks** - Multiple risk factors (Delta, Gamma, Theta, Vega)
- **Strategic flexibility** - Can express various market views