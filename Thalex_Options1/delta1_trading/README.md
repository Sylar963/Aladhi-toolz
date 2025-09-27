# Delta-1 Trading Strategies

Linear instrument trading strategies for perpetuals, futures, and related products.

## Scripts:

### Market Making
- **Simple_quoter.py** - Basic market maker for BTC-PERPETUAL with configurable spreads
- **perp_quoter.py** - Perpetual market maker with TAR calculation
- **perp_quoter_advanced.py** - Advanced perpetual quoter with risk management

### Arbitrage & Rolls
- **synth_arbitrage.py** - Synthetic-future arbitrage opportunities
- **roll_quoter.py** - Roll strategy implementation for future contracts

## Characteristics:
- **Linear payoff** - Delta of 1 (or close to 1)
- **No time decay** - Perpetuals have no expiration
- **Lower complexity** - Straightforward risk management
- **High liquidity** - Generally more liquid than options