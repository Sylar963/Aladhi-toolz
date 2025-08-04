# Enhanced MAD Straddle System

A comprehensive volatility analysis and straddle trading system for cryptocurrency options.

## Directory Structure

```
enhanced_mad_straddle_system/
├── core/                    # Core analysis modules
│   ├── enhanced_mad_straddle.py       # Main enhanced analyzer
│   ├── enhanced_straddle_pricing.py   # Enhanced pricing models
│   ├── volatility_regime.py           # Regime detection
│   ├── forward_volatility.py          # Forward vol estimation
│   ├── volatility_surface.py          # Vol surface modeling
│   └── financial_math.py              # Mathematical utilities
├── tests/                   # Test and validation scripts
│   ├── test_enhanced_pricing.py       # Pricing model tests
│   ├── test_pricing_fixes.py          # Bug fix validation
│   └── volatility_backtest.py         # Backtesting framework
├── demos/                   # Example and demo scripts
│   ├── demo_enhanced_straddle.py      # System demonstration
│   └── setup_enhanced_system.py       # Setup utilities
├── data/                    # Historical data
│   ├── straddle_data_20250721.csv     # Historical straddle data
│   └── straddle_data_20250727.csv     # More historical data
├── config/                  # Configuration files
│   ├── straddle_config.json           # Straddle analysis config
│   └── option_hedger_config.json      # Hedging parameters
└── docs/                    # Documentation
    ├── ENHANCED_SYSTEM_README.md      # System overview
    └── MAD_STRADDLE_ARCHITECTURE.md   # Architecture guide
```

## Usage

Import the system:

```python
from enhanced_mad_straddle_system.core import EnhancedMADStraddleAnalyzer
```

Run demos:

```bash
cd enhanced_mad_straddle_system/demos
python demo_enhanced_straddle.py
```

## Core Features

- **MAD Analysis**: Tail risk analysis using Mean Absolute Deviation
- **Volatility Regimes**: Automatic detection of market volatility states
- **Forward Volatility**: GARCH-based volatility forecasting
- **Volatility Surface**: Dynamic volatility surface modeling
- **Enhanced Pricing**: Advanced straddle pricing with regime awareness
- **Persistent Data**: Supabase integration for data management