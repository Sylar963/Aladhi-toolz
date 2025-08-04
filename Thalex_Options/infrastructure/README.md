# Infrastructure & Core Components

Core system components, API libraries, and utilities.

## Components:

### Core API
- **thalex_py/** - Main Thalex exchange API library
  - WebSocket client for real-time data
  - Authentication and order management
  - Market data subscriptions

### Authentication & Configuration
- **keys.py** - API keys and authentication credentials
  - RSA private keys for JWT signing
  - Network configuration (testnet/production)

### Utilities
- **debug_ticker.py** - Debugging and testing utilities
- **Instrument_counter.py** - Instrument tracking and utilities

## Security Note:
The `keys.py` file contains sensitive authentication data and should never be committed to version control.