
# API Keys Setup for Roll Cost Analyzer

## Important Security Notice

**This is an example file showing you how to set up your API keys. This is currently hardcoded for demonstration purposes, but in a production environment, you should NEVER store API keys directly in your code.**

## What You Need to Do

### Step 1: Create your keys.py file

You need to create a file called `keys.py` in this same directory with your actual Thalex API credentials.

**Create a new file called `keys.py` and copy this template:**

```python
#!/usr/bin/env python3
"""
API Keys for Thalex Roll Cost Analyzer

SECURITY WARNING: This file contains sensitive API credentials.
- Never commit this file to version control
- Never share these keys with anyone
- Keep this file secure and private
"""

import thalex_py.thalex as th

# Your Thalex API Key IDs for different networks
key_ids = {
    th.Network.TESTNET: "your_testnet_key_id_here",
    th.Network.PROD: "your_production_key_id_here"
}

# Your Thalex Private Keys for different networks
private_keys = {
    th.Network.TESTNET: "your_testnet_private_key_here", 
    th.Network.PROD: "your_production_private_key_here"
}
```

### Step 2: Get Your API Keys from Thalex

1. **Log into your Thalex account**
2. **Go to Account Settings**
3. **Navigate to API Management**
4. **Create a new API key pair**
5. **Copy both the Key ID and Private Key**
6. **Paste them into your keys.py file**

### Step 3: Replace the placeholder values

In your `keys.py` file, replace:
- `"your_testnet_key_id_here"` with your actual testnet key ID
- `"your_testnet_private_key_here"` with your actual testnet private key
- `"your_production_key_id_here"` with your actual production key ID
- `"your_production_private_key_here"` with your actual production private key

### Example (with fake keys for illustration):

```python
key_ids = {
    th.Network.TESTNET: "tk_1234567890abcdef",
    th.Network.PROD: "pk_abcdef1234567890"
}

private_keys = {
    th.Network.TESTNET: "ts_your_very_long_testnet_private_key_string_here",
    th.Network.PROD: "ps_your_very_long_production_private_key_string_here"
}
```

## Security Best Practices (What You Should Do Instead)

### The Right Way: Environment Variables

Instead of hardcoding keys, you should use environment variables:

```python
import os
import thalex_py.thalex as th

key_ids = {
    th.Network.TESTNET: os.getenv('THALEX_TESTNET_KEY_ID'),
    th.Network.PROD: os.getenv('THALEX_PROD_KEY_ID')
}

private_keys = {
    th.Network.TESTNET: os.getenv('THALEX_TESTNET_PRIVATE_KEY'),
    th.Network.PROD: os.getenv('THALEX_PROD_PRIVATE_KEY')
}
```

Then set your environment variables:
```bash
export THALEX_TESTNET_KEY_ID="your_testnet_key_id"
export THALEX_TESTNET_PRIVATE_KEY="your_testnet_private_key"
export THALEX_PROD_KEY_ID="your_production_key_id"
export THALEX_PROD_PRIVATE_KEY="your_production_private_key"
```

### Even Better: Use a .env file

Create a `.env` file (and add it to .gitignore):
```
THALEX_TESTNET_KEY_ID=your_testnet_key_id
THALEX_TESTNET_PRIVATE_KEY=your_testnet_private_key
THALEX_PROD_KEY_ID=your_production_key_id
THALEX_PROD_PRIVATE_KEY=your_production_private_key
```

Then load it in your code:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Important Files to Add to .gitignore

Make sure these files are NEVER committed to your repository:
```
keys.py
.env
*.key
*.pem
*_keys.py
```

## Testing Your Setup

The roll cost analyzer will validate your keys when it starts. If you see connection errors, double-check:

1. **Keys are correct** - Copy them again from Thalex
2. **No extra spaces** - Keys should have no leading/trailing spaces
3. **Quotes are correct** - Make sure you're using proper quotes in Python
4. **Network matches** - Make sure you're using the right keys for testnet vs production

## Troubleshooting

**"ImportError: No module named 'keys'"**
- You haven't created the keys.py file yet

**"KeyError: Network.TESTNET"**
- Your keys.py file is missing the network you're trying to use

**"Authentication failed"**
- Your API keys are incorrect or expired
- Check that you copied them correctly from Thalex

**"Permission denied"**
- Your API keys might not have the required permissions
- Check your API key settings in Thalex

Remember: Start with testnet first to make sure everything works before using production keys!
