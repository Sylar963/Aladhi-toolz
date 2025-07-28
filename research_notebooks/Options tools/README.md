# Options Roll Cost Analyzer

Hey there! Welcome to your personal options roll cost analyzer. If you're here, you probably trade options and want to make smarter decisions about when and how to roll your positions. Don't worry if you're not a coding expert - this guide will walk you through everything step by step.

## What the heck is "rolling" anyway?

Think of rolling like this: You have an option that's about to expire, but you want to keep the trade going. Instead of just letting it expire, you:
1. Close your current position (sell what you have)
2. Open a new position with a later expiration date (and maybe a different strike price)

It's like extending the lease on your apartment - you're basically saying "I want to stay in this trade longer."

## Why should I care about roll costs?

Rolling isn't free! Every time you roll, you're paying:
- **Bid-ask spreads** (the difference between buying and selling prices)
- **Commission fees** (what your broker charges)
- **Time decay** differences between old and new options

This tool helps you figure out if rolling is actually worth it or if you're just throwing money away.

## Getting Started (Don't Panic!)

### Step 1: Make sure you have Python installed
If you don't know what Python is, think of it as the engine that runs this tool. Here's how to check:

1. Open your computer's terminal/command prompt
   - **Windows**: Search for "cmd" or "Command Prompt"
   - **Mac**: Search for "Terminal" 
   - **Linux**: You probably already know what you're doing

2. Type this and hit Enter: `python --version`
3. If you see something like "Python 3.8.5", you're good to go!
4. If you get an error, head to [python.org](https://python.org) and download Python

### Step 2: Set up your API keys (CRITICAL!)

**Before you can use the roll cost analyzer, you MUST create a keys.py file with your Thalex API credentials.**

1. **Read the setup guide**: Open `_keys.example.md` in this folder for detailed instructions
2. **Create keys.py**: Follow the guide to create your keys.py file with your actual API keys
3. **Security note**: This example uses hardcoded keys for simplicity, but you should use environment variables in production (explained in the setup guide)

**Don't skip this step!** The script will not work without proper API keys.

### Step 3: Get the tool ready

1. Navigate to where you downloaded this folder
2. In your terminal, type: `cd "path/to/Options tools"`
3. Run the script: `python roll_cost_analyzer.py`

### Step 4: Follow the prompts

The tool will ask you some questions. Here's what each one means:

**"Enter your current option details"**
- **Symbol**: The stock ticker (like AAPL for Apple)
- **Strike price**: The price your option gives you the right to buy/sell at
- **Expiration date**: When your current option expires
- **Option type**: Call or Put
- **Current price**: What the option is trading for right now

**"Enter your target roll details"**
- **New expiration**: When you want the new option to expire
- **New strike**: The strike price for the new option (can be same as current)
- **Target price**: What the new option costs

**"Enter market conditions"**
- **Current stock price**: What the underlying stock is trading for
- **Your broker's commission**: How much your broker charges per trade

## What you'll get back

The tool will spit out some numbers that look scary but are actually pretty simple:

### The Good Stuff:
- **Net roll cost**: How much the roll will cost you (or save you!)
- **Cost per day**: How much you're paying for each extra day in the trade
- **Break-even movement**: How much the stock needs to move for the roll to pay off

### The Analysis:
- **Recommendation**: Should you roll or not?
- **Risk assessment**: What could go wrong?
- **Timing suggestion**: Is now a good time or should you wait?

## Real-World Example

Let's say you have:
- Apple (AAPL) $150 Call expiring this Friday
- Stock is at $148
- Your call is worth $0.50
- You want to roll to next week's $150 call for $2.00

The tool might tell you:
- Roll cost: $1.50 per share ($150 total for 1 contract)
- Cost per day: $0.21 per day
- Break-even: Stock needs to hit $151.50 by new expiration

## Common Scenarios

### Scenario 1: "My option is almost worthless"
- **When this happens**: Your option is way out-of-the-money
- **Tool helps by**: Calculating if rolling to a different strike makes sense
- **Tip**: Sometimes it's better to just take the loss

### Scenario 2: "I'm making money but want more time"
- **When this happens**: You're in-the-money but worried about time decay
- **Tool helps by**: Showing you the true cost of buying more time
- **Tip**: Consider taking profits vs. paying for more time

### Scenario 3: "The stock moved against me"
- **When this happens**: Your call went down or put went up in value
- **Tool helps by**: Analyzing if rolling to a better strike is worth it
- **Tip**: Don't throw good money after bad

## Troubleshooting

**"I get an error when running the script"**
- Make sure you created the keys.py file (see _keys.example.md)
- Make sure you're in the right folder
- Check that the file name is exactly `roll_cost_analyzer.py`
- Try `python3 roll_cost_analyzer.py` instead

**"ImportError: No module named 'keys'"**
- You haven't created the keys.py file yet
- Go back to Step 2 and follow the _keys.example.md guide

**"Authentication failed" or "Connection errors"**
- Your API keys are wrong or expired
- Double-check you copied them correctly from Thalex
- Make sure you're using the right network (testnet vs production)

**"The numbers seem weird"**
- Double-check your inputs (especially dates and prices)
- Make sure you're using the right option type (call vs put)
- Verify your commission structure with your broker

**"I don't understand the results"**
- Start with the recommendation - it's designed to be simple
- Focus on the "cost per day" number - is it worth paying that much?
- When in doubt, ask yourself: "Would I enter this trade fresh at these prices?"

## Pro Tips

1. **Don't roll just because you can** - Sometimes taking a loss is the smart move
2. **Watch the calendar** - Rolling on expiration day is usually expensive
3. **Consider the big picture** - Are you rolling because of analysis or emotions?
4. **Keep track** - Use this tool to build a database of your roll decisions
5. **Paper trade first** - Practice with fake money if you're new to rolling

## A Friendly Warning

This tool gives you math, not magic. It can't predict the future or guarantee profits. Always:
- Do your own research
- Consider your risk tolerance
- Don't risk money you can't afford to lose
- Remember that past performance doesn't predict future results

Think of this as a really smart calculator, not a crystal ball!

---

## Need Help?

If you're stuck or have questions:
1. Read through this guide again (seriously, most answers are here)
2. Check your inputs for typos
3. Make sure your market data is current
4. Remember: when in doubt, smaller position sizes are your friend

Happy trading!

---

*Disclaimer: This tool is for educational and analysis purposes only. Not financial advice. Options trading involves substantial risk and is not suitable for all investors.*