## Developer / Creator

tubakhxn

---

# Smart Backtesting Engine

This project is a fully self-contained Python script that demonstrates a simple backtesting engine for trading strategies using synthetic stock price data. It is intended for educational and research purposes only.

## What is this project about?

- **Generates synthetic stock price data** using a random walk model
- **Implements a simple machine learning signal** (moving average crossover)
- **Simulates a trading strategy** (buy/sell based on the signal)
- **Calculates key metrics**: cumulative returns, drawdown, and equity curve
- **Plots**:
  1. Price with buy/sell signals
  2. Equity curve
  3. Drawdown curve
- **Adds a smoothed trend line** using a rolling mean
- **Runs and displays all graphs automatically**

## How to fork and run

1. **Fork this repository** on GitHub (or download the code)
2. Make sure you have Python 3.x installed
3. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib
   ```
4. Run the script:
   ```bash
   python smart_backtest.py
   ```

## Relevant Wikipedia Links

- [Backtesting](https://en.wikipedia.org/wiki/Backtesting)
- [Moving Average](https://en.wikipedia.org/wiki/Moving_average)
- [Random Walk](https://en.wikipedia.org/wiki/Random_walk)
- [Drawdown (economics)](https://en.wikipedia.org/wiki/Drawdown_(economics))

---

**This project is for educational and research purposes only. Not for investment advice or live trading.**
