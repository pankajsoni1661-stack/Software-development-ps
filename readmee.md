# CLSTM-RL PPO: End-to-End Reinforcement Learning Cryptocurrency Trader

> **IITISoC 2026 — Advanced Problem Statement 1**
> *End-to-End RL-Based Strategy in Cryptocurrency Markets*

---

## Overview

This repository implements a complete, end-to-end pipeline for training a **Cascaded LSTM Reinforcement Learning (CLSTM-RL)** agent to trade BTC/USDT using historical OHLCV data. The agent uses **Proximal Policy Optimization (PPO)** with a continuous action space.

---

## Motivation & Research Foundation

Traditional deep reinforcement learning (DRL) methods were largely developed for gaming environments. When directly adapted to financial data—which is highly noisy, uneven, and influenced by dynamic external factors—standard models often fall short. 

To overcome these limitations, this project implements a **Cascaded LSTM (CLSTM)** architecture, heavily inspired by recent academic research. 

### The Reference Paper
The foundational concepts for our architecture are derived from the following paper, which is also included in this repository:

* **Source:** Jie Zou et al., ["A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks" (View/Download PDF)](https://arxiv.org/pdf/2212.02721.pdf)
* **Local File:** `2212.02721v2.pdf`

### How This Paper Helped Us
By adapting the methodology from this paper, we achieved significant architectural improvements over standard neural networks (MLPs):

* **Handling Partial Observability:** Financial trading is a partially observable Markov decision process (POMDP) because raw features do not represent the complete state of the market. The paper demonstrated that using an LSTM as a feature extractor uncovers hidden time-series patterns, making the environment behave much closer to a fully observable MDP.
* **The Cascaded Advantage:** Instead of a standard stacked LSTM, the paper introduced a *cascaded* approach. An initial LSTM extracts time-series features from raw daily data, and these refined, encoded features are passed alongside the raw data to the PPO agent (which utilizes a second LSTM). This improves gradient flow and allows learning at multiple temporal scales.
* **Proven Superiority:** The research proved that CLSTM-PPO models consistently outperform standard PPO, MLP baselines, and complex ensemble strategies in key metrics like cumulative returns, maximum earning rates, and Sharpe ratios.

---

## Architecture Flow

1. **Multi-Timeframe OHLCV Inputs:** Extracts 1h, 4h, and 1d candles.
2. **Technical Indicators:** Generates 25 to 50 indicators via `pandas_ta`.
3. **Stationarity Enforcement:** Stabilizes data using rolling z-scores and percentage distances.
4. **Three-Stage Feature Selection:**
   * Stage 1: Mutual Information Filter (drops bottom 30%)
   * Stage 2: Spearman Correlation Filter (drops redundant features where |ρ| > 0.80)
   * Stage 3: LightGBM + SHAP (selects top 4 to 8 "Golden Features")
5. **Custom Gymnasium Environment:**
   * Continuous actions bounded between -1 (short), 0 (flat), and +1 (long)
   * Dynamic ATR stop-loss and take-profit bounds
   * Turbulence gate to force-exit during sudden market crashes
   * Incorporates real-world transaction fees and slippage friction
6. **Cascaded LSTM Feature Extractor:** Built in PyTorch. Feeds LSTM Layer 1 output combined with raw inputs into LSTM Layer 2, which then flows to an MLP.
7. **PPO Actor/Critic Networks:** Implemented via Stable Baselines 3.
8. **Interactive Plotly Dashboard:** Outputs performance visuals with a synchronized crosshair.

---

## Folder Structure

```text
clstm_rl_pipeline/
├── config.py                      
├── phase1_data_extraction.py      
├── phase1_feature_engineering.py  
├── phase2_feature_selection.py    
├── phase3_environment.py          
├── phase3_model.py                
├── phase3_train.py                
├── phase4_test_and_visualize.py   
├── run_full_pipeline.py           
├── 2212.02721v2.pdf               
├── requirements.txt
└── README.md

Quick Start
1. Install dependencies
Bash
pip install -r requirements.txt
GPU Acceleration (Optional):

Bash
pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
2. Run the full pipeline
Bash
cd clstm_rl_pipeline
python run_full_pipeline.py
Pipeline Execution Steps:

Download 5 years of BTC/USDT data (1h, 4h, 1d) from Binance.

Engineer 25-50 indicators per timeframe.

Run 3-stage feature selection to print the "Golden State Space".

Train the CLSTM-PPO agent for 500,000 timesteps.

Backtest on the unseen test set.

Generate results/interactive_backtest.html.

3. Command-line options
Bash
# Force fresh data download
python run_full_pipeline.py --redownload

# Skip training, load saved model
python run_full_pipeline.py --skip-train

# Start from Phase 2 (data already downloaded)
python run_full_pipeline.py --phase 2

# Start from Phase 3 (features already computed)
python run_full_pipeline.py --phase 3

# Only re-run the visualization
python run_full_pipeline.py --phase 4
Output Files Generated
results/interactive_backtest.html

Interactive dashboard you can view right in your browser.

results/test_metrics.json

Final performance scores including Return, Sharpe, Drawdown, and Win Rate.

results/test_trades.csv

Logged trade actions detailing explicit entry and exit reasons.

results/mi_scores.png

Bar chart representing Stage 1 Mutual Information rankings.

results/correlation_heatmap.png

Visual matrix showing Stage 2 Spearman correlation filters.

results/shap_beeswarm.png & results/shap_bar.png

Displays feature impact rankings from Stage 3 SHAP evaluations.

results/training_reward_curve.png

Convergence and reward trends across the training history.

models/clstm_ppo_btcusdt.zip

The saved training checkpoint for the PPO model.

data/golden_features.json

Flat file preserving the names of the final selected features.

Interactive Visualization Features
Open results/interactive_backtest.html in your browser to inspect the strategy output.

Synchronized Crosshair: Hovering your mouse anywhere provides a vertical guide line intersecting all data panels simultaneously at that precise timestamp.

Panel 1 (Market View): Displays BTC/USDT price, live Fibonacci retracement steps, and distinct trade markers (▲ long, ▼ short, ✕ stop-loss, ★ take-profit, ⬡ turbulence).

Panel 2 (Position View): Shaded tracks tracking the continuous exposure weights (-1 to +1).

Panel 3 (Performance View): Real-time equity valuation growth contrasted directly against a static Buy & Hold benchmark.

Data Pipeline Details
Multi-Timeframe Merging (Anti-Lookahead)
The 4h and 1d candles are shifted by 1 period before forward-filling onto the 1h timeline. This ensures the agent at time t only sees the previously completed candle from higher timeframes — never the candle still forming.

Python
# 4h data shift to protect against look-ahead bias
htf_df = htf_df.shift(1)
htf_df = htf_df.reindex(h1_index, method="ffill")
Data Split Parameters
Train Set: 4 years (~35,040 hourly bars)

Validation Set: 6 months (~4,380 hourly bars)

Test Set: 6 months (~4,380 hourly bars)

Stationarity Enforcement Strategy
Bounded Oscillators (RSI, Stoch, MFI): Maintained in native forms.

Price Levels (SMA, EMA, Ichimoku): Transformed into percentage distance variables relative to moving averages.

Unbounded Series (OBV, ADL): Stabilized using a rolling z-score across a 50-bar window.

Scaling: Normalized using a StandardScaler fitted strictly on training data.

CLSTM Architecture
The Cascaded LSTM differs from standard stacked structures by passing the raw model input variables directly into the second sequence layer alongside the hidden outputs computed by the first layer.

Plaintext
Standard Stacked Layout:
x → LSTM Layer 1 → LSTM Layer 2 → Output

Cascaded Layout (This Project):
x → LSTM Layer 1 → [LSTM Layer 1 Output + Raw Inputs] → LSTM Layer 2 → Output
This structural cascade preserves clean backpropagation pathways for deeper layers and enables effective context capture across separate temporal intervals.

Reward Function Design
The system relies on a multi-faceted reward equation to optimize returns while discouraging aggressive structural over-trading or volatility spikes:

Sharpe Component: A rolling 20-step Sharpe ratio tracking risk-adjusted value growth.

Transaction Cost Penalty: Applies friction penalties to minimize execution churn.

Drawdown Penalty: Dampens scores if current equity drops away from historical peaks.

Reversal Penalty: penalizes rapid, high-frequency position directional flips.

Risk Management Guardrails
Fibonacci retracement intervals (23.6%, 38.2%, 50%, 61.8%, 78.6%) act as reference anchors, continuously generated dynamically from the preceding 200-bar high/low swing window.

ATR Stop-Loss: Positions are immediately forced closed if prices swing 2 × ATR out of alignment.

ATR Take-Profit: Profits are locked in when target movements touch 3 × ATR in favorable directions.

Turbulence Gate: Flattens outstanding asset exposures entirely if market metrics score in the extreme top 10% turbulence bracket.

Drawdown Circuit Breaker: Terminates training runs prematurely if overall equity balances degrade by 50%.

Hyperparameter Reference
SEQ_LEN: 24 (Rolling observation window in hours)

LSTM_HIDDEN_SIZE: 256 (Hidden units within the CLSTM)

MLP_HIDDEN_SIZE: 128 (Size of the output processing layer)

TOTAL_TIMESTEPS: 500,000 (Target loop steps for PPO training)

LEARNING_RATE: 3e-4 (Initial step size parameter for Adam optimization)

STOP_LOSS_ATR_MULT: 2.0 (ATR multiple scaling for Stop-Loss thresholds)

TAKE_PROFIT_ATR_MULT: 3.0 (ATR multiple scaling for Take-Profit targets)

TRANSACTION_FEE: 0.1% (Calculated maker baseline matching standard exchanges)

SLIPPAGE: 0.05% (Modeled execution friction buffers)

System Limitations
The system is calibrated to track and trade a single underlying market pair (BTC/USDT).

Running for 500k timesteps provides stable proofs but can underfit; scale TOTAL_TIMESTEPS higher for active deployments.

Historical records do not map active real-time order book depth profiles or variable funding metrics.

Core Dependencies
ccxt — Extracts uniform marketplace trading historical sets.

pandas_ta — Computes structural mathematical market indicators.

scikit-learn — Provides filtering modules and parsing structures.

lightgbm — Functions as an entry processing layer for target feature scoring.

shap — Explains asset state spaces by isolating specific data impact metrics.

gymnasium — Serves as the base framework interface tracking state logic steps.

stable-baselines3 — Provides verified, efficient PPO baseline wrappers.

torch — Powers the deep tensor graphs defining the internal network layouts.

plotly — Computes fully responsive front-end dashboard output rendering.
