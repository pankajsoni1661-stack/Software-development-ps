Markdown# CLSTM-RL PPO: End-to-End Reinforcement Learning Cryptocurrency Trader
  
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

* **Title:** A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks
* **Authors:** Jie Zou, Jiashu Lou, Baohua Wang, Sixue Liu
* **Source:** arXiv:2212.02721v2 [q-fin.CP]
* **Link:** [View / Download PDF on arXiv](https://arxiv.org/pdf/2212.02721.pdf)
* **Local File:** `2212.02721v2.pdf`

### How This Paper Helped Us
By adapting the methodology from this paper, we achieved significant architectural improvements over standard neural networks (MLPs):

* **Handling Partial Observability:** Financial trading is a partially observable Markov decision process (POMDP) because raw features do not represent the complete state of the market. The paper demonstrated that using an LSTM as a feature extractor uncovers hidden time-series patterns, making the environment behave much closer to a fully observable MDP.
* **The Cascaded Advantage:** Instead of a standard stacked LSTM, the paper introduced a *cascaded* approach. An initial LSTM extracts time-series features from raw daily data, and these refined, encoded features are passed alongside the raw data to the PPO agent (which utilizes a second LSTM). This improves gradient flow and allows learning at multiple temporal scales.
* **Proven Superiority:** The research proved that CLSTM-PPO models consistently outperform standard PPO, MLP baselines, and complex ensemble strategies in key metrics like cumulative returns, maximum earning rates, and Sharpe ratios.

---

## Architecture Summary

```text
Multi-Timeframe OHLCV (1h, 4h, 1d)
         ↓
  25-50 Technical Indicators (pandas_ta)
         ↓
  Stationarity Enforcement (rolling z-scores, pct-distances)
         ↓
  3-Stage Feature Selection
    Stage 1: Mutual Information Filter   → drop bottom 30%
    Stage 2: Spearman Correlation Filter → drop redundant (|ρ| > 0.80)
    Stage 3: LightGBM + SHAP             → select top 4-8 "Golden Features"
         ↓
  Custom Gymnasium Environment
    • Continuous action: [-1=short, 0=flat, +1=long]
    • ATR stop-loss & take-profit
    • Turbulence gate (force-exit in market crashes)
    • Transaction fees + slippage
         ↓
  Cascaded LSTM Feature Extractor (PyTorch)
    LSTM L1 → [L1 output | raw input] → LSTM L2 → MLP
         ↓
  PPO Actor / Critic Networks (Stable Baselines 3)
         ↓
  Interactive Plotly Dashboard (synchronized crosshair)
Folder StructurePlaintextclstm_rl_pipeline/
├── config.py                      # All hyperparameters
├── phase1_data_extraction.py      # CCXT multi-TF download & merging
├── phase1_feature_engineering.py  # 25-50 indicators + stationarity
├── phase2_feature_selection.py    # MI → Spearman → SHAP pipeline
├── phase3_environment.py          # Custom Gymnasium environment
├── phase3_model.py                # CLSTM PyTorch architecture
├── phase3_train.py                # PPO training loop + validation
├── phase4_test_and_visualize.py   # Backtest + interactive charts
├── run_full_pipeline.py           # ← Run this to start everything
├── 2212.02721v2.pdf               # Reference research paper
├── requirements.txt
└── README.md
Quick Start1. Install dependenciesBashpip install -r requirements.txt
GPU (optional): Install PyTorch with CUDA support for faster training:Bashpip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
2. Run the full pipelineBashcd clstm_rl_pipeline
python run_full_pipeline.py
Pipeline Execution Steps:Download 5 years of BTC/USDT data (1h, 4h, 1d) from Binance.Engineer 25-50 indicators per timeframe.Run 3-stage feature selection to print the "Golden State Space".Train the CLSTM-PPO agent for 500,000 timesteps.Backtest on the unseen test set.Generate results/interactive_backtest.html.3. Command-line optionsBash# Force fresh data download (ignore cached CSVs)
python run_full_pipeline.py --redownload

# Skip training, load saved model
python run_full_pipeline.py --skip-train

# Start from Phase 2 (data already downloaded)
python run_full_pipeline.py --phase 2

# Start from Phase 3 (features already computed)
python run_full_pipeline.py --phase 3

# Only re-run the visualization
python run_full_pipeline.py --phase 4
Output FilesFileDescriptionresults/interactive_backtest.html🌐 Interactive dashboard — open in browserresults/test_metrics.jsonReturn, Sharpe, Drawdown, Win Rateresults/test_trades.csvAnnotated trade log with entry/exit reasonsresults/mi_scores.pngStage 1: Mutual Information bar chartresults/correlation_heatmap.pngStage 2: Spearman correlation heatmapresults/shap_beeswarm.pngStage 3: SHAP beeswarm (feature impact)results/shap_bar.pngStage 3: Mean SHAP importance rankingresults/training_reward_curve.pngPPO training progressmodels/clstm_ppo_btcusdt.zipTrained PPO model checkpointdata/golden_features.jsonFinal selected feature namesInteractive VisualizationOpen results/interactive_backtest.html in any browser to view strategy performance.Synchronized crosshair: Move your mouse over any panel to see a vertical line intersecting all 3 panels simultaneously, showing the exact values at that timestamp.Panel 1: BTC/USDT price + Fibonacci retracement levels + trade entry/exit markers (▲ long, ▼ short, ✕ stop-loss, ★ take-profit, ⬡ turbulence).Panel 2: Agent position over time (−1 to +1, filled green/red).Panel 3: Portfolio value vs. Buy & Hold baseline.Data Pipeline DetailsMulti-Timeframe Merging (Anti-Lookahead)The 4h and 1d candles are shifted by 1 period before forward-filling onto the 1h timeline. This ensures the agent at time t only sees the previously completed candle from higher timeframes — never the candle still forming.Python# 4h data: shift by 1 (one 4h candle = 4h of look-back protection)
htf_df = htf_df.shift(1)
htf_df = htf_df.reindex(h1_index, method="ffill")
Data SplitSplitDurationBars (1h)Train4 years~35,040Val6 months~4,380Test6 months~4,380Stationarity EnforcementFeature TypeTransformationBounded oscillators (RSI, Stoch, MFI)Kept as-isPrice levels (SMA, EMA, Ichimoku)% distance from moving averageUnbounded series (OBV, ADL)Rolling z-score (50-bar window)ScalingStandardScaler fit on TRAIN onlyReward FunctionThe reward function guides the agent to prioritize stable, consistent growth while heavily penalizing reckless trading behavior.Plaintextreward = Sharpe_component
       - transaction_cost_penalty
       - drawdown_penalty
       - reversal_penalty
Sharpe component: Rolling 20-step Sharpe ratio (risk-adjusted return).Cost penalty: Discourages excessive trading.Drawdown penalty: Punishes losing from portfolio peak.Reversal penalty: Discourages rapid position flip-flopping.Risk ManagementFibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) are displayed as reference lines on the visualization, computed from the most recent 200-bar swing high and low.MechanismDescriptionATR Stop-LossExit if price moves 2 × ATR against positionATR Take-ProfitExit if price moves 3 × ATR in favourTurbulence GateForce-flatten all positions during market stress (top 10% turbulence)Max Drawdown KillEpisode ends early if portfolio drops 50%Hyperparameter ReferenceKey settings found in config.py:ParameterValueDescriptionSEQ_LEN24Rolling window (hours)LSTM_HIDDEN_SIZE256CLSTM hidden unitsMLP_HIDDEN_SIZE128Output MLP sizeTOTAL_TIMESTEPS500,000PPO training stepsLEARNING_RATE3e-4Adam optimizer LRSTOP_LOSS_ATR_MULT2.0ATR multiplier for SLTAKE_PROFIT_ATR_MULT3.0ATR multiplier for TPTRANSACTION_FEE0.1%Binance maker feeSLIPPAGE0.05%Execution slippageLimitationsThe agent currently trades a single asset (BTC/USDT). Multi-asset extension is possible but requires environment modification.Training on 500K steps (~hours on CPU, ~30min on GPU) may underfit; increase TOTAL_TIMESTEPS for production use.Historical data does not capture order book dynamics or funding rates.DependenciesSee requirements.txt. Key libraries utilized in this project:LibraryPurposeccxtExchange data downloadpandas_ta25-50 technical indicatorsscikit-learnMI filter, StandardScalerlightgbmSHAP proxy modelshapFeature importancegymnasiumRL environment interfacestable-baselines3PPO implementationtorchCLSTM neural networkplotlyInteractive visualization
