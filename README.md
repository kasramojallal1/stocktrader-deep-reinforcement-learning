# DRL Stock Trader 📈🤖

Train and evaluate **Deep Reinforcement Learning (DRL)** agents for stock trading and interact with them through a lightweight **Django** web UI. Built on **OpenAI Gym**–style environments and **Stable-Baselines3**, with datasets covering **Dow Jones (2009–2020)** and **Tehran Stock Exchange**.

---

##  Repository Description

End-to-end stock trading with Deep Reinforcement Learning (Gym + Stable-Baselines3) and a Django web UI for experiment control and live results.

---

##  🔬 Deep Reinforcement Learning Setup

At the heart of this repo is a **trading simulation environment** where DRL agents learn policies that maximize trading performance.  

### Environment Design
- **State (observation):**  
  Each step provides the agent with features such as current cash, portfolio value, owned shares, and historical technical indicators (moving averages, price changes, etc.).  
- **Action space:**  
  - *Buy* – purchase available shares within cash constraints  
  - *Sell* – liquidate owned shares  
  - *Hold* – maintain current position  
- **Reward function:**  
  Defined as change in portfolio value at each step, encouraging agents to maximize long-term return.  
- **Episodes:**  
  Simulated over a sliding window of historical stock data (train → validation → test splits).  

### Algorithms
We leverage **Stable-Baselines3** implementations of state-of-the-art DRL algorithms:
- **PPO (Proximal Policy Optimization)** – robust policy gradient algorithm suited for noisy markets.  
- **A2C (Advantage Actor-Critic)** – lightweight synchronous actor-critic baseline.  
- **DDPG / TD3** – deterministic policy gradients for continuous actions.  
- **SAC (Soft Actor-Critic)** – entropy-regularized exploration for better robustness.  

Each algorithm can be plugged into the same environment, allowing comparative experiments.

### Why DRL for Trading?
Stock trading is sequential decision-making under uncertainty. DRL fits this well because:
- It learns from **trial and error** interactions with a simulated market.  
- Balances **short-term rewards** (instant gains) with **long-term strategy** (overall portfolio growth).  
- Adapts to **non-stationary environments**, crucial for real-world markets.  

---

##  Features

- **Web UI for easy control**  
  Launch and monitor DRL experiments from a web interface. Select the market (Dow or Tehran), training/trading date ranges, hyperparameters, and watch real-time logs and results.
  
- **Gym-style environments**  
  Consistent, OpenAI Gym–compatible train, validation, and test environments for both Dow Jones and Tehran markets.  

- **Plug-and-play DRL algorithms**  
  PPO, A2C, SAC, TD3, DDPG, and more available from Stable-Baselines3.  

- **Reproducible and trackable**  
  All runs generate saved models under `trained_models/`, logs and results under `results/`.

---

##  Tech Stack

- **Python**: core logic, environments, training  
- **Django**: web UI and run orchestration  
- **Stable-Baselines3**: DRL algorithms  
- **Gym-style envs**: consistent RL API

---

##  Quickstart

### Prerequisites
- Python 3.x environment  
- Use `pip` to install dependencies (preferred over Conda)

### Installation
```bash
git clone https://github.com/kasramojallal1/stocktrader-deep-reinforcement-learning.git
cd stocktrader-deep-reinforcement-learning
pip install -r requirements.txt
```

### Launching the Web UI
```bash
python manage.py runserver
```
- Visit `http://127.0.0.1:8000/`
- Register an account
- Choose market (Dow or Tehran), specify training/trading date windows and hyperparameters (e.g. initial cash, epoch steps)
- View live logs and rolling results in the UI

### Command-Line (CLI) Mode
```bash
python stock_trader_website/trader/drl_stock_trader/main.py
```
- Run experiments without the UI
- CLI outputs logs—add `print(...)` alongside existing SocketIO messages if needed

---

##  Project Structure

```
stock_trader_website/
├── trader/
│   └── drl_stock_trader/
│       ├── dataset/            # Raw + preprocessed data (Dow, Tehran)
│       ├── RL_envs/            # Train/val/test Gym-style environments
│       ├── trained_models/     # Saved model checkpoints
│       ├── results/            # Logs and result outputs
│       ├── main.py             # Orchestration entry point
│       ├── data_retriever.py   # Fetch raw market data
│       ├── preprocess.py       # Data preprocessing pipeline
│       ├── models.py           # Environment and model builders; training loops
│       └── algorithms.py       # DRL algorithms and trading routines
└── ... (Django app code, settings, templates)
```

---

##  How It Works

1. **Fetch & preprocess data**  
   Pull raw data for Dow Jones (2009–2020) or Tehran Stock Exchange, transform it into structured inputs.

2. **Create environments**  
   Initialize Gym-style environments for training, validation, and testing according to specified windows.

3. **Train agents**  
   Use Stable‑Baselines3 algorithms (e.g., PPO, A2C) to train on the train window. Control via epoch/step parameters for robustness.

4. **Evaluate and save**  
   Trade in the test window with the trained policy. Persist model checkpoints to `trained_models/` and logs/results to `results/`.

5. **Visualize via UI**  
   View live performance metrics and results in the Django interface, or analyze outputs directly from saved files.

---

##  Datasets

- **Dow Jones Index (2009–2020)**  
- **Tehran Stock Exchange**  

Dataset directories include raw and processed data. Results from each experiment are saved under `results/`, and models in `trained_models/`.

---

##  Tips & Notes

- Use `pip install -r requirements.txt` to match the project’s environment expectations.  
- For GUI control and monitoring, use the Django web interface.  
- For automated or scriptable experiments, the CLI (`main.py`) is lean and effective.
