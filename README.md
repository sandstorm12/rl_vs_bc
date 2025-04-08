# Hybrid Reinforcement Learning with Expert Initialization

This project explores the combination of Behavior Cloning (BC) and Proximal Policy Optimization (PPO) to improve sample efficiency and training stability in reinforcement learning tasks.

## 🧠 Overview

Reinforcement learning (RL) algorithms often suffer from poor sample efficiency and high sensitivity to hyperparameters. This project implements a hybrid approach where PPO is initialized with a policy pretrained via BC, aiming to:

- Improve sample efficiency
- Enhance training stability
- Enable faster convergence

## 📂 Structure

- `agents/`: Implementations of PPO, BC, and BC + PPO
- `experiments/`: Scripts for running experiments on different Gym environments
- `notebooks/`: Visualizations and analysis of training curves
- `data/`: Expert demonstrations for BC
- `results/`: Logged rewards and summary plots

## 📊 Environments Tested

We evaluate the methods on three OpenAI Gym environments:

- `CartPole-v1`
- `LunarLander-v2`
- `CarRacing-v2`

## 🔍 Results Summary

| Environment     | Method    | Avg Reward | Samples to Converge | Warm-up Steps |
|----------------|-----------|------------|----------------------|----------------|
| CartPole-v1     | PPO       | ~500       | ~10,000              | 0              |
|                | BC        | ~475       | 5,000 (supervised)   | N/A            |
|                | BC + PPO  | ~500       | ~6,000               | 5,000          |
| LunarLander-v2  | PPO       | ~180       | ~300,000             | 0              |
|                | BC        | ~100       | 10,000 (supervised)  | N/A            |
|                | BC + PPO  | ~200       | ~150,000             | 10,000         |
| CarRacing-v2    | PPO       | ~600       | ~1,000,000           | 0              |
|                | BC        | ~300       | 20,000 (supervised)  | N/A            |
|                | BC + PPO  | ~650       | ~800,000             | 20,000         |

## ⚙️ Hyperparameter Sensitivity

BC + PPO was found to be more robust to hyperparameter variation compared to PPO alone, especially in complex environments like `CarRacing-v2`.

## 📦 Dependencies

- Python 3.8+
- `torch`
- `gym`
- `numpy`
- `matplotlib`

Install dependencies via:

```bash
pip install -r requirements.txt
