# PPO for Smart Grid Energy Management

**Complete implementation of Proximal Policy Optimization with Offline Pre-training for Multi-Objective Smart Grid Energy Management**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

This project implements a reinforcement learning system for smart grid energy management using Proximal Policy Optimization (PPO). The system optimizes multiple objectives simultaneously: electricity cost, carbon emissions, peak demand reduction, and battery health preservation.

### Key Features

- **Correct PPO Implementation**: Generalized Advantage Estimation (GAE), clipped surrogate objective, value function clipping
- **Multi-Objective Optimization**: Balances cost, carbon, peak demand, and battery health with true CityLearn metrics
- **Offline Pre-Training**: Implicit Q-Learning (IQL) for safe initial policy, followed by online PPO fine-tuning
- **Safety Constraints**: Battery SoC limits (0.1-0.9) with penalty-based enforcement
- **Comprehensive Baselines**: Random, Do-Nothing, Rule-Based TOU, Peak Shaving, and Simple MPC
- **Statistical Validation**: Multi-seed experiments with t-tests, Wilcoxon tests, and confidence intervals
- **Visualizations**: Learning curves, baseline comparisons, ablation studies

---

## Project Structure

```
ppo_smart_grid/
├── configs/                     # Experiment configurations
│   ├── default.yaml            # Default hyperparameters
│   ├── quick_test.yaml         # Quick test configuration
│   └── ablation_*.yaml         # Ablation study configs
│
├── src/
│   ├── agents/                 # RL agents
│   │   ├── ppo.py             # PPO implementation with GAE
│   │   ├── baselines.py       # 5 baseline controllers
│   │   └── offline_rl.py      # IQL for offline pre-training
│   │
│   ├── envs/                  # Environment wrappers
│   │   ├── citylearn_wrapper.py    # Multi-objective reward
│   │   ├── reward_extractor.py     # True metric extraction
│   │   ├── normalization.py        # State normalization
│   │   └── safety_constraints.py   # Safety enforcement
│   │
│   ├── models/                # Neural networks
│   │   └── networks.py        # Actor-Critic networks
│   │
│   ├── training/              # Training loops
│   │   └── trainer.py         # PPO training loop
│   │
│   └── utils/                 # Utilities
│       ├── logger.py          # TensorBoard + file logging
│       ├── plotting.py        # Result visualizations
│       ├── metrics.py         # Metric computation
│       ├── config.py          # YAML config loading
│       ├── seeding.py         # Reproducibility
│       └── statistics.py      # Statistical analysis
│
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── evaluate_baselines.py      # Baseline comparison
├── run_multi_seed.py          # Multi-seed experiments
├── run_ablation_study.py      # Ablation study runner
├── run_complete_pipeline.py  # Complete pipeline
├── aggregate_results.py        # Result aggregation
└── analyze_results.py         # Result analysis
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import citylearn; print('All dependencies installed!')"
```

---

## Quick Start

### Single Training Run (Test)
```bash
python train.py --config configs/default.yaml --total_timesteps 100000
```

### Full Training Run
```bash
python train.py --config configs/default.yaml --total_timesteps 1000000
```

### Multi-Seed Experiments
```bash
python run_multi_seed.py --n_seeds 5 --total_timesteps 1000000
```

### Complete Experiment Pipeline
```bash
# Quick test (2-3 hours)
python run_complete_pipeline.py --quick_test

# Full experiment suite (15-20 hours on CPU, 3-5 hours on GPU)
python run_complete_pipeline.py --full
```

### Evaluate Against Baselines
```bash
python evaluate_baselines.py \
    --ppo_checkpoint results/checkpoints/model_1000000.pt \
    --n_episodes 20 \
    --output_dir results/baselines
```

---

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
training:
  total_timesteps: 1000000     # Total training steps
  learning_rate: 3e-4          # PPO learning rate
  gamma: 0.99                  # Discount factor
  gae_lambda: 0.95             # GAE lambda
  clip_ratio: 0.2              # PPO clip range
  
reward:
  weight_cost: 0.4             # Cost objective weight
  weight_carbon: 0.2           # Carbon objective weight
  weight_peak: 0.2             # Peak demand weight
  weight_health: 0.2           # Battery health weight
  
offline_pretraining:
  enabled: false               # Enable offline pre-training
  n_episodes: 100              # Episodes to collect
  n_training_steps: 5000       # IQL training steps
```

---

## Results

The project includes comprehensive results in the `results/` directory:

- **final_results/**: Complete experimental results with visualizations
- **multi_seed/**: Multi-seed training data (5 seeds)
- **logs/**: Training logs and TensorBoard data
- **plots/**: Result visualizations

Key results include:
- Training curves with confidence intervals
- Baseline comparison charts
- Ablation study analysis
- Statistical significance tests

---

## Computational Requirements

- **CPU**: 12-24 hours for 1M steps per seed
- **GPU (Google Colab T4)**: 1-2 hours for 1M steps per seed
- **Memory**: ~4GB RAM
- **Full experimental suite**: ~12-15 GPU hours (available for free on Google Colab)

---

## References

1. Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. Vázquez-Canteli et al., "CityLearn: Standardizing Research in Multi-Agent RL for Demand Response" (2020)
3. Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning" (2021)

---

## License

MIT License

---

## Authors

**Created by:** Syed Bilal Hassan, Hassan Waqar, Abdul Haadi  
**Course:** Reinforcement Learning (Fall 2025)  
**Institution:** FAST School of Computing
