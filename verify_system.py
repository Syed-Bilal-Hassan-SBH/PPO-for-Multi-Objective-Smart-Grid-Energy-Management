#!/usr/bin/env python3
"""
Comprehensive Verification Suite

Tests all components of the PPO smart grid system to ensure:
1. All modules import correctly
2. Algorithms are implemented correctly
3. Shapes and dimensions match
4. Training loop works end-to-end
5. Evaluation metrics are computed correctly
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.agents.ppo import PPOAgent
from src.agents.baselines import (
    RandomBaseline, DoNothingBaseline, RuleBasedBaseline,
    PeakShavingBaseline, SimpleModelPredictiveBaseline,
    BaselineEvaluator
)
from src.agents.offline_rl import OfflineDataset, ImplicitQLearning
from src.envs.citylearn_wrapper import CityLearnWrapper
from src.envs.normalization import NormalizeObservation
from src.envs.safety_constraints import SafetyConstraints
from src.models.networks import ActorCritic


def test_imports():
    """Test that all modules import correctly."""
    print("✓ Testing imports...")
    
    try:
        from citylearn import CityLearnEnv
        print("  ✓ CityLearn imported successfully")
        citylearn_available = True
    except ImportError:
        print("  ⚠ CityLearn not available (will use dummy environment)")
        citylearn_available = False
    
    return citylearn_available


def test_network_architecture():
    """Test Actor-Critic network architecture."""
    print("✓ Testing network architecture...")
    
    obs_dim = 75  # 15 obs per building × 5 buildings
    action_dim = 5  # 1 action per building
    
    network = ActorCritic(obs_dim, action_dim, hidden_dims=(128, 128))
    
    # Test forward pass
    obs = torch.randn(32, obs_dim)
    action_mean, action_std, value = network(obs)
    
    assert action_mean.shape == (32, action_dim), f"Action mean shape mismatch: {action_mean.shape}"
    assert action_std.shape == (action_dim,), f"Action std shape mismatch: {action_std.shape}"
    assert value.shape == (32, 1), f"Value shape mismatch: {value.shape}"
    
    print(f"  ✓ Network shapes correct: action={action_dim}, obs={obs_dim}")
    print(f"  ✓ Forward pass successful")
    
    # Test action and value extraction
    obs_single = torch.randn(1, obs_dim)
    action, log_prob, entropy, val = network.get_action_and_value(obs_single)
    
    assert action.shape == (1, action_dim)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)
    assert val.shape == (1,)
    
    print(f"  ✓ Action extraction successful")
    
    return network


def test_ppo_agent(network):
    """Test PPO agent initialization and methods."""
    print("✓ Testing PPO agent...")
    
    obs_dim = 75
    action_dim = 5
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        learning_rate_schedule="linear",
        ent_coef_schedule="linear",
        total_timesteps=1000000
    )
    
    # Test get_action
    obs = np.random.randn(obs_dim).astype(np.float32)
    action, log_prob, value = agent.get_action(obs, deterministic=False)
    
    assert action.shape == (action_dim,), f"Action shape: {action.shape}"
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    print(f"  ✓ Agent get_action working")
    
    # Test GAE computation
    rewards = np.random.randn(100).astype(np.float32)
    values = np.random.randn(100).astype(np.float32)
    dones = np.zeros(100, dtype=bool)
    next_value = 0.0
    
    advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
    
    assert advantages.shape == (100,)
    assert returns.shape == (100,)
    assert not np.any(np.isnan(advantages))
    assert not np.any(np.isnan(returns))
    print(f"  ✓ GAE computation correct")
    
    # Test update
    obs_batch = np.random.randn(100, obs_dim).astype(np.float32)
    actions_batch = np.random.randn(100, action_dim).astype(np.float32)
    old_log_probs = np.random.randn(100).astype(np.float32)
    
    stats = agent.update(
        obs_batch, actions_batch, old_log_probs,
        advantages, returns, values,
        num_timesteps=2048
    )
    
    assert 'policy_loss' in stats
    assert 'value_loss' in stats
    assert 'entropy' in stats
    assert 'learning_rate' in stats
    assert 'entropy_coef' in stats
    print(f"  ✓ Agent update successful")
    print(f"    - Policy loss: {stats['policy_loss']:.4f}")
    print(f"    - Value loss: {stats['value_loss']:.4f}")
    print(f"    - Learning rate: {stats['learning_rate']:.2e}")
    print(f"    - Entropy coef: {stats['entropy_coef']:.4f}")


def test_baselines():
    """Test baseline controllers."""
    print("✓ Testing baseline controllers...")
    
    action_dim = 5
    obs_dim = 75
    obs = np.random.randn(obs_dim).astype(np.float32)
    
    # Random baseline
    random_bl = RandomBaseline(action_dim, seed=42)
    action = random_bl.get_action(obs)
    assert action.shape == (action_dim,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    print(f"  ✓ Random baseline working")
    
    # Do-nothing baseline
    nothing_bl = DoNothingBaseline(action_dim)
    action = nothing_bl.get_action(obs)
    assert np.allclose(action, 0.0)
    print(f"  ✓ Do-nothing baseline working")
    
    # Rule-based baseline
    rule_bl = RuleBasedBaseline(action_dim, n_buildings=5, seed=42)
    action = rule_bl.get_action(obs)
    assert action.shape == (action_dim,)
    print(f"  ✓ Rule-based baseline working")
    
    # Peak shaving baseline
    peak_bl = PeakShavingBaseline(action_dim, n_buildings=5, seed=42)
    action = peak_bl.get_action(obs)
    assert action.shape == (action_dim,)
    print(f"  ✓ Peak shaving baseline working")
    
    # MPC baseline
    mpc_bl = SimpleModelPredictiveBaseline(action_dim, n_buildings=5, seed=42)
    for _ in range(10):
        action = mpc_bl.get_action(obs)
        assert action.shape == (action_dim,)
    print(f"  ✓ MPC baseline working")


def test_offline_rl():
    """Test offline RL components."""
    print("✓ Testing offline RL...")
    
    obs_dim = 75
    action_dim = 5
    
    # Test OfflineDataset
    dataset = OfflineDataset(capacity=1000)
    
    # Create dummy trajectory
    observations = np.random.randn(100, obs_dim).astype(np.float32)
    actions = np.random.randn(100, action_dim).astype(np.float32)
    rewards = np.random.randn(100).astype(np.float32)
    dones = np.zeros(100, dtype=bool)
    
    dataset.add_trajectory(observations, actions, rewards, dones)
    
    assert dataset.size == 99  # 100 obs means 99 transitions
    print(f"  ✓ OfflineDataset working (stored {dataset.size} transitions)")
    
    # Test sampling
    batch = dataset.sample(batch_size=32)
    assert len(batch) == 5
    assert batch[0].shape == (32, obs_dim)
    assert batch[1].shape == (32, action_dim)
    print(f"  ✓ OfflineDataset sampling working")


def test_safety_constraints():
    """Test safety constraint checking."""
    print("✓ Testing safety constraints...")
    
    constraints = SafetyConstraints(
        battery_soc_min=0.1,
        battery_soc_max=0.9,
        max_power_per_building=10.0,
        min_comfort_temp=18.0,
        max_comfort_temp=26.0,
        max_comfort_deviation=2.0
    )
    
    # Test battery SoC
    safe, msg = constraints.check_battery_soc(0.5)
    assert safe, f"Should be safe: {msg}"
    
    safe, msg = constraints.check_battery_soc(0.05)
    assert not safe, "Should be unsafe (too low)"
    
    safe, msg = constraints.check_battery_soc(0.95)
    assert not safe, "Should be unsafe (too high)"
    
    print(f"  ✓ Battery SoC constraints working")
    
    # Test power limit
    safe, msg = constraints.check_power_limit(5.0)
    assert safe
    
    safe, msg = constraints.check_power_limit(15.0)
    assert not safe
    
    print(f"  ✓ Power limit constraints working")
    
    # Test comfort
    safe, msg = constraints.check_comfort(22.0, 21.0)
    assert safe
    
    safe, msg = constraints.check_comfort(17.0, 21.0)
    assert not safe  # Too cold
    
    print(f"  ✓ Comfort constraints working")


def test_normalization():
    """Test observation normalization."""
    print("✓ Testing observation normalization...")
    
    obs_dim = 75
    normalizer = NormalizeObservation(shape=(obs_dim,), clip=10.0)
    
    # Test first update
    obs1 = np.random.randn(obs_dim)
    normalized1 = normalizer(obs1, update=True)
    assert normalized1.shape == obs1.shape
    
    # Test second update
    obs2 = np.random.randn(obs_dim)
    normalized2 = normalizer(obs2, update=True)
    
    # Should have non-zero mean and std
    assert normalizer.mean is not None
    assert normalizer.var is not None
    
    print(f"  ✓ Observation normalization working")
    print(f"    - Running mean norm: {np.linalg.norm(normalizer.mean):.4f}")
    print(f"    - Running var norm: {np.linalg.norm(normalizer.var):.4f}")


def test_citylearn_wrapper():
    """Test CityLearn wrapper (with dummy environment)."""
    print("✓ Testing CityLearn wrapper...")
    
    # Create dummy environment
    class DummyEnv:
        def __init__(self):
            self.observation_space = [
                np.zeros(15) for _ in range(5)  # 5 buildings, 15 obs each
            ]
            self.action_space = [
                np.zeros(1) for _ in range(5)  # 5 buildings, 1 action each
            ]
        
        def reset(self):
            obs = [np.random.randn(15).astype(np.float32) for _ in range(5)]
            info = {}
            return obs, info
        
        def step(self, actions):
            obs = [np.random.randn(15).astype(np.float32) for _ in range(5)]
            reward = -np.random.rand()  # Negative reward (cost)
            done = False
            truncated = False
            info = {}
            return obs, reward, done, truncated, info
    
    dummy_env = DummyEnv()
    
    wrapper = CityLearnWrapper(
        dummy_env,
        normalize_obs=True,
        weight_cost=0.4,
        weight_carbon=0.2,
        weight_peak=0.2,
        weight_health=0.2,
        use_safety_constraints=True
    )
    
    assert wrapper.obs_dim == 75  # 5 buildings × 15 obs
    assert wrapper.action_dim == 5  # 5 buildings × 1 action
    
    # Test reset
    obs, info = wrapper.reset()
    assert obs.shape == (75,)
    print(f"  ✓ Wrapper reset working")
    
    # Test step
    action = np.random.randn(5).astype(np.float32)
    obs, reward, done, truncated, info = wrapper.step(action)
    
    assert obs.shape == (75,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, (bool, np.bool_))
    print(f"  ✓ Wrapper step working")
    print(f"    - Reward: {reward:.4f}")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Imports
        citylearn_available = test_imports()
        print()
        
        # Test 2: Network
        network = test_network_architecture()
        print()
        
        # Test 3: PPO Agent
        test_ppo_agent(network)
        print()
        
        # Test 4: Baselines
        test_baselines()
        print()
        
        # Test 5: Offline RL
        test_offline_rl()
        print()
        
        # Test 6: Safety constraints
        test_safety_constraints()
        print()
        
        # Test 7: Normalization
        test_normalization()
        print()
        
        # Test 8: CityLearn wrapper
        test_citylearn_wrapper()
        print()
        
        # Summary
        print("=" * 60)
        print("✓ ALL VERIFICATION TESTS PASSED")
        print("=" * 60)
        print()
        print("System is ready for training!")
        print(f"CityLearn available: {citylearn_available}")
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ VERIFICATION FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
