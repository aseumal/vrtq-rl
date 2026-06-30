"""
environment/test_prioritization_env.py
--------------------------------------
Custom Gymnasium environment for RL-based test prioritization.

The agent learns to select which test to run next, given:
- Current state of all unselected tests (10 features each)
- Budget constraint (max tests to run per episode)

Episode dynamics:
- State: flattened state matrix of remaining tests
- Action: index of next test to select (from unselected tests)
- Reward: based on fault detection, risk coverage, cost
- Done: when budget exhausted or all tests selected

Author: Anthony Seumal
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from environment.state_builder import (
    build_state_matrix,
    apply_selection_mask,
    STATE_DIM,
)


class TestPrioritizationEnv(gym.Env):
    """
    Gymnasium environment for VRTQ-RL test prioritization.

    Observation space: Box(n_tests * STATE_DIM,) — flattened state matrix
    Action space: Discrete(n_tests) — index of test to run next

    The environment uses action masking: already-selected tests
    have zero state vectors and the agent learns to avoid them.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        test_suite_df: pd.DataFrame,
        budget: int = 50,
        git_diff_features: Optional[Dict] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            test_suite_df: DataFrame with test metadata + has_fault column
            budget: Max number of tests to run per episode
            git_diff_features: Optional git change context
            render_mode: 'human' for verbose output
        """
        super().__init__()

        self.df = test_suite_df.reset_index(drop=True)
        self.n_tests = len(self.df)
        self.budget = min(budget, self.n_tests)
        self.git_diff_features = git_diff_features
        self.render_mode = render_mode

        # Build base state matrix (does not change within an episode
        # unless git_diff_features changes)
        self.base_state = build_state_matrix(self.df, self.git_diff_features)

        # Gymnasium spaces
        obs_dim = self.n_tests * STATE_DIM
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_tests)

        # Episode state (reset on each episode)
        self._selected = []
        self._steps = 0
        self._total_reward = 0.0
        self._faults_found = 0
        self._invalid_attempts = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._selected = []
        self._steps = 0
        self._total_reward = 0.0
        self._faults_found = 0
        self._invalid_attempts = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one test selection.

        Args:
            action: Index of test to run (0 to n_tests-1)

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Guard: if agent selects already-selected test, penalize and skip.
        # A deterministic (eval) policy can get stuck repeatedly choosing the
        # same already-selected action forever, since this branch never
        # advances self._steps. Truncate once that's clearly happening so
        # episodes always terminate.
        if action in self._selected:
            self._invalid_attempts += 1
            reward = -0.2
            obs = self._get_obs()
            info = self._get_info()
            truncated = self._invalid_attempts > self.n_tests
            return obs, reward, False, truncated, info

        # Record selection
        self._selected.append(action)
        self._steps += 1

        # Get selected test
        test = self.df.iloc[action]
        fault_found = bool(test["has_fault"])

        if fault_found:
            self._faults_found += 1

        # Compute reward
        reward = self._compute_reward(test, fault_found)

        # Check termination
        terminated = self._steps >= self.budget
        truncated = False  # we don't use time limits beyond budget

        if terminated:
            # Terminal bonus rewards sustained recall across the whole budget
            # (quadratic so 80%->100% FDR is worth much more than 0%->20%),
            # directly targeting FDR@100% rather than first-hit speed alone.
            total_faults = int(self.df["has_fault"].sum())
            episode_fdr = self._faults_found / max(1, total_faults)
            reward += 5.0 * (episode_fdr ** 2)

        self._total_reward += reward

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_step(action, test, fault_found, reward)

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, test: pd.Series, fault_found: bool) -> float:
        """
        VRTQ-RL reward function.

        Components:
        - Base cost: -0.05 per test (encourages selectivity)
        - Fault bonus: 1/sqrt(N) for the Nth unique fault found this episode —
          diminishing returns, but every new fault still adds positive reward,
          so the optimal policy keeps finding faults across the whole budget
          instead of stopping after the first one (see also the terminal FDR
          bonus applied in step() once the episode ends).
        - Risk/value bonuses are small relative to the fault bonus so the
          agent isn't incentivized to just re-derive the VRTQ heuristic's
          risk weighting instead of learning from fault outcomes directly.
        """
        reward = -0.05  # base cost: running any test has a cost

        if fault_found:
            reward += 1.0 / np.sqrt(self._faults_found)

        if test["vrtq_risk_score"] > 0.7:
            reward += 0.1

        if test["vrtq_value_score"] > 0.8:
            reward += 0.05

        return reward

    def _get_obs(self) -> np.ndarray:
        """
        Return current observation: state matrix with selected tests zeroed out.
        Flattened to 1D for SB3 compatibility.
        """
        masked = apply_selection_mask(self.base_state, self._selected)
        return masked.flatten()

    def _get_info(self) -> Dict[str, Any]:
        """Return episode metadata for logging."""
        total_faults = self.df["has_fault"].sum()
        return {
            "steps": self._steps,
            "budget": self.budget,
            "selected": list(self._selected),
            "faults_found": self._faults_found,
            "total_faults": int(total_faults),
            "fdr": self._faults_found / max(1, total_faults),
            "total_reward": self._total_reward,
            "budget_remaining": self.budget - self._steps,
        }

    def _render_step(
        self,
        action: int,
        test: pd.Series,
        fault_found: bool,
        reward: float,
    ) -> None:
        fault_str = "🐛 FAULT" if fault_found else "  pass"
        print(
            f"Step {self._steps:3d} | {test['test_id']} | "
            f"{test['module']:<22} | {test['test_type']:<12} | "
            f"VRTQ={test['vrtq_composite']:.3f} | "
            f"{fault_str} | reward={reward:+.2f}"
        )

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean mask: True = valid action, False = already selected.
        Name/signature expected by sb3-contrib's MaskablePPO.
        """
        mask = np.ones(self.n_tests, dtype=bool)
        for idx in self._selected:
            mask[idx] = False
        return mask

    def render(self) -> None:
        if self.render_mode == "human":
            info = self._get_info()
            print(f"\n--- Episode State ---")
            print(f"Steps: {info['steps']} / {info['budget']}")
            print(f"Faults found: {info['faults_found']} / {info['total_faults']}")
            print(f"FDR: {info['fdr']:.1%}")
            print(f"Total reward: {info['total_reward']:.3f}")

    def close(self) -> None:
        pass
