"""
data/synthetic_test_suite.py
----------------------------
Generates a synthetic test suite of N tests with realistic metadata,
VRTQ scores, and historical failure patterns.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_TESTS = int(os.getenv("N_TESTS", 200))
FAULT_RATE = float(os.getenv("FAULT_RATE", 0.15))

# Modules in a realistic enterprise codebase
MODULES = [
    "payment_service",
    "auth_service",
    "user_management",
    "order_processing",
    "inventory_service",
    "notification_service",
    "reporting_engine",
    "api_gateway",
    "data_pipeline",
    "search_service",
]

# Module criticality weights (higher = more business-critical)
MODULE_CRITICALITY = {
    "payment_service": 0.95,
    "auth_service": 0.90,
    "order_processing": 0.85,
    "api_gateway": 0.80,
    "user_management": 0.70,
    "inventory_service": 0.65,
    "data_pipeline": 0.60,
    "reporting_engine": 0.55,
    "search_service": 0.50,
    "notification_service": 0.40,
}

TEST_TYPES = ["unit", "integration", "e2e"]
TEST_TYPE_WEIGHTS = [0.60, 0.30, 0.10]  # realistic distribution


@dataclass
class TestCase:
    test_id: str
    test_name: str
    test_type: str
    module: str
    execution_time_seconds: float
    historical_failure_rate: float
    days_since_last_run: int
    vrtq_value_score: float
    vrtq_risk_score: float
    vrtq_time_score: float
    vrtq_quality_score: float
    vrtq_composite: float
    has_fault: bool


def compute_vrtq_scores(
    module: str,
    test_type: str,
    execution_time: float,
    failure_rate: float,
    days_since_run: int,
    max_exec_time: float = 120.0,
    max_days: int = 30,
) -> dict:
    """
    Compute VRTQ component scores.

    Weights: Value=0.30, Risk=0.35, Time=0.20, Quality=0.15
    These are Anthony Seumal's proprietary VRTQ weights.
    """
    # Value: how much does this test protect business value?
    # Higher for critical modules and integration/e2e tests
    type_multiplier = {"unit": 0.6, "integration": 0.85, "e2e": 1.0}
    value_score = MODULE_CRITICALITY[module] * type_multiplier[test_type]

    # Risk: how likely is this area to have a defect?
    # Driven by historical failure rate
    risk_score = min(1.0, failure_rate * 2.5 + 0.1)

    # Time: inverse of execution time (faster tests score higher)
    # Normalized against max expected execution time
    time_score = max(0.0, 1.0 - (execution_time / max_exec_time))

    # Quality: how recently was this test run? (freshness)
    # Tests run recently have lower quality score (less urgent to re-run)
    quality_score = min(1.0, days_since_run / max_days)

    # Composite VRTQ score
    composite = (
        0.30 * value_score
        + 0.35 * risk_score
        + 0.20 * time_score
        + 0.15 * quality_score
    )

    return {
        "vrtq_value_score": round(value_score, 4),
        "vrtq_risk_score": round(risk_score, 4),
        "vrtq_time_score": round(time_score, 4),
        "vrtq_quality_score": round(quality_score, 4),
        "vrtq_composite": round(composite, 4),
    }


def generate_test_suite(
    n_tests: int = N_TESTS,
    fault_rate: float = FAULT_RATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic test suite with realistic distributions.

    Fault clustering: faults are concentrated in high-criticality modules
    to simulate realistic bug patterns (not random noise).

    Args:
        n_tests: Number of test cases to generate
        fault_rate: Overall proportion of tests that detect a fault
        seed: Random seed for reproducibility

    Returns:
        DataFrame with one row per test case
    """
    rng = np.random.default_rng(seed)
    tests = []

    for i in range(n_tests):
        # Select module with slight bias toward critical ones
        # (critical modules tend to have more tests written for them)
        module_probs = np.array(list(MODULE_CRITICALITY.values()))
        module_probs = module_probs / module_probs.sum()
        module = rng.choice(MODULES, p=module_probs)

        # Test type
        test_type = rng.choice(TEST_TYPES, p=TEST_TYPE_WEIGHTS)

        # Execution time: unit=fast, integration=medium, e2e=slow
        exec_time_params = {
            "unit": (5.0, 3.0),
            "integration": (20.0, 10.0),
            "e2e": (60.0, 25.0),
        }
        mean, std = exec_time_params[test_type]
        execution_time = float(np.clip(rng.normal(mean, std), 0.5, 120.0))

        # Historical failure rate: higher for critical modules
        base_failure = MODULE_CRITICALITY[module] * 0.3
        failure_rate = float(
            np.clip(rng.beta(base_failure * 5 + 0.5, 5.0), 0.0, 1.0)
        )

        # Days since last run: some tests stale, most recent
        days_since_run = int(rng.choice(
            [0, 1, 2, 3, 5, 7, 14, 30],
            p=[0.15, 0.20, 0.20, 0.15, 0.10, 0.10, 0.07, 0.03]
        ))

        vrtq = compute_vrtq_scores(
            module=module,
            test_type=test_type,
            execution_time=execution_time,
            failure_rate=failure_rate,
            days_since_run=days_since_run,
        )

        test = TestCase(
            test_id=f"TEST_{i+1:03d}",
            test_name=f"test_{module}_{test_type}_{i+1:03d}",
            test_type=test_type,
            module=module,
            execution_time_seconds=round(execution_time, 2),
            historical_failure_rate=round(failure_rate, 4),
            days_since_last_run=days_since_run,
            has_fault=False,  # will be set by fault_injection
            **vrtq,
        )
        tests.append(asdict(test))

    df = pd.DataFrame(tests)
    return df


def save_test_suite(df: pd.DataFrame, path: str = "data/test_suite.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Test suite saved to {path} ({len(df)} tests)")


def load_test_suite(path: str = "data/test_suite.csv") -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    print(f"Generating {N_TESTS} synthetic tests (seed={RANDOM_SEED})...")
    df = generate_test_suite()
    print(f"\nTest suite shape: {df.shape}")
    print(f"\nModule distribution:\n{df['module'].value_counts()}")
    print(f"\nTest type distribution:\n{df['test_type'].value_counts()}")
    print(f"\nVRTQ composite stats:\n{df['vrtq_composite'].describe()}")
    print(f"\nSample (first 3 rows):")
    print(df[['test_id', 'module', 'test_type', 'vrtq_composite']].head(3))
    save_test_suite(df)
