"""
data/fault_injection.py
-----------------------
Injects faults into the synthetic test suite using realistic patterns:
- Faults cluster in high-risk modules (not random)
- High historical failure rate increases fault probability
- High VRTQ composite score correlates with fault presence

Author: Anthony Vallente
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
FAULT_RATE = float(os.getenv("FAULT_RATE", 0.15))


def inject_faults(
    df: pd.DataFrame,
    fault_rate: float = FAULT_RATE,
    seed: int = RANDOM_SEED,
    cluster_in_high_risk: bool = True,
) -> pd.DataFrame:
    """
    Inject faults into test suite with realistic clustering.

    Strategy:
    - Base fault probability derived from VRTQ risk score
    - Scaled so overall fault_rate matches target
    - Faults cluster in payment_service, auth_service, order_processing

    Args:
        df: Test suite DataFrame from synthetic_test_suite.py
        fault_rate: Target overall fault rate (default 0.15)
        seed: Random seed
        cluster_in_high_risk: If True, bias faults to high-risk modules

    Returns:
        DataFrame with has_fault column populated
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if cluster_in_high_risk:
        # Fault probability is a function of risk score + module criticality
        # This makes the RL agent's job meaningful — it CAN learn the pattern
        fault_probs = (
            0.6 * df["vrtq_risk_score"]
            + 0.3 * df["historical_failure_rate"]
            + 0.1 * rng.uniform(0, 1, len(df))  # noise
        )

        # Scale to hit target fault_rate
        scale_factor = fault_rate / fault_probs.mean()
        fault_probs = np.clip(fault_probs * scale_factor, 0.0, 0.9)

        df["has_fault"] = rng.uniform(0, 1, len(df)) < fault_probs
    else:
        # Pure random injection (for ablation study)
        df["has_fault"] = rng.uniform(0, 1, len(df)) < fault_rate

    actual_rate = df["has_fault"].mean()
    print(f"Fault injection complete: {df['has_fault'].sum()} faults "
          f"({actual_rate:.1%} of {len(df)} tests)")

    return df


def get_fault_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Show fault distribution across modules for analysis."""
    return (
        df.groupby("module")
        .agg(
            total_tests=("test_id", "count"),
            faulty_tests=("has_fault", "sum"),
            fault_rate=("has_fault", "mean"),
            avg_risk_score=("vrtq_risk_score", "mean"),
        )
        .sort_values("fault_rate", ascending=False)
        .round(3)
    )


def create_training_dataset(
    n_tests: int = 200,
    fault_rate: float = FAULT_RATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Full pipeline: generate + inject faults.
    This is the primary entry point for the RL environment.
    """
    from data.synthetic_test_suite import generate_test_suite
    df = generate_test_suite(n_tests=n_tests, seed=seed)
    df = inject_faults(df, fault_rate=fault_rate, seed=seed)
    return df


if __name__ == "__main__":
    from data.synthetic_test_suite import generate_test_suite, save_test_suite

    print("Generating test suite with fault injection...")
    df = generate_test_suite()
    df = inject_faults(df)

    print("\nFault distribution by module:")
    print(get_fault_distribution(df).to_string())

    save_test_suite(df, "data/test_suite_with_faults.csv")
    print(f"\nTotal faults: {df['has_fault'].sum()} / {len(df)}")
