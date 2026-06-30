"""
evaluation/metrics.py
----------------------
Core evaluation metrics for VRTQ-RL paper.

Metrics:
- FDR@k  : Fault Detection Rate at k% of suite run
- TTFF   : Time to First Failure (lower = better)
- TSR    : Test Suite Reduction % (higher = better)

Author: Anthony Seumal
Project: VRTQ-RL
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def fault_detection_rate(
    selected_df: pd.DataFrame,
    total_faults: int,
    at_k: float = 1.0,
) -> float:
    """
    FDR@k: proportion of total faults detected in first k fraction of budget.

    Args:
        selected_df: Tests in selection order (index = selection rank)
        total_faults: Total faults in the full suite
        at_k: Fraction of budget to evaluate at (0.25, 0.50, 1.0)

    Returns:
        FDR as float 0-1
    """
    k_tests = max(1, int(len(selected_df) * at_k))
    faults_in_k = selected_df.iloc[:k_tests]["has_fault"].sum()
    return faults_in_k / max(1, total_faults)


def time_to_first_failure(
    selected_df: pd.DataFrame,
    budget: int,
) -> int:
    """
    TTFF: 1-indexed position of the first test that reveals a fault.
    Returns budget+1 if no fault found within budget.
    """
    fault_positions = np.where(selected_df["has_fault"].values)[0]
    if len(fault_positions) == 0:
        return budget + 1
    return int(fault_positions[0]) + 1


def test_suite_reduction(
    selected_df: pd.DataFrame,
    total_tests: int,
    total_faults: int,
    coverage_target: float = 0.80,
) -> float:
    """
    TSR: fraction of tests saved while still hitting coverage_target % of faults.

    Higher TSR = more efficient (fewer tests needed).

    Args:
        total_faults: Total faults in the full suite (not just the selected
            subset) — coverage_target must be measured against the real total,
            otherwise a method that finds few faults but happens to find them
            early in its own selection scores an artificially high TSR.
    """
    target_faults = int(coverage_target * total_faults)

    if target_faults == 0:
        return 0.0

    cumulative = selected_df["has_fault"].cumsum().values
    indices = np.where(cumulative >= target_faults)[0]

    if len(indices) == 0:
        return 0.0

    tests_needed = int(indices[0]) + 1
    return 1.0 - (tests_needed / total_tests)


def compute_all_metrics(
    selected_indices: List[int],
    full_df: pd.DataFrame,
    budget: int,
) -> Dict[str, float]:
    """
    Compute all three metrics for a given prioritization.

    Args:
        selected_indices: Ordered list of test indices
        full_df: Full test suite DataFrame
        budget: Number of tests run

    Returns:
        Dict with fdr_25, fdr_50, fdr_100, ttff, tsr
    """
    selected_df = full_df.iloc[selected_indices].reset_index(drop=True)
    total_faults = int(full_df["has_fault"].sum())
    total_tests = len(full_df)

    return {
        "fdr_25": round(fault_detection_rate(selected_df, total_faults, at_k=0.25), 4),
        "fdr_50": round(fault_detection_rate(selected_df, total_faults, at_k=0.50), 4),
        "fdr_100": round(fault_detection_rate(selected_df, total_faults, at_k=1.00), 4),
        "ttff": time_to_first_failure(selected_df, budget),
        "tsr": round(test_suite_reduction(selected_df, total_tests, total_faults), 4),
        "faults_found": int(selected_df["has_fault"].sum()),
        "total_faults": total_faults,
        "budget": budget,
    }


def print_metrics_table(results: List[Dict]) -> None:
    """Pretty-print a comparison table of all methods."""
    print(f"\n{'Method':<20} {'FDR@25%':>8} {'FDR@50%':>8} {'FDR@100%':>9} {'TTFF':>6} {'TSR':>7}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['method']:<20} "
            f"{r.get('fdr_25', 0):>8.1%} "
            f"{r.get('fdr_50', 0):>8.1%} "
            f"{r.get('fdr_100', 0):>9.1%} "
            f"{r.get('ttff', 0):>6.1f} "
            f"{r.get('tsr', 0):>7.1%}"
        )
