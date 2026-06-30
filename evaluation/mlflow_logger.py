"""
evaluation/mlflow_logger.py
----------------------------
MLflow logging utilities for VRTQ-RL experiments.

Author: Anthony Seumal
Project: VRTQ-RL
"""
import os
import mlflow
from dotenv import load_dotenv
load_dotenv()

def get_client():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","./mlruns"))
    return mlflow.tracking.MlflowClient()

def start_experiment_run(run_name, params):
    """
    Open an MLflow run under the shared vrtq-rl-experiments experiment and
    log its hyperparameters. Used by both train_ppo.py and train_dqn.py so
    the run-opening/param-logging boilerplate can't drift between the two.

    Returns the active mlflow.ActiveRun (use as a context manager); callers
    are free to keep logging metrics/artifacts against it (e.g. from a
    training callback) for the rest of the run.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("vrtq-rl-experiments")
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    return run

def log_training_run(run_name, params, metrics, artifact_path=None):
    mlflow.set_experiment("vrtq-rl-experiments")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifact_path and os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)

def get_best_ppo_fdr():
    """Return best eval_fdr from all PPO runs."""
    try:
        client = get_client()
        exp = client.get_experiment_by_name("vrtq-rl-experiments")
        if not exp:
            return None
        runs = client.search_runs(exp.experiment_id,
            filter_string="tags.mlflow.runName = 'ppo_vrtq_rl'",
            order_by=["metrics.final_fdr DESC"], max_results=1)
        if runs:
            return runs[0].data.metrics.get("final_fdr")
    except Exception:
        pass
    return None
