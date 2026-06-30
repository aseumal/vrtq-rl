"""
api/main.py
-----------
FastAPI backend for VRTQ-RL React dashboard.

Endpoints:
  POST /api/prioritize        - Run full agent pipeline
  GET  /api/compare           - All baseline comparisons
  GET  /api/status            - Model + dataset status
  GET  /api/learning-curve    - PPO training curve (MLflow or simulated)
  GET  /api/health            - Health check

Run with:
  uvicorn api.main:app --reload --port 8000

Author: Anthony Vallente
Project: VRTQ-RL
"""

import os
import sys
import math
import numpy as np
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

app = FastAPI(
    title="VRTQ-RL API",
    description="Reinforcement Learning-Powered Test Prioritization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_dataset_cache = None

def get_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        from data.fault_injection import create_training_dataset
        from data.dataset_splits import EVAL_SEEDS
        # Use a held-out eval seed, not RANDOM_SEED — RANDOM_SEED (42) is one
        # of TRAIN_SEEDS, so the live demo would otherwise show PPO inference
        # on a dataset it was directly trained on (inflated, misleading FDR).
        _dataset_cache = create_training_dataset(seed=EVAL_SEEDS[0])
    return _dataset_cache


# ── Pydantic models ───────────────────────────────────────────────────────────

class DiffRequest(BaseModel):
    modules: List[str] = ["payment_service"]
    churn: str = "medium"
    budget: int = 50
    use_llm: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serialize(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj

def _run_rl_episode(model, df, budget, masked=False):
    from environment.test_prioritization_env import TestPrioritizationEnv
    from evaluation.metrics import compute_all_metrics
    env = TestPrioritizationEnv(df, budget=budget)
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        if masked:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
    env.close()
    return compute_all_metrics(info["selected"], df, budget)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "vrtq-rl-api", "version": "1.0.0"}


@app.get("/api/status")
def status():
    ppo_path = os.getenv("MODEL_PATH", "./models/ppo_vrtq_rl.zip")
    dqn_path = os.getenv("DQN_MODEL_PATH", "./models/dqn_vrtq_rl.zip")
    df = get_dataset()
    return {
        "ppo_model_ready": os.path.exists(ppo_path),
        "dqn_model_ready": os.path.exists(dqn_path),
        "dataset": {
            "n_tests": len(df),
            "n_faults": int(df["has_fault"].sum()),
            "fault_rate": round(float(df["has_fault"].mean()), 3),
            "modules": sorted(df["module"].unique().tolist()),
        },
    }


@app.post("/api/prioritize")
def prioritize(req: DiffRequest):
    try:
        from agents.change_analyzer_agent import simulate_git_diff
        from agents.orchestrator import run_pipeline
        df = get_dataset()
        diff = simulate_git_diff(modules=req.modules, churn=req.churn)
        report = run_pipeline(
            diff_input=diff,
            test_suite_df=df,
            budget=req.budget,
            use_llm=req.use_llm,
            verbose=False,
        )
        return _serialize(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AgenticDiffRequest(BaseModel):
    modules: List[str] = ["payment_service"]
    churn: str = "medium"
    budget: int = 50


@app.get("/api/agentic-status")
def agentic_status():
    from agents.agentic.llm_config import get_llm_config
    return {"openai_configured": get_llm_config() is not None}


@app.post("/api/prioritize-agentic")
def prioritize_agentic(req: AgenticDiffRequest):
    try:
        from agents.change_analyzer_agent import simulate_git_diff
        from agents.agentic_orchestrator import run_agentic_pipeline
        df = get_dataset()
        diff = simulate_git_diff(modules=req.modules, churn=req.churn)
        report = run_agentic_pipeline(
            diff_input=diff,
            test_suite_df=df,
            budget=req.budget,
            verbose=False,
        )
        return _serialize(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compare")
def compare(budget: int = 50, episodes: int = 5):
    try:
        from evaluation.metrics import compute_all_metrics
        from rl.baselines.random_selector import RandomSelector
        from rl.baselines.vrtq_heuristic import VRTQHeuristicSelector
        df = get_dataset()
        n = len(df)
        results = []

        # 1. Random baseline
        rand_ep = []
        for ep in range(episodes):
            rng = np.random.default_rng(RANDOM_SEED + ep)
            idx = list(range(n))
            rng.shuffle(idx)
            rand_ep.append(compute_all_metrics(idx[:budget], df, budget))
        results.append({
            "method": "Random",
            **{k: round(float(np.mean([r[k] for r in rand_ep])), 4)
               for k in ["fdr_25","fdr_50","fdr_100","ttff","tsr"]},
        })

        # 2. VRTQ heuristic
        vrtq = VRTQHeuristicSelector()
        vm = compute_all_metrics(vrtq.select(df, budget=budget), df, budget)
        results.append({"method": "VRTQ Heuristic",
                         **{k: vm[k] for k in ["fdr_25","fdr_50","fdr_100","ttff","tsr"]}})

        # 3. DQN
        dqn_path = os.getenv("DQN_MODEL_PATH", "./models/dqn_vrtq_rl.zip")
        if os.path.exists(dqn_path):
            from stable_baselines3 import DQN
            model = DQN.load(dqn_path)
            ep_m = [_run_rl_episode(model, df, budget) for _ in range(episodes)]
            results.append({"method": "DQN",
                             **{k: round(float(np.mean([r[k] for r in ep_m])), 4)
                                for k in ["fdr_25","fdr_50","fdr_100","ttff","tsr"]}})
        else:
            results.append({"method": "DQN",
                             "fdr_25":0,"fdr_50":0,"fdr_100":0,"ttff":0,"tsr":0,
                             "note":"python -m rl.train_dqn"})

        # 4. PPO
        ppo_path = os.getenv("MODEL_PATH", "./models/ppo_vrtq_rl.zip")
        if os.path.exists(ppo_path):
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(ppo_path)
            ep_m = [_run_rl_episode(model, df, budget, masked=True) for _ in range(episodes)]
            results.append({"method": "PPO (VRTQ-RL)",
                             **{k: round(float(np.mean([r[k] for r in ep_m])), 4)
                                for k in ["fdr_25","fdr_50","fdr_100","ttff","tsr"]}})
        else:
            results.append({"method": "PPO (VRTQ-RL)",
                             "fdr_25":0,"fdr_50":0,"fdr_100":0,"ttff":0,"tsr":0,
                             "note":"python -m rl.train_ppo"})

        return {"comparison": results, "budget": budget}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learning-curve")
def learning_curve():
    """Return MLflow training history or simulated curve."""
    mlruns = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    try:
        import mlflow
        mlflow.set_tracking_uri(mlruns)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("vrtq-rl-experiments")
        if exp:
            # Pull baseline FDR@100% from the latest comparison run instead
            # of hardcoding stale numbers that drift once models are retrained.
            vrtq_baseline, random_baseline = 0.405, 0.243
            cmp_runs = client.search_runs(
                exp.experiment_id,
                filter_string="tags.mlflow.runName = 'baseline_comparison'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if cmp_runs:
                m = cmp_runs[0].data.metrics
                vrtq_baseline = m.get("vrtq_heuristic_fdr_100", vrtq_baseline)
                random_baseline = m.get("random_fdr_100", random_baseline)

            runs = client.search_runs(
                exp.experiment_id,
                filter_string="tags.mlflow.runName = 'ppo_vrtq_rl'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                history = client.get_metric_history(runs[0].info.run_id, "eval_fdr")
                if history:
                    return {"source": "mlflow", "data": [
                        {"step": m.step, "ppo": round(m.value, 4),
                         "vrtq": round(vrtq_baseline, 4), "random": round(random_baseline, 4)}
                        for m in history
                    ]}
    except Exception:
        pass

    # Simulated fallback
    data = [
        {"step": (i+1)*5000,
         "ppo": round(min(0.72, 0.15 + 0.57*(1-math.exp(-i/6))), 3),
         "vrtq": 0.405, "random": 0.243}
        for i in range(20)
    ]
    return {"source": "simulated", "data": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
