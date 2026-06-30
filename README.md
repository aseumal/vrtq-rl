# VRTQ-RL: Reinforcement Learning-Powered Test Prioritization

> *Extending a Heuristic Test Prioritization Framework with Reinforcement Learning in Agentic QA Pipelines*

**Author:** Anthony Vallente  
**Affiliation:** Asian Institute of Management | Cambridge University Press & Assessment  
**Target venues:** Software Testing, Verification & Reliability (STVR, Wiley) — primary, direct topical
fit; Journal of Systems and Software (JSS, Elsevier, Q1); Information and Software Technology
(IST, Elsevier, Q1)

---

## Abstract

VRTQ-RL extends the proprietary VRTQ test prioritization framework (Value 0.30, Risk 0.35, Time 0.20, Quality 0.15) with Proximal Policy Optimization (PPO), enabling a self-optimizing agentic QA system. Four AutoGen agents orchestrate the pipeline: git diff analysis → VRTQ state construction → RL-based test selection → human-readable report. We compare PPO and DQN against the VRTQ heuristic and random baselines using Fault Detection Rate (FDR), Time to First Failure (TTFF), and Test Suite Reduction (TSR).

---

## Highlights

- Found PPO scoring *worse than random* on fault detection, root-caused it to 4 distinct issues
  (data leakage, reward misalignment, no action masking, undertraining), fixed each, and reported
  the honest before/after — not a cherry-picked number.
- Validated via a 5-seed training sweep (mean ± std), a held-out train/eval dataset split, and an
  automated minimum-FDR acceptance gate — the evaluation discipline standard in ML, applied to RL.
- A genuinely multi-agent AutoGen layer (Supervisor + Critic) that negotiates over real tool calls —
  the Critic can trigger an actual re-run with different parameters, not just generate more text.
- Full-stack: Gymnasium RL environment, FastAPI backend, React dashboard, MLflow experiment tracking.

---

## Architecture

```
git diff → ChangeAnalyzerAgent → RiskScorerAgent → TestSelectorAgent → ReportAgent
                                                          ↓
                                              PPO Agent (Gymnasium + SB3)
                                              trained on VRTQ state features
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/vrtq-rl.git
cd vrtq-rl
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2. Generate data
python -m data.synthetic_test_suite
python -m data.fault_injection

# 3. Train agents
python -m rl.train_ppo
python -m rl.train_dqn

# 4. Compare all methods
python -m evaluation.compare_baselines

# 5. Run Streamlit dashboard
streamlit run dashboard/app.py
```

---

## Results

| Method | FDR@25% | FDR@50% | FDR@100% | TTFF | TSR |
|--------|---------|---------|----------|------|-----|
| Random | 6.1% | 12.8% | 26.1% | 6.4 | 0.0% |
| VRTQ Heuristic | 10.3% | 19.2% | 39.9% | 2.7 | 0.0% |
| DQN | 0.9% | 0.9% | 4.1% | 2.3 | 0.0% |
| **PPO (VRTQ-RL)** | 6.6% ± 3.6% | 14.0% ± 4.6% | **25.6% ± 5.2%** | 4.4 ± 4.0 | 0.0% |

*PPO trained for 500k timesteps per run (`sb3-contrib` `MaskablePPO`, real action masking) across a 5-seed sweep
(`evaluation/run_seed_sweep.py`, train seeds [42,142,242,342,442]), evaluated on 3 held-out dataset seeds
([1042,1142,1242]) it never saw during training — PPO's numbers above are mean ± std across all 5 trained models.
Random/VRTQ Heuristic/DQN are evaluated on the same held-out seeds via `evaluation/compare_baselines.py`. Raw data
in `evaluation/seed_sweep_results.json` and `evaluation/results.json`.

**Honest read: PPO has improved dramatically but still hasn't clearly beaten Random.** Before this round of fixes,
PPO scored 13.5% FDR@100% — *worse* than Random (24.6%) and far behind VRTQ Heuristic (40.5%), caused by a cluster
of real bugs: training and evaluating on the literal same dataset (no train/test split), a reward function that
only rewarded finding *any* fault once rather than sustained detection, no real action masking, and only 100k
training steps. Fixing all of that (reward reshaping, `MaskablePPO`, a proper train/eval seed split, 5x more
training, higher entropy) raised PPO to **25.6% ± 5.2%** — now roughly on par with Random's 26.1% rather than
losing to it outright (`PPO_mean - Random_mean` is well within one PPO std-dev, i.e. not a clear win by the simple
non-overlap check this project uses), but not a clean win either. VRTQ Heuristic (39.9%) is still the strongest
method here. `evaluation/validate_model.py`'s acceptance gate (PPO must beat Random by ≥20%
relative on the *same* held-out data, with Random itself averaged over 10 episodes per seed to avoid a noisy
threshold) correctly fails even the best of the 5 sweep models (27.8%) against the required 31.3% — this is
deliberately strict and currently honest about where the model stands, not yet a result worth overstating.

Next steps if continuing this work: the diminishing-returns + terminal-FDR reward (`environment/test_prioritization_env.py::_compute_reward`)
may still need retuning (try doubling the terminal bonus coefficient), entropy could go higher than 0.05 if MLflow's
`eval_fdr` curve shows early convergence, and/or training budget could go beyond 500k steps — see MLflow run history
under the `vrtq-rl-experiments` experiment for the eval-FDR-over-time curves per seed.

TSR is 0.0% for every method because `coverage_target=0.80` (80% of all faults) is structurally unreachable with a
25%-of-suite budget — no method here gets remotely close to 80% recall. An earlier version of this table showed
TSR up to 98%, which was a bug in `evaluation/metrics.py::test_suite_reduction()`: it measured "80% of faults *found
within the selection*" instead of "80% of faults in the whole suite," so a method that found very few faults but
happened to rank them early in its own list scored an artificially perfect TSR. Now fixed — `coverage_target` likely
needs lowering (e.g. 50%) or the metric needs reframing around achieved FDR@budget to be useful at this budget size.

DQN's 4.1% separately reflects that standard SB3 `DQN` has no masking equivalent to `MaskablePPO` (`sb3-contrib`
doesn't ship a `MaskableDQN`) — its greedy eval policy still gets stuck proposing already-picked tests until the
episode truncates. DQN is treated as the secondary/cautionary baseline here, not a focus of the masking fix.

**Known confound:** fault injection (`data/fault_injection.py`) biases fault probability by
`0.6 * vrtq_risk_score + 0.3 * historical_failure_rate`, and the VRTQ Heuristic weights risk at 35% of its
composite score — this gives the heuristic a structural advantage worth keeping in mind when interpreting
PPO-vs-heuristic comparisons. PPO's task is harder than "rediscover the known risk-score weighting," since it
must learn from all 10 raw state features rather than the pre-computed composite.*

---

## Agentic Mode (opt-in)

The default pipeline (`agents/orchestrator.py::run_pipeline()`) is a deterministic 4-stage Python
pipeline — see the dashboard's Architecture tab for the honest breakdown of what is and isn't
"agentic" about it. A separate, **opt-in** agentic mode adds genuine multi-turn AutoGen
(`pyautogen==0.2.35`) collaboration on top of the same four stages, without changing them:

- A **Supervisor** agent (`agents/agentic/supervisor.py`) drives `analyze_diff` → `score_risk` →
  `select_tests` → `get_report_metrics` via real tool/function calls it decides to make.
- A **Critic** agent (`agents/agentic/critic.py`) reviews the resulting metrics and can
  autonomously issue `REQUEST: increase budget to X` (or request a different module focus) if
  FDR@50% is poor or selection is too concentrated in one module — a real decision driven by the
  LLM reasoning over the actual numbers, not a scripted branch.
- The Supervisor can apply that feedback with one real re-run (`select_tests` + `get_report_metrics`
  again) — capped at one re-run by a hard Python counter in `agents/agentic/conversation.py`,
  regardless of what the LLM says.

**Setup**: requires a real `OPENAI_API_KEY` in `.env` (the default `gpt-4o-mini`, configurable via
`AGENTIC_OPENAI_MODEL`). Without one, `agents/agentic_orchestrator.py::run_agentic_pipeline()`
falls back to the plain deterministic `run_pipeline()` automatically — verified to produce
byte-identical output to calling `run_pipeline(use_llm=False)` directly.

**Usage**:
```bash
python -m agents.orchestrator --agentic --modules payment_service auth_service --budget 50
# or via the dashboard's "Agentic" tab, or POST /api/prioritize-agentic
```

**Guardrails**: 30s per-LLM-request timeout, 90s whole-conversation wall-clock timeout
(`ThreadPoolExecutor`-enforced), max 12/5 turns per chat phase, falls back to the deterministic
pipeline on any exception. Cost: a few cents at most per run with `gpt-4o-mini` and the round caps
above — there's no native pre-call token budget in `pyautogen==0.2.35`, so the real ceiling is the
turn/round caps, not a token limit.

**Scope**: strictly additive. `evaluation/compare_baselines.py`, `evaluation/validate_model.py`,
and `evaluation/run_seed_sweep.py` never import anything from `agents/` and are unaffected by this
mode — confirmed via `grep` and a before/after diff of `evaluation/results.json`.

---

## Project Structure

```
vrtq-rl/
├── data/               # Synthetic test suite + fault injection
├── environment/        # Gymnasium RL environment + state builder
├── rl/                 # PPO + DQN training + baselines
├── agents/             # Deterministic 4-stage pipeline + agents/agentic/ (opt-in AutoGen mode)
├── evaluation/         # Metrics + baseline comparison + MLflow
├── dashboard/          # Streamlit demo
├── paper/              # Research paper draft
└── notebooks/          # Jupyter walkthroughs
```

---

## Citation

```bibtex
@article{vallente2026vrtqrl,
  title   = {VRTQ-RL: Extending a Heuristic Test Prioritization Framework
             with Reinforcement Learning in Agentic QA Pipelines},
  author  = {Vallente, Anthony},
  journal = {Under review},
  note    = {Target venues: STVR, JSS, IST},
  year    = {2026}
}
```

---

## License

MIT License — see LICENSE file.
