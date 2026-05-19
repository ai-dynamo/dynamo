# Power-Aware Planner — E2E Pipeclean Runbook

This runbook lets you validate the end-to-end power planner pipeline **right now**,
using the single-TDP power data already collected in `aic_[h,b]200_power_data`,
before the AIC team delivers the full multi-power-level sweep (Phase 4).

What is pipecleaned:
1. Data integration path (CSV → AIC database → `estimate_perf()` returns `power_w > 0`)
2. Optimizer power path (real `power_w`, not TDP fallback)
3. EMA feedback loop (15-tick convergence, stable state, drift detection)
4. Budget constraint pipeline (replica capping)
5. **Synthetic Phase 4 multi-level simulation** — exercises the exact code path
   Phase 4 will use, using 50%/75%/100% TDP profiles derived from the actual data

---

## Step 1 — Run the unit / integration tests (no cluster needed)

These tests run against a mocked AIC estimator parameterised from the real H200 data:

```bash
# From the dynamo repo root
pytest components/src/dynamo/planner/tests/integration/test_aic_power_e2e_sim.py \
       -v --tb=short -m "pre_merge or integration"
```

Expected: 15 tests across 4 classes, all green.

| Test class | What it validates |
|---|---|
| `TestFeedbackLoopH200` | EMA converges with H200-realistic power values |
| `TestBudgetConstraintPipeline` | Replica capping within 40 kW budget |
| `TestPhase4SyntheticMultiLevel` | Three power profiles, 12 budget×level combos |
| `TestFullDeploymentScenario` | Cold-start → calibration → surge → re-optimize |

---

## Step 2 — Integrate real H200/B200 power data into AIC

```bash
# Dry-run first to see what would be copied
python tools/integrate_aic_power_data.py \
    --aic-checkout  /path/to/aiconfigurator \
    --h200-data     <path-to-h200-power-data> \
    --b200-data     <path-to-b200-power-data> \
    --dry-run

# Apply
python tools/integrate_aic_power_data.py \
    --aic-checkout  /path/to/aiconfigurator \
    --h200-data     <path-to-h200-power-data> \
    --b200-data     <path-to-b200-power-data> \
    --overwrite

# Reinstall AIC from the updated checkout
cd /path/to/aiconfigurator && pip install -e .
```

---

## Step 3 — Validate the integration (no cluster needed)

```bash
# H200 TRTLLm
python tools/validate_aic_power_integration.py \
    --system h200_sxm --backend trtllm --hf-id Qwen/Qwen3-32B

# B200 + vLLM
python tools/validate_aic_power_integration.py \
    --system b200_sxm --backend vllm --hf-id Qwen/Qwen3-32B
```

Key assertions:
- `prefill power_w` in `[100, 710]` W (not 0.0, not exactly 700.0)
- `decode power_w` in `[100, 710]` W
- No `"power_w unavailable"` WARNING in the log → real data path active

---

## Step 4 — Deploy and run the on-cluster smoke test

```bash
# Apply example config (Phase 3 power-aware disagg)
kubectl apply -f examples/deployments/powerplanner/disagg-power-aware.yaml

# Wait for planner to stabilise
kubectl rollout status deployment/planner -n <ns>

# Run all checks including the Phase 4 preview check (section 8)
./examples/deployments/powerplanner/verify_poweraware.bash \
    -n <ns> -d <dgd-name> -p http://<prometheus>:9090 -m phase3
```

### What "Phase 4 preview check" (section 8) verifies

Once the AIC data is integrated:
- `dynamo_aic_optimizer_power_w_prefill > 0` — the optimizer used real power data
- `dynamo_aic_optimizer_power_w_decode > 0`
- No `"power_w unavailable"` in planner logs (grep `kubectl logs`)

Before data integration, section 8 prints `[SKIP]` (informational only, does not fail the check).

---

## What remains for full Phase 4

| Deliverable | Owner | ETA |
|---|---|---|
| Multi-power-level AIC sweep API | AIC team | ~2–3 weeks |
| `cli_power_optimize` planner integration | Dynamo planner | Post-AIC delivery |
| Regression reset on cap changes | Dynamo planner | Post-AIC delivery |
| `total_power_budget_w` in TaskRunner | Dynamo planner | Post-AIC delivery |
| H200/B200 Phase 5 hardware validation | Platform | TBD |

Everything above steps 1–4 is implementable and testable today.
