<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cognitive Agent

Cognitive Agent is a scaling agent built around the open decision model
`PelaAI/KnowLine-31B-Gen1`. The model's weights and its author's serving code
are used unchanged. The agent adds everything described here: the observations,
the feasible fleet options, the objective, the decision rules and the recorded
evidence.

It runs on the Gym's existing [`jev` adapter](jev.md) with
`provider: self_hosted`, so it shares that adapter's tick seam, validation and
decision log. It sends no paid API calls and never falls back to a hosted
provider. [match.cognitive-agent.example.yaml](../configs/match.cognitive-agent.example.yaml)
runs it beside Planner and the CPU baselines.

> [!IMPORTANT]
> Like [cloudAI-MPC](mpc.md#runtime-requirement), Cognitive Agent reads measured
> outcomes from replay telemetry and needs the `completion_feedback` block.
> The AISimulate release pinned by Dynamo does not emit that block yet.

## Select it

```yaml
autoscalers:
  - name: cognitive-agent
    type: jev
    start: {prefill: 1, decode: 1}
    config:
      provider: self_hosted
      model: PelaAI/KnowLine-31B-Gen1
      base_url: http://127.0.0.1:18080
      timeout_s: 180
      poll_interval_s: 15
      max_calls: 1000
```

With the model served (see [Serve the model](#serve-the-model)):

```bash
python -m pip install -e '.[jev]'
python scripts/run_match_config.py configs/match.cognitive-agent.example.yaml --validate-only --print-matrix
python scripts/run_match_config.py configs/match.cognitive-agent.example.yaml
```

`base_url` must be an HTTP(S) origin without credentials or a path. If the
endpoint needs a bearer token, name its environment variable with
`api_key_env`; `TYPESAFE_API_KEY` is reserved for the hosted provider. A token
is sent only over HTTPS or to a loopback address, so a remote endpoint with
`api_key_env` needs an `https` origin. The adapter bypasses ambient HTTP
proxies for a self-hosted endpoint.

Plain `http` without a token is meant for the loopback setup above. Over any
other network it sends the workload and fleet measurements unencrypted, and
the model's answers can be altered in transit, so use an `https` origin there.

## Joint fleet actions

Each decision asks one Choice question about the joint fleet. Options specify
absolute prefill/decode worker counts, weighted GPU cost, and the GPUs needed
immediately. The candidate set intersects bounded per-pool steps with the
shared GPU budget and the pool bounds. Workers remain indivisible: adding a
four-GPU prefill worker uses four GPUs; adding an eight-GPU decode worker uses
eight. Because every option fits the budget, a disaggregated Match Config
needs only a starting fleet that fits; `max_prefill` and `max_decode` may
together exceed it.

Active, starting and draining workers occupy the budget. A reduction cannot
fund growth until its workers release their GPUs. Confidence gating holds the
whole joint fleet atomically. A single feasible fleet needs no model call.

The objective is requests satisfying every configured SLO per GPU-second.
The prompt supplies recent measured SLO success, goodput/GPU, TTFT and ITL,
resident allocation, and before/after outcome windows for the previous action.
Completion-window observations can include requests started under earlier
fleets; they are distinct from the final arrival-cohort leaderboard score.
Missing observations remain null. Future arrivals, request content and the
native Planner's recommendation are excluded.

The policy identifier is `joint-fleet-measured-efficiency-v2`. Five-second
native telemetry supplies `completion_feedback.schema_version == 1`; missing,
stale or inconsistent feedback fails the run. The replay runtime must emit
that block; see the runtime requirement in [mpc.md](mpc.md#runtime-requirement).
Model decisions occur every
`poll_interval_s` simulated seconds (15 in the example); keep it at or above
the five-second telemetry cadence so each decision sees its previous action.

## Serve the model

The endpoint accepts `POST /v1/systemone` and returns the pinned model identity,
choice distributions, selected choices and confidence. The model repository
ships the server. The reported results used revision
`8f052a8130e599b53e5324c34f0f5ff90696e1c6` on two 80 GiB GPUs:

```bash
export MODEL_DIR="$HOME/models/knowline-31b-gen1"  # any local directory
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "PelaAI/KnowLine-31B-Gen1",
    revision="8f052a8130e599b53e5324c34f0f5ff90696e1c6",
    local_dir=os.environ["MODEL_DIR"],
)
PY
CUDA_VISIBLE_DEVICES=0,1 python "$MODEL_DIR/knowline_server.py" \
  --model "$MODEL_DIR" --backend hf --temperature 1 --workers 1 \
  --host 127.0.0.1 --port 18080 --temperatures "$MODEL_DIR/temperatures.json" \
  --box-prompt gemma --box-coords yxyx
```

Install the serving dependencies the model repository lists, and the Gym's
`jev` extra for the HTTP client.

## Evidence and interpretation

Each run records exact requests, validated answers, targets, inference latency
and errors in `jev-decisions.jsonl`, with a `runtime.jev` summary. A resumed
run can replay a saved decision prefix (`replay_decisions` with its SHA-256 and
`prior_calls`); it rejects changed observations, and replayed decisions do not
call the model. `prior_calls` is the number of model calls the prefix used; a
decision forced by a single feasible fleet uses none. A cancelled or otherwise
interrupted attempt is logged with status `interrupted` and is never replayed.

API wall time does not advance replay time. Results measure decision quality
under ideal actuation timing; physical model GPU use and call latency are
reported separately from simulated serving costs. Timeouts, malformed answers,
model mismatches and exhausted call budgets fail explicitly.
