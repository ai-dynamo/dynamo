<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf load-generator calibration (EXPERIMENT)

This harness measures how much load one AIPerf instance can generate, separately from the system under test.
AIPerf drives `stub_server.py`, an OpenAI-compatible chat stub that answers at once. The stub streams one
token per SSE event and reports `usage`; with `--itl-ms` it paces the tokens. The harness samples the CPU of
every AIPerf process tree and of the stub.

- `setup_node.sh`: creates a node-local venv with `aiperf==0.13.0` and fetches the `Qwen/Qwen3-0.6B` tokenizer snapshot.
- `run_trial.sh`: runs one trial with the stub, K pinned AIPerf instances (`AIPERF_CPUS="0-31;32-63"`) and `cpu_sampler.py`, then writes `summarize.py` output.
- `sweep.py TRIALS.tsv`: a resumable driver. Each trial runs under `node-gate.sh`, the co-tenant gate from the cluster tooling. Copy the gate into `$CALIB_ROOT/scripts`, or replace it with a pass-through.
- `trials/*.tsv`: the trial lists that were run. The columns are name, series, offered rate per instance, stub CPUs, AIPerf CPU sets, env, and AIPerf arguments.
- `tables.py RESULTS [PREFIX...]`: renders the per-trial summaries as a markdown table.

Usage, inside an allocation:

```bash
export CALIB_ROOT=/shared/dir   # must contain scripts/ with these files
bash $CALIB_ROOT/scripts/setup_node.sh
python3 $CALIB_ROOT/scripts/sweep.py $CALIB_ROOT/scripts/trials_A.tsv
```

How to read the metrics:

- **CPU ms/req incl. drain**: AIPerf CPU-seconds from profile start until the record processors go idle, divided by completed requests. Dataset build and warmup are excluded.
- **record drain s**: the time record processors stay busy after the profile window ends. A nonzero value means records were processed more slowly than requests were sent.
- **ISL**: the stub reports body bytes / 4 as `prompt_tokens`, so ISL is a size proxy, not a token count.
- With a paced stub (`STUB_ITL_MS`), the request-throughput metric includes the stream tail. Judge sending with the credit-to-start latency (c2s) instead.
