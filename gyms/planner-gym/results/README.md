# Leaderboard results: Planner, Jev, KEDA, reactive, static, CloudAI MPC, CloudAI RL

Final evaluation of seven autoscalers in DynoSim on `openai/gpt-oss-120b`
(H200, vLLM 0.24.0 AIS performance data, one GPU per worker, 30 s cold start)
over 14 traces (the eight registry workloads and the six Astra GPT-OSS Golden
Set traces), three repetitions each, SLA TTFT 2000 ms / ITL 50 ms, in the
aggregated and the disaggregated topology. Produced on 2026-10-02 from
`configs/match.sim.agg.all-datasets.yaml` and
`configs/match.sim.disagg.all-datasets.yaml`.

| File | Contents |
|---|---|
| `evaluation.md` | How to read the tables, then one summary table per topology (every column as `all (golden)`, goodput per GPU-second) and the CloudAI RL row by mode |
| `evaluation.xlsx` | The same summaries plus, per topology, the method x trace `goodput_per_gpu` matrix, the method x trace rank matrix, and the long-form per-run means (goodput per GPU, good rate, average GPUs, TTFT, P99 TTFT, ITL/TPOT, oscillations, scale events) |
| `rank_tables.agg.md`, `rank_tables.disagg.md` | Mean rank and pairwise win rate on `goodput_per_gpu`, all traces and Golden Set |

The interactive HTML reports (`report.agg.html`, `report.disagg.html`, about
45 MB each, every run's timelines) are attached to the pull request that added
this directory rather than committed; `scripts/run_match_config.py` writes them
next to the results JSON when the sweeps are rerun.

## Reproduce

Environment: the native Dynamo simulation runtime (see
[`docs/getting-started.md`](../docs/getting-started.md)), the `cloudai` extra
(`pip install -e '.[cloudai]'`: numpy and torch) and, for the tables, the
`xlsx` extra (`openpyxl`). The published numbers were produced on a Dynamo
build at the replay-telemetry revision (ai-dynamo/dynamo#14042) with the vLLM
0.24.0 AIS performance data; a rerun on the current runtime reproduces the
comparison up to small simulator differences. Datasets: the Mooncake anchor
ships with Dynamo (`lib/bench/testdata/`); the six Golden Set traces go under
the gitignored `data/` as laid out in [`data/README.md`](../data/README.md).
The Jev entry calls the hosted TypeSafe API and needs `TYPESAFE_API_KEY` in
the environment.

```bash
cd gyms/planner-gym
export TYPESAFE_API_KEY=...
python scripts/run_match_config.py configs/match.sim.agg.all-datasets.yaml
python scripts/run_match_config.py configs/match.sim.disagg.all-datasets.yaml
python scripts/rank_tables.py runs/match-sim-agg-all-datasets/results.agg.json --out results/rank_tables.agg.md
python scripts/rank_tables.py runs/match-sim-disagg-all-datasets/results.disagg.json --out results/rank_tables.disagg.md
python scripts/evaluation_table.py \
  --results agg=runs/match-sim-agg-all-datasets/results.agg.json \
  --results disagg=runs/match-sim-disagg-all-datasets/results.disagg.json \
  --out results/evaluation.md          # also writes results/evaluation.xlsx
```

Each sweep is 294 cells (7 autoscalers x 14 traces x 3 repetitions) and takes
roughly half an hour; results JSON, HTML reports and per-run artifacts
(telemetry, trace reports, RL decision logs) land under the gitignored
`runs/`. Single entries can be redone and folded into an existing results
file with `--autoscaler <name> --merge`. Synthetic registry traces regenerate
per repetition seed; the recorded traces are fixed inputs, so their
repetitions measure simulator determinism only.
