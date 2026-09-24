# Experiment 7: Heuristic Whole-Prefix Retention

This experiment compares three conditions in one paired run:

1. no retention hints;
2. heuristic whole-prefix retention with a fixed 10-second TTL;
3. the same heuristic request and block targets with an exact future-derived TTL.

The no-hint control supplies two online-observable selector inputs:

```text
missing_blocks = total_prefix_blocks - cached_prefix_blocks
removal_pressure_5s = mean successful BlockRemoved rate over the preceding 5 seconds

select when missing_blocks >= 24 and removal_pressure_5s >= 4/s
```

Both policy conditions target the complete current request prefix. They do not use the baseline-oracle block ranges from Experiment 6. The fixed-TTL and future-derived-TTL schedules contain the same request positions and block ranges; only lease duration differs. vLLM enforces a 25% retained-block cap.

Every case starts a new Dynamo vLLM container. The runner rejects a case when the selected GPU already has a compute process, records GPU state before and after the benchmark, clears the container's Prometheus multiprocess directory, and waits 15 seconds between cases.

Run:

```bash
./experiments/continuum-kv-hints/weka-small-qwen3-0.6b/heuristic-validation-20260923/run_experiment.sh
```

Results are written under `artifacts/heuristic-validation-20260923/`.
