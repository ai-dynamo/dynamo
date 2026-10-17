# Local datasets (gitignored)

Everything under `data/` except this file is excluded from Git (`data/*` in
`.gitignore`, `*.jsonl` at the repository root). The leaderboard Match
Configs, `configs/match.sim.agg.all-datasets.yaml` and
`configs/match.sim.disagg.all-datasets.yaml`, select the eight registry
workloads plus the six Astra GPT-OSS Golden Set traces laid out here.

## Layout

```text
data/
└── golden-set/
    └── astra-gpt-oss/
        ├── steady-near-capacity.jsonl      # 3,558 rows
        ├── step-and-recovery.jsonl         # 4,259 rows
        ├── gradual-ramp.jsonl              # 4,518 rows
        ├── repeated-cycles.jsonl           # 5,282 rows
        ├── composition-shift.jsonl         # 6,005 rows
        ├── random-bursts.jsonl             # 3,901 rows
        ├── manifest.json                   # seed, reference RPS, per-file SHA-256
        ├── match-config.fragment.yaml      # builder-emitted evaluations fragment
        └── validation.json                 # independent validation (status: pass)
```

## Registry workloads (8)

The seven synthetic workloads (`flat`, `staircase`, `square_wave`,
`flash_crowd`, `diurnal`, `decode_heavy`, `shared_prefix`) are generated on
demand from `src/autoscaling_arena/workloads/registry.py` and need no files.
`mooncake` is the recorded anchor that ships with Dynamo at
`lib/bench/testdata/mooncake_trace_1000.jsonl` (1,000 rows, block size 512);
the registry resolves it from this checkout (or `$DYNAMO_DIR`).

## Astra GPT-OSS Golden Set (6)

Built on 2026-09-22 with `scripts/build_golden_set.py` from the recipe
`configs/golden-set.example.yaml`; every file's SHA-256 and row count match
`manifest.json`.

- Source: the complete 15-shard Astra GPT-OSS `dynamo.request.trace.v1`
  collection (675,633 requests, block size 16). Source paths and request IDs
  are redacted; rows keep only `timestamp`, `input_length`, `output_length`,
  `hash_ids`.
- Seed: 7. Reference rate: 2.370491433501 req/s, the observed mean arrival
  rate of the source, not a calibrated serving capacity. Treat recipe load
  factors as schedule-shape labels until a steady-load capacity sweep replaces
  that anchor.
- Traces are 30 to 60 simulated minutes each. Cap them with
  `evaluations.overrides.<name>.max_requests` when a shorter run is needed.

Rebuilding from the same source, recipe, seed and reference RPS is
deterministic and byte-identical.

## Verify

```bash
cd data/golden-set/astra-gpt-oss && shasum -a 256 -c <(python3 -c '
import json; [print(w["sha256"], " ", w["file"]) for w in json.load(open("manifest.json"))["workloads"]]')
```
