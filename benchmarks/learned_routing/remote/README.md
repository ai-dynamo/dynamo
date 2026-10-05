# Slurm CPU lane for `lr-eval` and `lr-train`

These scripts run learned-routing replays on the zero-GPU x86_64 nodes of a Slurm CPU cluster and
bring the results back into the campaign cache (`CR/runs/cache`). They wrap the harness's
self-contained bundle (`lr-eval --bundle-out`), so a remote record has the same cache key as a local
one and is bit-identical to it. Each run records its Slurm jobs, the evidence for parity and the
measured throughput in `CR/facts/remote.json` (`CR` is the campaign root). The campaign's own
results, with cluster details removed, are summarized in
`notes/learned-routing/campaign/facts/compute_summary.md`.

Run every script from the campaign workstation. Each script records its Slurm jobs in
`CR/facts/remote.json` (`allocations`, with the exact `scancel` command) and prints the cancel
commands. Batch jobs release their node when they finish; there are no idle holds.

## Site settings

Nothing site-specific is built in. Copy `site.env.example` to `site.env` (git-ignored) and fill in
the campaign root `LR_CR`, the cluster's ssh alias `LR_SSH_ALIAS`, the remote scratch root
`LR_REMOTE_ROOT`, and the Slurm `LR_ACCOUNT`, `LR_QOS` and `LR_PARTITIONS`. `LR_SITE_ENV` can point
at a file elsewhere, and variables already in the environment win over the file. `p2orch.py` also
needs `LR_P2ORCH_TARGETS`, a JSON list of placement targets (`p2orch_targets.example.json`).

## Scripts

| Script | What it does |
|---|---|
| `submit_eval.sh` | Builds an `lr-eval` bundle of a cells x policies x replicates batch, pushes it to the cluster's scratch, and submits one batch job per shard (`--nodes N`, shard `i/N` by `(cell, k)`, so every policy of a CRN pair runs on the same node). |
| `submit_train.sh` | Builds an `lr-train` bundle (the eval bundle plus numpy and cma in `site-train/`, the spaces, and `train_jobs.jsonl`) and submits one batch job (one node) per training run. Resubmitting resumes from the checkpoint synced back to scratch. |
| `fetch_ingest.sh` | Refreshes the batch's jobs from `sacct`, rsyncs the returned shards to `CR/runs/remote/returned/<name>/`, runs `lr-eval --ingest` on each shard, and records a summary under `batches.<name>` in `CR/facts/remote.json`. Works on partial results while jobs still run. |
| `node_eval.sh`, `node_train.sh`, `node_common.sh` | Node side (Slurm batch scripts): stage the bundle node-local, verify trace SHA-256s, materialize CRN replicates in parallel, run, and sync results back every 5 minutes and at exit. |
| `prematerialize.py` | Parallel CRN replicate materialization. The harness otherwise materializes replicates one at a time while planning tasks, which took minutes on a fresh node. Output files are byte-identical to the harness's. |
| `parity_check.py` | Field-by-field and per-request byte comparison of remote records against local cache records by cache key. |
| `lane.py` | Bookkeeping: the `remote.json` allocation records (flock + atomic replace), `traces.sha256` and trace provenance, the train-bundle assembly, and the bundle index used for `rsync --link-dest`. |
| `p2orch.py` | Phase-2 orchestrator: places the `CR/runs/phase2/MANIFEST.json` training runs on Slurm nodes (one per node, `submit_train.sh --only RUN` pinned with `--nodelist`) and the workstation, fetches and ingests finished runs, resumes paused ones, enforces an optional job-end cutoff (`--cutoff` or `LR_CUTOFF`, in `LR_TZ`; no built-in default), counts evaluations and runs the LR-10 selection. Documented in `CR/runs/phase2/ORCHESTRATOR.md`. A second instance, run from another worktree so its bundle carries that worktree's build, keeps its own manifest and state under `LR_P2_DIR` and, with `init --yield-to OTHER/state.json[:MAX_WAVE]`, never takes the other instance's targets and waits while it has eligible runs (`CR/runs/phase2-ais/README.md`). |
| `common.sh`, `siteenv.py` | Shared settings: the site values from `site.env` (no built-in defaults), plus `LR_MEM`, `LR_TIME` and the other overridable knobs. |

## Typical use

Shard a batch over 3 nodes, then ingest it:

```bash
R=$WT/benchmarks/learned_routing/remote   # WT = your checkout of this branch
$R/submit_eval.sh --name pilot-baselines-r1 --nodes 3 --repeats 3 \
  --specs CR/runs/pilot/specs.jsonl --cells CR/cells/val.jsonl --time 02:00:00
# ... later (also fine while running; finished shards sync every 5 min):
$R/fetch_ingest.sh --name pilot-baselines-r1 --out CR/runs/pilot/r1/results.jsonl
```

Run one CMA-ES job per node:

```bash
cat > jobs.jsonl <<'EOF'
{"run": "default-tuned-s1", "space": "/abs/path/spaces/default_cost_fn.yaml", "args": ["--budget-evals", "400", "--popsize", "12", "--seed", "1"]}
{"run": "learned-m1-s1", "space": "/abs/path/spaces/learned_choice_m1.yaml", "args": ["--budget-evals", "400", "--popsize", "12", "--seed", "1"]}
EOF
$R/submit_train.sh --name phase2-r1 --jobs jobs.jsonl --cells CR/cells/train.jsonl \
  --val CR/cells/val.jsonl --time 08:00:00
$R/fetch_ingest.sh --name phase2-r1
# run directories: CR/runs/remote/returned/phase2-r1/<run>/runs/train/<run>/
# resume after a wall-limit stop (exit 3 in timing.json):
$R/submit_train.sh --name phase2-r1 --bundle-dir CR/runs/remote/bundles/phase2-r1 --time 08:00:00
```

Cancel a job (exact ID only; the scripts print these):

```bash
ssh "$LR_SSH_ALIAS" scancel <job_id>
```

## Shape and placement

- **Account, QoS and time:** set in `site.env`. Some QoS settings let Slurm start a job with less
  than the requested time (`TimeMin`), and some clusters require jobs to end before a site
  cutoff, so the node scripts clamp their budget to `SLURM_JOB_END_TIME` rather than
  trusting the requested `--time`.
- **Partitions:** zero-GPU x86_64 partitions whose image meets the bundle's glibc floor. On
  partitions that share nodes between jobs (`OverSubscribe=FORCE`), `--exclusive` may be rejected
  and only memory is really reserved, so two of your own jobs, or other users' jobs, can land on the
  same node. To keep one job per node, request more than half the node's memory (for example
  `--mem 160G` on 256 GB nodes and `--mem 400G` on 768 GB nodes), one partition per submission
  (`LR_PARTITIONS=... submit_*.sh ... --mem ...`).
- **Slots:** the default is one replay per physical core (64 on an AMD EPYC 7702P, 96 on an EPYC 9654P).
  Filling the SMT siblings doubled each replay's wall time and added under 10% node throughput, and
  a longer replay lengthens every lr-train generation barrier. Override with `--slots`.
  Throughput per slot count is in `remote.json` `throughput`.
- **Memory:** a replay worker's RSS is the replay's own, about 0.2-1.1 GiB. The harness's
  `worker_peak_rss_mib` field is not a per-replay measure (see the `remote.json` caveats). The node
  scripts sample the RSS sum of the user's processes into `mem.tsv`.

## Reproducibility

- The bundle ships the exact local `_core.abi3.so`, so the build ID and the cache keys match. Nodes
  run `/usr/bin/python3.12 -S` with `PYTHONPATH=site` (CPython 3.12.3, glibc 2.39, the same as the
  workstation).
- **Replays are bit-identical** across the workstation, EPYC 7702P and EPYC 9654P nodes. Every
  record field except timing, PID and path fields matches, and so do the gzip per-request rows,
  byte for byte. The evidence is in `remote.json` `parity`.
- **lr-train numerics:** on AVX-512 nodes, numpy and OpenBLAS dispatch to different kernels, and the
  CMA-ES state differs in the last bits from generation 0. `node_train.sh` pins `OPENBLAS_CORETYPE=Haswell`,
  `OPENBLAS_NUM_THREADS=1` and `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` (set
  `LR_PIN_NUMERICS=0` to opt out). With the pins, CMA-ES trajectories are bit-identical to the
  workstation's. To resume a remote run locally, export the same three variables; they match the
  workstation's own default dispatch.
- Results are ingested only if the harness version and bindings build ID match (`lr-eval --ingest`
  recomputes every cache key from the record's own fields). A bindings rebuild invalidates the old
  bundles: build a new one.

## Layout

| Where | What |
|---|---|
| `CR/runs/remote/bundles/<name>/` | Local bundle (`site/` 365 MB, traces, cells, specs, `traces.sha256`, and for training `site-train/`, `spaces/`, `train_jobs.jsonl`) |
| `$LR_SSH_ALIAS:$LR_REMOTE_ROOT/bundles/<name>/` | Pushed copy; `site/` is hard-linked to the previous bundle of the same build (`rsync --link-dest`) |
| `.../returned/<name>/<shard or run>/` | Node outputs: `runs/cache/results` (records + per-request gz), `results.jsonl`, `runs/train/<run>/`, `node.json` (host, CPU, glibc, Python), `timing.json`, `mem.tsv`, `trace_check.txt`, `prematerialize.json`, `node.log` |
| `.../jobs/<job_id>.out` | Slurm output |
| node `/tmp/lr-$USER-<job>-<bundle>[-<run>]/` | Node-local working copy (not deleted; list it for your end-of-campaign cleanup) |
| `CR/runs/remote/returned/<name>/` | Fetched outputs and `batch_summary.json` |

Nothing is deleted locally or remotely. Keep a list of the remote scratch paths for the
end-of-campaign cleanup.
