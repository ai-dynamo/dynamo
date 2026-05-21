# In-Tree Benchmark Suites

Under `ai-dynamo/dynamo` `benchmarks/`, each suite covers a specialized
workload AIPerf alone does not directly produce. Suite set per the
release line is documented in [`DYNAMO_REPO_SURVEY.md` §9.5](../../../docs/DYNAMO_REPO_SURVEY.md).

---

## When to Use a Suite Instead of AIPerf

- **Synthetic synthetic-token distributions are not enough.** You need
  real-world prompt traces.
- **The benchmark must include arrival-rate variation.** AIPerf is
  steady-state by default.
- **The workload is non-chat.** Multimodal, audio, video, embedding.
- **You're measuring a specific Dynamo subsystem.** KV-router, Frontend,
  Planner.

Otherwise, AIPerf is the default. The suites are complements, not
replacements.

---

## Suite Inventory

Per release branch, verify with:

```bash
ls /Users/dagil/dynamo/benchmarks/
```

| Suite | Workload | When to use |
|---|---|---|
| `agent_trace` | Recorded agent prompts (multi-turn, tool-use) | Agent / function-calling workloads |
| `burstgpt_loadgen` | BurstGPT trace replay | Variable arrival-rate workloads |
| `frontend` | HTTP-layer microbenchmark | Isolate Frontend overhead from worker overhead |
| `incluster` | In-cluster benchmark scaffolding | Run benchmarks from inside the cluster (no port-forward) |
| `llm` | Standard LLM benchmarks (MMLU, etc.) | Accuracy regression checks |
| `multimodal` | Image / video / audio | vLLM-Omni, multimodal models |
| `nat_trace` | NAT-derived workload trace | Specialized NVIDIA workload |
| `omni` | vLLM-Omni / Wan2.2 video | Video generation deployments |
| `prefix_data_generator` | Synthetic prefix dataset | Pre-warm KV cache for KV-router benchmarks |
| `router` | KV-router benchmark harness | KV-aware routing efficacy |
| `sin_load_generator` | Sinusoidal load pattern | Test Planner autoscaling under known-shape variation |

Each suite has its own `README.md`; read it before running.

---

## Recipe-Attached Benchmarks

Every recipe under `recipes/<model>/<framework>/<config>/benchmark/`
ships with the AIPerf invocation that produced its published numbers.
These are not in the `benchmarks/` directory; they live alongside the
recipe DGD.

To reproduce a recipe's published numbers exactly:

```bash
RECIPE=/Users/dagil/dynamo/recipes/<model>/<framework>/<config>
bash $RECIPE/benchmark/run.sh
```

The `run.sh` typically:

1. Port-forwards to the deployed Frontend.
2. Invokes AIPerf with the recipe's specific ISL/OSL/concurrency.
3. Writes `./results.json` and compares to a `expected.json` in the
   recipe.

Diverging from the recipe's `run.sh` flags is fine for exploratory
runs, but a regression baseline run must use the recipe's exact flags.

---

## In-Tree vs Recipe Coverage

| Concern | In-tree suite | Recipe benchmark |
|---|---|---|
| What model? | Generic | Specific to the recipe |
| What hardware? | Generic | Specific to the recipe |
| Published numbers? | Each suite documents its own | The recipe `benchmark/` has them |
| Use case | Exploring a Dynamo subsystem | Reproducing published performance |
