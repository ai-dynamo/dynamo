# Definitions

Shared terms for Dynamo agentic deployment, benchmarking, and optimization.

## Deployment

- **Recipe family**: a model recipe directory under `recipes/`, such as `recipes/kimi-k2.6`.
- **Recipe variant**: one deployable recipe configuration, usually a specific `deploy.yaml`.
- **DGD**: `DynamoGraphDeployment`, the Kubernetes resource that defines a Dynamo serving graph.
- **Component**: a named DGD serving unit, such as `frontend`, `worker`, `prefill`, or `decode`.
- **Aggregated serving**: workers handle both prefill and decode.
- **Disaggregated serving**: prefill and decode run as separate components.
- **Model cache**: PVC and supporting jobs/manifests used to stage model files and related artifacts.
- **Endpoint**: OpenAI-compatible HTTP service exposed by the DGD frontend.

## Runs

- **Target workload**: the requested model, hardware, serving shape, load shape, and success constraints.
- **Smoke test**: `/v1/models` plus one `/v1/chat/completions` request. It proves serving, not performance.
- **AIPerf trace**: benchmark input, usually JSONL.
- **AIPerf artifacts**: raw benchmark outputs, logs, parsed metrics, and copied run metadata.
- **Baseline**: first valid deployment and benchmark result used for comparison.
- **Candidate**: recipe, manifest, or generated config being evaluated.

## Metrics

- `active_gpu_count`: GPUs on the serving request path. For disaggregated serving, include both prefill and decode GPUs.
- `output_tput_per_gpu = output_token_throughput / active_gpu_count`.
- `total_tput_per_gpu = total_token_throughput / active_gpu_count`.
- `tokens_per_sec_per_user = 1000 / TPOT_ms`.
- **TTFT**: Time To First Token.
- **ITL**: Inter-Token Latency.
- **TPOT**: Time Per Output Token (same as **ITL**).
- **Goodput**: throughput that satisfies the target SLA.

Default Pareto axes are `output_tput_per_gpu` and `tokens_per_sec_per_user`.

## Artifacts

- `EXP_ROOT`: `runs/<EXP_ID>/`, created once for an optimization job.
- `DEPLOY_ROOT`: `runs/<EXP_ID>/artifacts/deploy-iter-<NNN>/`, created for one candidate recipe.
- `recipe_inventory.json`: structured recipe inventory.
- `deployment_ledger.json`: deployment and smoke-test record.
- `benchmark_summary.json`: parsed AIPerf metrics and comparison record.

## Knobs

- **Dynamo knobs**: DGD or Kubernetes-level choices, including but not limited to topology, replicas, router mode, PVCs, resources, node placement, container images, and component env vars.
- **Engine knobs**: backend runtime args/env, including but not limited to TP, max model length, max sequences, batching limits, GPU memory utilization, prefix caching, speculative decoding, and KV transfer config.
- **Benchmark knobs**: AIPerf load and measurement inputs, including but not limited to trace, concurrency, request rate, warmup, duration, target model, tokenizer path, and output directory.

## Agent Roles

- **Recipe Explorer**: inventories existing recipe variants and exposed knobs.
- **Recipe Deployer**: deploys one selected recipe variant and verifies the endpoint.
- **Performance Analyzer**: interprets AIPerf artifacts.
- **Hypothesis Generator**: proposes evidence-backed candidate changes.
- **Hypothesis Challenger**: critiques candidates before more GPU time is spent.
