<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom Worker Selection Policies

Use this example as the build guide for a custom Rust worker-selection policy. It covers the full path from `WorkerFilter`, `WorkerScorer`, and `WorkerPicker` implementations to a linked Python frontend or standalone Endpoint Picker Provider (EPP).

## What You Are Building

```text
policy crate -> catalog crate -> router-policy YAML -> frontend or EPP binary
```

- The policy crate owns the filtering, scoring, and picking algorithm.
- The catalog gives each policy type a stable name.
- The YAML file creates named policy instances and supplies parameters.
- The frontend or EPP links the catalog at compile time.

Dynamo owns discovery, eligibility, queueing, validation, reservations, accounting, and metrics. A policy sees only eligible workers and returns one candidate row.

## Pick a Starting Point

| Crate | Use it for |
|---|---|
| `simple-filter-score-pick` | One filter, one scorer, and one picker show the complete policy flow |
| `disagg-filter-score-pick` | Prefill and decode workers each need the complete policy flow |
| `simple-stacked-score-pick` | Multiple scorer costs compose before one picker runs |
| `catalog` | You need to register policy types for configuration |
| `epp` | Worker selection runs in a standalone EPP |

The `simple-filter-score-pick` policy shows the complete pipeline. It filters on minimum device overlap and scores active requests. Its picker normally selects the lowest cost. Tool-result turns select the worker with the most device overlap through `session_context().input_trigger()`.

The `disagg-filter-score-pick` policy applies the overlap filter to both worker types. Prefill and decode workers then use separate scorers and pickers.

The `simple-stacked-score-pick` policy has no custom filter. It adds active-request and uncached-request costs before its picker selects the lowest total.

## 1. Create the Policy Crate

Add `dynamo-kv-router` from the same Dynamo checkout that builds the host process. Enable `standalone-selection`:

```toml
[dependencies]
dynamo-kv-router = { path = "/work/dynamo/lib/kv-router", features = ["standalone-selection"] }
serde = { version = "1", features = ["derive"] }
```

A policy that lives in the Dynamo workspace can use the workspace dependencies shown in [`simple-filter-score-pick/Cargo.toml`](simple-filter-score-pick/Cargo.toml).

## 2. Implement the Filter, Scorers, and Picker

A filter receives one host-eligible worker and returns whether to keep it. Use filters for hard requirements. A scorer receives one kept worker and returns a finite cost. Lower costs are better. A picker receives all scored rows and returns one row index.

Each example keeps its implemented stages in matching modules: `filter.rs`, `scorer.rs`, and `picker.rs`. The crate's `lib.rs` parses parameters, composes those stages, and registers the policy. The stacked example has no filter, so it omits `filter.rs`. Its [`scorer.rs`](simple-stacked-score-pick/src/scorer.rs) groups the scorer stage and re-exports each scorer from a separate file under [`scorer/`](simple-stacked-score-pick/src/scorer/).

The [filter-score-pick policy](simple-filter-score-pick/src/lib.rs) is the shortest complete implementation. Its [filter](simple-filter-score-pick/src/filter.rs) uses raw device-overlap data. The [disaggregated policy](disagg-filter-score-pick/src/lib.rs) creates the complete flow for both worker types.

The [stacked policy](simple-stacked-score-pick/src/lib.rs) shows why scorers use a `Vec`. The `Vec` stores an ordered stack. Each `Box<dyn WorkerScorer>` can hold a different scorer type:

```rust
let scorers: Vec<Box<dyn WorkerScorer>> = vec![
    Box::new(ActiveRequestsScorer),
    Box::new(UncachedBlocksScorer),
];
```

Dynamo calls the scorers in order and adds their costs. A cost of `3` plus a cost of `5` gives the picker a total cost of `8`.

The `simple-filter-score-pick` example exercises one of each policy stage:

```text
host eligibility -> MinimumDeviceOverlapFilter -> ActiveRequestsScorer -> RequestAwarePicker
```

The `disagg-filter-score-pick` factory creates the same three stages for prefill and decode worker sets. Its scorer and picker types differ by worker type.

Declare each optional input group that a component reads:

```rust
fn required_worker_inputs(&self) -> WorkerInputs {
    WorkerInputs::LOAD
}
```

If a component needs both groups, use `WorkerInputs::CACHE | WorkerInputs::LOAD`. Do not request unused groups because Dynamo calculates and retains those columns for each eligible worker.

Keep these rules in every filter, scorer, and picker:

- Return an error instead of panicking.
- Return finite scorer costs.
- Treat candidate order as unspecified.
- Keep blocking I/O out of `keep`, `score`, and `pick`.
- Keep mutable policy state inside the factory-created policy.

## 3. Parse Parameters and Build the Factory

The registry calls the provider once at startup for the selected policy instance. The provider parses and validates the YAML parameters. It then returns a factory that captures the validated values:

```rust
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    min_device_overlap_blocks: f64,
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    if !parameters.min_device_overlap_blocks.is_finite()
        || parameters.min_device_overlap_blocks < 0.0
    {
        return Err(WorkerSelectionPolicyProviderError::new(
            "min_device_overlap_blocks must be a finite non-negative number",
        ));
    }
    let min_device_overlap_blocks = parameters.min_device_overlap_blocks;

    Ok(Arc::new(move |config, worker_type, _partition| {
        let filters: Vec<Box<dyn WorkerFilter>> = vec![
            Box::new(MinimumDeviceOverlapFilter {
                min_device_overlap_blocks,
            }),
        ];

        WorkerSelectionPolicy::new_with_filters(
            config.clone(),
            worker_type,
            filters,
            vec![Box::new(ActiveRequestsScorer)],
            Box::new(RequestAwarePicker),
        )
    }))
}
```

The provider and factory have different lifetimes:

1. The registry matches the YAML `type` to a provider.
2. The provider parses one named instance and returns one shared factory.
3. Dynamo calls the factory once for each model and routing-group partition.
4. Each factory call creates a new filter, scorer, and picker set for that partition.

The factory also receives `worker_type`. The Python frontend supplies `prefill` or `decode`. A standalone EPP supplies `select`. Use the partition value when models or routing groups need separate policy state.

## 4. Register the Policy Type

Expose one registration function from the policy crate:

```rust
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("simple-filter-score-pick", Arc::new(provider))
}
```

Choose a stable, unique type name. The name becomes part of the YAML contract.

Add the policy dependency and registration call to the catalog that ships with the host process. See [`catalog/src/lib.rs`](catalog/src/lib.rs).

## 5. Configure a Policy Instance

Create a YAML file outside the source tree:

```yaml
worker_selection:
  default: simple-filter-score-pick
  instances:
    - name: simple-filter-score-pick
      type: simple-filter-score-pick
      parameters:
        min_device_overlap_blocks: 0
    - name: filter-score-pick-cache-affinity
      type: simple-filter-score-pick
      parameters:
        min_device_overlap_blocks: 8
    - name: disagg-filter-score-pick
      type: disagg-filter-score-pick
      parameters:
        min_device_overlap_blocks: 0
    - name: simple-stacked-score-pick
      type: simple-stacked-score-pick
      parameters: {}
```

- `type` selects a registered provider.
- `name` identifies one configured instance.
- `worker_selection.default` selects an instance at startup.
- `DYN_ROUTER_WORKER_SELECTION_POLICY` overrides the selected instance by name.
- The override value `default` selects Dynamo's built-in policy.

Unknown policy types, duplicate registrations, and invalid parameters stop startup.

The `min_device_overlap_blocks` parameter is a hard filter. A value of `0` keeps cold workers for a smoke test. If every worker is below a positive threshold, Dynamo returns HTTP 503.

## 6. Build and Test

Run these commands from the Dynamo repository root:

```bash
cargo test \
  -p dynamo-custom-policy-example-simple-filter-score-pick \
  -p dynamo-custom-policy-example-disagg-filter-score-pick \
  -p dynamo-custom-policy-example-simple-stacked-score-pick \
  -p dynamo-custom-policy-example-catalog
cargo build -p dynamo-custom-policy-example-epp
```

Add one focused test for the policy decision and one registration test for each new type. If a new input adds calculation, allocation, storage, or scans, add a worker-selection benchmark.

## Run With the Python Frontend

The Python extension uses the dependency alias `dynamo-worker-selection-policy-catalog`. Point that alias at the catalog in the checkout that you build:

```bash
export DYNAMO_DIR="$(pwd)"

cargo add \
  --manifest-path "$DYNAMO_DIR/lib/bindings/python/Cargo.toml" \
  --optional \
  --rename dynamo-worker-selection-policy-catalog \
  --path "$DYNAMO_DIR/examples/router/custom-policy-example/catalog" \
  dynamo-custom-policy-example-catalog
```

Build the extension with the linked catalog:

```bash
cd "$DYNAMO_DIR/lib/bindings/python"
CARGO_TARGET_DIR="$DYNAMO_DIR/target" maturin develop --uv --features custom-policy

cd "$DYNAMO_DIR"
uv pip install -e .
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config /path/to/worker-selection.yaml
```

For a private catalog, keep the dependency alias and change the package name and path. Linked policies apply to the embedded frontend selection service. They do not apply to `python3 -m dynamo.router`.

## Run With EPP

The example EPP links the example catalog and registers it before the standard runner starts:

```rust
let mut registry = WorkerSelectionPolicyRegistry::default();
dynamo_custom_policy_example_catalog::register(&mut registry)?;
run_with_worker_selection_policy_registry(registry).await
```

Run the binary in standalone mode:

```bash
DYN_EPP_MODE=standalone \
DYN_ROUTER_POLICY_CONFIG=/path/to/worker-selection.yaml \
DYN_ROUTER_WORKER_SELECTION_POLICY=simple-filter-score-pick \
  cargo run --release -p dynamo-custom-policy-example-epp
```

Standalone EPP supplies `select` as `worker_type` because it selects from one worker pool. A policy that branches on `worker_type` must handle `select`.

Follow the [standalone EPP guide](../../../docs/fern/pages/kubernetes/kv-aware-routing/vanilla-vllm-onramp.mdx) for discovery, KV events, tokenization, and Kubernetes resources.

## Before You Ship

- Build the policy against the same Dynamo revision as the frontend or EPP.
- Make sure that each component declares every input group that it reads.
- Check every parameter before the factory is created.
- Exercise each `worker_type` branch.
- Prove that filter failures, all-filtered candidate sets, scorer failures, and invalid picker rows do not reserve a worker.
- Benchmark stateful or input-heavy policies at the expected worker count.

The [custom routing API reference](../../../docs/fern/pages/developer-guide/advanced-customizations/custom-worker-selection.mdx) lists the available context and worker signals.

## Try the Policies End to End With Mocker

Use the embedded Python frontend for this local test. The standalone EPP uses Kubernetes `InferencePool` discovery. Complete [Run With the Python Frontend](#run-with-the-python-frontend) first so that the extension links this example catalog.

Create `/tmp/worker-selection.yaml` with the policy instances from [Configure a Policy Instance](#5-configure-a-policy-instance).

### Aggregated Policy

In the first terminal, start the frontend with the `simple-filter-score-pick` policy:

```bash
DYN_ROUTER_WORKER_SELECTION_POLICY=simple-filter-score-pick \
python -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config /tmp/worker-selection.yaml \
  --discovery-backend file \
  --http-port 8000
```

In the second terminal, start two aggregated Mocker workers:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --num-workers 2
```

In the third terminal, send a request:

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

A response confirms that the linked provider created the policy and selected a Mocker worker. The frontend log records the selected worker and its policy cost as `logit`.

Keep the Mocker process active to try `simple-stacked-score-pick`. Stop the frontend and restart it with:

```bash
DYN_ROUTER_WORKER_SELECTION_POLICY=simple-stacked-score-pick \
python -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config /tmp/worker-selection.yaml \
  --discovery-backend file \
  --http-port 8000
```

Send the same request. This time, `logit` is the sum of the active-request and uncached-request costs.

### Disaggregated Policy

Stop the frontend and Mocker processes. Start the frontend with the disaggregated policy:

```bash
DYN_ROUTER_WORKER_SELECTION_POLICY=disagg-filter-score-pick \
python -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config /tmp/worker-selection.yaml \
  --discovery-backend file \
  --http-port 8000
```

In the second terminal, start two prefill Mocker workers:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --disaggregation-mode prefill \
  --bootstrap-ports 50100,50101 \
  --num-workers 2
```

`--bootstrap-ports` takes a comma-separated port for each prefill worker.

In the third terminal, start two decode Mocker workers:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --disaggregation-mode decode \
  --num-workers 2
```

Send the same `curl` request from a fourth terminal. The frontend log records separate prefill and decode selections. Each worker set runs its own filter, scorer, and picker.
