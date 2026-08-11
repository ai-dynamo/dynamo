<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom Worker Selection Policies

Use this example as the build guide for a custom Rust worker-selection policy. It covers the full path from `WorkerScorer` and `WorkerPicker` implementations to a linked Python frontend or standalone Endpoint Picker Provider (EPP).

## What You Are Building

```text
policy crate -> catalog crate -> router-policy YAML -> frontend or EPP binary
```

- The policy crate owns the scoring and picking algorithm.
- The catalog gives each policy type a stable name.
- The YAML file creates named policy instances and supplies parameters.
- The frontend or EPP links the catalog at compile time.

Dynamo owns discovery, eligibility, queueing, validation, reservations, accounting, and metrics. A policy sees only eligible workers and returns one candidate row.

## Pick a Starting Point

| Crate | Use it for |
|---|---|
| `basic` | Multiple scorers and one picker work for every worker type |
| `disaggregated` | Prefill and decode workers need different components |
| `catalog` | You need to register policy types for configuration |
| `epp` | Worker selection runs in a standalone EPP |

The basic policy adds an active-request cost and an uncached-request cost for every worker. The disaggregated policy scores active prefill work plus uncached request blocks for prefill workers, and projected decode blocks for decode workers. Both own their picker implementation.

## 1. Create the Policy Crate

Add `dynamo-kv-router` from the same Dynamo checkout that builds the host process. Enable `standalone-selection`:

```toml
[dependencies]
dynamo-kv-router = { path = "/work/dynamo/lib/kv-router", features = ["standalone-selection"] }
serde = { version = "1", features = ["derive"] }
```

A policy that lives in the Dynamo workspace can use the workspace dependencies shown in [`basic/Cargo.toml`](basic/Cargo.toml).

## 2. Implement the Scorer and Picker

A scorer receives one eligible worker and returns a finite cost. Lower costs are better. A picker receives all scored rows and returns one row index.

The [basic policy](basic/src/lib.rs) is the shortest complete implementation. The [disaggregated policy](disaggregated/src/lib.rs) shows how one factory selects different component types from `worker_type`.

A policy can apply multiple scorers to every candidate. Dynamo calls them in order and adds their costs before the picker runs. The basic policy composes two scorers:

```rust
vec![Box::new(LeastBusyScorer), Box::new(UncachedBlocksScorer)]
```

For example, a worker with an active-request cost of `3` and an uncached-request cost of `5` reaches the picker with a total cost of `8`.

Declare each optional input group that a component reads:

```rust
fn required_worker_inputs(&self) -> WorkerInputs {
    WorkerInputs::LOAD
}
```

If a component needs both groups, use `WorkerInputs::CACHE | WorkerInputs::LOAD`. Do not request unused groups because Dynamo calculates and retains those columns for each eligible worker.

Keep these rules in the scorer and picker:

- Return an error instead of panicking.
- Return finite scorer costs.
- Treat candidate order as unspecified.
- Keep blocking I/O out of `score` and `pick`.
- Keep mutable policy state inside the factory-created policy.

## 3. Parse Parameters and Build the Factory

The provider runs at startup. Parse all parameters there and reject unknown fields:

```rust
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let _: Parameters = parameters.deserialize()?;

    Ok(Arc::new(|config, worker_type, _partition| {
        WorkerSelectionPolicy::new(
            config.clone(),
            worker_type,
            vec![Box::new(LeastBusyScorer)],
            Box::new(LowestCostPicker),
        )
    }))
}
```

Dynamo calls the returned factory once per routing partition. Use `worker_type` to choose prefill, decode, or standalone `select` behavior. Use the partition identity for distinct model or routing-group state.

## 4. Register the Policy Type

Expose one registration function from the policy crate:

```rust
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("least-busy", Arc::new(provider))
}
```

Choose a stable, unique type name. The name becomes part of the YAML contract.

Add the policy dependency and registration call to the catalog that ships with the host process. See [`catalog/src/lib.rs`](catalog/src/lib.rs).

## 5. Configure a Policy Instance

Create a YAML file outside the source tree:

```yaml
worker_selection:
  default: disaggregated-load
  instances:
    - name: least-busy
      type: least-busy
      parameters: {}
    - name: disaggregated-load
      type: disaggregated-load
      parameters: {}
```

- `type` selects a registered provider.
- `name` identifies one configured instance.
- `worker_selection.default` selects an instance at startup.
- `DYN_ROUTER_WORKER_SELECTION_POLICY` overrides the selected instance by name.
- The override value `default` selects Dynamo's built-in policy.

Unknown policy types, duplicate registrations, and invalid parameters stop startup.

## 6. Build and Test

Run these commands from the Dynamo repository root:

```bash
cargo test \
  -p dynamo-custom-policy-example-basic \
  -p dynamo-custom-policy-example-disaggregated \
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
DYN_ROUTER_WORKER_SELECTION_POLICY=least-busy \
  cargo run --release -p dynamo-custom-policy-example-epp
```

Standalone EPP supplies `select` as `worker_type` because it selects from one worker pool. A policy that branches on `worker_type` must handle `select`.

Follow the [standalone EPP guide](../../../docs/fern/pages/kubernetes/kv-aware-routing/vanilla-vllm-onramp.mdx) for discovery, KV events, tokenization, and Kubernetes resources.

## Before You Ship

- Build the policy against the same Dynamo revision as the frontend or EPP.
- Make sure that each component declares every input group that it reads.
- Check every parameter before the factory is created.
- Exercise each `worker_type` branch.
- Prove that scorer failures and invalid picker rows do not reserve a worker.
- Benchmark stateful or input-heavy policies at the expected worker count.

The [custom routing API reference](../../../docs/fern/pages/developer-guide/advanced-customizations/custom-worker-selection.mdx) lists the available context and worker signals.
