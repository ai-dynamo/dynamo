<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom Worker Selection Policies

This example shows how to link custom Rust worker-selection policies into the Dynamo frontend or a standalone Endpoint Picker Provider (EPP).

## Crates

| Crate | Purpose |
|---|---|
| `basic` | Optionally requires cache overlap, then scores active requests and picks the lowest cost. |
| `disaggregated` | Uses separate scorer and picker types for prefill and decode workers. |
| `catalog` | Registers both policy types with Dynamo. Add new policy crates here. |
| `epp` | Runs the catalog in a standalone EPP binary. |

Each policy crate keeps configuration, factory creation, and registration in `lib.rs`. The filter, scorer, and picker algorithms live in focused source files.

The disaggregated policy keeps the algorithm deliberately small:

| Worker type | Score | Pick |
|---|---|---|
| Prefill | Active prefill tokens | Lowest score |
| Decode | Projected decode blocks | Lowest score |

The Python frontend supplies `prefill` or `decode` as `worker_type`. Standalone EPP supplies `select` because it selects from one worker pool.

## Build and Test

Run these commands from the Dynamo repository root:

```bash
cargo test -p dynamo-custom-policy-example-basic \
  -p dynamo-custom-policy-example-disaggregated \
  -p dynamo-custom-policy-example-catalog
cargo build -p dynamo-custom-policy-example-epp
```

## Configure a Policy

Create a policy file outside the repository. This example selects the disaggregated policy:

```yaml
worker_selection:
  default: disaggregated-load
  instances:
    - name: least-busy
      type: least-busy
      parameters: {}
    - name: cache-affinity
      type: least-busy
      parameters:
        min_effective_overlap_blocks: 8
    - name: disaggregated-load
      type: disaggregated-load
      parameters: {}
```

Change `default` to `least-busy` to use the basic policy without a filter. Change it to `cache-affinity` to require at least eight effective cache-overlap blocks. Set a positive `min_effective_overlap_blocks` value. Set `DYN_ROUTER_WORKER_SELECTION_POLICY` to an instance name to override the YAML default. Use `default` to select Dynamo's built-in policy.

The cache-affinity filter is a hard requirement. If no eligible worker meets the threshold, Dynamo returns HTTP 503.

## Run With the Python Frontend

The Python extension uses the dependency alias `dynamo-worker-selection-policy-catalog`. Point that alias at this example catalog in the checkout you build:

```bash
export DYNAMO_DIR="$(pwd)"

cargo add \
  --manifest-path "$DYNAMO_DIR/lib/bindings/python/Cargo.toml" \
  --optional \
  --rename dynamo-worker-selection-policy-catalog \
  --path "$DYNAMO_DIR/examples/router/custom-policy-example/catalog" \
  dynamo-custom-policy-example-catalog
```

Build the extension with the custom catalog and start the frontend:

```bash
cd "$DYNAMO_DIR/lib/bindings/python"
CARGO_TARGET_DIR="$DYNAMO_DIR/target" maturin develop --uv --features custom-policy

cd "$DYNAMO_DIR"
uv pip install -e .
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config /path/to/worker-selection.yaml
```

For a private catalog, run the same `cargo add` command with your catalog path and package name. The alias stays the same.

## Run With EPP

The example EPP already links the example catalog. Use the basic policy for its single worker pool:

```bash
DYN_EPP_MODE=standalone \
DYN_ROUTER_POLICY_CONFIG=/path/to/worker-selection.yaml \
DYN_ROUTER_WORKER_SELECTION_POLICY=least-busy \
cargo run --release -p dynamo-custom-policy-example-epp
```

Follow the [standalone EPP guide](../../../docs/fern/pages/kubernetes/kv-aware-routing/vanilla-vllm-onramp.mdx) for discovery, KV events, tokenization, and Kubernetes setup. Use this binary instead of the stock EPP executable. Policies that branch on `worker_type` must handle `select` for this path.

## Add a Policy

Start with a new Rust library crate. The `basic` and `disaggregated` crates are runnable references, not templates.

1. Implement filters, scorers, and a picker. Declare each signal with `required_worker_inputs`.
2. Parse and validate policy parameters in the provider.
3. Return a factory that builds the policy for each worker type.
4. Register a unique policy type name.
5. Add the crate dependency and `register` call to the catalog that ships with your frontend or EPP.

The [custom routing guide](../../../docs/fern/pages/developer-guide/advanced-customizations/custom-worker-selection.mdx) explains the traits, available signals, factory lifecycle, and registration flow.

`keep`, `score`, and `pick` run in the scheduler queue actor. Keep them free of blocking I/O. Return finite costs and a valid candidate row.

For each candidate, filters run in declaration order before scorers. Candidate order and callback order across different candidates are unspecified. Dynamo normally scores kept candidates directly. It buffers them only when the built-in scorer needs a minimum active prefill load across the filtered set.
