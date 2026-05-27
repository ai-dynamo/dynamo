---
name: dynamo-interconnect-check
description: Validate that a Dynamo deployment's NIXL/UCX/NCCL interconnect is ready for disaggregated serving over RDMA/NVLink. Use after recipe-runner brings a deployment up (especially disagg/multi-node) to confirm the KV transport is correct; use troubleshoot for diagnosing already-failed pods.
---

# Dynamo Interconnect Check

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Goal

Confirm that the transport disaggregated serving depends on actually works. A
deployment can pass an endpoint smoke test while disagg is silently wrong: if
NIXL/UCX cannot reach the peer worker over RDMA or NVLink, KV transfer falls
back to a slow or broken path. Catch that with read-only checks before trusting
a disagg deployment or its benchmark numbers.

This skill is read-only. It never mutates the cluster and never prints secrets.

## When To Use

- After `dynamo-recipe-runner` deploys a **disagg** or multi-node recipe.
- Before reporting disagg throughput/latency, so numbers reflect the real
  transport.
- When agg works but disagg is slow, hangs, or returns wrong output and you
  suspect the fabric rather than the model.

For diagnosing pods that are already crashing or unschedulable, use
`dynamo-troubleshoot` first.

## Workflow

### 1. Check Transport Env Vars On The Recipe

```bash
python3 scripts/check_interconnect.py env recipes/<model>/<framework>/<mode>
```

Reports which NIXL/UCX/NCCL transport variables are set and flags
disagg-critical ones (e.g. `UCX_TLS`, `UCX_NET_DEVICES`, `NCCL_IB_HCA`) that are
absent. Missing here is only a warning — they may be baked into the image — so
confirm with the node and NIXL checks. See
`references/interconnect-env-vars.md` for what each variable does.

### 2. Check Node Capabilities

Locally on a GPU node, or inside a running worker pod:

```bash
python3 scripts/check_interconnect.py node \
  --namespace "${NAMESPACE}" --pod <worker-pod>
```

Probes (read-only) for: InfiniBand devices and Active links, GPUDirect RDMA
(`nvidia_peermem`), GDRCopy, and NVLink in the GPU topology. Missing tools are
reported as `skipped`, not failures.

### 3. Validate NIXL Reachability

```bash
python3 scripts/check_interconnect.py nixl \
  --namespace "${NAMESPACE}" --pod <worker-pod>
```

Looks for NIXL test tooling in the pod and surfaces the exact next step to run a
pairwise prefill↔decode transfer test. A full cross-pod transfer test requires
two scheduled GPU pods on the fabric.

## Output Contract

Each check returns `ok` / `warn` / `fail` / `skipped` with a one-line detail,
plus a rolled-up verdict on disagg transport readiness. Report:

- transport env vars present vs. disagg-critical ones missing
- RDMA / GPUDirect / NVLink capability status
- whether NIXL reachability was validated, and the next command if not
- a clear statement of whether disagg can be trusted, or what to fix first

## References

- `references/interconnect-env-vars.md` — NIXL/UCX/NCCL env var catalog and IB
  capability checklist.
- Use `scripts/check_interconnect.py` for all read-only checks.
