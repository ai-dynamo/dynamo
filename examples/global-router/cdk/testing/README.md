<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Global Router tests

Testing owns workload recipes, environment configuration, request generation and results.
It consumes `GlobalStoresStack`, `GlobalLayerStack` and `PoolStack` from `lib/`. There is no
separate mock implementation of infrastructure. `legacy/` preserves the original combined
CPU/POC deployment for existing stacks.

## Two-region environment

Run commands from `cdk`. Copy `testing/environments/two-regions.example.json` to a
`*.local.json` file. Replace cluster references, image tags, model and resource names.
Choose unique namespaces and **both** table names for each independent environment.
The example IDs/images are placeholders, not a deployable preset.

The environment entrypoint selects one lifecycle at a time:

```bash
pnpm synth:test -c configPath=testing/environments/two-regions.example.json -c layer=stores
pnpm synth:test -c configPath=testing/environments/two-regions.example.json -c layer=global -c member=east
pnpm synth:test -c configPath=testing/environments/two-regions.example.json -c layer=pool -c member=east -c pool=blue
```

After completing phase 1 of the [three-phase deployment](../README.md#configure-and-deploy),
use `pnpm exec cdk diff` and then `pnpm exec cdk deploy` with
`--app 'node --require ts-node/register bin/test-environment.ts'` and the same context flags.
Phase 2 deploys stores once, then global services for each member (`east`, `west`).
Phase 3 deploys each member's pools. All examples select DynamoDB-backed registrations,
with no cluster-federation dependency. Supply private frontend URLs reachable from both regions,
or use `frontendEndpoint.sourceCidrs` for a pool-owned internal NLB whose address is registered
automatically. Supply private pod connectivity for router/relay state traffic. Routers register their pod
UID/IP automatically; select images containing the router-registration runtime change.
See the [endpoint requirements](../README.md#dynamodb-discovery-and-endpoint-requirements).
Changing a pool synthesizes no global services, storage, networking or compute.

`workloads/mocker.ts` produces a DGD `spec.components` recipe with a frontend and CPU mocker
workers, Kubernetes/pod discovery, TCP request plane, ZMQ events and serving-load publication.
An optional `computeGroup` schedules these components onto nodes labeled
`dynamo.nvidia.com/compute-group=<value>`. Otherwise normal scheduling applies; provision
sufficient CPU/memory separately. The chosen image must contain Dynamo frontend, mocker and
PoolRelay binaries/modules and be compatible with the installed operator.
Set `workload.runtimeVersionOverride` to the image's Dynamo semantic version when using a
digest or custom tag; the operator cannot infer the runtime version from an image digest.
GPU serving pools use `bin/deploy.ts` with a tested DGD recipe instead of this mocker factory.

## Request smoke check

From a host that can reach the private ingress, send one unary and one streaming chat request:

```bash
python3 testing/runners/smoke.py \
  --base-url http://YOUR-INTERNAL-ROUTER \
  --model Qwen/Qwen3-0.6B \
  --session smoke-session-1 \
  --output testing/results/smoke.json
```

The runner checks completion choices and the SSE terminal event, records timing and errors,
and returns nonzero on failure. `OPENAI_API_KEY` is sent as a bearer token if set. Results
are ignored by Git. Both requests share a session ID, but this smoke check does not establish
cross-region affinity or session continuity; those need routing/state observations.

For end-to-end validation, record evidence for discovery convergence, requests
through both regions, session affinity, pool add/remove, relay/router restart, regional
failure and recovery, and blue/green pool replacement. Keep deployment-specific evidence
in ignored `results/` or `experiments/` directories; reusable runners belong in `runners/`.
Private campaign results are not included in the public source distribution.

## Cleanup

Destroy selected pool stacks first and allow their registration leases to expire,
then destroy regional global stacks. Keep the foundation/shared-services stacks for reuse.
The stores are protected and retained: deleting an environment's application stacks does
not remove its DynamoDB data or ongoing table costs. Retiring tables/replicas requires an
explicit storage cleanup after all consumers are gone.

For the original combined deployment, use [legacy instructions](legacy/README.md) and
[CPU fleet instructions](../CPU_FLEET.md); do not point those releases at the new stacks.
