<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Historical Grove upgrade replay

Part of #16623; implementation work is tracked in #16624.

These CPU tests feed release-generated DGD and PCS snapshots through the current
Grove workload reconciler. They exercise worker-generation planning, rendering,
PCS synchronization and DGD hash projection, without overriding the hash-change
decision. They compare raw historical worker PodSpecs, labels, annotations and
replica counts with every candidate reconciliation round. Raw input prevents a
newer Go API type from silently discarding an old Pod field before comparison.

## Operator upgrade paths with fixed runtimes

Every unchanged-workload profile upgrades only the operator. Its DGD spec and
runtime image string stay identical before and after takeover. The catalog's
`release` identifies the old operator that generates the baseline; `image`
identifies the runtime held fixed throughout that profile. `candidate` is the
operator code checked out for the CI run, currently on the 1.6 release line.

| Profile | Old operator | Candidate operator | Fixed runtime image | Historical worker state |
| --- | --- | --- | --- | --- |
| `operator-1.4-to-candidate_sglang-runtime-1.4-fixed` | 1.4.0 | 1.6 candidate | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0` | Unsuffixed |
| `operator-1.4-to-candidate_sglang-runtime-1.5-custom-fixed` | 1.4.0 | 1.6 candidate | `registry.example.com/dynamo/sglang-runtime:v1.5.0-9aa3-repro` | Unsuffixed |
| `operator-1.5-to-candidate_sglang-runtime-1.5-fixed` | 1.5.0 | 1.6 candidate | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.0` | Already suffixed |

The first row tests a 1.6 candidate operator taking over existing runtime 1.4
workers. In the second row, the custom runtime 1.5 tag is already present under
operator 1.4; it is never upgraded by the replay. That profile detects whether
the candidate's interpretation of the same tag changes the hash inputs and
worker templates. It reproduces the runtime-version hash transition behind the
reported operator 1.2-to-1.5 incident, rather than that exact version path.

Operator 1.4 and 1.5 are the initial N-2 and N-1 source baselines for candidate
1.6. These three profiles are not a complete backend/provider/runtime matrix.

## Generate immutable snapshots

From the repository root, ensure the catalog's release commits are present in
the object database (fetch those exact commits from ai-dynamo/dynamo if needed),
then run:

~~~sh
python3 deploy/operator/hack/generate_upgrade_fixtures.py
python3 deploy/operator/hack/generate_upgrade_fixtures.py --check
~~~

The generator exports released source into temporary directories and adds a
capture test. Hashes and workload PodSpecs come from that release's production
functions. The 1.5 adapter uses its production Grove suffix helper. DGD/PCS UIDs,
generation and ownership are deterministic test stand-ins for API-server metadata;
these are not snapshots from a live cluster. The generator uses Go 1.26.8 for
reproducible source-level captures and does not pull images or load models.

The lock file records SHA-256 for the catalog, input, capture adapter and generated
snapshots. CI regenerates all snapshots from released source using --check.
Do not regenerate historical expected state with candidate rendering, or edit a
snapshot to match a failing candidate. Review changes to the input, catalog or
release adapter as changes to the compatibility contract.

## Run the candidate contract

From deploy/operator:

~~~sh
DYNAMO_UPGRADE_REPLAY_OUTPUT_DIR=/tmp/operator-upgrade-replay \
  go test ./internal/controller -run '^TestOperatorUpgradeReplay' -count=1 -v
~~~

The unchanged-workload test asserts no worker template or identity changes,
including the custom runtime-tag incident. Known failures remain failures; there
is no skip, xfail or automatic baseline-update mode. Hash equality is diagnostic,
not the assertion: a safe comparison-hash migration is allowed if worker identity
and templates remain unchanged.

The separate intentional-change test is a control case: after loading the
operator 1.5 baseline, it explicitly changes a worker environment variable while
keeping the runtime image fixed. It injects a failed PCS write, retries
reconciliation and checks that the new worker generation converges without
advancing the DGD hash before the failed write is retried. This distinguishes
an intentional DGD edit from the unchanged-workload upgrade profiles above.

The standalone pull-request workflow executes these CPU checks independently of
the GPU deploy-test switch. Aggregate N-2 inventory/merge enforcement, live
operator replacement, service selectors and historical-runtime inference remain
later work in #16623. A passing CPU result does not prove that real Pods were not
recreated or that historical runtimes can serve traffic.
