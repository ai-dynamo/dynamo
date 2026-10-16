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

The initial catalog selects operator 1.4.0 and 1.5.0 as historical release lines
for candidate 1.6. It covers ordinary legacy workers, a parseable custom runtime
1.5 tag under operator 1.4, and already-suffixed operator 1.5 workers. It is not
a complete backend/provider/runtime matrix. The custom-tag profile reproduces
the same runtime-version hash transition as the reported 1.2-to-1.5 incident,
using a supported 1.4-operator/1.5-runtime starting pair.

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

The intentional-change test modifies a real worker field, injects a failed PCS
write, retries reconciliation and checks that the new worker generation converges
without advancing the DGD hash before the failed write is retried.

The standalone pull-request workflow executes these CPU checks independently of
the GPU deploy-test switch. Aggregate N-2 inventory/merge enforcement, live
operator replacement, service selectors and historical-runtime inference remain
later work in #16623. A passing CPU result does not prove that real Pods were not
recreated or that historical runtimes can serve traffic.
