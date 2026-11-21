<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frozen environment-sync predecessor

These fixtures were captured by running the actual renderers and spec-hash
implementation from commit `8f1fdc45f043356fa2e15bfcd28f576ce44db555`, immediately
before the list-order-sensitive resource-sync change in `a97bc86907`.
They are independent of the implementation under test. Ordinary tests read them;
they do not regenerate a predecessor with the current renderer or hash function.

`provenance.json` records the exact source revision, capture-template checksum,
fixture checksums, and the predecessor's persisted bookkeeping hashes. The
`missing` case has no origin annotation, not an empty annotation value.

Each file retains complete, unnormalized objects:

- A frontend-only DGD input, its generated child DCD, and its generated Grove PCS.
- Standalone matching-origin DCD inputs and their generated Deployment and LWS.
- Original owner references, selectors, names, complete PodSpecs, env order, and
  the predecessor's `nvidia.com/last-applied-hash` annotations.

The capture harness assigns deterministic test UIDs, generation `7`, and its
matching last-applied-generation bookkeeping before serialization. These are
render-only fixtures, not claims about a live API server or a released operator
history. No fields are removed or normalized. No existing upgrade history,
goldens, or normalizers were changed.

The sync tests compare the complete predecessor specs with the current result,
preserve identity and generation assertions, verify an unchanged second
reconcile, and check that an order-only edit changes new-origin workloads but
not older-origin workloads. The frozen 1.8 case checks the hash bookkeeping
transition without a spec change. Separate conflicting standalone/inherited
origin cases remain current-code policy tests; they are not labelled as
independent predecessor-lowering evidence.

## Semantic Review

The frontend-only input declares `B=$(A)` before `A=x`. The predecessor's DCD
keeps the authored list, while old-origin rendered PodSpecs sort `A` before `B`;
1.8 PodSpecs retain `B` before `A`. Every injected variable remains present.
The independent serialized fixtures freeze that complete result, not only the
two illustrative variables. Matching-origin standalone Deployment and LWS
inputs exercise the same ordering change with unchanged runtime images and
commands. No expected resource field was changed to make the candidate pass.

A negative-control Go overlay removed only the injected `DYN_COMPONENT` entry
from the current frontend renderer. The frozen 1.7 Grove case failed at its
complete-spec equality assertion during the upgrade phase (`14` env entries
versus `13`), while the unmodified renderer passes. This demonstrates that a
lowering regression cannot be copied into both sides of the comparison.
The overlay modifies no tracked production source or fixture.

## Reproduce the capture

Run from the repository root, using a new, ignored artifact directory. The
captured production source must remain unchanged; the only added source is the
supplied test harness:

```bash
capture_revision=8f1fdc45f043356fa2e15bfcd28f576ce44db555
capture_artifacts=deploy/operator/bin/review-baseline/8f1fdc45f
capture_output="$(pwd)/deploy/operator/internal/controller/testdata/environment-sync"
mkdir -p "$capture_artifacts"
git archive "$capture_revision" deploy/operator | tar -x -C "$capture_artifacts"
cp deploy/operator/internal/controller/testdata/environment-sync/capture_test.go.tmpl \
  "$capture_artifacts/deploy/operator/internal/controller/environment_sync_capture_test.go"
cd "$capture_artifacts/deploy/operator"
GOWORK=off DYNAMO_ENV_SYNC_CAPTURE_DIR="$capture_output" \
  go test ./internal/controller -run '^TestCaptureEnvironmentSyncPredecessor$' -count=1 -v
```

Recapture is not a routine golden update. Review any changed field and its
upgrade consequences before accepting different bytes, under the parent
[golden review rules](../README.md).
