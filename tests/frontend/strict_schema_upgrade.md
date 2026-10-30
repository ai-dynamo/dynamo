<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Strict-schema upgrade acceptance

This test compares one unchanged Chat Completions request between an older
and a newer SGLang runtime image. CI selects N-1/N-2 baselines from the community
release catalog; manual runs can choose any increasing release pair. Each image
runs its own frontend and SGLang worker. Both the native Rust (`dynamo`) and
Python SGLang processors use the same pinned Qwen3-0.6B snapshot as the
[component compatibility suite](../deploy/n2/README.md).

The request has `strict: true` and a `unit` property omitted from `required`.
The baseline must accept it and generate tokens. The target is tested first
with the compatibility flag disabled, then enabled, in fresh deployments.
A default-mode change from HTTP 200 to HTTP 400 is an **upgrade failure**, even
if enabling `DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION` restores success.
Both target results are collected before reporting that failure. The existing
same-version processor e2es separately verify the migration contract.

## N-2 CI

`cross-version-compatibility.yml` collects this test alongside the existing
component suite in one serial pytest session. With `--run-strict-schema-upgrade`
and no explicit image pair, `--image` supplies the candidate SGLang runtime.
The existing `version_matrix` helper selects the historical worker images from
`tests/deploy/n2/releases.json`, using `N2_RELEASE_LINE` or
`DEFAULT_RELEASE_LINE`. Each historical runtime supplies both its frontend and
worker; the candidate runtime does the same for both target stages. Native
Rust runs declare tool/reasoning parsers on the worker; Python SGLang runs
configure them on the frontend.

Two baselines and two processors produce four upgrade comparisons, adding
twelve serial deployments to the eight existing mixed-component deployments.
They use the same prepared snapshots, native
`--model-cache-pvc` / `--model-cache-mount` options, and deployment artifacts.
A compatibility assertion failure allows the next release comparison to run;
a deployment or cleanup failure stops the session because isolation is no
longer established. The workflow's outer cleanup tears down the vCluster.

The default-mode upgrade assertion is required: an old HTTP 200 becoming a
candidate HTTP 400 fails N-2 CI even when the compatibility stage succeeds.
See the [component compatibility guide](../deploy/n2/README.md) to run the same
combined suite locally.

## Image selection and prerequisites

- Choose two version-tagged SGLang runtime images, for example
  `registry/sglang-runtime:1.5.0` and `registry/sglang-runtime:1.7.0-custom`.
  Ordinary version tags are accepted, matching the community N-2 suite.
  Optionally append `@sha256:<64 hex digits>` to either image for immutable
  selection. When both images specify a digest, the digests must differ.
  The test does not select `latest` or infer images from the checked-out source.
- The target's numeric runtime version must be greater than the baseline's.
  Any increasing patch, minor or major release pair is accepted. Version
  parsing uses the `X.Y.Z` prefix of the image tag, matching the existing deploy
  helper; CI suffixes do not order two builds of the same release.
- Both images must support the selected processor and pinned model. Python
  SGLang runs require the SGLang runtime on the frontend; a framework-independent
  frontend image cannot supply that processor.
  When the target implements the migration switch (for example via #17163),
  the enabled stage verifies recovery. Older targets may ignore the variable;
  successful inference alone does not prove that they implement the switch.
- Use Linux amd64 Kubernetes with a compatible Dynamo operator, GPU capacity,
  and a shared model-cache PVC. The PVC must contain `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, prepared as described in the
  component compatibility guide. The aggregate template also references
  `hf-token-secret`; create it in the test namespace.

## Explicit image comparison

Set `BASELINE_IMAGE` and `TARGET_IMAGE` to the desired version-tagged images
(with optional digests), then run from the repository root with deployment-test
dependencies:

```bash
: "${BASELINE_IMAGE:?Set the older version-tagged SGLang runtime image}"
: "${TARGET_IMAGE:?Set the newer version-tagged SGLang runtime image}"
python3 -m pytest -xvv -n 0 \
  tests/frontend/test_strict_schema_upgrade.py \
  --run-strict-schema-upgrade \
  --schema-upgrade-baseline-image "$BASELINE_IMAGE" \
  --schema-upgrade-target-image "$TARGET_IMAGE" \
  --namespace default \
  --schema-upgrade-model-cache-pvc model-cache \
  --schema-upgrade-model-cache-mount /models
```

Without `--run-strict-schema-upgrade`, the test explicitly skips and creates no
deployments. With the switch enabled, missing image/namespace/PVC arguments,
missing version tags, malformed optional digests, non-increasing versions or
identical explicitly supplied digests fail setup. Explicit image options select
one image pair instead of the N-1/N-2 matrix, tested with both processors.
Add `-k rust` or `-k sglang` to select one processor. The dedicated cache options
override the shared N-2 options; the mount defaults to `/models`. The N-2
workflow passes the opt-in switch, while ordinary frontend runs still skip
this test.

Each processor comparison runs three serial deployments using the existing
`ManagedDeployment` helper. They share cached model files but each loads a
worker; readiness checks may send additional warmup requests. Leftover Pods
stop the next deployment.
Explicit image selection runs a static release comparison, not a rolling
upgrade.

Under `DYN_TEST_OUTPUT_PATH`, artifacts include the shared request, per-stage
raw responses, requested image references, processor, model revision and
`comparison.json`. Normal deployment artifacts retain Pod manifests with actual
resolved image IDs (`status.containerStatuses[].imageID`) and process logs, including when
ordinary tags are used. Tags may move between deployments; the recorded image
IDs identify what each stage actually ran. Partial results survive failure;
skipped/interrupted runs do not establish compatibility. CPU-only checks are
not GPU upgrade evidence.
