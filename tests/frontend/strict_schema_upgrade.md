<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Strict-schema upgrade acceptance

This manual test compares one unchanged Chat Completions request between an
older and a newer SGLang runtime image. The release pair is chosen at run time,
not hardcoded to 1.5/1.6. Each image runs its own frontend and worker using the
Python SGLang processor and the same pinned Qwen3-0.6B snapshot as the
[component compatibility suite](../deploy/n2/README.md).

The request has `strict: true` and a `unit` property omitted from `required`.
The baseline must accept it and generate tokens. The target is tested first
with the compatibility flag disabled, then enabled, in fresh deployments.
A default-mode change from HTTP 200 to HTTP 400 is an **upgrade failure**, even
if enabling `DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION` restores success.
Both target results are collected before reporting that failure. The existing
same-version processor e2es separately verify the migration contract.

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
- Both images must support the same Python SGLang processor and pinned model.
  A framework-independent frontend image cannot supply that processor.
  When the target implements the migration switch (for example via #17163),
  the enabled stage verifies recovery. Older targets may ignore the variable;
  successful inference alone does not prove that they implement the switch.
- Use Linux amd64 Kubernetes with a compatible Dynamo operator, GPU capacity,
  and a shared model-cache PVC. The PVC must contain `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, prepared as described in the
  component compatibility guide. The aggregate template also references
  `hf-token-secret`; create it in the test namespace.

## Run

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
identical explicitly supplied digests fail setup. Dedicated options are
registered in the frontend conftest; no deploy conftest or CI-filter changes
are required. Normal core CI still applies
to frontend tests, but this manual comparison is not automatically executed by
PR or N-2 workflows.

Three deployments run serially using the existing `ManagedDeployment` helper.
They share cached model files but each loads a worker; readiness checks may
send additional warmup requests. Leftover Pods stop the next deployment.
This is a static release comparison, not a rolling upgrade or N-2 matrix.

Under `DYN_TEST_OUTPUT_PATH`, artifacts include the shared request, per-stage
raw responses, requested image references, model revision and `comparison.json`.
Normal deployment artifacts retain Pod manifests with actual resolved image
IDs (`status.containerStatuses[].imageID`) and process logs, including when
ordinary tags are used. Tags may move between deployments; the recorded image
IDs identify what each stage actually ran. Partial results survive failure;
skipped/interrupted runs do not establish compatibility. CPU-only checks are not GPU upgrade evidence.
