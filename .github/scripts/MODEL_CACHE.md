<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Approved test model caches

The reviewed inventory is `.github/approved-test-models.json`. It starts with
the four model IDs in `tests.utils.constants.TEST_MODELS` and the adapter used
by the existing pre-merge LoRA tests. Revisions are full HF commit IDs.
The inventory excludes duplicate original/Metal, GGUF, ONNX and other
non-PyTorch exports to keep hydration within the test-node storage budget.

`populate-test-model-caches.yml` runs only from canonical main on the dedicated
publisher pool. It downloads one model at a time, hashes every snapshot file,
and publishes files followed by a completion index into **both** the PR store
and `approved_test_models`. An incomplete publication has no usable index.
The inventory's SHA-256 is part of each object namespace, so an inventory edit
requires new publication before tests can consume it. HF tokens stay in this
publisher; gated models still require the token holder's upstream access.

`hydrate-test-models` checks each file's size and SHA-256 and materializes a
standard HF_HOME layout under `/models`. It exports `MODELS_DIR`; the existing
pytest runner passes `--models-dir` and enables pytest's offline cache mode.
Missing assets and LoRA adapters fail rather than suppressing tests.

Canonical `shared-test.yml` opts in with the repository variable
`APPROVED_TEST_MODELS_ENABLED=true` after the approved store is populated.
This rollout enables hydration only for west jobs, including ARM jobs when
the caller selects east AMD64 runners. East jobs keep their existing model
source and receive no new S3 permissions.
The contributor lane always uses the separate PR store and never receives
the publisher identity. New test model IDs need explicit inventory review,
an immutable revision, and publication before those tests can run offline.

Local pytest remains unchanged unless `--models-dir` is supplied. To use S3
explicitly with a reader identity:

```bash
python .github/scripts/model-cache.py hydrate \
  --bucket dynamo-ci-210086341041-us-west-2-approved-test-models \
  --destination /tmp/approved-test-models
python -m pytest --models-dir=/tmp/approved-test-models tests/
```

The unit tests use tiny synthetic snapshots and real offline HF cache lookup;
they do not download model weights or contact AWS. GPU/LoRA canaries and
model inventory coverage for the selected test matrix remain rollout checks.
