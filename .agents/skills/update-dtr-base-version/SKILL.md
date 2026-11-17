---
name: update-dtr-base-version
description: >
  Moves the Dynamo Triton Runtime (DTR) image to a new upstream Triton Inference Server release (nvcr.io/nvidia/tritonserver:<xx.yy>-py3) by updating the pinned runtime_image_tag in container/context.yaml and every doc that quotes it.
  Use when asked to bump, update, or move the Triton base or runtime image to a new xx.yy release.
license: Apache-2.0
user-invocable: true
disable-model-invocation: true
argument-hint: "<xx.yy>  (for example 26.11)"
metadata:
  author: NVIDIA
  tags:
  - dynamo
  - triton
  - container
  - release
---

# Update the DTR Base Version

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Moves the Dynamo Triton Runtime image to a new upstream Triton release.
Takes the new release as `xx.yy`, for example `/update-dtr-base-version 26.11`.
Below, `NEW` is that value and `OLD` is the release pinned today.

## Step 1: Validate the input and find OLD

The requested version is: `$ARGUMENTS`

That is `NEW`. If it is empty, ask the user for the new `xx.yy` release before doing anything else; do not guess it from the latest Triton release or from the working tree.
Strip a trailing `-py3` if the user gave one.
`NEW` must then match `^[0-9]{2}\.[0-9]{2}$`; stop and ask if it does not.

```bash
NEW='<NEW from above>'
OLD=$(sed -n 's/^ *runtime_image_tag: *\([0-9][0-9]\.[0-9][0-9]\)-py3.*/\1/p' container/context.yaml)
echo "OLD=$OLD NEW=$NEW"
```

`OLD` comes from the `triton:` section of [`container/context.yaml`](../../../container/context.yaml); check that the match is the line under `triton.cuda13.4`, not another framework's.
If `OLD == NEW`, report that nothing needs to change and stop.

## Step 2: Find the CUDA version and driver for NEW

The docs table records the CUDA version and minimum driver of each Triton release.
Get both for `NEW` from the [Triton release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/) or the [framework support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).
If the image can be pulled, this also works:

```bash
docker run --rm nvcr.io/nvidia/tritonserver:${NEW}-py3 bash -c 'echo CUDA=$CUDA_VERSION; echo DRIVER=$CUDA_DRIVER_VERSION'
```

If neither source answers, ask the user rather than guessing; do not carry the old values forward silently.

## Step 3: Edit the files

Make these edits, and only these. Replace `OLD` with `NEW` in each tag; leave every `base_image_tag` (`26.02-cuda13.1-devel-ubuntu24.04`) alone:
it is the build toolchain, pinned to CUDA 13.1 on purpose.

| File                                                                                                                                             | Change                                                                                                                                                                          |
| :----------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`container/context.yaml`](../../../container/context.yaml)                                                                                      | `triton.cuda13.4.runtime_image_tag: OLD-py3` → `NEW-py3`                                                                                                                        |
| [`container/README.md`](../../../container/README.md)                                                                                            | Triton build block: `(defaults to OLD-py3)` and `--build-arg RUNTIME_IMAGE_TAG=OLD-py3` → `NEW-py3`                                                                             |
| [`docs/.../backends/triton/overview.md`](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/triton/overview.md) | Release table row: tag → `nvcr.io/nvidia/tritonserver:NEW-py3`, CUDA and driver columns → the Step 2 values                                                                     |
| same file, "Configuring the Triton Version"                                                                                                      | `(default OLD-py3)` → `NEW-py3`                                                                                                                                                 |
| same file, build example                                                                                                                         | `--build-arg RUNTIME_IMAGE_TAG=OLD-py3` → `NEW-py3`, and the image tag `-t dynamo:triton-<anything>` → `-t dynamo:triton-NEW` (this tag has drifted before, so match any value) |

Then search for stragglers, and fix only references to the *default* Triton runtime tag:

```bash
git grep -nE "${OLD//./\\.}(-py3)?" -- container components/src/dynamo/triton docs/fern/pages/developer-guide tests deploy .github
```

Do **not** change:

- **Release notes** under `docs/fern/pages/reference/general/releases/`.
  They record what shipped in a past release and stay as written.
  The bump belongs in the *next* release's notes, which the release process writes.
- **The `cuda13.4` key** in `context.yaml` and the `triton.cuda13.4` references in the docs, even when `NEW` ships a different CUDA. The key is how `container/render.py` selects the Triton config; renaming it is a separate change across the renderer and CI.
- **Feature minimums** such as `26.08+` in [`components/src/dynamo/triton/main.py`](../../../components/src/dynamo/triton/main.py) (the `log_callback` API).
  They name the first release with a feature, not the default.
- **QA examples** such as `26.07` in `components/src/dynamo/triton/tests/qa/`.
  They are docstring examples of the release format.

## Step 4: Verify

```bash
git diff --stat
git diff
git grep -nE "${OLD//./\\.}-py3" -- container docs/fern/pages/developer-guide   # expect no output
container/render.py --framework=triton --target=runtime --output-short-filename
grep -n 'RUNTIME_IMAGE_TAG=' container/rendered.Dockerfile                      # expect NEW-py3
pre-commit run --files $(git diff --name-only)
```

The diff should touch exactly `container/context.yaml`, `container/README.md`, and the Triton `overview.md`, unless the straggler search in Step 3 found a real default-tag reference elsewhere.
Remove `container/rendered.Dockerfile` if it is not tracked.

If the user asks for a full build check, and the machine has Docker and network access:

```bash
docker build --network=host -t dynamo:triton-${NEW} -f container/rendered.Dockerfile .
docker run --rm dynamo:triton-${NEW} python3 -c "import tritonserver, dynamo; print('ok')"
```

## Step 5: Report

Tell the user `OLD` → `NEW`, the CUDA and driver values used and where they came from, the files changed, and anything from the straggler search that was left alone and why.
Do not commit unless asked; when asked, sign off (`git commit -s`) with a title such as `chore(triton): move DTR runtime image to NEW-py3`.
