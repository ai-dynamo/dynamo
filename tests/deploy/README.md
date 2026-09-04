<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment tests

This directory contains live-cluster pytest suites. `test_dgd.py` validates
example `DynamoGraphDeployment` manifests. `test_dgdr.py` validates the operator's
`DynamoGraphDeploymentRequest` API, and `test_dgdr_h100.py` contains its real-GPU
H100 support matrix.

The CPU DGDR lane uses mocker workers and needs a cluster with the Dynamo
operator installed and a planner image:

```bash
python -m pytest tests/deploy/test_dgdr.py \
  --namespace=dgdr-test \
  --image=registry.example/dynamo-planner:tag \
  -m gpu_0 -v -s
```

The GPU lane uses the `gpu_1` cases in `test_dgdr.py` and the H100 support matrix
in `test_dgdr_h100.py`. The latter additionally requires eight H100 GPUs and the
cluster-specific model access settings:

```bash
python -m pytest tests/deploy/test_dgdr_h100.py \
  --namespace=dgdr-test \
  --image=registry.example/dynamo-planner:tag \
  --dgdr-no-mocker \
  --dgdr-hf-token-secret=hf-token-secret \
  -m 'h100 and gpu_8' -v -s
```

Every pytest item owns its DGDR, profiling Job, output ConfigMap, generated DGD,
and DGD pods. Teardown stops DGDR reconciliation first and then waits for child
resources to disappear. CI also runs the suite in a per-run namespace and
deletes that namespace unconditionally.

`test_deploy_efa.py` and `test_deploy_efa_sglang.py` are the EFA lanes: each
deploys a 1P1D stack from `efa/` on an EFA-capable cluster (p5/H100) and proves
the prefill/decode KV transfer rode NIXL -> LIBFABRIC -> EFA rather than falling
back to UCX. They share their flow and assertions through `efa_utils.py`; what
differs per framework -- how LIBFABRIC is pinned, which worker initiates the
transfer, and therefore which NIXL and EFA adapter counters move -- lives in an
`EfaFrameworkProfile` in each test file. They are excluded from the
auto-discovered matrix and selected by the `framework_with_efa` marker:

```bash
python -m pytest tests/deploy/test_deploy_efa_sglang.py \
  -m framework_with_efa \
  --image=registry.example/sglang-runtime:tag-efa \
  --namespace=efa-test -v -s
```

The Go package at `deploy/operator/test/e2e` is a deprecated, manual Kind smoke
test for the legacy Kustomize installation path. New operator deployment tests
belong here.
