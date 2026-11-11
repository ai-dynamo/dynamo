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

`test_slime_kuberay.py` runs the Slime reinforcement-learning tutorial end to
end with the tutorial's manifests in `slime/`. A setup Job stages Qwen3-0.6B and
GSM8K on a shared volume, a DGD serves rollouts from two SGLang engines, and a
KubeRay RayJob trains 10 GRPO steps against them. The test then runs the
tutorial's checks:

- The RayJob succeeds.
- The frontend routes all 640 rollout requests round-robin.
- Each engine loads all 11 weight updates.
- The engines' token log-probabilities match the trainer's.

No CI job selects this test. It runs against the current kubectl context and
needs:

- Three free GPUs, with 16 CPUs and 200 GiB of memory free on the Ray head's
  node.
- The Dynamo platform and the KubeRay operator.
- A ReadWriteMany StorageClass or an existing ReadWriteMany claim.

```bash
python -m pytest tests/deploy/test_slime_kuberay.py \
  --namespace=slime-e2e \
  --slime-storage-class=your-rwx-storage-class \
  -v -s
```

If the namespace does not exist, the test creates it and deletes it at the end.
In an existing namespace, the test first deletes any of the tutorial's objects
that an earlier run left behind, and it keeps the namespace. The setup Job,
driver, and DGD logs go to `test_slime_tutorial/` under `DYN_TEST_OUTPUT_PATH`,
which defaults to `dynamo_tests/` in the system temporary directory. Other
options:

| Option | Effect |
|---|---|
| `--slime-pvc` | Mounts an existing ReadWriteMany claim, such as one bound to a static volume, instead of creating one. The test does not delete it. Replaces `--slime-storage-class`. |
| `--image`, `--frontend-image` | Replace the DGD's images, which default to the tutorial's Dynamo 1.6.1 images. `--image` sets every component, then `--frontend-image` sets the frontend. |
| `--slime-image` | Replaces the Slime image in the setup Job and the Ray head. |
| `--slime-gpu-toleration` | Adds a toleration for this taint key to every Pod that requests a GPU, for clusters that do not add one themselves. |

New operator deployment tests belong in this directory.
