<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native vllm-rs worker rendered by 1.6.0

This independent compatibility snapshot holds a Grove worker that runs `vllm-rs`
directly: a `runtime` native sidecar (`restartPolicy: Always`) and a `main`
container with command `/usr/local/bin/vllm-rs`. The DGD omits
`spec.backendFramework`. The released 1.6.0 operator created `create.yaml` with
mutating admission enabled, so the DGD carries operator origin `1.6.0`. It did not
detect a backend from the `vllm-rs` command, resolved the worker as `noop`, and
persisted worker hash `e2a112ed` on the DGD and on the worker clique.

On a single-node Grove clique the `noop` and `vllm` backends render the same pod,
so the resolved backend is visible only through the worker hash. Detecting
`vllm-rs` as vLLM, as #15923 did before #18589 reverted it, resolves `vllm`, moves
the worker hash, and rolls the worker on an operator-only upgrade (#17543).

`provenance.json` records the pinned release revision, the rendered harness
digest, the observation digest, the worker hash, the worker clique hash, that the
persisted DGD still omits `backendFramework`, and the final input digest.
`../../compatibility.lock.json` records the active input digest and last released
reconciler alongside the other compatibility inputs.

Reproduce from the repository root with Go, PyYAML, GitHub CLI access, and envtest
Kubernetes binaries:

```bash
KUBEBUILDER_ASSETS=/absolute/path/to/envtest/binaries \
  python3 deploy/operator/hack/capture_compatibility.py native-vllm-rs --output /new/staging/directory
```

`TestClusterOperatorUpgrade/native-vllm-rs` seeds the snapshot through the shared
permissive historical phase, runs production takeover, checks the independent
safety assertion, and compares the complete owned graph with the reviewed
`output.yaml`. The golden differs from the released graph only in the worker
status: the current operator does not predict `servedModelName` from the engine
flags of a native-sidecar worker, so the field is absent. Before and after
takeover it also recomputes the worker hash with
the current operator and requires it to equal the persisted DGD annotation and the
worker clique label. That check fails on an operator that resolves `vllm-rs` as
vLLM, independent of the golden. This is a CPU admission/controller/API-server
check, not an inference test.
