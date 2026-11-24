<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DGDR Output-Copier Image

This image adds the v1beta1 DGDR output-copier's shell utilities to the official
`registry.k8s.io/kubectl` image. The kubectl v1.37.2 base and static BusyBox
1.37.0-musl donor are pinned by multi-architecture manifest digest. BusyBox
provides `/bin/sh` with `pipefail`, `date`, `grep`, `awk`, `sed`, `tr`, `cat` and
`sleep`; kubectl and CA certificates come from the Kubernetes image. The default
user is UID/GID 1000, matching the profiling Job.

From the repository root, build and test either supported platform:

```bash
docker buildx build --platform linux/amd64 --load \
  -t dgdr-output-copier:local deploy/operator/output-copier
deploy/operator/output-copier/smoke-test.sh dgdr-output-copier:local linux/amd64
```

Repeat with `linux/arm64` for the other architecture. CI builds and tests both
platforms on native runners. The smoke test runs without network access, with a
read-only root filesystem and writable `/tmp`; it checks the shell pipelines
and uses the real kubectl client to generate a ConfigMap from profiler output.
It does not contact a Kubernetes API server.

For operator-wide image configuration, see
[#18201](https://github.com/ai-dynamo/dynamo/pull/18201). Default image selection
is discussed in [#19463](https://github.com/ai-dynamo/dynamo/pull/19463).
