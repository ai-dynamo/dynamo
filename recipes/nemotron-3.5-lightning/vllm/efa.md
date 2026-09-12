<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EFA setup

AWS EFA-backed disaggregated vLLM recipes use NIXL LIBFABRIC for prefill/decode
transport. The manifests request `vpc.amazonaws.com/efa`, but they do not
install AWS EFA packages during pod startup because non-AWS clusters should not
run AWS-specific setup.

Prepare the complete matching EFA userspace stack before deployment, either in
the runtime image, through cluster bootstrap, or through a local deployment
overlay. Validate the final runtime from a pod that requests EFA resources:

```bash
kubectl exec -i -n "${NAMESPACE}" <worker-pod> -- \
  bash -s -- check < check-efa-userspace.sh
```

## Install helper

On AWS, `check-efa-userspace.sh install` can be used while preparing a derived
runtime image, cluster bootstrap layer, or local deployment overlay. Pin the
installer version and checksum:

```bash
EFA_INSTALLER_VERSION=1.49.0 \
EFA_INSTALLER_SHA256=cf2e9281a2328a243c76f911a490faed43ca0fecfe4733c25e34b2e92a32c309 \
  bash check-efa-userspace.sh install
```

If EFA is installed under `/opt/amazon/efa` but is not registered with
`ldconfig`, add these environment variables to the disaggregated worker
components before applying the manifest:

```yaml
- name: PATH
  value: /opt/amazon/efa/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
- name: LD_LIBRARY_PATH
  value: /opt/amazon/efa/lib:/opt/amazon/efa/lib64:/usr/lib/x86_64-linux-gnu
```

## Deployment-time wrapper

For deployment-time setup, use a wrapper entrypoint on the worker containers.
Run the install in the actual worker container before
`python3 -m dynamo.vllm`, so the installed `/opt/amazon/efa` and system
libraries are visible to vLLM:

```bash
#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
if [ ! -x /opt/amazon/efa/bin/fi_info ]; then
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl tar gzip
  EFA_INSTALLER_VERSION=1.49.0 \
  EFA_INSTALLER_SHA256=cf2e9281a2328a243c76f911a490faed43ca0fecfe4733c25e34b2e92a32c309 \
    bash /path/to/check-efa-userspace.sh install
fi

export PATH="/opt/amazon/efa/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi_info -p efa | head -80

exec "$@"
```

Mount the wrapper into both prefill and decode workers and move the original
`python3 -m dynamo.vllm ...` command into `args`:

```yaml
command:
- /opt/dynamo/efa/efa-entrypoint.sh
args:
- python3
- -m
- dynamo.vllm
# Keep the existing recipe arguments here.
volumeMounts:
- name: efa-entrypoint
  mountPath: /opt/dynamo/efa
  readOnly: true
volumes:
- name: efa-entrypoint
  configMap:
    name: <recipe-configmap>
    defaultMode: 0555
    items:
    - key: efa-entrypoint.sh
      path: efa-entrypoint.sh
```

Do not rely on a one-off init container that installs EFA into an `emptyDir`
mounted only at `/opt/amazon/efa`: the installer also installs ABI-matched
libraries and provider configuration under system paths such as
`/usr/lib/x86_64-linux-gnu`, and the final `fi_info -p efa` validation must run
inside a container that has been allocated EFA devices.
