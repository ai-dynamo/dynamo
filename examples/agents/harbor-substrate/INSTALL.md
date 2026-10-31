<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Install the pinned Substrate pieces for the
[Harbor tutorial](../../../docs/fern/pages/use-cases/agents/harbor-substrate.mdx).
Run these commands from the example directory. Set `EXAMPLE_DIR`,
`KUBE_CONTEXT`, and `IMAGE_REPO` as described there.

The Harbor adapter talks to Substrate directly. It creates and deletes actors
through `kubectl-ate`, and it reaches the guest daemon inside each actor
through the Substrate router. There is no environment service to deploy.

## Check cluster capabilities

Use Substrate `v1.0.0-rc.1`, commit
`f86fde9dfb5424c8d16f613de80469bdbef06541`. Read its
[operator installation guide](https://github.com/agent-substrate/substrate/blob/v1.0.0-rc.1/docs/guides/operator-install.md)
for database, storage, runtime assets, certificate issuance, and platform settings.

Check that Kubernetes serves both native certificate APIs:

```bash
kubectl --context "$KUBE_CONTEXT" api-resources --api-group=certificates.k8s.io
```

Require `PodCertificateRequest` and `ClusterTrustBundle`, and their respective
Pod volume projections. Check API-server and kubelet feature configuration
with your cluster administrator. A server-side dry run is insufficient if
unsupported fields are silently pruned. Verify that the admitted Pod retains
both projections and that a real Pod receives its files.

The CPU node, gVisor build, and task image architecture must match. The supplied
SWE-bench task uses `linux/amd64`. Every node selected by a worker pool needs
`atelet` and the runtime assets.

## Install or check the control plane

Build the pinned tools:

```bash
mkdir -p deps/bin
git clone --branch v1.0.0-rc.1 https://github.com/agent-substrate/substrate deps/substrate
(cd deps/substrate && go build -o "$EXAMPLE_DIR/deps/bin/ate-setup" ./cmd/ate-setup)
(cd deps/substrate && go build -o "$EXAMPLE_DIR/deps/bin/kubectl-ate" ./cmd/kubectl-ate)
export PATH="$EXAMPLE_DIR/deps/bin:$PATH"
```

Prepare `.local/substrate-install.yaml` using the upstream installation guide.
Select `ghcr.io/agent-substrate/substrate` and `v1.0.0-rc.1` for the runtime
images. The router admits only clients whose SPIFFE ID it is given. Include the
runner's identity:

```yaml
ingressAllowedSPIFFEIDs:
  - spiffe://cluster.local/ns/ate-env/sa/harbor-runner
```

Use your installation's trust domain if it differs from `cluster.local`.
Preserve existing permitted identities. Install the configured control plane:

```bash
ate-setup deploy ate-system --context "$KUBE_CONTEXT" \
  --config .local/substrate-install.yaml
```

An S3-compatible snapshot store also needs its endpoint, credentials, and storage
plugin configured. `substrate-s3-install.patch` only removes the installer's
GCS service-account requirement for `s3://` snapshot paths. Apply it before
building `ate-setup` if that validation blocks an otherwise configured S3
installation. It does not configure the storage backend. Omit or adapt optional
Google `PodMonitoring` resources on clusters without that API.

For an existing installation, merge the identity into the router's allowlist
through its normal deployment configuration. The installer renders it as the
`--ingress-allowed-spiffe-ids` argument of the `atenet-router` container. Do
not replace the whole shared install.

## Build the guest image

`ate-env-guest` is the daemon that runs commands and moves files inside an
actor. It comes from the env repository and does not depend on the Substrate Go
module, so it builds from the pinned commit without changes:

```bash
git clone https://github.com/agent-substrate/env deps/env
git -C deps/env checkout 281e86085bb0b6d3090ff3fa82d722cf0d9f2f0b
(
	cd deps/env
	umask 022
	KO_DOCKER_REPO="$IMAGE_REPO/env-guest" ko build --bare \
	  --platform=linux/amd64 ./cmd/ate-env-guest
)
```

Save the printed digest reference as `GUEST_IMAGE` in `.local/config.env`. The
actor template mounts this image as a volume at `/ate`, so the task image stays
untouched. Configure image-pull credentials separately if the registry is
private. The same checkout supplies the Python client the runner installs
later; the adapter uses its generated gRPC stubs for the guest.

## Create the namespace, atespace, and access policy

Run the Substrate commands as an administrator with global ownership. An
atespace-scoped owner cannot create an atespace:

```bash
kubectl --context "$KUBE_CONTEXT" create namespace ate-env
kubectl-ate --context "$KUBE_CONTEXT" create atespace ate-env
kubectl-ate --context "$KUBE_CONTEXT" create access-policy \
  -a ate-env -f runner-access-policy.yaml
```

The policy grants the runner's ServiceAccount token the editor role on the
atespace. That covers creating and deleting actors, creating egress policies,
and listing templates. The ServiceAccount itself is created with the runner
Pod in the tutorial. If these objects already exist, inspect and reuse them.
For an existing access policy, preserve its current bindings and update with
its `uid` and `version`.

Two identities, one ServiceAccount: `kubectl-ate` sends the projected token,
whose principal is `user:system:serviceaccount:ate-env:harbor-runner`. The
adapter's guest calls present the projected pod certificate, whose SPIFFE ID is
the one in `ingressAllowedSPIFFEIDs`. The access policy covers the first. The
router allowlist covers the second.

## Align request timeouts

The rc1 router defaults to a five-minute route timeout, which also bounds guest
process-output streams. In the `atenet-router` container's deployment arguments,
set `--route-timeout=60m` for the unmodified 3,000-second task deadlines:

```bash
kubectl --context "$KUBE_CONTEXT" -n ate-system edit deployment atenet-router
kubectl --context "$KUBE_CONTEXT" -n ate-system rollout status \
  deployment/atenet-router --timeout=300s
```

Preserve the other arguments. Make this change before running actors. If you
choose shorter task deadlines, choose a route budget longer than the longest
command. The parking retry budget is a separate setting and does not suspend
actors to free capacity.

## Resolve low-port permission failures when needed

If a Substrate API or worker Pod cannot bind port 443, the installation may
need a Pod-local sysctl:

```yaml
spec:
  template:
    spec:
      securityContext:
        sysctls:
          - name: net.ipv4.ip_unprivileged_port_start
            value: "0"
```

Apply it only to the affected deployment through the installation's supported
configuration. The rc1 WorkerPool API does not expose Pod sysctls. When a manual
patch to its generated Deployment is necessary, record it and reapply after
recreation. The generated deployment has the same name as its WorkerPool.
This is a Pod setting, not a host sysctl change.
