<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Install the pinned Substrate environment integration for the
[Harbor tutorial](../../../docs/fern/pages/use-cases/agents/harbor-substrate.mdx).
Run these commands from the example directory. Set `EXAMPLE_DIR`,
`KUBE_CONTEXT`, and `IMAGE_REPO` as described there.

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
images. Include the runner's identity in `ingressAllowedSPIFFEIDs`:

```yaml
ingressAllowedSPIFFEIDs:
  - spiffe://cluster.local/ns/ate-env/sa/ate-env-api
```

Use your installation's trust domain if it differs from `cluster.local`, and
update `env-access-policy.yaml` to match. Preserve existing permitted identities.
Install the configured control plane:

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

For an existing installation, merge the required identity and settings through
its normal deployment configuration. Do not replace the whole shared install.

## Build the environment service and guest

The pinned env source predates Substrate rc1. Apply the included compatibility
patch, which updates its Go API bindings and authenticated transport:

```bash
git clone https://github.com/agent-substrate/env deps/env
git -C deps/env checkout 281e86085bb0b6d3090ff3fa82d722cf0d9f2f0b
git -C deps/env apply "$EXAMPLE_DIR/env-rc1-compat.patch"
(
	cd deps/env
	umask 022
	KO_DOCKER_REPO="$IMAGE_REPO/env-api" ko build --bare \
	  --platform=linux/amd64,linux/arm64 ./cmd/ate-env-api
	KO_DOCKER_REPO="$IMAGE_REPO/env-guest" ko build --bare \
	  --platform=linux/amd64 ./cmd/ate-env-guest
)
```

Save the printed digest references as `ENV_API_IMAGE` and `GUEST_IMAGE` in
`.local/config.env`. These are images you build and publish to your registry.
Configure image-pull credentials separately if the registry is private.

## Create the namespace and atespace

Run the Substrate commands as an administrator with global ownership. An
atespace-scoped owner cannot create an atespace:

```bash
kubectl --context "$KUBE_CONTEXT" create namespace ate-env
kubectl --context "$KUBE_CONTEXT" -n ate-env create serviceaccount ate-env-api
kubectl-ate --context "$KUBE_CONTEXT" create atespace ate-env
kubectl-ate --context "$KUBE_CONTEXT" create access-policy \
  -a ate-env -f env-access-policy.yaml
```

If these objects already exist, inspect and reuse them. For an existing access
policy, preserve its current bindings and update with its `uid` and `version`.
The namespace, atespace, and owner binding must exist before the env API starts.

Render and deploy the environment service:

```bash
set -a
source .local/config.env
set +a
envsubst '${ENV_API_IMAGE}' < env-api.yaml > .local/env-api.yaml
kubectl --context "$KUBE_CONTEXT" apply -f .local/env-api.yaml
kubectl --context "$KUBE_CONTEXT" apply -f env-network-policy.yaml
kubectl --context "$KUBE_CONTEXT" -n ate-env rollout status \
  deployment/ate-env-api --timeout=300s
```

Keep `api.ate-system.svc:443` and `atenet-router.ate-system.svc:443` consistent
with the service certificates. The projected identity bundle authenticates
those downstream connections. The env Service's port 7777 is plaintext inside
the cluster; do not expose it publicly. NetworkPolicy enforcement requires a
network plugin that supports it.

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
