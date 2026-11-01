---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Install Substrate
subtitle: Prepare a Kubernetes cluster to run agent sandboxes with Agent Substrate.
---

Install Agent Substrate's control plane, node runtime, and snapshot storage.
Workloads configure their own client identities, worker pools, and actor images
after installation. This procedure does not require Harbor, Pi, or a model server.

> [!WARNING]
> This integration is experimental and uses Substrate `v1.0.0-rc.1`, commit
> `f86fde9dfb5424c8d16f613de80469bdbef06541`. The S3 instructions below work
> around limitations in that release's installer.

## Prerequisites

- A Kubernetes cluster with permission to install controllers, CRDs, and the
  native certificate APIs and projections described below.
- Linux nodes compatible with your selected sandbox runtime and its assets.
- PostgreSQL, object storage, and network access configured according to the
  pinned [Substrate operator installation guide](https://github.com/agent-substrate/substrate/blob/v1.0.0-rc.1/docs/guides/operator-install.md).
- Git, Go 1.27 or newer, and `kubectl` on the installation workstation, with the
  intended Kubernetes context already selected.

Run commands from a working directory of your choice. Paths are relative to
that directory. Keep the installer configuration for future updates.

## Check cluster capabilities

Check that Kubernetes serves both native certificate APIs:

```bash
kubectl api-resources --api-group=certificates.k8s.io
```

Require `PodCertificateRequest` and `ClusterTrustBundle`, and their respective
Pod volume projections. Check API-server and kubelet feature configuration
with your cluster administrator. A server-side dry run is insufficient if
unsupported fields are silently pruned. Verify that the admitted Pod retains
both projections and that a real Pod receives its files.

The node architecture, sandbox runtime build, and workload image architecture
must match. Every node eligible for sandbox placement needs `atelet` and the
assets required by its sandbox runtime.

## Build the installation tools

Build the tools from the same release as the control-plane images:

```bash
mkdir -p deps/bin
git clone --branch v1.0.0-rc.1 https://github.com/agent-substrate/substrate deps/substrate
go -C deps/substrate build -o ../bin/ate-setup ./cmd/ate-setup
go -C deps/substrate build -o ../bin/kubectl-ate ./cmd/kubectl-ate
export PATH="$PWD/deps/bin:$PATH"
```

Keep this `bin` directory on `PATH` in shells where you manage Substrate or run
its tutorials. `ate-setup` installs the platform; `kubectl-ate` manages its API
objects.

## Configure the installation

Create `substrate-install.yaml` using the pinned upstream
[operator installation guide](https://github.com/agent-substrate/substrate/blob/v1.0.0-rc.1/docs/guides/operator-install.md).
That guide owns the database, storage, runtime asset, certificate issuance, and
platform-specific settings. Set these fields in your installation document:

```yaml
apiVersion: install.ate.dev/v1alpha1
kind: SubstrateInstall
namespace: ate-system
images:
  repo: ghcr.io/agent-substrate/substrate
  tag: v1.0.0-rc.1
```

This fragment selects the release; it is not a complete installation document.
Supply database credentials through the upstream installer's supported secret
settings. Configure your administrative identity using Substrate's
[authorization guide](https://github.com/agent-substrate/substrate/blob/v1.0.0-rc.1/docs/guides/authorization.md).

The router's `ingressAllowedSPIFFEIDs` setting controls which clients can
reach actors. It has no default: without an allowlist, the platform accepts no
actor ingress. Add approved client identities through your installation
configuration when you configure a workload. Preserve existing identities when
adding another client.

For Google Cloud Storage, follow the upstream bucket and transfer-service-account
setup. For an S3-compatible store, use the release-specific settings below.

### Use an S3-compatible snapshot store

In `v1.0.0-rc.1`, the installer assumes Google Cloud Storage on every cluster
that is not kind.
It refuses to deploy `ate-api-server` unless `storage.transferGSAName` and
`gcp.projectID` are set, because on GCS the object-store plugin impersonates
that service account to mint per-actor credentials. The S3 backend never reads
the value. Its plugin transfers with the Pod's own credentials and answers the
credential exchange as unimplemented. Give the two settings placeholder values
to pass the check:

```yaml
storage:
  snapshotLocation: s3://your-bucket/
  transferGSAName: unused
gcp:
  projectID: not-a-gcp-project
```

The deploy command below passes the current Kubernetes context explicitly,
which skips the installer's GKE credential lookup. Any S3-compatible service
works, including the small in-cluster RustFS from
upstream's `manifests/ate-install/kind/rustfs.yaml`. Create the bucket before
the first actor.

## Deploy the control plane

Pass the current Kubernetes context explicitly so the installer uses your
existing cluster credentials:

```bash
ate-setup deploy ate-system --context "$(kubectl config current-context)" \
  --config substrate-install.yaml
```

Omit or adapt optional Google `PodMonitoring` resources on clusters without
that API. For an existing installation, update its managed configuration
instead of replacing it with a new document.

### Configure S3 plugin credentials

Complete this step only when using S3-compatible snapshot storage.

The installer does not carry S3 endpoint or credential settings yet, so point
the object-store plugin at your store after the deploy. The plugin runs as the
`objectstore-plugin` init container of both the `ate-api-server` Deployment and
the `atelet` DaemonSet:

```bash
kubectl -n ate-system create secret generic ate-objectstore \
  --from-literal=AWS_ACCESS_KEY_ID=your-access-key \
  --from-literal=AWS_SECRET_ACCESS_KEY=your-secret-key
cat > objectstore-s3.json <<'EOF'
{"spec":{"template":{"spec":{"initContainers":[{"name":"objectstore-plugin","env":[
  {"name":"ATE_STORAGE_BACKEND","value":"s3"},
  {"name":"AWS_REGION","value":"us-east-1"},
  {"name":"AWS_ENDPOINT_URL","value":"http://rustfs.ate-system.svc:9000"},
  {"name":"AWS_S3_USE_PATH_STYLE","value":"true"},
  {"name":"AWS_ACCESS_KEY_ID","valueFrom":{"secretKeyRef":{"name":"ate-objectstore","key":"AWS_ACCESS_KEY_ID"}}},
  {"name":"AWS_SECRET_ACCESS_KEY","valueFrom":{"secretKeyRef":{"name":"ate-objectstore","key":"AWS_SECRET_ACCESS_KEY"}}}
]}]}}}}
EOF
kubectl -n ate-system patch deployment ate-api-server --patch-file objectstore-s3.json
kubectl -n ate-system patch daemonset atelet --patch-file objectstore-s3.json
kubectl -n ate-system rollout status deployment/ate-api-server --timeout=300s
kubectl -n ate-system rollout status daemonset/atelet --timeout=600s
```

Replace the endpoint and region with your store's values. Running
`ate-setup deploy` again rewrites both workloads, so reapply the patch after
each deploy.

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

## Verify the installation

Wait for the control plane and node runtime:

```bash
kubectl -n ate-system rollout status deployment/ate-api-server --timeout=600s
kubectl -n ate-system rollout status deployment/atenet-router --timeout=600s
kubectl -n ate-system rollout status deployment/ate-controller --timeout=600s
kubectl -n ate-system rollout status daemonset/atelet --timeout=600s
kubectl -n ate-system get pods -o wide
kubectl get clustertrustbundles
```

Check that `atelet` is ready on every node intended for sandbox placement and
that the admitted Pods retain their certificate projections and receive their
credential files. Readiness alone does not verify a workload's client
permissions, guest image, or durable snapshot restoration; check those when
configuring the workload.

For a coding-agent workload, continue with
[Run SWE-bench Verified with Harbor, Pi, and Substrate](../../use-cases/agents/harbor-substrate.mdx).
That tutorial configures its runner identity, guest image, worker capacity, and
request timeouts, then validates them with a task.
