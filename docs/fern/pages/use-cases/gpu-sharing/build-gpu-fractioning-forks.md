---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Build the KAI-Scheduler and GPU Fractioning Forks
sidebar-title: Build the GPU Fractioning Forks
subtitle: Build, push, and install the two forks that give each model a hard compute and memory share of a GPU
---

This guide builds two forks from source, pushes the images to a registry you control, and installs them on a Kubernetes cluster. Together they give each pod a hard share of a GPU's streaming multiprocessors (SMs) and memory, enforced by CUDA Multi-Process Service (MPS) v3 namespaces. You need them for the "KAI + GPU fractions" experiment in [Interpreting GPU Sharing Results](gpu-sharing-results.mdx): 16 Qwen3-4B models on 8 A100 GPUs, each model capped at 50% of the SMs and 50% of the memory of one GPU.

When the install is done, a pod that asks for `gpu-fraction: "0.5"` lands on a shared GPU in its own MPS namespace. The MPS server enforces the pod's SM share, and the driver enforces its memory cap, so the pod cannot raise either one. The next step is [deploying the models](deploy-kai-gpu-fractions.md).

## What You Are Building

| Fork | What it provides | Images | Helm chart |
|---|---|---|---|
| [KAI-Scheduler](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) | Scheduler, binder, admission controller, and operator. Tracks per-GPU compute as well as memory, and writes the `gpu-compute.portion` and `gpu-memory.request` annotations on each fractional pod | 15 service images | `deployments/kai-scheduler` |
| [kai-gpu-fractioning](https://github.com/kai-scheduler/gpu-fractioning/pull/147) | Node agents. `fractiond` (an NRI plugin) injects the caps into each container, `mpsd` runs the MPS control daemon and creates one namespace per container, `metricsd` exports metrics, and `operator` manages them | 4 images: `operator`, `mpsd`, `fractiond`, `metricsd` | `operator/charts` |

```text
pod (gpu-fraction: "0.5")
  │
  ▼
KAI-Scheduler ── picks a GPU, writes gpu-compute.portion / gpu-memory.request / gpus.devices
  │
  ▼
kai-gpu-fractioning
  ├── fractiond   reads the annotations, injects env + the namespace's MPS pipe directory
  └── mpsd        creates the MPS namespace, sets its active-thread percentage on the server
```

> [!IMPORTANT]
> The two forks are a matched pair. The scheduler writes annotations that only this fork of the node agent reads, and the node agent provisions namespaces that only this fork of the scheduler accounts for. Build and deploy both from the commits below. Do not mix a fork image with the upstream release chart, or the reverse. The symptom of a mismatch is a pod that schedules but receives no GPU. See [Troubleshooting](#troubleshooting).

### Pinned Revisions

Both changes are open pull requests against the upstream repositories. The results published for this experiment were produced by exactly these commits, so check them out by SHA rather than following the pull request.

| Repository | Pull request | Commit |
|---|---|---|
| `KAI-Scheduler` | [kai-scheduler/KAI-Scheduler#2368](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) | `44d1d0a966465c69b1fd0d24b40b9a1ebb73699e` |
| `kai-gpu-fractioning` | [kai-scheduler/gpu-fractioning#147](https://github.com/kai-scheduler/gpu-fractioning/pull/147) | `af0544e4dbd1d8e03ca14fc1e06d6265fabed99b` |

## Prerequisites

### Build Machine: KAI-Scheduler

| Requirement | Version | Notes | References |
|---|---|---|---|
| Docker with `buildx` | Recent | The Makefile compiles each service inside a `golang:1.26.3-bookworm` builder container and then builds the image with `docker buildx build` | [Install Docker Engine](https://docs.docker.com/engine/install/), [Docker Build and Buildx](https://docs.docker.com/build/concepts/overview/), [Builders](https://docs.docker.com/build/builders/) |
| Go | 1.26.3 (`go.mod`) | Only needed if you build outside Docker. The Makefile does not use a host Go toolchain | [Go downloads](https://go.dev/dl/), [Installing Go](https://go.dev/doc/install), [`go.mod`](https://github.com/kai-scheduler/KAI-Scheduler/blob/44d1d0a966465c69b1fd0d24b40b9a1ebb73699e/go.mod) |
| GNU `make` | Any recent | `make build` builds all 15 services for amd64 and arm64, then assembles the image for the host architecture only | [GNU Make](https://www.gnu.org/software/make/) |
| `helm` | 3.x or 4.x | Used for `helm dependency build` and the install. The chart rendered cleanly with Helm 4.1 | [Installing Helm](https://helm.sh/docs/intro/install/) |
| `kubectl` | Within one minor version of your cluster | | [Install kubectl](https://kubernetes.io/docs/tasks/tools/), [kubectl version skew](https://kubernetes.io/releases/version-skew-policy/#kubectl) |
| `git` | Any recent | | [Git downloads](https://git-scm.com/downloads) |
| Network access | | Docker Hub (`golang`), `nvcr.io` (`nvcr.io/nvidia/distroless/go:v3.2.1`, the runtime base image), and `ghcr.io` (`helm dependency build` pulls the `gpu-fractioning` subchart that `Chart.yaml` declares) | [`golang` on Docker Hub](https://hub.docker.com/_/golang), [`Dockerfile`](https://github.com/kai-scheduler/KAI-Scheduler/blob/44d1d0a966465c69b1fd0d24b40b9a1ebb73699e/Dockerfile), [`Chart.yaml`](https://github.com/kai-scheduler/KAI-Scheduler/blob/44d1d0a966465c69b1fd0d24b40b9a1ebb73699e/deployments/kai-scheduler/Chart.yaml) |
| A container registry you can push to and your cluster can pull from | | Referred to below as `$REGISTRY` | [What is a registry?](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-registry/), [`docker image push`](https://docs.docker.com/reference/cli/docker/image/push/) |

### Build Machine: kai-gpu-fractioning

| Requirement | Version | Notes | References |
|---|---|---|---|
| Docker | Recent | Plain `docker build` and `docker push`. `buildx` is needed only for the multi-architecture `docker-buildx` targets | [Install Docker Engine](https://docs.docker.com/engine/install/), [Docker Build and Buildx](https://docs.docker.com/build/concepts/overview/), [Builders](https://docs.docker.com/build/builders/) |
| Go | 1.26.4 (`go.mod`) | Images build inside `golang:1.27.x` containers, so a host toolchain is not required for them. The `operator/Makefile` calls `go` while evaluating variables, so keep Go on your `PATH` to avoid a `go: No such file or directory` warning | [Go downloads](https://go.dev/dl/), [Installing Go](https://go.dev/doc/install), [`go.mod`](https://github.com/kai-scheduler/gpu-fractioning/blob/af0544e4dbd1d8e03ca14fc1e06d6265fabed99b/go.mod) |
| GNU `make` | Any recent | | [GNU Make](https://www.gnu.org/software/make/) |
| `helm` | 3.x or 4.x | | [Installing Helm](https://helm.sh/docs/intro/install/) |
| `git` | Any recent | | [Git downloads](https://git-scm.com/downloads) |
| Network access | | Docker Hub (`golang`, `nvidia/cuda:13.3.1-base-ubuntu24.04` for `mpsd`) and `nvcr.io` (`nvcr.io/nvidia/distroless/go:v4.1.3`, the runtime base image for the other three) | [`golang` on Docker Hub](https://hub.docker.com/_/golang), [`nvidia/cuda` on Docker Hub](https://hub.docker.com/r/nvidia/cuda), [`mpsd` Dockerfile](https://github.com/kai-scheduler/gpu-fractioning/blob/af0544e4dbd1d8e03ca14fc1e06d6265fabed99b/fractioning-manager/mpsd/build/Dockerfile), [operator Dockerfile](https://github.com/kai-scheduler/gpu-fractioning/blob/af0544e4dbd1d8e03ca14fc1e06d6265fabed99b/operator/Dockerfile) |

The images target `linux/amd64` by default (`PLATFORM ?= linux/amd64`), which matches A100 and H100 nodes. Override `PLATFORM` if you build on another architecture.

### Cluster and Nodes

Check every row on a GPU node before you install anything. The defaults of a stock GPU Operator install are not enough for compute enforcement.

| Requirement | Detail | Check | References |
|---|---|---|---|
| NVIDIA driver r615 or newer (CUDA 13.4) | Provides the MPS v3 control daemon with `namespace` support. This is not the GPU Operator's default driver (the chart defaults to 595.91.07), so pin it. This guide used 615.71.09 | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` prints `615.x` or newer, and `nvidia-cuda-mps-control -p 3 namespace --help` lists `create`, `delete`, `list`, `set`, and `get`. With a GPU Operator-managed driver, the MPS binary is in the driver container: run the second command as `kubectl -n gpu-operator exec ds/nvidia-driver-daemonset -- nvidia-cuda-mps-control -p 3 namespace --help` | [Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html), [CUDA apt repository (Ubuntu 24.04)](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/), [MPS v3 Interface](https://docs.nvidia.com/deploy/mps/latest/mpsv3-interface.html) |
| NVIDIA GPU Operator v26.7.1 or newer | The `gpu-fractioning` chart refuses to start below `gpuOperator.minimumVersion` (default `v26.7.1`). v26.7.1 ships the container toolkit's `apply-cuda-memory-limits` hook, which turns the memory cap into a driver-enforced limit. v26.7.0 also works if you set that value to `v26.7.0`, but the memory cap then rests only on the `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` variable that `fractiond` injects into the container's environment, which a process inside the container can change | `helm list -n gpu-operator` | [Installing the GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html), [Release Notes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/release-notes.html), [Platform Support](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/platform-support.html) |
| containerd 2.0 or newer with NRI enabled, or CRI-O with NRI | `fractiond` is an NRI plugin. Without NRI nothing is injected into containers | `containerd --version`, and `ls -l /var/run/nri/nri.sock` shows the socket | [containerd releases](https://github.com/containerd/containerd/releases), [NRI Support in containerd](https://github.com/containerd/containerd/blob/main/docs/NRI.md), [CRI-O `enable_nri`](https://github.com/cri-o/cri-o/blob/main/docs/crio.conf.5.md), [NRI project](https://github.com/containerd/nri) |
| `nvidia.com/gpu.present=true` node label | The `mpsd` and `fractiond` DaemonSets select GPU nodes by this label. The GPU Operator sets it; a node without it gets no daemons and reports no error | `kubectl get nodes -l nvidia.com/gpu.present=true` lists every GPU node | [GPU Operator installation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) |
| `nvidia.com/gpu.memory` node label | KAI's binder reads each GPU's memory size from this label to compute a pod's memory cap. GPU Feature Discovery, part of the GPU Operator, sets it. Without it, fractional pods stay `Pending` and only a binder event explains why | `kubectl get nodes -L nvidia.com/gpu.memory` shows a value in MiB for every GPU node | [GPU Feature Discovery labels](https://github.com/NVIDIA/k8s-device-plugin/blob/main/docs/gpu-feature-discovery/README.md) |
| `nvidia` RuntimeClass | Needed by GPU pods for driver libraries, and by KAI's reservation pod, which calls NVML | `kubectl get runtimeclass nvidia` | [Kubernetes RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/), [GPU Operator installation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) |
| `nvidia-fabricmanager` | Required on multi-GPU NVLink and NVSwitch nodes such as HGX A100. Match the driver branch | `systemctl is-active nvidia-fabricmanager` | [Fabric Manager User Guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html) |
| Host memory for `mpsd` | `mpsd` holds one CUDA server context per GPU at about 50 MiB of host memory each. This fork's default pod limit is `1Gi`, enough for 16 GPUs. Raise it on denser nodes | `kubectl -n gpu-fractioning get ds -o yaml` after install, under `resources.limits.memory` | [MPS documentation](https://docs.nvidia.com/deploy/mps/index.html), [`mpsDaemon` chart value](https://github.com/kai-scheduler/gpu-fractioning/blob/af0544e4dbd1d8e03ca14fc1e06d6265fabed99b/operator/charts/values.yaml) |
| Cluster-admin | The charts create CRDs, ClusterRoles, ClusterRoleBindings, PriorityClasses, and a second namespace, `kai-resource-reservation` | The five `kubectl auth can-i create ...` checks below | [Kubernetes RBAC user-facing roles](https://kubernetes.io/docs/reference/access-authn-authz/rbac/#user-facing-roles), [`kubectl auth can-i`](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_auth/kubectl_auth_can-i/) |

```bash
for r in customresourcedefinitions clusterroles clusterrolebindings priorityclasses namespaces; do
  printf '%s: ' "$r"; kubectl auth can-i create "$r" --all-namespaces
done
```

#### Installing the r615 Driver

All the packages and images below exist publicly; choose the path that matches how the cluster manages drivers.

- **Host-installed driver (Ubuntu 24.04, NVIDIA CUDA apt repository).** The `nvidia-driver-pinning-615` package exists only in the NVIDIA CUDA repository, so add that repository first. Then install the branch pin, then the driver and, on NVSwitch systems, Fabric Manager:

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install nvidia-driver-pinning-615
  sudo apt-get install nvidia-driver-open nvidia-fabricmanager
  sudo reboot
  ```

  If a different `nvidia-driver-pinning-*` package is already installed, remove it first, or the 615 pin does not take effect and `nvidia-driver-open` resolves to the older branch. If the driver install appears to do nothing, check `apt-cache policy nvidia-driver-open` for an `/etc/apt/preferences.d/` file that pins the NVIDIA origin to a negative priority.

- **GPU Operator-managed driver.** Pass the version when you install or upgrade the operator:

  ```bash
  helm upgrade --install gpu-operator nvidia/gpu-operator \
    -n gpu-operator --create-namespace \
    --version v26.7.1 \
    --set driver.version=615.71.09
  ```

  The driver image `nvcr.io/nvidia/driver:615.71.09-ubuntu24.04` is published. Add the chart repository first with `helm repo add nvidia https://helm.ngc.nvidia.com/nvidia`.

## Step 1: Clone and Check Out the Pinned Commits

```bash
mkdir kai-forks && cd kai-forks

git clone https://github.com/kai-scheduler/KAI-Scheduler.git
git -C KAI-Scheduler fetch origin pull/2368/head
git -C KAI-Scheduler checkout 44d1d0a966465c69b1fd0d24b40b9a1ebb73699e

git clone https://github.com/kai-scheduler/gpu-fractioning.git kai-gpu-fractioning
git -C kai-gpu-fractioning fetch origin pull/147/head
git -C kai-gpu-fractioning checkout af0544e4dbd1d8e03ca14fc1e06d6265fabed99b
```

Confirm that both trees are clean, so no local edit is baked into an image:

```bash
git -C KAI-Scheduler status --porcelain          # prints nothing
git -C kai-gpu-fractioning status --porcelain    # prints nothing
```

## Step 2: Build and Push the Images

Set the registry and one tag for both forks. The tag records the pairing by naming both commits:

```bash
export REGISTRY=registry.example.com/team/kai-fork   # replace with a registry you can push to
export TAG=smshare-44d1d0a-af0544e
```

> [!WARNING]
> Always pass an explicit tag. `KAI-Scheduler` defaults `DOCKER_TAG` to `0.0.0`, and both Helm charts fall back to their `Chart.AppVersion` (`0.0.0` for the scheduler, `0.1.0` for the node agents) when an image tag is empty. A forgotten override does not fail. It deploys the wrong code, and you then benchmark the wrong thing.

### kai-gpu-fractioning

```bash
cd kai-gpu-fractioning
make docker-build docker-push \
  DOCKER_REPO_BASE="$REGISTRY/kai-gpu-fractioning" \
  VERSION="$TAG"
cd ..
```

This builds and pushes `operator`, `mpsd`, `fractiond`, and `metricsd` as `$REGISTRY/kai-gpu-fractioning/<name>:$TAG`. For a multi-architecture build, run `make docker-buildx DOCKER_REPO_BASE="$REGISTRY/kai-gpu-fractioning" VERSION="$TAG" DOCKER_BUILD_PLATFORM=linux/amd64,linux/arm64`, which pushes directly. Pass `DOCKER_REPO_BASE` and `VERSION` again: without them the target tags the images for the upstream `ghcr.io/kai-scheduler` registry with a `git describe` version. It also needs a `docker-container` buildx builder and QEMU emulation for the non-native architecture. Install the emulator with `docker run --privileged --rm tonistiigi/binfmt --install arm64` (or `amd64` on an Arm host), create a dedicated builder with `docker buildx create --name kai-multiarch --driver docker-container`, and prefix the `make` command with `BUILDX_BUILDER=kai-multiarch` so your default builder is not changed. A `docker-container` builder runs in its own network namespace, so it cannot push to a registry on `localhost`; use a registry it can reach over the network.

### KAI-Scheduler

`make build` uses your current buildx builder. Check which one that is before you build:

```bash
docker buildx ls    # the current builder is marked with *; note its DRIVER
```

> [!NOTE]
> With the default `docker` driver, images built without `--push` stay in your local Docker daemon. With a `docker-container` builder they stay only in the build cache: `make` still exits 0 and prints `No output specified with docker-container driver`, but no image appears in `docker images`. A registry on `localhost` is also unreachable from inside the builder container, so the push fails with `connection refused`. In either case, prefix the command with `BUILDX_BUILDER=<builder>`, naming a builder that `docker buildx ls` lists with the `docker` driver, usually `default`.

```bash
cd KAI-Scheduler
make build \
  DOCKER_REPO_BASE="$REGISTRY/kai-scheduler" \
  DOCKER_TAG="$TAG" \
  DOCKER_BUILDX_ADDITIONAL_ARGS=--push
cd ..
```

To build without pushing, drop `DOCKER_BUILDX_ADDITIONAL_ARGS=--push` and select a `docker` driver builder so the images land in your local Docker daemon:

```bash
cd KAI-Scheduler
BUILDX_BUILDER=default make build \
  DOCKER_REPO_BASE="$REGISTRY/kai-scheduler" \
  DOCKER_TAG="$TAG"
cd ..
```

> [!NOTE]
> In this Makefile `make push` is only an alias for `make build`: it has no recipe of its own and does not push. Pushing happens because `DOCKER_BUILDX_ADDITIONAL_ARGS=--push` is passed to `docker buildx build`. Without it, `make build push` leaves the images in your local Docker daemon only. The images match your host architecture. Set `DOCKER_BUILD_PLATFORM=linux/amd64` when you build on an Arm machine for x86 nodes.

The build compiles 15 services in sequence, each for both architectures, so the first run takes a while. Go caches are kept in `~/.cache/go-build-docker-gocache` and `~/.cache/go-build-docker-gopath`, and the build leaves a `builder:1.26.3-bookworm` image in your local Docker daemon. Remove them when you no longer need fast rebuilds.

### Verify the Images Were Pushed

```bash
for svc in operator mpsd fractiond metricsd; do
  docker buildx imagetools inspect "$REGISTRY/kai-gpu-fractioning/$svc:$TAG" >/dev/null \
    && echo "ok   kai-gpu-fractioning/$svc" || echo "MISSING kai-gpu-fractioning/$svc"
done

for svc in podgrouper scheduler binder resourcereservation snapshot-tool scalingpod \
    nodescaleadjuster podgroupcontroller queuecontroller fairshare-simulator admission \
    operator time-based-fairshare-simulator numa-placement-exporter helm-hooks; do
  docker buildx imagetools inspect "$REGISTRY/kai-scheduler/$svc:$TAG" >/dev/null \
    && echo "ok   kai-scheduler/$svc" || echo "MISSING kai-scheduler/$svc"
done
```

Every line must print `ok`: four node-agent images and fifteen scheduler images.

## Step 3: Install KAI-Scheduler

Fetch the chart dependency, then install with the fork images. The `binder.runtimeClassName` value gives KAI's GPU reservation pod the `nvidia` RuntimeClass at install time. Without it, the reservation pod cannot load NVML and every fractional pod stays `Pending` for a reason that is invisible in the workload's own events.

```bash
cd KAI-Scheduler
helm dependency build ./deployments/kai-scheduler

helm upgrade --install kai-scheduler ./deployments/kai-scheduler \
  -n kai-scheduler --create-namespace \
  --set global.gpuSharingMode=NvFractions \
  --set global.registry="$REGISTRY/kai-scheduler" \
  --set global.tag="$TAG" \
  --set binder.runtimeClassName=nvidia
cd ..
```

- `global.gpuSharingMode` is case-sensitive: `NvFractions`.
- Leave `global.nvFractions.set` at its default, `false`. Setting it to `true` installs the upstream `kai-gpu-fractioning` release from `ghcr.io` as a subchart, which is not your fork. You install the fork as its own release in Step 4. `helm dependency build` still has to download that subchart, because Helm checks that every declared dependency is present.
- For a private registry, add `--set 'global.imagePullSecrets[0].name=<secret>'` and create the secret in `kai-scheduler` first.

Verify before moving on. Wait for the deployments first, because `helm upgrade --install` returns before the pods are up:

```bash
kubectl -n kai-scheduler wait --for=condition=Available deploy --all --timeout=300s
helm status kai-scheduler -n kai-scheduler                   # STATUS: deployed
kubectl -n kai-scheduler get pods                            # all Running or Completed
kubectl -n kai-scheduler get deploy \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}'
kubectl get config.kai.scheduler kai-config -o jsonpath='{.spec.global.gpuSharingMode}{"\n"}'   # NvFractions
```

Until you finish Step 4, `kai-config` reports `DependenciesFulfilled=False` with `GpuFractioningConfig not found`. That is expected at this point.

Every image must show your `$REGISTRY` and `$TAG`. Do not continue with a pod that is not running.

## Step 4: Install kai-gpu-fractioning

```bash
cd kai-gpu-fractioning
helm upgrade --install gpu-fractioning ./operator/charts \
  -n gpu-fractioning --create-namespace \
  --set images.operator.repository="$REGISTRY/kai-gpu-fractioning/operator" \
  --set images.fractiond.repository="$REGISTRY/kai-gpu-fractioning/fractiond" \
  --set images.metricsd.repository="$REGISTRY/kai-gpu-fractioning/metricsd" \
  --set images.mpsd.repository="$REGISTRY/kai-gpu-fractioning/mpsd" \
  --set images.operator.tag="$TAG" \
  --set images.fractiond.tag="$TAG" \
  --set images.metricsd.tag="$TAG" \
  --set images.mpsd.tag="$TAG"
cd ..
```

Set every image tag. An empty tag falls back to the chart's `appVersion`, which is not your build.

Adjust for your environment:

| Situation | Add |
|---|---|
| GPU Operator v26.7.0 | `--set gpuOperator.minimumVersion=v26.7.0` (or `""` to skip the check) |
| More than 16 GPUs on a node | `--set mpsDaemon.resources.limits.memory=<larger>`, budgeting about 50 MiB per GPU plus overhead |
| MicroK8s | Symlink the CRI socket under its own parent directory first, then add `--set fractioningAgent.criSocketPath=/opt/kai-runtime/containerd.sock --set fractioningAgent.nriSocketPath=/var/snap/microk8s/common/run/nri.sock`; see the note below |
| k3s or RKE2 | `--set fractioningAgent.criSocketPath=/run/k3s/containerd/containerd.sock` |
| Private registry | `--set 'imagePullSecrets[0].name=<secret>'`, with the secret created in `gpu-fractioning` first |

> [!NOTE]
> On MicroK8s, containerd and its NRI socket live under the snap directory, so the default `/var/run/nri/nri.sock` is unreachable. Pointing both socket paths at `/var/snap/microk8s/common/run/` does not work either. The operator mounts the parent directory of each socket into the `fractiond` DaemonSet, so two sockets in the same directory produce two volume mounts at the same path, and the DaemonSet is rejected. Give the CRI socket its own parent directory with a symlink before you run the `helm upgrade --install` command above:
>
> ```bash
> sudo mkdir -p /opt/kai-runtime
> sudo ln -sf /var/snap/microk8s/common/run/containerd.sock /opt/kai-runtime/containerd.sock
> ```
>
> Then add both socket paths to that command:
>
> ```bash
>   --set fractioningAgent.criSocketPath=/opt/kai-runtime/containerd.sock \
>   --set fractioningAgent.nriSocketPath=/var/snap/microk8s/common/run/nri.sock
> ```
>
> Set the paths through the chart rather than by editing the `default` `GpuFractioningConfig`. The chart recreates that object from its values on every `helm upgrade`, which overwrites a manual `kubectl apply`. [`microk8s-gpu-fractioning-config.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-gpu-fractions/microk8s-gpu-fractioning-config.yaml) shows the resulting configuration for reference.
>
> MicroK8s can also register the containerd runtime as `nvidia-container-runtime` while the GPU Operator's RuntimeClass handler is `nvidia`. If operands fail with `no runtime for "nvidia" is configured`, align the two names.

Verify:

Wait for the operator, then for the DaemonSets it creates. The DaemonSets appear a few seconds after the operator starts:

```bash
kubectl -n gpu-fractioning wait --for=condition=Available deploy --all --timeout=300s
sleep 15
kubectl -n gpu-fractioning rollout status ds --timeout=300s
```

```bash
helm status gpu-fractioning -n gpu-fractioning               # STATUS: deployed
kubectl -n gpu-fractioning get pods -o wide                  # operator, plus mpsd and fractiond (2/2, with a metricsd sidecar) on each GPU node
kubectl -n gpu-fractioning get ds \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.desiredNumberScheduled}{"/"}{.status.numberReady}{"\n"}{end}'
kubectl get gpufractioningconfig default \
  -o jsonpath='{range .status.conditions[*]}{.type}={.status} ({.reason}){"\n"}{end}'
```

For every DaemonSet, desired must equal ready. A mismatch usually means a node failed the NRI or CRI socket check. Then confirm each GPU node reports ready, which is what gates scheduling:

```bash
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="gpu-fractioning.nvidia.com/Ready")].status}{"\n"}{end}'
```

Confirm that MPS started in multi-user mode. Without it, a server accepts only clients with the owner's uid, which would force every pod to run as that user:

```bash
kubectl -n gpu-fractioning logs ds/gpu-fractioning-mpsd | grep -E "multiuser|active thread percentage"
```

Expect `Adding server 'shared' [multiuser enabled]` and a line capping the `default` namespace at 10%. That cap is defense in depth for a pod that receives the server's pipe directory instead of its own namespace's.

## Step 5: Verify With a Two-Pod Smoke Test

This step proves that the caps are enforced, not just requested. It puts two half-GPU pods on one GPU. The manifests for the full experiment are in [`examples/gpu-sharing/kai-gpu-fractions`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing/kai-gpu-fractions), and a ready-made version of this test, with the same pod names, is [`smoke-test-half-gpu.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-gpu-fractions/smoke-test-half-gpu.yaml). The minimal form is below, for readers without a checkout:

```bash
for n in a b; do
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: fraction-smoke-$n
  labels:
    kai.scheduler/queue: default-queue
  annotations:
    gpu-fraction: "0.5"
    nvidia.com/container.main.gpu-compute.mode: sm-sharing
spec:
  schedulerName: kai-scheduler
  runtimeClassName: nvidia
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
  containers:
  - name: main
    image: nvidia/cuda:13.3.1-base-ubuntu24.04
    command: ["sleep", "3600"]
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
EOF
done

kubectl wait --for=condition=Ready pod/fraction-smoke-a pod/fraction-smoke-b --timeout=180s
```

The container must be named `main` because the annotation names the container it applies to. The chart creates `default-queue` for you.

> [!WARNING]
> Keep `runAsNonRoot: true`. MPS authorizes control commands as the server owner or root, and in multi-user mode the server is owned by root. A fractional pod that runs as root can raise its own cap or delete a neighbor's namespace. A non-root pod cannot. Enforce this with Pod Security Admission or a policy engine instead of trusting each manifest.

Run these four checks. All four must hold:

```bash
# 1. What the binder decided: portion, memory, and the physical GPU UUID
for n in a b; do
  kubectl get pod fraction-smoke-$n -o jsonpath='{.metadata.annotations}' \
    | grep -o '"nvidia\.com/[^"]*":"[^"]*"' | sort -u
done

# 2. What fractiond injected into the container
kubectl exec fraction-smoke-a -- env | grep -E 'CUDA_MPS|NVIDIA_GPU_MEMORY|NVIDIA_VISIBLE_DEVICES'

# 3. What the MPS server enforces (ground truth, not the client's environment)
kubectl -n gpu-fractioning exec ds/gpu-fractioning-mpsd -- \
  nvidia-cuda-mps-control -p 3 namespace list --server=shared

# 4. The pod's own view of memory, with a hostile client-side override
kubectl exec fraction-smoke-a -- env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
  nvidia-smi --query-gpu=memory.total --format=csv,noheader
```

| Check | Pass condition |
|---|---|
| 1 | Both pods show non-empty `gpu-compute.portion` and `gpu-memory.request` values and a real `gpus.devices` UUID. By default the scheduler bin-packs fractional pods, so the two UUIDs should match; matching UUIDs mean both pods share one GPU. An empty UUID means the pair is mismatched ([Troubleshooting](#troubleshooting)) |
| 2 | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=<N>M`, where N = (GPU memory in MiB − 1024) × 0.5. The 1 GiB reserve is the MPS server's own context. Change it with the binder's `reservedGpuMemory` argument |
| 3 | Two `kai_*` namespaces at `50.00` active thread percentage, plus `default` at `10.00`. If you re-ran the test, namespaces from the deleted pods can linger for up to about a minute before `mpsd` removes them; run the command again |
| 4 | Reports the pod's capped memory, not the physical card's. This confirms what the pod sees, not that the cap is enforced: the value comes from `fractiond`'s NVML shim. The memory cap is not a namespace setting either (check 3 shows `unlimited` in the pinned-memory column); the container toolkit's memory-limit hook and the `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` variable from check 2 apply it. The MPS server enforces the SM share through the pod's namespace; the next section measures it |

Clean up:

```bash
kubectl delete pod fraction-smoke-a fraction-smoke-b
```

### Measure the Effective SM Share

The checks above show the caps that were requested and configured. To show that the SM share is enforced, run CUDA's `deviceQuery` sample as a half-GPU pod that tries to override its own share to 100%:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: fraction-devicequery
  labels:
    kai.scheduler/queue: default-queue
  annotations:
    gpu-fraction: "0.5"
    nvidia.com/container.main.gpu-compute.mode: sm-sharing
spec:
  schedulerName: kai-scheduler
  runtimeClassName: nvidia
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
  containers:
  - name: main
    image: nvcr.io/nvidia/k8s/cuda-sample:devicequery
    env:
    - name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
      value: "100"
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
EOF

kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/fraction-devicequery --timeout=180s
kubectl logs fraction-devicequery | grep -E 'Multiprocessors|Result'
kubectl delete pod fraction-devicequery
```

The `Multiprocessors` count must be half the GPU's SM count, rounded down, despite the override: for example, `(054) Multiprocessors` on an A100 with 108 SMs. `fractiond` replaces the variable when the container is created, and the MPS server caps the namespace whatever the client asks for. `deviceQuery` still reports the physical memory total, as the note below explains.

> [!NOTE]
> `nvidia-smi` and `pynvml` in a pod report the pod's slice, but CUDA's `cudaMemGetInfo` total stays at the physical card size. Engines that size their KV cache from it, such as vLLM with `--gpu-memory-utilization`, therefore size against the whole GPU. Set that flag against the physical total, not the quota, when you [deploy the models](deploy-kai-gpu-fractions.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Fractional pod stays `Pending` and its events say nothing useful | The reservation pod has no `nvidia` RuntimeClass and fails with `unable to initialize NVML: ERROR_LIBRARY_NOT_FOUND`. Look at `kubectl -n kai-resource-reservation logs <gpu-reservation-pod>` | Reinstall with `--set binder.runtimeClassName=nvidia`, or patch a running install: `kubectl patch config.kai.scheduler kai-config --type=merge -p '{"spec":{"binder":{"resourceReservation":{"runtimeClassName":"nvidia"}}}}'`, then `kubectl -n kai-resource-reservation delete pod --all` |
| Binder logs `received-resource-type: "Regular"`, no `gpus.devices` annotation, container gets `NVIDIA_VISIBLE_DEVICES=void` | A fork node agent is running against an upstream scheduler, or the reverse | Rebuild and reinstall both from the pinned commits (Steps 1 to 4) |
| `make build push` finishes but the registry has no images | `push` is an alias and does not push | Add `DOCKER_BUILDX_ADDITIONAL_ARGS=--push` |
| `helm install` fails with `found in Chart.yaml, but missing in charts/ directory: gpu-fractioning` | `helm dependency build` was skipped | Run it in `KAI-Scheduler`; the machine needs access to `ghcr.io` |
| Pods run the wrong code, or pull `0.0.0` images | A tag value was left empty | Set `global.tag` for the scheduler and all four `images.*.tag` values for the node agents |
| `mpsd` exits with `CUDA_ERROR_OUT_OF_MEMORY` while `nvidia-smi` shows idle GPUs | The pod's host-memory limit is too small for the number of GPUs, not GPU memory. About 50 MiB per GPU is needed | Raise `mpsDaemon.resources.limits.memory` (the default is `1Gi`) |
| `nvidia-cuda-mps-control -p 3 namespace --help` has no `namespace` subcommand | Driver older than r615, or `mpsd` is not running protocol 3 | Check the driver version and the `mpsd` logs |
| Container has no `CUDA_MPS_*` variables | `fractiond` did not register as an NRI plugin | `kubectl -n gpu-fractioning logs ds/gpu-fractioning-fractiond \| grep -i "registering plugin\|configured NRI"`; on MicroK8s, set the socket paths as described in Step 4 |
| `sm-sharing` pod will not start, and `mpsd` logs a namespace error | Namespace provisioning fails closed, so a container is refused rather than started uncapped | `kubectl -n gpu-fractioning logs ds/gpu-fractioning-mpsd \| grep -iE "namespace\|provision"` |
| Fractional pods stay `Pending`; `kubectl get events` shows `BindingError ... failed to set NvFractions memory annotation: node does not include nvidia.com/gpu.memory label` | The node has no `nvidia.com/gpu.memory` label | Run GPU Feature Discovery (the GPU Operator includes it), which sets the label. Confirm with `kubectl get nodes -L nvidia.com/gpu.memory` |
| Pod fails to start with `error running createRuntime hook ... failed to set memory limits for gpu ...: Insufficient Permissions` | The container toolkit's memory-limit hook cannot set a limit on this node | Check that the driver and toolkit versions support the hook. To fall back to the MPS-only memory cap, run GPU Operator v26.7.0 with `--set gpuOperator.minimumVersion=v26.7.0` If the container toolkit is installed on the host rather than by the GPU Operator, use a toolkit version that does not ship the `apply-cuda-memory-limits` hook |
| An MPS server is wedged after a run, or pods will not start after repeated teardown | MPS state survives pod deletion | Delete the `gpu-fractioning-mpsd` pods to restart the daemons |

## Uninstall

Delete the `default` `GpuFractioningConfig` first, while the operator is still running. The operator then stops the `mpsd` and `fractiond` DaemonSets and clears the node conditions. If you uninstall the chart first, the DaemonSets keep running, MPS stays on the GPUs, and the object can no longer be deleted: its `gpu-fractioning.nvidia.com/cleanup-node-conditions` finalizer is removed only by the operator.

```bash
kubectl delete gpufractioningconfig default --wait
kubectl -n gpu-fractioning wait --for=delete pod -l app.kubernetes.io/component=mpsd --timeout=180s
helm uninstall gpu-fractioning -n gpu-fractioning
helm uninstall kai-scheduler -n kai-scheduler
```

The `mpsd` pods take about a minute to terminate. Helm keeps the `default-queue` and `default-parent-queue` queues, the `default` `SchedulingShard`, the CRDs, and the `kai-resource-reservation`, `kai-scheduler`, and `gpu-fractioning` namespaces. For a clean slate, remove them too. Delete the CRDs only if nothing else in the cluster uses them:

```bash
kubectl get crd -o name | grep -E 'kai\.scheduler|scheduling\.run\.ai' | xargs kubectl delete
kubectl delete namespace kai-resource-reservation kai-scheduler gpu-fractioning
```

## Next Steps

- [Deploy the 16-model experiment](deploy-kai-gpu-fractions.md) on the cluster you just built.
- [Per-GPU Compute and Memory Limits with MPS](mps-per-gpu-limits.md) covers the MPS v3 mechanism that these forks drive for you.
- [Interpreting GPU Sharing Results](gpu-sharing-results.mdx) explains how to read the benchmark.
