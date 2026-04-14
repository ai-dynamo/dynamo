# Volcano v1.13.2 local install overlay

This directory contains a local Kustomize overlay for installing Volcano `v1.13.2`.

Why this form:

- pins Volcano to the requested `v1.13.2` version
- keeps the official upstream installer as the base
- applies local scheduling constraints so Volcano control-plane pods prefer CPU nodes
- rewrites Volcano images from `docker.io/volcanosh/*` to `harbor.weizhipin.com/cloud-native/volcanosh/*`

## Files

- `kustomization.yaml`

## What this overlay changes

It imports the official Volcano installer from:

- `https://raw.githubusercontent.com/volcano-sh/volcano/v1.13.2/installer/volcano-development.yaml`

And patches these workloads to run on CPU-labeled nodes:

- `volcano-scheduler`
- `volcano-controllers`
- `volcano-admission`
- `volcano-admission-init`

Each patched workload gets:

```yaml
nodeSelector:
  workload-type: cpu
```

This matches the labels we already placed on:

- `yj-testaiinfrawork-01`
- `yj-testaiinfrawork-02`

## Render locally

```bash
kubectl kustomize /Users/admin/work/go/src/github.com/ai-dynamo/dynamo/cxl/volcano
```

## Install

```bash
kubectl apply -k /Users/admin/work/go/src/github.com/ai-dynamo/dynamo/cxl/volcano
```

## Verify

```bash
kubectl get pods -n volcano-system -o wide
kubectl get deploy -n volcano-system -o wide
kubectl get job -n volcano-system
kubectl get crd | grep volcano
```

## Important note for air-gapped or internal-registry environments

This overlay already rewrites these images to your Harbor registry:

- `docker.io/volcanosh/vc-webhook-manager:v1.13.2`
- `docker.io/volcanosh/vc-controller-manager:v1.13.2`
- `docker.io/volcanosh/vc-scheduler:v1.13.2`

They become:

- `harbor.weizhipin.com/cloud-native/volcanosh/vc-webhook-manager:v1.13.2`
- `harbor.weizhipin.com/cloud-native/volcanosh/vc-controller-manager:v1.13.2`
- `harbor.weizhipin.com/cloud-native/volcanosh/vc-scheduler:v1.13.2`
