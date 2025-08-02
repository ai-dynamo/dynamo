# NIXL Benchmark Technical Documentation (Kubernetes)

This guide describes how to run the NIXL benchmark using the provided Docker image on a Kubernetes (K8s) cluster.

---

## Prerequisites

- A running Kubernetes cluster with access to NVIDIA GPUs (e.g., using NVIDIA GPU Operator or device plugin)
- `kubectl` configured to access your cluster
- deploy dynamo cloud in a namespace

---

## 1. Prepare the Kubernetes Deployment

A sample deployment YAML is provided in this repository:
`benchmarks/nixl/nixl-benchmark-deployment.yaml`

Update the image field in sample yaml to appropiate image in your registry.

You can use the `yq` tool to update the image field in the deployment YAML
```bash
yq -i '.spec.template.spec.containers[] |= select(.name == "nixl-benchmark") .image = "your-registry/your-nixl-benchmark:your-tag"' benchmarks/nixl/nixl-benchmark-deployment.yaml > nixl-benchmark-deployment.yaml
```

## 2. Deploy using kubectl
Launch using the command below:

```bash
kubectl apply -f  nixl-benchmark-deployment.yaml
```