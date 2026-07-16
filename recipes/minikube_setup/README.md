# Minikube Setup

Provisions and configures a local minikube cluster with NVIDIA GPU support for running Dynamo workloads.

Two steps:

1. **`install.yml`** — Ansible playbook that installs the minikube binary to `/usr/local/bin`.
2. **`configure.py`** — Python script that starts minikube with Docker + GPU passthrough and installs the NVIDIA GPU Operator via Helm.

## Prerequisites

- Linux host with an NVIDIA GPU
- NVIDIA driver installed and `nvidia-smi` working
- Docker installed and the current user in the `docker` group
- `ansible`, `helm`, and `kubectl` available in `PATH`
- Python 3.10+ with `typer` installed (`pip install typer`)

## Step 1 — Install the minikube binary

Run against localhost:

```bash
ansible-playbook -i localhost, -c local --become install.yml
```

Run against a remote host:

```bash
ansible-playbook -i <host>, --become install.yml
```

The playbook is idempotent — it skips download and install if minikube is already present.

## Step 2 — Configure the cluster

### Full setup (recommended for a fresh machine)

Tears down any existing cluster, then starts minikube and installs the GPU Operator:

```bash
python configure.py setup
```

You will be prompted for confirmation before an existing cluster is deleted.

### Individual commands

| Command | Description |
|---|---|
| `python configure.py start` | Start minikube (Docker driver, `--gpus all`), no teardown |
| `python configure.py delete` | Stop and delete the existing cluster (with confirmation) |
| `python configure.py install-gpu-operator` | Install NVIDIA GPU Operator into a running cluster |

## What `setup` does

1. Checks that `nvidia-smi` reports a working driver — aborts if not.
2. Stops and deletes any existing minikube cluster (with confirmation).
3. Starts minikube: `--driver docker --container-runtime docker --gpus all`.
4. Sets `kubectl` context to `minikube`.
5. Installs NVIDIA GPU Operator v25.3.0 via Helm into the `gpu-operator` namespace.
   - If the host driver is already loaded, `driver.enabled=false` is passed so the operator does not install a second driver.

## Verifying the cluster

```bash
minikube status
kubectl get nodes
kubectl get pods -n gpu-operator
```
