# NIXL Benchmark Documentation

This guide describes how to build and deploy the NIXL benchmark using the provided scripts on a Kubernetes (K8s) cluster.

> **Note**: NIXL benchmark is part of the Dynamo platform. Before proceeding, ensure your cluster meets the basic Dynamo requirements by running the pre-deployment check script located in the parent directory (`../pre-deployment-check.sh`).

---

## Prerequisites

### Cluster Requirements
Before deploying NIXL benchmark, ensure your cluster meets the Dynamo platform requirements by running the pre-deployment check:

```bash
# Run from the parent directory
../pre-deployment-check.sh
```

This script verifies:
- `kubectl` connectivity and cluster access
- GPU nodes availability (`nvidia.com/gpu.present=true` label)
- GPU Operator installation and status

### NIXL-Specific Requirements
In addition to the cluster requirements above, NIXL benchmark requires:
- **Docker** installed and configured on your local machine (for building images)
- **Docker registry access** to push the built nixlbench images
- **ETCD service** deployed and accessible as `etcd:2379`
- **Build utilities**: `wget` and `unzip` for downloading NIXL source code

### Verification Steps
1. **Run pre-deployment check** (recommended):
   ```bash
   ../pre-deployment-check.sh
   ```
   Ensure all checks pass before proceeding.

2. **Verify ETCD availability** (NIXL-specific):
   ```bash
   kubectl get svc etcd
   ```

3. **Confirm Docker registry access**:
   ```bash
   docker login your-registry.com  # if using private registry
   ```

---

## Quick Start

For the easiest experience, use the interactive build and deploy script:

```bash
./build_and_deploy.sh
```

This script provides a flexible workflow where you can:
1. **Select architecture**: Choose between x86_64 (Intel/AMD 64-bit) or aarch64 (ARM64)
2. **Choose which steps to execute**: Select any combination of:
   - Build nixlbench Docker image
   - Update deployment YAML file
   - Deploy to Kubernetes
3. **Provide Docker registry** (only when needed for building or updating deployment)

---

## Interactive Script Features

### Architecture Selection
The script supports two architectures:
- **Option 1**: x86_64 (Intel/AMD 64-bit)
- **Option 2**: aarch64 (ARM64)

You can select by entering:
- `1` or `x86_64` for x86_64 architecture
- `2` or `aarch64` for aarch64 architecture

### Step Selection
Choose which steps to execute by entering comma-separated numbers:

- **All steps**: `1,2,3`
- **Build and update only**: `1,2` (skips Kubernetes deployment)
- **Deploy only**: `3` (useful if image is already built and deployment file exists)
- **Build only**: `1` (useful for just creating the Docker image)
- **Update deployment only**: `2` (useful for updating deployment file with new registry/version)

### Smart Registry Prompting
The script only prompts for Docker registry information when needed:
- **Steps 1 or 2**: Registry required for building image or updating deployment
- **Step 3 only**: No registry prompt (uses existing deployment file)

---

## What Each Step Does

### Step 1: Build nixlbench Docker Image
- Downloads NIXL source code (version 0.6.0) from GitHub
- Extracts and navigates to the build directory
- Pauses for user confirmation before building
- Builds Docker image with specified registry and architecture
- Tags image as: `{registry}/nixlbench:0.6.0-{arch}`

### Step 2: Update Deployment YAML File
- Copies base deployment template (`nixlbench-deployment.yaml`)
- Creates architecture-specific deployment file (`nixlbench-deployment-{arch}.yaml`)
- Updates image reference with your registry and architecture
- Preserves all other deployment configurations

### Step 3: Deploy to Kubernetes
- Validates deployment file exists
- Applies deployment to Kubernetes cluster
- Provides monitoring commands for checking status

---

## Deployment Configuration

The deployment creates:
- **2 replicas** of the nixlbench pod
- **Resource requests/limits**:
  - CPU: 10 cores
  - Memory: 5Gi
  - GPU: 1 NVIDIA GPU per pod
- **Environment variables**:
  - `ETCD_ENDPOINTS`: Points to `etcd:2379`
- **Command**: Runs nixlbench with VRAM segments and keeps container alive

---

## Usage Examples

### Example 1: Complete Workflow
```bash
./build_and_deploy.sh
# Select: 1 (x86_64)
# Steps: 1,2,3
# Registry: docker.io/myusername
# Confirm: y
```

### Example 2: Build Image Only
```bash
./build_and_deploy.sh
# Select: 2 (aarch64)
# Steps: 1
# Registry: my-private-registry.com
# Confirm: y
```

### Example 3: Deploy Existing Image
```bash
./build_and_deploy.sh
# Select: 1 (x86_64)
# Steps: 3
# Confirm: y
```

### Example 4: Update Deployment File Only
```bash
./build_and_deploy.sh
# Select: 2 (aarch64)
# Steps: 2
# Registry: new-registry.com
# Confirm: y
```

---

## Generated Files

The script creates architecture-specific deployment files:
- `nixlbench-deployment-x86_64.yaml` - For x86_64 builds
- `nixlbench-deployment-aarch64.yaml` - For aarch64 builds

These files are customized versions of the base template with your specific:
- Docker registry
- Image tag
- Architecture

---

## Monitoring Your Deployment

After deployment, monitor your NIXL benchmark:

```bash
# Check pod status
kubectl get pods -l app=nixl-benchmark

# View logs
kubectl logs -l app=nixl-benchmark -f

# Check resource usage
kubectl top pods -l app=nixl-benchmark

# Get detailed pod information
kubectl describe pods -l app=nixl-benchmark
```

If deployed to a specific namespace:
```bash
kubectl get pods -l app=nixl-benchmark -n your-namespace
kubectl logs -l app=nixl-benchmark -f -n your-namespace
```

---


## Troubleshooting

### Cluster-Level Issues
For cluster-related problems, first run the pre-deployment check to identify issues:

```bash
../pre-deployment-check.sh
```

This will help diagnose:
- kubectl connectivity problems
- Missing default StorageClass
- GPU node availability issues
- GPU Operator status problems

### NIXL-Specific Issues

1. **ETCD Connection**:
   - Ensure etcd service is running: `kubectl get svc dynamo-platform-etcd`
   - Verify etcd endpoints are accessible from pods
   - Check if etcd is in the correct namespace

2. **Image Pull Issues**:
   - Verify registry credentials are configured
   - Check image exists: `docker pull {registry}/nixlbench:0.6.0-{arch}`
   - Ensure image was pushed successfully after build

3. **Build Failures**:
   - Ensure Docker daemon is running
   - Check available disk space in `/tmp`
   - Verify network connectivity to GitHub
   - Confirm build utilities are installed: `which wget unzip`

4. **Deployment File Not Found**:
   - Run step 2 to create deployment file before step 3
   - Check file permissions in script directory
   - Verify script directory path is correct

### Debug Commands
```bash
# Check script-generated files
ls -la nixlbench-deployment-*.yaml

# Verify deployment status
kubectl get deployment nixl-benchmark -o yaml

# Check events for issues
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Cleanup

To remove the deployment:
```bash
kubectl delete deployment nixl-benchmark
```

Or if deployed to a specific namespace:
```bash
kubectl delete deployment nixl-benchmark -n your-namespace
```

To clean up generated files:
```bash
rm -f nixlbench-deployment-*.yaml
```

---

## Script Reference

### build_and_deploy.sh
Interactive script that provides flexible build and deployment workflow:
- **Architecture selection**: x86_64 or aarch64
- **Step selection**: Choose any combination of build, update, deploy
- **Validation**: Checks for deployment files before deploying

### nixlbench-deployment.yaml
Base Kubernetes deployment template that gets customized by the script:
- **Template image**: `my-registry/nixlbench:version-arch`
- **Resource allocation**: 10 CPU, 5Gi memory, 1 GPU per pod
- **ETCD integration**: Pre-configured environment variables
- **Benchmark command**: Runs with VRAM segment configuration