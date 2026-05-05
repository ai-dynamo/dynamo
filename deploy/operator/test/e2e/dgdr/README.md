# DGDR e2e Tests (Go)

End-to-end tests for the DynamoGraphDeploymentRequest (DGDR) v1beta1 API using
Go, Ginkgo v2, and typed CRD structs.

## Prerequisites

1. Kubernetes cluster with Dynamo operator CRDs and webhooks installed.
   The operator can be deployed via Helm (`deploy/helm/`) or via Tilt for
   development branches. If running against a fix branch, build and push your
   own operator image and install with:
   ```bash
   helm upgrade --install dynamo-platform deploy/helm/dynamo-platform \
     --set operator.image.repository=<your-registry>/kubernetes-operator \
     --set operator.image.tag=<your-branch-tag>
   ```
2. `kubectl` configured for the cluster
3. Go 1.25+

## Image Override

The `-dgdr-image` flag specifies the container image used for profiling jobs.

- **Operator ≤ 1.0.x**: uses `nvcr.io/nvidia/ai-dynamo/dynamo-frontend:<tag>`
- **Operator on main (≥ 1.1.0)**: uses `nvcr.io/nvidia/ai-dynamo/dynamo-planner:<tag>`

To test with your own image (e.g. from a fix branch), build and push it, then
pass the image via the `-dgdr-image` flag or `DGDR_IMAGE` Makefile variable:

```bash
# Using your own registry/tag
make test-e2e-dgdr DGDR_IMAGE=myregistry/dynamo-planner:my-fix-branch
```

## Running

```bash
cd deploy/operator

# Set the image to match your operator version:
#   Operator 1.0.x  → dynamo-frontend:1.0.2
#   Operator ≥1.1.0 → dynamo-planner:1.1.0
IMAGE=nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.0

# Mocker mode (no GPU, default)
make test-e2e-dgdr DGDR_NAMESPACE=default DGDR_IMAGE=$IMAGE

# Or directly with go test:
go test ./test/e2e/dgdr/ -v -ginkgo.v \
  -dgdr-namespace=default \
  -dgdr-image=$IMAGE

# Real GPU mode
go test ./test/e2e/dgdr/ -v -ginkgo.v \
  -dgdr-namespace=dynamo-test \
  -dgdr-image=$IMAGE \
  -dgdr-no-mocker

# Validation only (fastest — no profiling jobs)
go test ./test/e2e/dgdr/ -v -ginkgo.v \
  -dgdr-namespace=default \
  -dgdr-image=$IMAGE \
  -ginkgo.label-filter="gpu_0"
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `-dgdr-namespace` | _(required)_ | Kubernetes namespace for test resources |
| `-dgdr-image` | _(required)_ | Container image for profiling/deployment |
| `-dgdr-model` | `Qwen/Qwen3-0.6B` | HuggingFace model ID |
| `-dgdr-backend` | `vllm` | Backend (auto/vllm/sglang/trtllm) |
| `-dgdr-no-mocker` | `false` | Disable mocker (requires real GPUs) |
| `-dgdr-profiling-timeout` | `3600` | Max seconds for profiling |
| `-dgdr-deploy-timeout` | `600` | Max seconds for deployment |
| `-kubeconfig` | default | Path to kubeconfig |

## Test Matrix

| # | File | Context | Test | Verifies |
|---|---|---|---|---|
| 1 | `validation_test.go` | Webhook Validation | should reject a DGDR with missing model | CRD required field |
| 2 | | | should reject thorough + auto backend | Webhook logic |
| 3 | | | should reject an invalid backend | CRD enum |
| 4 | | | should reject an invalid searchStrategy | CRD enum |
| 5 | | | should reject an invalid sla.optimizationType | CRD/webhook enum |
| 6 | | | should accept a valid minimal DGDR | Minimal spec passes |
| 7 | | | should accept a fully-specified DGDR | Full v1beta1 spec passes |
| 8 | | CRD Metadata | should have v1beta1 as the storage version | CRD storage version |
| 9 | | | should support the dgdr shortname | CRD shortName |
| 10 | | | should show expected columns in kubectl output | CRD PrintColumns |
| 11 | | Version Conversion | should accept a v1alpha1 DGDR | Conversion webhook |
| 12 | | | should serve a v1alpha1 view of a v1beta1 object | Conversion webhook |
| 13 | `lifecycle_test.go` | Rapid profiling | should reach Ready with autoApply=false | Profiling lifecycle |
| 14 | | | should reach Deployed with autoApply=true | Deploy lifecycle (non-mocker) |
| 15 | `profiling_test.go` | Rapid search strategy | should emit an output ConfigMap with final_config.yaml | Profiling output |
| 16 | | | should include Planner service when planner feature is enabled | Feature flag |
| 17 | | | should respect totalGpus budget [xfail #8583] | GPU budget guard |
