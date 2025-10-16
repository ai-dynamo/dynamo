# Operator Default Values Injection

The Dynamo operator automatically applies default values to various fields when they are not explicitly specified in your deployments. These defaults include:

- **Health Probes**: Startup, liveness, and readiness probes are configured differently for frontend, worker, and planner components. For example, worker components receive a startup probe with a 2-hour timeout (720 failures × 10 seconds) to accommodate long model loading times.

- **Shared Memory**: All components receive an 8Gi shared memory volume mounted at `/dev/shm` by default (can be disabled or resized via the `sharedMemory` field).

- **Environment Variables**: Components automatically receive environment variables like `DYN_NAMESPACE`, `DYN_PARENT_DGD_K8S_NAME`, `DYNAMO_PORT`, and backend-specific variables.

- **Pod Configuration**: Default `terminationGracePeriodSeconds` of 60 seconds and `restartPolicy: Always`.

- **Autoscaling**: When enabled without explicit metrics, defaults to CPU-based autoscaling with 80% target utilization.

- **Backend-Specific Behavior**: For multinode deployments, probes are automatically modified or removed for worker nodes depending on the backend framework (VLLM, SGLang, or TensorRT-LLM).

## Pod Specification Defaults

All components receive the following pod-level defaults unless overridden:

- **`terminationGracePeriodSeconds`**: `60` seconds
- **`restartPolicy`**: `Always`

## Shared Memory Configuration

Shared memory is enabled by default for all components:

- **Enabled**: `true` (unless explicitly disabled via `sharedMemory.disabled`)
- **Size**: `8Gi`
- **Mount Path**: `/dev/shm`
- **Volume Type**: `emptyDir` with `memory` medium

To disable shared memory or customize the size, use the `sharedMemory` field in your component specification.

## Health Probes by Component Type

The operator applies different default health probes based on the component type.

### Frontend Components

Frontend components receive the following probe configurations:

**Liveness Probe:**
- **Type**: HTTP GET
- **Path**: `/health`
- **Port**: `http` (8000)
- **Initial Delay**: 60 seconds
- **Period**: 60 seconds
- **Timeout**: 30 seconds
- **Failure Threshold**: 10

**Readiness Probe:**
- **Type**: Exec command
- **Command**: `curl -s http://localhost:${DYNAMO_PORT}/health | jq -e ".status == \"healthy\""`
- **Initial Delay**: 60 seconds
- **Period**: 60 seconds
- **Timeout**: 30 seconds
- **Failure Threshold**: 10

### Worker Components

Worker components receive the following probe configurations:

**Liveness Probe:**
- **Type**: HTTP GET
- **Path**: `/live`
- **Port**: `system` (9090)
- **Period**: 5 seconds
- **Timeout**: 30 seconds
- **Failure Threshold**: 1

**Readiness Probe:**
- **Type**: HTTP GET
- **Path**: `/health`
- **Port**: `system` (9090)
- **Period**: 10 seconds
- **Timeout**: 30 seconds
- **Failure Threshold**: 60

**Startup Probe:**
- **Type**: HTTP GET
- **Path**: `/live`
- **Port**: `system` (9090)
- **Period**: 10 seconds
- **Timeout**: 5 seconds
- **Failure Threshold**: 720 (allows up to 2 hours for startup: 10s × 720 = 7200s)

:::{note}
For larger models (typically >70B parameters) or slower storage systems, you may need to increase the `failureThreshold` to allow more time for model loading. Calculate the required threshold based on your expected startup time: `failureThreshold = (expected_startup_seconds / period)`. Override the startup probe in your component specification if the default 2-hour window is insufficient.
:::

### Multinode Deployment Probe Modifications

For multinode deployments, the operator modifies probes based on the backend framework and node role:

#### VLLM Backend

The operator automatically selects between two deployment modes based on parallelism configuration:

**Ray-Based Mode** (when `world_size > GPUs_per_node`):
- **Worker nodes**: All probes (liveness, readiness, startup) are removed
- **Leader nodes**: All probes remain active

**Data Parallel Mode** (when `world_size × data_parallel_size > GPUs_per_node`):
- **Worker nodes**: All probes (liveness, readiness, startup) are removed
- **Leader nodes**: All probes remain active

#### SGLang Backend
- **Worker nodes**: All probes (liveness, readiness, startup) are removed

#### TensorRT-LLM Backend
- **Leader nodes**: All probes remain unchanged
- **Worker nodes**:
  - Liveness and startup probes are removed
  - Readiness probe is replaced with a TCP socket check on SSH port (2222):
    - **Initial Delay**: 20 seconds
    - **Period**: 20 seconds
    - **Timeout**: 5 seconds
    - **Failure Threshold**: 10

## Environment Variables

The operator automatically injects environment variables based on component type and configuration:

### All Components

- **`DYN_NAMESPACE`**: The Dynamo namespace for the component
- **`DYN_PARENT_DGD_K8S_NAME`**: The parent DynamoGraphDeployment Kubernetes resource name
- **`DYN_PARENT_DGD_K8S_NAMESPACE`**: The parent DynamoGraphDeployment Kubernetes namespace

### Frontend Components

- **`DYNAMO_PORT`**: `8000`
- **`DYN_HTTP_PORT`**: `8000`

### Worker Components

- **`DYN_SYSTEM_ENABLED`**: `true`
- **`DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS`**: `["generate"]`
- **`DYN_SYSTEM_PORT`**: `9090`

### Planner Components

- **`PLANNER_PROMETHEUS_PORT`**: `9085`

### VLLM Backend (with compilation cache)

When a volume mount is configured with `useAsCompilationCache: true`:
- **`VLLM_CACHE_ROOT`**: Set to the mount point of the cache volume

## Service Account

Planner components automatically receive the following service account:

- **`serviceAccountName`**: `planner-serviceaccount`

## Image Pull Secrets

The operator automatically discovers and injects image pull secrets for container images. When a component specifies a container image, the operator:

1. Scans all Kubernetes secrets of type `kubernetes.io/dockerconfigjson` in the component's namespace
2. Extracts the docker registry server URLs from each secret's authentication configuration
3. Matches the container image's registry host against the discovered registry URLs
4. Automatically injects matching secrets as `imagePullSecrets` in the pod specification

This eliminates the need to manually specify image pull secrets for each component. The operator maintains an internal index of docker secrets and their associated registries, refreshing this index periodically.

**To disable automatic image pull secret discovery** for a specific component, add the following annotation:

```yaml
annotations:
  nvidia.com/disable-image-pull-secret-discovery: "true"
```

## Autoscaling Defaults

When autoscaling is enabled but no metrics are specified, the operator applies:

- **Default Metric**: CPU utilization
- **Target Average Utilization**: `80%`

## Port Configurations

Default container ports are configured based on component type:

### Frontend Components
- **Port**: 8000
- **Protocol**: TCP
- **Name**: `http`

### Worker Components
- **Port**: 9090
- **Protocol**: TCP
- **Name**: `system`

### Planner Components
- **Port**: 9085
- **Protocol**: TCP
- **Name**: `metrics`

## Backend-Specific Configurations

### VLLM
- **Ray Head Port**: 6379 (for Ray-based multinode deployments)
- **Data Parallel RPC Port**: 13445 (for data parallel multinode deployments)

### SGLang
- **Distribution Init Port**: 29500 (for multinode deployments)

### TensorRT-LLM
- **SSH Port**: 2222 (for multinode MPI communication)
- **OpenMPI Environment**: `OMPI_MCA_orte_keep_fqdn_hostnames=1`

## Implementation Reference

For users who want to understand the implementation details or contribute to the operator, the default values described in this document are set in the following source files:

- **Health Probes & Pod Specifications**: [`internal/dynamo/graph.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/dynamo/graph.go) - Contains the main logic for applying default probes, environment variables, shared memory, and pod configurations
- **Component-Specific Defaults**:
  - [`internal/dynamo/component_frontend.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/dynamo/component_frontend.go)
  - [`internal/dynamo/component_worker.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/dynamo/component_worker.go)
  - [`internal/dynamo/component_planner.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/dynamo/component_planner.go)
- **Image Pull Secrets**: [`internal/secrets/docker.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/secrets/docker.go) - Implements the docker secret indexer and automatic discovery
- **Backend-Specific Behavior**:
  - [`internal/dynamo/backend_vllm.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/dynamo/backend_vllm.go)
  - [`internal/dynamo/backend_sglang.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/dynamo/backend_sglang.go)
  - [`internal/dynamo/backend_trtllm.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/dynamo/backend_trtllm.go)
- **Constants & Annotations**: [`internal/consts/consts.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/cloud/operator/internal/consts/consts.go) - Defines annotation keys and other constants

## Notes

- All these defaults can be overridden by explicitly specifying values in your DynamoComponentDeployment or DynamoGraphDeployment resources
- User-specified probes (via `livenessProbe`, `readinessProbe`, or `startupProbe` fields) take precedence over operator defaults
- For multinode deployments, some defaults are modified or removed as described above to accommodate distributed execution patterns
- The `extraPodSpec.mainContainer` field can be used to override probe configurations set by the operator