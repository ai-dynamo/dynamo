# DynamoGraphDeployment (DGD) Architecture Analysis

## 1. DynamoGraphDeployment Definition and Structure

### Core Definition
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/deploy/cloud/operator/api/v1alpha1/dynamographdeployment_types.go` (lines 30-69)

The `DynamoGraphDeployment` is a Kubernetes CRD (Custom Resource Definition) that encapsulates an entire inference deployment graph:

```go
type DynamoGraphDeploymentSpec struct {
    // List of persistent volume claims that can be referenced by components
    PVCs []PVC `json:"pvcs,omitempty"`
    
    // Services to deploy as part of this deployment (map of service name -> component spec)
    Services map[string]*DynamoComponentDeploymentSharedSpec `json:"services,omitempty"`
    
    // Environment variables applied to all services
    Envs []corev1.EnvVar `json:"envs,omitempty"`
    
    // Backend framework: "sglang", "vllm", "trtllm"
    BackendFramework string `json:"backendFramework,omitempty"`
}

type DynamoGraphDeploymentStatus struct {
    State string `json:"state,omitempty"`
    Conditions []metav1.Condition `json:"conditions,omitempty"`
}
```

### Key Observations:
- **Single deployment unit**: Encapsulates an entire deployment graph with multiple components
- **Services are heterogeneous**: Each service in the map can have different configurations via `DynamoComponentDeploymentSharedSpec`
- **Services field is a map**: Allows multiple instances of different component types (e.g., Frontend, Planner, multiple worker types)

### Service Configuration Structure
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/deploy/cloud/operator/api/v1alpha1/dynamocomponentdeployment_types.go` (lines 48-106)

```go
type DynamoComponentDeploymentSharedSpec struct {
    Annotations map[string]string
    Labels map[string]string
    ServiceName string
    ComponentType string              // "frontend", "worker", "planner"
    SubComponentType string           // "prefill", "decode" for workers
    DynamoNamespace *string
    Resources *dynamoCommon.Resources
    Autoscaling *Autoscaling
    Envs []corev1.EnvVar
    EnvFromSecret *string
    VolumeMounts []VolumeMount
    Ingress *IngressSpec
    SharedMemory *SharedMemorySpec
    ExtraPodMetadata *dynamoCommon.ExtraPodMetadata
    ExtraPodSpec *dynamoCommon.ExtraPodSpec
    LivenessProbe *corev1.Probe
    ReadinessProbe *corev1.Probe
    Replicas *int32              // Desired replicas for this service
    Multinode *MultinodeSpec      // Multi-node configuration (NodeCount field)
}

type MultinodeSpec struct {
    NodeCount int32 `json:"nodeCount"`  // Number of nodes for multinode components
}
```

---

## 2. Current Planner Implementation

### Architecture Overview
**Files:** 
- `/Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/utils/planner_core.py` (lines 61-634)
- `/Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/kubernetes_connector.py` (lines 48-287)

### Planner Class Structure
```python
class Planner:
    def __init__(self, runtime: Optional[DistributedRuntime], args: argparse.Namespace, dryrun: bool = False)
```

**Scaling Targets (Hard-coded):**
- **Prefill Workers**: Single component name per backend (e.g., "VllmPrefillWorker")
- **Decode Workers**: Single component name per backend (e.g., "VllmDecodeWorker")

### Planner Workflow

1. **Observation Phase** (`observe_metrics`, line 241-292):
   - Polls Prometheus metrics every `adjustment_interval` seconds
   - Metrics observed:
     - Number of requests (num_req)
     - Input sequence length (isl)
     - Output sequence length (osl)
     - Time to First Token (ttft)
     - Inter Token Latency (itl)
   - Fetches current endpoint list from runtime.namespace().component().endpoint().client().instance_ids()

2. **Prediction Phase** (`predict_load`, line 294-306):
   - Uses configurable load predictor: "arima", "constant", or "prophet"
   - Window size: configurable (default 50 samples)
   - Predicts: next_num_req, next_isl, next_osl

3. **Compute Requirements Phase** (`_compute_replica_requirements`, line 313-407):
   - **Prefill calculation**:
     ```python
     pred_prefill_throughput = next_num_req * next_isl / adjustment_interval * p_correction_factor
     next_num_p = ceil(pred_prefill_throughput / interpolator.thpt_per_gpu(isl) / prefill_engine_num_gpu)
     ```
   - **Decode calculation**:
     ```python
     corrected_itl = itl_sla / d_correction_factor
     pred_decode_thpt_per_gpu = find_best_throughput_per_gpu(itl=corrected_itl, context_length)
     pred_decode_throughput = next_num_req * next_osl / adjustment_interval
     next_num_d = ceil(pred_decode_throughput / pred_decode_thpt_per_gpu / decode_engine_num_gpu)
     ```
   - **GPU Budget Enforcement**: Scales down if total GPUs exceed max_gpu_budget

4. **Decision Phase** (`make_adjustments`, line 409-470):
   - Computes correction factors based on observed vs. predicted metrics
   - Calls `connector.set_component_replicas(target_replicas, blocking=False)`

### Decision Making Assumptions
- **Homogeneous workers**: Assumes all prefill workers are identical, all decode workers are identical
- **Single deployment**: Works with one DGD at a time
- **Throughput-based**: Assumes linear scaling of throughput with replica count
- **No heterogeneous configurations**: Cannot differentiate between workers with different GPU types or compute capabilities

### Defaults Configuration
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/defaults.py` (lines 44-139)

```python
class SLAPlannerDefaults(BasePlannerDefaults):
    adjustment_interval = 180          # seconds
    max_gpu_budget = 8                 # GPUs
    min_endpoint = 1                   # minimum replicas for both prefill and decode
    decode_engine_num_gpu = 1          # GPUs per decode replica
    prefill_engine_num_gpu = 1         # GPUs per prefill replica
    isl = 3000                         # input sequence length SLA (tokens)
    osl = 150                          # output sequence length SLA (tokens)
    ttft = 0.5                         # time to first token SLA (seconds)
    itl = 0.05                         # inter-token latency SLA (seconds)
```

---

## 3. Current Router Implementation

### Router Types and Responsibilities

#### KV Router (Key-Value Cache Router)
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/lib/llm/src/kv_router.rs` (lines 204-318)

```rust
pub struct KvRouter {
    indexer: Indexer,                          // Tracks cached KV blocks
    scheduler: KvScheduler,                    // Selects best worker
    block_size: u32,
    kv_router_config: KvRouterConfig,
    cancellation_token: tokio_util::sync::CancellationToken,
}
```

**Router Configuration:**
```rust
pub struct KvRouterConfig {
    pub overlap_score_weight: f64,             // Weight for KV cache overlap scoring
    pub router_temperature: f64,               // Randomization in selection
    pub use_kv_events: bool,                   // Track KV events vs. approximate TTL-based
    pub router_replica_sync: bool,             // Sync state across replicas
    pub router_track_active_blocks: bool,      // Track active blocks per worker
    pub router_snapshot_threshold: Option<u32>,// Snapshot state periodically
    pub router_reset_states: bool,             // Reset router state on startup
}
```

**Core Decision Logic** (`find_best_match`, line 323-329):
1. Computes block hashes for input tokens
2. Finds overlapping blocks in KV cache via Indexer
3. Uses Scheduler to select best worker based on:
   - Overlap scores (KV cache hit potential)
   - Active load on workers
   - Temperature-based randomization

#### Indexer Strategies
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/lib/llm/src/kv_router.rs` (lines 160-200)

```rust
pub enum Indexer {
    KvIndexer(KvIndexer),              // Exact tracking of cached blocks
    ApproxKvIndexer(ApproxKvIndexer),  // TTL-based approximation (120 seconds)
    None,                              // No indexing (overlap_score_weight == 0)
}
```

#### KV Scheduler
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/lib/llm/src/kv_router/scheduler.rs` (lines 90-180)

```rust
pub struct KvScheduler {
    request_tx: tokio::sync::mpsc::Sender<SchedulingRequest>,
    slots: Arc<ActiveSequencesMultiWorker>,
}

pub struct SchedulingRequest {
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub overlaps: OverlapScores,
    pub decode_blocks: HashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: HashMap<WorkerWithDpRank, usize>,
    pub router_config_override: Option<RouterConfigOverride>,
    pub update_states: bool,
}
```

**Worker Selection Process:**
1. Monitors instances_rx (watch receiver) for worker list changes
2. Monitors runtime_configs_rx for ModelRuntimeConfig updates
3. Maintains HashMap<WorkerId, Option<ModelRuntimeConfig>> for all workers
4. DefaultWorkerSelector uses overlap scores + active load + temperature for decision

### Router Assumptions
- **Homogeneous workers within a cohort**: All workers in a worker component expected to have same ModelRuntimeConfig
- **Single component per role**: One set of prefill workers, one set of decode workers
- **Distributed Routers**: Each router instance maintains its own state of active blocks
- **Static worker discovery**: Workers discovered via etcd, updated dynamically

---

## 4. Worker Configuration and Deployment Logic

### Worker Components Structure

**Example Deployment:**
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/components/backends/vllm/deploy/disagg.yaml` (lines 17-81)

```yaml
services:
  VllmDecodeWorker:
    dynamoNamespace: vllm-disagg
    componentType: worker
    subComponentType: decode          # Identifies decode worker
    replicas: 1                        # Desired replicas
    resources:
      limits:
        gpu: "1"
    extraPodSpec:
      mainContainer:
        image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
        args:
          - -m
          - dynamo.vllm
          - --model
          - Qwen/Qwen3-0.6B
          - --is-prefill-worker
          
  VllmPrefillWorker:
    dynamoNamespace: vllm-disagg
    componentType: worker
    subComponentType: prefill         # Identifies prefill worker
    replicas: 1
    resources:
      limits:
        gpu: "1"
```

### Worker Configuration Handling

**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/deploy/cloud/operator/internal/dynamo/graph.go` (lines 119-188)

```go
func GenerateDynamoComponentsDeployments(ctx context.Context, 
    parentDynamoGraphDeployment *v1alpha1.DynamoGraphDeployment, 
    defaultIngressSpec *v1alpha1.IngressSpec) 
    (map[string]*v1alpha1.DynamoComponentDeployment, error) {
    
    // For each service in the DGD...
    for componentName, component := range parentDynamoGraphDeployment.Spec.Services {
        deployment := &v1alpha1.DynamoComponentDeployment{}
        deployment.Spec.DynamoComponentDeploymentSharedSpec = *component
        deployment.Name = GetDynamoComponentName(parentDynamoGraphDeployment, componentName)
        deployment.Spec.BackendFramework = parentDynamoGraphDeployment.Spec.BackendFramework
        // ... set labels, annotations, etc.
    }
}
```

### Scaling via Kubernetes Connector

**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/kube.py` (lines 81-93)

```python
def update_graph_replicas(self, graph_deployment_name: str, 
                         component_name: str, replicas: int) -> None:
    """Update the replicas count for a component in a DynamoGraphDeployment"""
    patch = {"spec": {"services": {component_name: {"replicas": replicas}}}}
    self.custom_api.patch_namespaced_custom_object(
        group="nvidia.com",
        version="v1alpha1",
        namespace=self.current_namespace,
        plural="dynamographdeployments",
        name=graph_deployment_name,
        body=patch,
    )
```

**Scaling Workflow:**
1. Planner calls `connector.set_component_replicas(target_replicas, blocking=False)`
2. Connector patches DGD spec.services[component_name].replicas
3. Kubernetes operator watches the DGD and creates/deletes Pods accordingly
4. Worker discovery system updates etcd with new instance information

---

## 5. Existing Multi-Deployment and Heterogeneous Worker Support

### Current State: MINIMAL SUPPORT

#### Multi-Deployment Support
**CURRENT:** Single DGD at a time
- Planner hardcoded to work with one DGD specified via environment variable `DYN_PARENT_DGD_K8S_NAME`
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/kubernetes_connector.py` (lines 63-69)

```python
graph_deployment_name = os.getenv("DYN_PARENT_DGD_K8S_NAME")
if not graph_deployment_name:
    raise DeploymentValidationError(
        ["DYN_PARENT_DGD_K8S_NAME environment variable is not set"]
    )
self.graph_deployment_name = graph_deployment_name
```

#### Heterogeneous Worker Support
**CURRENT:** Framework-level heterogeneity only
- Different backend frameworks (vllm, sglang, trtllm) supported
- Within a framework, all prefill workers assumed identical
- Within a framework, all decode workers assumed identical

**Supported Differentiation:**
1. **Backend framework** (global to entire DGD)
2. **Component type** (frontend, worker, planner)
3. **Sub-component type** (prefill, decode for workers)
4. **Resource limits** (different GPU configs per service possible)
5. **Multinode configuration** (NodeCount per service)

**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/defaults.py` (lines 99-139)

```python
WORKER_COMPONENT_NAMES = {
    "vllm": VllmComponentName,
    "sglang": SGLangComponentName,
    "trtllm": TrtllmComponentName,
}
```

**NOT currently supported:**
- Multiple independent DGDs with separate planners coordinating
- Mixed GPU types within the same prefill/decode worker pool
- Different model versions in same deployment
- Per-worker resource heterogeneity visible to planner

#### Virtual Connector (Non-Kubernetes Deployments)
**File:** `/Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/virtual_connector.py` (lines 28-143)

The VirtualConnector provides an abstraction for non-Kubernetes environments:

```python
class VirtualConnector(PlannerConnector):
    def __init__(self, runtime: DistributedRuntime, 
                 dynamo_namespace: str, 
                 model_name: Optional[str] = None):
        self.connector = VirtualConnectorCoordinator(...)
```

**Capabilities:**
- Communicates scaling decisions to external orchestration systems
- Still maintains one DynamoNamespace at a time
- Provides abstraction layer for different orchestration backends

---

## 6. Boundary Analysis for Hierarchical Planner Support

### Current Boundaries and Constraints

**Planner Boundaries:**
1. **Fixed to single DGD**: Environment variable `DYN_PARENT_DGD_K8S_NAME`
2. **Component discovery hard-coded**: Looks for "prefill" and "decode" subComponentTypes
3. **Flat worker pool assumption**: No distinction between worker groups
4. **Single adjustment loop**: Monolithic planner loop processes all metrics

**Router Boundaries:**
1. **Dynamic worker discovery**: Watches etcd for instance changes
2. **Worker pool agnostic**: KvRouter can handle variable number of workers
3. **Multi-worker configuration**: scheduler.rs tracks HashMap<WorkerId, ModelRuntimeConfig>
4. **No hierarchy awareness**: Treats all workers equally

**Scaling Mechanism Boundaries:**
1. **Kubernetes Connector**: Patches DGD services directly
2. **All-or-nothing updates**: Cannot scale partial worker groups
3. **Global state check**: `is_deployment_ready()` checks entire DGD status

### Key Integration Points for Hierarchical Planner

**Would need to change:**

1. **Planner initialization** (line 62-90 in planner_core.py):
   - Replace single environment variable with multi-DGD configuration
   - Support multiple parent DGD references

2. **Component discovery** (line 76-90 in kubernetes_connector.py):
   - Replace hardcoded prefill/decode lookup with multi-level discovery
   - Support grouping workers by deployment

3. **Replica setting** (line 219-258 in kubernetes_connector.py):
   - Extend to target multiple DGDs
   - Support partial scaling within DGD

4. **Metrics observation** (line 190-239 in planner_core.py):
   - Would need to filter metrics by deployment/worker group
   - Support per-deployment correction factors

5. **Decision computation** (line 313-407 in planner_core.py):
   - Would benefit from per-group computation
   - Could support different SLAs per deployment

---

## 7. File Locations Reference

### Core Architecture Files
```
DGD Definition:
  /Users/anishmaddipoti/Desktop/repos/dynamo/deploy/cloud/operator/api/v1alpha1/dynamographdeployment_types.go (lines 30-121)
  /Users/anishmaddipoti/Desktop/repos/dynamo/deploy/cloud/operator/api/v1alpha1/dynamocomponentdeployment_types.go (lines 38-272)

Kubernetes Operator/Controller:
  /Users/anishmaddipoti/Desktop/repos/dynamo/deploy/cloud/operator/internal/controller/dynamographdeployment_controller.go (lines 71-550)
  /Users/anishmaddipoti/Desktop/repos/dynamo/deploy/cloud/operator/internal/dynamo/graph.go (lines 48-300+)
```

### Planner Files
```
Main Planner Class:
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/utils/planner_core.py (lines 61-634)

Kubernetes Connector:
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/kubernetes_connector.py (lines 48-287)
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/kube.py (lines 40-137)

Virtual Connector:
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/virtual_connector.py (lines 28-143)

Planner Entry Points:
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/planner_sla.py (lines 36-56)
```

### Router Files
```
KV Router:
  /Users/anishmaddipoti/Desktop/repos/dynamo/lib/llm/src/kv_router.rs (lines 1-350+)

Router Scheduler:
  /Users/anishmaddipoti/Desktop/repos/dynamo/lib/llm/src/kv_router/scheduler.rs (lines 90-200+)

Router Protocols:
  /Users/anishmaddipoti/Desktop/repos/dynamo/lib/llm/src/kv_router/protocols.rs

Disagg Router:
  /Users/anishmaddipoti/Desktop/repos/dynamo/lib/llm/src/disagg_router.rs (lines 1-200+)
```

### Configuration Files
```
Planner Defaults:
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/src/dynamo/planner/defaults.py (lines 43-237)

Example Deployments:
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/backends/vllm/deploy/disagg.yaml
  /Users/anishmaddipoti/Desktop/repos/dynamo/components/backends/vllm/deploy/disagg_planner.yaml
```
