package consts

import (
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	DefaultUserId = "default"
	DefaultOrgId  = "default"

	DynamoServicePort       = 8000
	DynamoServicePortName   = "http"
	DynamoContainerPortName = "http"

	DynamoPlannerMetricsPort = 9085
	DynamoMetricsPortName    = "metrics"

	DynamoSystemPort     = 9090
	DynamoSystemPortName = "system"

	// EPP (Endpoint Picker Plugin) ports
	EPPGRPCPort     = 9002
	EPPGRPCPortName = "grpc"

	DynamoNixlPort     = 19090
	DynamoNixlPortName = "nixl"

	MpiRunSshPort = 2222

	// Default security context values
	// These provide secure defaults for running containers as non-root
	// Users can override these via extraPodSpec.securityContext in their DynamoGraphDeployment
	DefaultSecurityContextFSGroup = 1000

	EnvDynamoServicePort = "DYNAMO_PORT"

	KubeLabelDynamoSelector = "nvidia.com/selector"

	KubeAnnotationEnableGrove = "nvidia.com/enable-grove"

	KubeAnnotationDisableImagePullSecretDiscovery = "nvidia.com/disable-image-pull-secret-discovery"
	KubeAnnotationDynamoDiscoveryBackend          = "nvidia.com/dynamo-discovery-backend"

	KubeLabelDynamoGraphDeploymentName = "nvidia.com/dynamo-graph-deployment-name"
	KubeLabelDynamoComponent           = "nvidia.com/dynamo-component"
	KubeLabelDynamoNamespace           = "nvidia.com/dynamo-namespace"
	KubeLabelDynamoComponentType       = "nvidia.com/dynamo-component-type"
	KubeLabelDynamoSubComponentType    = "nvidia.com/dynamo-sub-component-type"
	KubeLabelDynamoBaseModel           = "nvidia.com/dynamo-base-model"
	KubeLabelDynamoBaseModelHash       = "nvidia.com/dynamo-base-model-hash"
	KubeAnnotationDynamoBaseModel      = "nvidia.com/dynamo-base-model"
	KubeLabelDynamoDiscoveryBackend    = "nvidia.com/dynamo-discovery-backend"
	KubeLabelDynamoDiscoveryEnabled    = "nvidia.com/dynamo-discovery-enabled"
	KubeLabelDynamoWorkerHash          = "nvidia.com/dynamo-worker-hash"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelDynamoComponentPod = "nvidia.com/dynamo-component-pod"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	DynamoDeploymentConfigEnvVar      = "DYN_DEPLOYMENT_CONFIG"
	DynamoNamespaceEnvVar             = "DYN_NAMESPACE"
	DynamoNamespacePrefixEnvVar       = "DYN_NAMESPACE_PREFIX"
	DynamoNamespaceWorkerSuffixEnvVar = "DYN_NAMESPACE_WORKER_SUFFIX"
	DynamoComponentEnvVar             = "DYN_COMPONENT"
	DynamoDiscoveryBackendEnvVar      = "DYN_DISCOVERY_BACKEND"

	GlobalDynamoNamespace = "dynamo"

	ComponentTypePlanner      = "planner"
	ComponentTypeFrontend     = "frontend"
	ComponentTypeWorker       = "worker"
	ComponentTypePrefill      = "prefill"
	ComponentTypeDecode       = "decode"
	ComponentTypeEPP          = "epp"
	ComponentTypeDefault      = "default"
	PlannerServiceAccountName = "planner-serviceaccount"
	EPPServiceAccountName     = "epp-serviceaccount"
	EPPClusterRoleName        = "epp-cluster-role"

	DefaultIngressSuffix = "local"

	DefaultGroveTerminationDelay = 15 * time.Minute

	// Operator origin version: stamped on DGD at creation time by mutating webhook.
	// Records which operator version created the resource, enabling version-gated behavior changes.
	KubeAnnotationDynamoOperatorOriginVersion = "nvidia.com/dynamo-operator-origin-version"

	// vLLM distributed executor backend override annotation.
	// Users can set this on a DGD to explicitly choose "mp" or "ray" for multi-node vLLM deployments.
	// When present, takes priority over the version-based default.
	KubeAnnotationVLLMDistributedExecutorBackend = "nvidia.com/vllm-distributed-executor-backend"

	// VLLMMpMasterPort is the default port for vLLM multiprocessing coordination between nodes.
	VLLMMpMasterPort = "29500"

	// VLLMNixlSideChannelHostEnvVar is the env var that tells vLLM which host IP to use for the NIXL side channel.
	VLLMNixlSideChannelHostEnvVar = "VLLM_NIXL_SIDE_CHANNEL_HOST"

	// Metrics related constants
	KubeAnnotationEnableMetrics  = "nvidia.com/enable-metrics"  // User-provided annotation to control metrics
	KubeLabelMetricsEnabled      = "nvidia.com/metrics-enabled" // Controller-managed label for pod selection
	KubeValueNameSharedMemory    = "shared-memory"
	DefaultSharedMemoryMountPath = "/dev/shm"
	DefaultSharedMemorySize      = "8Gi"

	// Compilation cache default mount points
	DefaultVLLMCacheMountPoint = "/root/.cache/vllm"

	// Kai-scheduler related constants
	KubeAnnotationKaiSchedulerQueue = "nvidia.com/kai-scheduler-queue" // User-provided annotation to specify queue name
	KubeLabelKaiSchedulerQueue      = "kai.scheduler/queue"            // Label injected into pods for kai-scheduler
	KaiSchedulerName                = "kai-scheduler"                  // Scheduler name for kai-scheduler
	DefaultKaiSchedulerQueue        = "dynamo"                         // Default queue name when none specified

	// Grove multinode role suffixes
	GroveRoleSuffixLeader = "ldr"
	GroveRoleSuffixWorker = "wkr"

	MainContainerName = "main"

	RestartAnnotation = "nvidia.com/restartAt"

	// Resource type constants - match Kubernetes Kind names
	// Used consistently across controllers, webhooks, and metrics
	ResourceTypeDynamoGraphDeployment               = "DynamoGraphDeployment"
	ResourceTypeDynamoComponentDeployment           = "DynamoComponentDeployment"
	ResourceTypeDynamoModel                         = "DynamoModel"
	ResourceTypeDynamoGraphDeploymentRequest        = "DynamoGraphDeploymentRequest"
	ResourceTypeDynamoGraphDeploymentScalingAdapter = "DynamoGraphDeploymentScalingAdapter"

	// Resource state constants - used in status reporting and metrics
	ResourceStateReady    = "ready"
	ResourceStateNotReady = "not_ready"
	ResourceStateUnknown  = "unknown"

	// Checkpoint/restore constants
	// CROSS-REFERENCE: Some constants below are duplicated in the chrek package at
	// deploy/chrek/pkg/config/constants.go. If you change a value here, update there too.

	// Kubernetes labels
	KubeLabelIsCheckpointSource = "nvidia.com/chrek-is-checkpoint-source" // Pod label that triggers DaemonSet auto-checkpoint
	KubeLabelCheckpointHash     = "nvidia.com/chrek-checkpoint-hash"      // Checkpoint identity hash (= DynamoCheckpoint CR name)
	KubeLabelIsRestoreTarget    = "nvidia.com/chrek-is-restore-target"    // Pod label that triggers DaemonSet auto-restore

	// Environment variables injected into pods
	EnvCheckpointStorageType  = "DYN_CHECKPOINT_STORAGE_TYPE"   // Storage backend (pvc, s3, oci) — checkpoint job pods only
	EnvCheckpointLocation     = "DYN_CHECKPOINT_LOCATION"       // Full checkpoint URI — future S3/OCI; for PVC, use PATH+HASH instead
	EnvCheckpointPath         = "DYN_CHECKPOINT_PATH"           // Base checkpoint directory (e.g., /checkpoints) — PVC restored pods
	EnvCheckpointHash         = "DYN_CHECKPOINT_HASH"           // Identity hash — all checkpoint-related pods
	EnvReadyForCheckpointFile = "DYN_READY_FOR_CHECKPOINT_FILE" // Ready-for-checkpoint file path — checkpoint job pods
	EnvSkipWaitForCheckpoint  = "SKIP_WAIT_FOR_CHECKPOINT"      // Skip polling, check once — restored/DGD pods
	// Checkpoint pod-internal constants
	CheckpointVolumeName = "checkpoint-storage" // Pod-internal volume name for checkpoint PVC

	// SeccompProfilePath is the localhost seccomp profile that blocks io_uring syscalls.
	// Deployed to nodes by the chrek DaemonSet init container.
	SeccompProfilePath = "profiles/block-iouring.json"

	// Pod identity (Downward API) ---
	// After CRIU restore, env vars contain stale values from the checkpoint pod.
	// The Downward API files at /etc/podinfo always reflect the current pod.
	PodInfoVolumeName = "podinfo"
	PodInfoMountPath  = "/etc/podinfo"

	// Downward API field paths
	PodInfoFieldPodName      = "metadata.name"
	PodInfoFieldPodUID       = "metadata.uid"
	PodInfoFieldPodNamespace = "metadata.namespace"

	// Downward API file names for DGD annotations
	PodInfoFileDynNamespace        = "dyn_namespace"
	PodInfoFileDynComponent        = "dyn_component"
	PodInfoFileDynParentDGDName    = "dyn_parent_dgd_name"
	PodInfoFileDynParentDGDNS      = "dyn_parent_dgd_namespace"
	PodInfoFileDynDiscoveryBackend = "dyn_discovery_backend"

	// Annotation keys for DGD info (exposed via Downward API)
	AnnotationDynNamespace        = "nvidia.com/dyn-namespace"
	AnnotationDynComponent        = "nvidia.com/dyn-component"
	AnnotationDynParentDGDName    = "nvidia.com/dyn-parent-dgd-name"
	AnnotationDynParentDGDNS      = "nvidia.com/dyn-parent-dgd-namespace"
	AnnotationDynDiscoveryBackend = "nvidia.com/dyn-discovery-backend"

	// Rolling update annotations
	AnnotationCurrentWorkerHash = "nvidia.com/current-worker-hash"

	// LegacyWorkerHash is a sentinel value used during migration from pre-rolling-update
	// operator versions. Legacy worker DCDs (those without a worker hash label) are tagged
	// with this value so the existing rolling update machinery can manage the transition.
	LegacyWorkerHash = "legacy"
)

type MultinodeDeploymentType string

const (
	MultinodeDeploymentTypeGrove MultinodeDeploymentType = "grove"
	MultinodeDeploymentTypeLWS   MultinodeDeploymentType = "lws"
)

// GroupVersionResources for external APIs
var (
	// Grove GroupVersionResources for scaling operations
	PodCliqueGVR = schema.GroupVersionResource{
		Group:    "grove.io",
		Version:  "v1alpha1",
		Resource: "podcliques",
	}
	PodCliqueScalingGroupGVR = schema.GroupVersionResource{
		Group:    "grove.io",
		Version:  "v1alpha1",
		Resource: "podcliquescalinggroups",
	}

	// KAI-Scheduler GroupVersionResource for queue validation
	QueueGVR = schema.GroupVersionResource{
		Group:    "scheduling.run.ai",
		Version:  "v2",
		Resource: "queues",
	}
)
