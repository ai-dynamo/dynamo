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

	MainContainerName            = "main"
	FrontendSidecarContainerName = "sidecar-frontend"

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
	// CROSS-REFERENCE: Some constants below are duplicated in the snapshot package at
	// deploy/snapshot/pkg/config/constants.go. If you change a value here, update there too.

	// Kubernetes labels
	KubeLabelIsCheckpointSource             = "nvidia.com/snapshot-is-checkpoint-source" // Pod label that triggers DaemonSet auto-checkpoint
	KubeLabelCheckpointHash                 = "nvidia.com/snapshot-checkpoint-hash"      // Checkpoint identity hash used for lookup/reuse (may differ from DynamoCheckpoint metadata.name)
	KubeLabelIsRestoreTarget                = "nvidia.com/snapshot-is-restore-target"    // Pod label that triggers DaemonSet auto-restore
	KubeAnnotationCheckpointArtifactVersion = "nvidia.com/snapshot-artifact-version"     // Checkpoint artifact generation; changing it triggers a new immutable capture attempt
	DefaultCheckpointArtifactVersion        = "1"
	KubeAnnotationCheckpointLocation        = "nvidia.com/snapshot-checkpoint-location"     // Pod annotation that tells snapshot-agent where the checkpoint lives
	KubeAnnotationCheckpointStorageType     = "nvidia.com/snapshot-checkpoint-storage-type" // Pod annotation that tells snapshot-agent which storage backend owns the checkpoint

	// Environment variables injected into pods
	EnvReadyForCheckpointFile = "DYN_READY_FOR_CHECKPOINT_FILE" // Ready-for-checkpoint file path — checkpoint job pods
	// Checkpoint pod-internal constants
	CheckpointVolumeName = "checkpoint-storage" // Pod-internal volume name for checkpoint PVC

	// SeccompProfilePath is the localhost seccomp profile that blocks io_uring syscalls.
	// Deployed to nodes by the snapshot DaemonSet init container.
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

	// Downward API file names for restore identity
	PodInfoFileDynNamespace             = "dyn_namespace"
	PodInfoFileDynNamespaceWorkerSuffix = "dyn_namespace_worker_suffix"
	PodInfoFileDynComponent             = "dyn_component"
	PodInfoFileDynParentDGDName         = "dyn_parent_dgd_k8s_name"
	PodInfoFileDynParentDGDNamespace    = "dyn_parent_dgd_k8s_namespace"

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
