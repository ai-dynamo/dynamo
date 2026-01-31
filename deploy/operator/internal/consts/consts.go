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

	KubeLabelDynamoGraphDeploymentName  = "nvidia.com/dynamo-graph-deployment-name"
	KubeLabelDynamoComponent            = "nvidia.com/dynamo-component"
	KubeLabelDynamoNamespace            = "nvidia.com/dynamo-namespace"
	KubeLabelDynamoDeploymentTargetType = "nvidia.com/dynamo-deployment-target-type"
	KubeLabelDynamoComponentType        = "nvidia.com/dynamo-component-type"
	KubeLabelDynamoSubComponentType     = "nvidia.com/dynamo-sub-component-type"
	KubeLabelDynamoBaseModel            = "nvidia.com/dynamo-base-model"
	KubeLabelDynamoBaseModelHash        = "nvidia.com/dynamo-base-model-hash"
	KubeAnnotationDynamoBaseModel       = "nvidia.com/dynamo-base-model"
	KubeLabelDynamoDiscoveryBackend     = "nvidia.com/dynamo-discovery-backend"
	KubeLabelDynamoDiscoveryEnabled     = "nvidia.com/dynamo-discovery-enabled"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelDynamoComponentPod = "nvidia.com/dynamo-component-pod"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	DynamoDeploymentConfigEnvVar = "DYN_DEPLOYMENT_CONFIG"
	DynamoNamespaceEnvVar        = "DYN_NAMESPACE"
	DynamoComponentEnvVar        = "DYN_COMPONENT"
	DynamoDiscoveryBackendEnvVar = "DYN_DISCOVERY_BACKEND"

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
	// Checkpoint related constants
	KubeLabelCheckpointSource = "nvidia.com/checkpoint-source"
	KubeLabelCheckpointHash   = "nvidia.com/checkpoint-hash"
	KubeLabelCheckpointName   = "nvidia.com/checkpoint-name"

	// EnvCheckpointStorageType indicates the storage backend type (pvc, s3, oci)
	EnvCheckpointStorageType = "DYN_CHECKPOINT_STORAGE_TYPE"
	// EnvCheckpointLocation is the source location of the checkpoint
	// For PVC: same as path (e.g., /checkpoints/{hash}.tar)
	// For S3: s3://bucket/prefix/{hash}.tar
	// For OCI: oci://registry/repo:{hash}
	EnvCheckpointLocation = "DYN_CHECKPOINT_LOCATION"
	// EnvCheckpointPath is the local path to the checkpoint tar file
	// For PVC: same as location
	// For S3/OCI: download destination (e.g., /tmp/{hash}.tar)
	EnvCheckpointPath = "DYN_CHECKPOINT_PATH"
	// EnvCheckpointHash is the identity hash (for debugging/observability)
	EnvCheckpointHash = "DYN_CHECKPOINT_HASH"
	// EnvCheckpointSignalFile is the full path to the signal file
	// The DaemonSet writes this file after checkpoint is complete
	// The checkpoint job pod waits for this file, then exits successfully
	EnvCheckpointSignalFile = "DYN_CHECKPOINT_SIGNAL_FILE"

	// EnvCheckpointReadyFile is the full path to a file the worker creates
	// when the model is loaded and ready for checkpointing.
	// The readiness probe watches this file to trigger DaemonSet checkpoint.
	EnvCheckpointReadyFile = "DYN_CHECKPOINT_READY_FILE"

	// CRIU-related environment variables for restore operations
	// EnvRestoreMarkerFile is the file created by CRIU after successful restore
	EnvRestoreMarkerFile = "DYN_RESTORE_MARKER_FILE"
	// EnvCRIUWorkDir is the working directory for CRIU operations
	EnvCRIUWorkDir = "CRIU_WORK_DIR"
	// EnvCRIULogDir is the directory where CRIU writes logs
	EnvCRIULogDir = "CRIU_LOG_DIR"
	// EnvCUDAPluginDir is the directory containing CRIU CUDA plugins
	EnvCUDAPluginDir = "CUDA_PLUGIN_DIR"
	// EnvCRIUTimeout is the timeout for CRIU operations
	EnvCRIUTimeout = "CRIU_TIMEOUT"

	// CheckpointReadyFilePath is the default path for the ready file
	CheckpointReadyFilePath = "/tmp/checkpoint-ready"
	// RestoreMarkerFilePath is the default path for the restore marker
	RestoreMarkerFilePath = "/tmp/dynamo-restored"
	// CRIUWorkDirPath is the default CRIU work directory
	CRIUWorkDirPath = "/var/criu-work"
	// CRIULogDirPath is the default CRIU log directory
	CRIULogDirPath = "/checkpoints/restore-logs"
	// CUDAPluginDirPath is the default CUDA plugin directory
	CUDAPluginDirPath = "/usr/local/lib/criu"
	// DefaultCRIUTimeout is the default CRIU timeout in seconds (6 hours)
	DefaultCRIUTimeout = "21600"

	CheckpointVolumeName       = "checkpoint-storage"
	CheckpointSignalVolumeName = "checkpoint-signal"
	CheckpointBasePath         = "/checkpoints"
	CheckpointSignalHostPath   = "/var/lib/dynamo-checkpoint/signals"
	CheckpointSignalMountPath  = "/checkpoint-signal"

	// PodInfo volume for Downward API (critical for CRIU restore)
	// After CRIU restore, environment variables contain stale values from checkpoint pod.
	// The Downward API files at /etc/podinfo always have current pod identity.
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
