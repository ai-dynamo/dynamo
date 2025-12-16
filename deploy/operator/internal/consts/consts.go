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
	ComponentTypeDefault      = "default"
	PlannerServiceAccountName = "planner-serviceaccount"

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
	// Checkpoint related constants
	KubeLabelCheckpointSource = "nvidia.com/checkpoint-source"
	KubeLabelCheckpointHash   = "nvidia.com/checkpoint-hash"
	KubeLabelCheckpointName   = "nvidia.com/checkpoint-name"

	// EnvCheckpointStorageType indicates the storage backend type (pvc, s3, oci)
	EnvCheckpointStorageType = "DYNAMO_CHECKPOINT_STORAGE_TYPE"
	// EnvCheckpointLocation is the source location of the checkpoint
	// For PVC: same as path (e.g., /checkpoints/{hash}.tar)
	// For S3: s3://bucket/prefix/{hash}.tar
	// For OCI: oci://registry/repo:{hash}
	EnvCheckpointLocation = "DYNAMO_CHECKPOINT_LOCATION"
	// EnvCheckpointPath is the local path to the checkpoint tar file
	// For PVC: same as location
	// For S3/OCI: download destination (e.g., /tmp/{hash}.tar)
	EnvCheckpointPath = "DYNAMO_CHECKPOINT_PATH"
	// EnvCheckpointHash is the identity hash (for debugging/observability)
	EnvCheckpointHash = "DYNAMO_CHECKPOINT_HASH"
	// EnvCheckpointSignalFile is the full path to the signal file
	// The DaemonSet writes this file after checkpoint is complete
	// The checkpoint job pod waits for this file, then exits successfully
	EnvCheckpointSignalFile = "DYNAMO_CHECKPOINT_SIGNAL_FILE"

	CheckpointVolumeName       = "checkpoint-storage"
	CheckpointSignalVolumeName = "checkpoint-signal"
	CheckpointBasePath         = "/checkpoints"
	CheckpointSignalHostPath   = "/var/lib/dynamo-checkpoint/signals"
	CheckpointSignalMountPath  = "/checkpoint-signal"
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
