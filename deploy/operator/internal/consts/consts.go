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

	DynamoFPMBasePort = 20380

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
	KubeAnnotationDynamoKubeDiscoveryMode         = "nvidia.com/dynamo-kube-discovery-mode"

	KubeLabelDynamoGraphDeploymentName = "nvidia.com/dynamo-graph-deployment-name"
	KubeLabelDynamoComponent           = "nvidia.com/dynamo-component"
	KubeLabelDynamoNamespace           = "nvidia.com/dynamo-namespace"
	// KubeLabelDynamoComponentType is the workload selector contract stamped on
	// DCDs and rendered onto pods/services. Native v1beta1 prefill/decode worker
	// DCDs use "prefill" or "decode". A DCD generation that is already serving
	// alpha-era selectors uses "worker" and pairs it with
	// KubeLabelDynamoSubComponentType so a no-op upgrade keeps matching existing
	// pods. This selector contract is separate from worker-hash currentness:
	// matching current-worker-hash does not imply legacy selectors.
	KubeLabelDynamoComponentType    = "nvidia.com/dynamo-component-type"
	KubeLabelDynamoSubComponentType = "nvidia.com/dynamo-sub-component-type"
	KubeLabelDynamoComponentClass   = "nvidia.com/dynamo-component-class"
	KubeLabelDynamoBaseModel        = "nvidia.com/dynamo-base-model"
	KubeLabelDynamoBaseModelHash    = "nvidia.com/dynamo-base-model-hash"
	KubeAnnotationDynamoBaseModel   = "nvidia.com/dynamo-base-model"
	KubeLabelDynamoDiscoveryBackend = "nvidia.com/dynamo-discovery-backend"
	KubeLabelDynamoDiscoveryEnabled = "nvidia.com/dynamo-discovery-enabled"
	// KubeLabelDynamoWorkerHash is the worker generation label on worker DCDs
	// and worker pods. During v1/v2 hash compatibility the label key remains
	// stable and the value may be either the active v1 hash or the active v2 hash
	// recorded on the parent DGD. Older operators understand only the v1 value,
	// so v1-compatible releases continue to generate new DCDs with the v1 value.
	KubeLabelDynamoWorkerHash = "nvidia.com/dynamo-worker-hash"

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

	// DynamoOperatorAllowGMSSnapshotEnvVar enables the temporary internal
	// GMS + Snapshot admission gate when set to "1".
	DynamoOperatorAllowGMSSnapshotEnvVar = "DYN_OPERATOR_ALLOW_GMS_SNAPSHOT"

	GlobalDynamoNamespace = "dynamo"

	ComponentTypePlanner  = "planner"
	ComponentTypeFrontend = "frontend"
	ComponentTypeWorker   = "worker"
	ComponentTypePrefill  = "prefill"
	ComponentTypeDecode   = "decode"
	ComponentTypeEPP      = "epp"
	ComponentTypeDefault  = "default"

	ComponentClassWorker      = "worker"
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
	GroveRoleSuffixGMS    = "gms"

	// MaxCombinedGroveResourceNameLength is the maximum allowed combined length for Grove
	// resource names (PCS name + PCSG config name + PCLQ template name).
	// This constraint comes from Grove's PodCliqueSet webhook validation.
	// Pod names follow: <pcs-name>-<pcs-index>-<pcsg-name>-<pcsg-index>-<pclq-name>-<random>
	// The hyphens, indices, and random suffix consume additional characters beyond this limit.
	MaxCombinedGroveResourceNameLength = 45

	KubeLabelDynamoFailoverEngineGroupMember = "nvidia.com/dynamo-failover-engine-group-member"

	DiscoveryBackendKubernetes   = "kubernetes" // label value for KubeLabelDynamoDiscoveryBackend
	MainContainerName            = "main"
	FrontendSidecarContainerName = "sidecar-frontend"

	RestartAnnotation = "nvidia.com/restartAt"

	// Resource type constants - match Kubernetes Kind names
	// Used consistently across controllers, webhooks, and metrics
	ResourceTypeDynamoGraphDeployment               = "DynamoGraphDeployment"
	ResourceTypeDynamoComponentDeployment           = "DynamoComponentDeployment"
	ResourceTypeDynamoCheckpoint                    = "DynamoCheckpoint"
	ResourceTypeDynamoModel                         = "DynamoModel"
	ResourceTypeDynamoGraphDeploymentRequest        = "DynamoGraphDeploymentRequest"
	ResourceTypeDynamoGraphDeploymentScalingAdapter = "DynamoGraphDeploymentScalingAdapter"

	// Resource state constants - used in status reporting and metrics
	ResourceStateReady    = "ready"
	ResourceStateNotReady = "not_ready"
	ResourceStateUnknown  = "unknown"

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

	// Worker hash rolling-update annotations are controller-owned annotations on
	// DynamoGraphDeployment. They record the active worker generation and must not
	// be treated as user-configurable inputs. During a managed rolling update,
	// these annotations remain on the previously serving worker generation until
	// the new generation is fully ready and old workers have drained.
	//
	// The compatibility contract is intentionally additive: existing annotation
	// and label keys keep their old meaning. AnnotationCurrentWorkerHash stores
	// the v1alpha1-compatible worker hash so a downgrade can still understand the
	// active generation. AnnotationCurrentWorkerHashV2 stores the v2 worker hash
	// for the same active generation. A worker DCD whose
	// KubeLabelDynamoWorkerHash value matches either annotation is current. While
	// v1 compatibility is required, generated worker DCDs use the v1 hash as the
	// label value. If a worker change is visible only to v2, the controller
	// removes the v1 annotation and rolls to a v2-labeled DCD because the v1 hash
	// can no longer prove pod-template compatibility. A future v2-only release
	// can start using the v2 value with the same label key and keep accepting the
	// v1 annotation until the next v2 generation change drains old workers.

	// AnnotationCurrentWorkerHash stores the active v1alpha1-compatible worker
	// generation hash.
	AnnotationCurrentWorkerHash = "nvidia.com/current-worker-hash"

	// AnnotationCurrentWorkerHashV2 stores the active v2 worker generation hash.
	AnnotationCurrentWorkerHashV2 = "nvidia.com/current-worker-hash-v2"

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
