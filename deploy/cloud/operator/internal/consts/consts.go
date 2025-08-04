package consts

import "time"

const (
	HPACPUDefaultAverageUtilization = 80

	DefaultUserId = "default"
	DefaultOrgId  = "default"

	DynamoServicePort       = 8000
	DynamoServicePortName   = "http"
	DynamoContainerPortName = "http"

	DynamoHealthPort     = 5000
	DynamoHealthPortName = "health"

	EnvDynamoServicePort = "DYNAMO_PORT"

	KubeLabelDynamoSelector = "nvidia.com/selector"

	KubeAnnotationEnableGrove = "nvidia.com/enable-grove"

	KubeLabelDynamoComponent            = "nvidia.com/dynamo-component"
	KubeLabelDynamoNamespace            = "nvidia.com/dynamo-namespace"
	KubeLabelDynamoDeploymentTargetType = "nvidia.com/dynamo-deployment-target-type"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelDynamoComponentPod = "nvidia.com/dynamo-component-pod"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	DynamoDeploymentConfigEnvVar = "DYN_DEPLOYMENT_CONFIG"

	ComponentTypePlanner      = "planner"
	ComponentTypeMain         = "main"
	PlannerServiceAccountName = "planner-serviceaccount"

	// DynamoConfig componentType values
	ComponentTypeWorker        = "worker"        // Aggregated serving workers
	ComponentTypePrefillWorker = "prefillWorker" // Disaggregated prefill workers
	ComponentTypeDecodeWorker  = "decodeWorker"  // Disaggregated decode workers

	DefaultIngressSuffix = "local"

	DefaultGroveTerminationDelay = 15 * time.Minute
	KubeValueNameSharedMemory    = "shared-memory"
)

type MultinodeDeploymentType string

const (
	MultinodeDeploymentTypeGrove MultinodeDeploymentType = "grove"
	MultinodeDeploymentTypeLWS   MultinodeDeploymentType = "lws"
)
