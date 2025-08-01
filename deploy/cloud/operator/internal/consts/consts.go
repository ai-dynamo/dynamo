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

	DefaultIngressSuffix = "local"

	DefaultGroveTerminationDelay = 15 * time.Minute
)
