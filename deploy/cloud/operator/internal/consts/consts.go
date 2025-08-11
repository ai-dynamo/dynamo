package consts

import "time"

const (
	HPACPUDefaultAverageUtilization = 80

	DefaultUserId = "default"
	DefaultOrgId  = "default"

	DynamoServicePort       = 8000
	DynamoServicePortName   = "http"
	DynamoContainerPortName = "http"

	DynamoSystemPort     = 9090
	DynamoSystemPortName = "system"

	EnvDynamoServicePort = "DYNAMO_PORT"

	KubeLabelDynamoSelector = "nvidia.com/selector"

	KubeAnnotationEnableGrove = "nvidia.com/enable-grove"

	KubeLabelDynamoComponent            = "nvidia.com/dynamo-component"
	KubeLabelDynamoNamespace            = "nvidia.com/dynamo-namespace"
	KubeLabelDynamoDeploymentTargetType = "nvidia.com/dynamo-deployment-target-type"
	KubeLabelDynamoComponentType        = "nvidia.com/dynamo-component-type"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelDynamoComponentPod = "nvidia.com/dynamo-component-pod"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	DynamoDeploymentConfigEnvVar = "DYN_DEPLOYMENT_CONFIG"

	ComponentTypePlanner      = "planner"
	ComponentTypeMain         = "main"
	ComponentTypeWorker       = "worker"
	PlannerServiceAccountName = "planner-serviceaccount"

	DefaultIngressSuffix = "local"

	DefaultGroveTerminationDelay = 15 * time.Minute

	// Metrics related constants
	KubeAnnotationEnableMetrics = "nvidia.com/enable-metrics"  // User-provided annotation to control metrics
	KubeLabelMetricsEnabled     = "nvidia.com/metrics-enabled" // Controller-managed label for pod selection
)
