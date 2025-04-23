package consts

const (
	HPACPUDefaultAverageUtilization = 80

	DefaultUserId = "default"
	DefaultOrgId  = "default"

	DynamoServicePort       = 3000
	DynamoServicePortName   = "http"
	DynamoContainerPortName = "http"

	DynamoImageBuilderComponentName = "dynamo-image-builder"

	DynamoDeploymentComponentApiServer = "api-server"

	InternalImagesDynamoComponentsDownloaderDefault = "quay.io/bentoml/bento-downloader:0.0.3"
	InternalImagesKanikoDefault                     = "gcr.io/kaniko-project/executor:debug"
	InternalImagesMetricsTransformerDefault         = "quay.io/bentoml/yatai-bento-metrics-transformer:0.0.3"
	InternalImagesBuildkitDefault                   = "moby/buildkit:v0.20.2"
	InternalImagesBuildkitRootlessDefault           = "moby/buildkit:v0.20.2-rootless"

	EnvApiStoreEndpoint    = "API_STORE_ENDPOINT"
	EnvApiStoreClusterName = "API_STORE_CLUSTER_NAME"
	// nolint: gosec
	EnvApiStoreApiToken = "API_STORE_API_TOKEN"

	EnvDynamoServicePort = "PORT"

	// tracking envars
	EnvDynamoDeploymentUID = "DYNAMO_DEPLOYMENT_UID"

	EnvDynamoDeploymentName      = "DYNAMO_DEPLOYMENT_NAME"
	EnvDynamoDeploymentNamespace = "DYNAMO_DEPLOYMENT_NAMESPACE"

	EnvDockerRegistryServer          = "DOCKER_REGISTRY_SERVER"
	EnvDockerRegistryInClusterServer = "DOCKER_REGISTRY_IN_CLUSTER_SERVER"
	EnvDockerRegistryUsername        = "DOCKER_REGISTRY_USERNAME"
	// nolint:gosec
	EnvDockerRegistryPassword                       = "DOCKER_REGISTRY_PASSWORD"
	EnvDockerRegistrySecure                         = "DOCKER_REGISTRY_SECURE"
	EnvDockerRegistryDynamoComponentsRepositoryName = "DOCKER_REGISTRY_DYNAMO_COMPONENTS_REPOSITORY_NAME"

	EnvInternalImagesDynamoComponentsDownloader = "INTERNAL_IMAGES_DYNAMO_COMPONENTS_DOWNLOADER"
	EnvInternalImagesKaniko                     = "INTERNAL_IMAGES_KANIKO"
	EnvInternalImagesMetricsTransformer         = "INTERNAL_IMAGES_METRICS_TRANSFORMER"
	EnvInternalImagesBuildkit                   = "INTERNAL_IMAGES_BUILDKIT"
	EnvInternalImagesBuildkitRootless           = "INTERNAL_IMAGES_BUILDKIT_ROOTLESS"

	EnvDynamoSystemNamespace       = "DYNAMO_SYSTEM_NAMESPACE"
	EnvDynamoImageBuilderNamespace = "DYNAMO_IMAGE_BUILDER_NAMESPACE"

	KubeLabelDynamoSelector   = "nvidia.com/selector"
	KubeLabelDynamoRepository = "nvidia.com/dynamo-repository"
	KubeLabelDynamoVersion    = "nvidia.com/dynamo-version"

	KubeLabelDynamoDeployment              = "nvidia.com/dynamo-deployment"
	KubeLabelDynamoDeploymentComponentType = "nvidia.com/dynamo-deployment-component-type"
	KubeLabelDynamoDeploymentTargetType    = "nvidia.com/dynamo-deployment-target-type"
	KubeLabelDynamoCreator                 = "nvidia.com/dynamo-creator"

	KubeLabelIsDynamoImageBuilder = "nvidia.com/is-dynamo-image-builder"
	KubeLabelDynamoRequest        = "nvidia.com/dynamo-request"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelDynamoImageBuilderPod = "nvidia.com/dynamo-image-builder-pod"
	KubeLabelDynamoDeploymentPod   = "nvidia.com/dynamo-deployment-pod"

	KubeAnnotationDynamoRepository             = "nvidia.com/dynamo-repository"
	KubeAnnotationDynamoVersion                = "nvidia.com/dynamo-version"
	KubeAnnotationDynamoDockerRegistryInsecure = "nvidia.com/docker-registry-insecure"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	// nolint: gosec
	KubeSecretNameRegcred = "dynamo-regcred"

	KubeAnnotationDynamoNimRequestHash            = "nvidia.com/dynamo-request-hash"
	KubeAnnotationDynamoNimRequestImageBuiderHash = "nvidia.com/dynamo-request-image-builder-hash"
	KubeAnnotationDynamoNimStorageNS              = "nvidia.com/dynamo-storage-namespace"
)
