package consts

const (
	HPACPUDefaultAverageUtilization = 80

	DefaultUserId = "default"
	DefaultOrgId  = "default"

	DynamoServicePort       = 3000
	DynamoServicePortName   = "http"
	DynamoContainerPortName = "http"

	DynamoImageBuilderComponentName = "dynamo-image-builder"

	DynamoApiServerComponentName = "api-server"

	InternalImagesDynamoComponentsDownloaderDefault = "rapidfort/curl:latest"
	InternalImagesKanikoDefault                     = "gcr.io/kaniko-project/executor:debug"
	InternalImagesBuildkitDefault                   = "moby/buildkit:v0.20.2"
	InternalImagesBuildkitRootlessDefault           = "moby/buildkit:v0.20.2-rootless"

	EnvApiStoreEndpoint    = "API_STORE_ENDPOINT"
	EnvApiStoreClusterName = "API_STORE_CLUSTER_NAME"
	// nolint: gosec
	EnvApiStoreApiToken = "API_STORE_API_TOKEN"

	EnvDynamoServicePort = "PORT"

	EnvDockerRegistryServer                         = "DOCKER_REGISTRY_SERVER"
	EnvDockerRegistrySecret                         = "DOCKER_REGISTRY_SECRET_NAME"
	EnvDockerRegistrySecure                         = "DOCKER_REGISTRY_SECURE"
	EnvDockerRegistryDynamoComponentsRepositoryName = "DOCKER_REGISTRY_DYNAMO_COMPONENTS_REPOSITORY_NAME"

	EnvInternalImagesDynamoComponentsDownloader = "INTERNAL_IMAGES_DYNAMO_COMPONENTS_DOWNLOADER"
	EnvInternalImagesKaniko                     = "INTERNAL_IMAGES_KANIKO"
	EnvInternalImagesBuildkit                   = "INTERNAL_IMAGES_BUILDKIT"
	EnvInternalImagesBuildkitRootless           = "INTERNAL_IMAGES_BUILDKIT_ROOTLESS"

	EnvDynamoSystemNamespace       = "DYNAMO_SYSTEM_NAMESPACE"
	EnvDynamoImageBuilderNamespace = "DYNAMO_IMAGE_BUILDER_NAMESPACE"

	KubeLabelDynamoSelector = "nvidia.com/selector"

	KubeLabelDynamoComponent            = "nvidia.com/dynamo-component"
	KubeLabelDynamoNamespace            = "nvidia.com/dynamo-namespace"
	KubeLabelDynamoDeploymentTargetType = "nvidia.com/dynamo-deployment-target-type"

	KubeLabelDynamoComponentType = "nvidia.com/dynamo-component-type"

	KubeLabelIsDynamoImageBuilder = "nvidia.com/is-dynamo-image-builder"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelDynamoImageBuilderPod = "nvidia.com/dynamo-image-builder-pod"
	KubeLabelDynamoDeploymentPod   = "nvidia.com/dynamo-deployment-pod"

	KubeAnnotationDynamoRepository             = "nvidia.com/dynamo-repository"
	KubeAnnotationDynamoVersion                = "nvidia.com/dynamo-version"
	KubeAnnotationDynamoDockerRegistryInsecure = "nvidia.com/docker-registry-insecure"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	KubeAnnotationDynamoComponentHash            = "nvidia.com/dynamo-request-hash"
	KubeAnnotationDynamoComponentImageBuiderHash = "nvidia.com/dynamo-request-image-builder-hash"
	KubeAnnotationDynamoComponentStorageNS       = "nvidia.com/dynamo-storage-namespace"
)
