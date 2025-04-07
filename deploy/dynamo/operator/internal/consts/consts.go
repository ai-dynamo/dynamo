package consts

const (
	HPACPUDefaultAverageUtilization = 80

	// nolint: gosec
	YataiApiTokenHeaderName = "X-YATAI-API-TOKEN"

	NgcOrganizationHeaderName = "Nv-Ngc-Org"
	NgcUserHeaderName         = "Nv-Actor-Id"

	DefaultUserId = "default"
	DefaultOrgId  = "default"

	BentoServicePort       = 3000
	BentoServicePortName   = "http"
	BentoContainerPortName = "http"

	YataiImageBuilderComponentName = "yatai-image-builder"
	YataiDeploymentComponentName   = "yatai-deployment"

	YataiBentoDeploymentComponentApiServer = "api-server"

	InternalImagesBentoDownloaderDefault    = "quay.io/bentoml/bento-downloader:0.0.3"
	InternalImagesKanikoDefault             = "quay.io/bentoml/kaniko:1.9.1"
	InternalImagesMetricsTransformerDefault = "quay.io/bentoml/yatai-bento-metrics-transformer:0.0.3"
	InternalImagesBuildkitDefault           = "quay.io/bentoml/buildkit:master"
	InternalImagesBuildkitRootlessDefault   = "quay.io/bentoml/buildkit:master-rootless"

	EnvYataiEndpoint    = "YATAI_ENDPOINT"
	EnvYataiClusterName = "YATAI_CLUSTER_NAME"
	// nolint: gosec
	EnvYataiApiToken = "YATAI_API_TOKEN"

	EnvBentoServicePort = "PORT"

	// tracking envars
	EnvYataiDeploymentUID = "YATAI_T_DEPLOYMENT_UID"

	EnvYataiBentoDeploymentName      = "YATAI_BENTO_DEPLOYMENT_NAME"
	EnvYataiBentoDeploymentNamespace = "YATAI_BENTO_DEPLOYMENT_NAMESPACE"

	EnvDockerRegistryServer          = "DOCKER_REGISTRY_SERVER"
	EnvDockerRegistryInClusterServer = "DOCKER_REGISTRY_IN_CLUSTER_SERVER"
	EnvDockerRegistryUsername        = "DOCKER_REGISTRY_USERNAME"
	// nolint:gosec
	EnvDockerRegistryPassword            = "DOCKER_REGISTRY_PASSWORD"
	EnvDockerRegistrySecure              = "DOCKER_REGISTRY_SECURE"
	EnvDockerRegistryBentoRepositoryName = "DOCKER_REGISTRY_BENTO_REPOSITORY_NAME"
	EnvDockerRegistryModelRepositoryName = "DOCKER_REGISTRY_MODEL_REPOSITORY_NAME"

	EnvInternalImagesBentoDownloader    = "INTERNAL_IMAGES_BENTO_DOWNLOADER"
	EnvInternalImagesKaniko             = "INTERNAL_IMAGES_KANIKO"
	EnvInternalImagesMetricsTransformer = "INTERNAL_IMAGES_METRICS_TRANSFORMER"
	EnvInternalImagesBuildkit           = "INTERNAL_IMAGES_BUILDKIT"
	EnvInternalImagesBuildkitRootless   = "INTERNAL_IMAGES_BUILDKIT_ROOTLESS"

	EnvYataiSystemNamespace       = "YATAI_SYSTEM_NAMESPACE"
	EnvYataiImageBuilderNamespace = "YATAI_IMAGE_BUILDER_NAMESPACE"
	EnvYataiDeploymentNamespace   = "YATAI_DEPLOYMENT_NAMESPACE"
	EnvBentoDeploymentNamespaces  = "BENTO_DEPLOYMENT_NAMESPACES"
	EnvImageBuildersNamespace     = "IMAGE_BUILDERS_NAMESPACE"

	KubeLabelYataiSelector        = "yatai.ai/selector"
	KubeLabelYataiBentoRepository = "yatai.ai/bento-repository"
	KubeLabelYataiBento           = "yatai.ai/bento"
	KubeLabelYataiModelRepository = "yatai.ai/model-repository"
	KubeLabelYataiModel           = "yatai.ai/model"

	KubeLabelYataiBentoDeployment              = "yatai.ai/bento-deployment"
	KubeLabelYataiBentoDeploymentComponentType = "yatai.ai/bento-deployment-component-type"
	KubeLabelYataiBentoDeploymentTargetType    = "yatai.ai/bento-deployment-target-type"
	KubeLabelBentoRepository                   = "yatai.ai/bento-repository"
	KubeLabelBentoVersion                      = "yatai.ai/bento-version"
	KubeLabelCreator                           = "yatai.ai/creator"

	KubeLabelIsBentoImageBuilder = "yatai.ai/is-bento-image-builder"
	KubeLabelIsModelSeeder       = "yatai.ai/is-model-seeder"
	KubeLabelBentoRequest        = "yatai.ai/bento-request"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelYataiImageBuilderPod = "yatai.ai/yatai-image-builder-pod"
	KubeLabelBentoDeploymentPod   = "yatai.ai/bento-deployment-pod"

	KubeAnnotationBentoRepository        = "yatai.ai/bento-repository"
	KubeAnnotationBentoVersion           = "yatai.ai/bento-version"
	KubeAnnotationDockerRegistryInsecure = "yatai.ai/docker-registry-insecure"

	KubeAnnotationYataiImageBuilderSeparateModels = "yatai.ai/yatai-image-builder-separate-models"
	KubeAnnotationIsMultiTenancy                  = "yatai.ai/is-multi-tenancy"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	// nolint: gosec
	KubeSecretNameRegcred = "yatai-regcred"
)
