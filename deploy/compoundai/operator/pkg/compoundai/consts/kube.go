package consts

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
)

const (
	KubeIngressClassName = "yatai-ingress"

	KubeLabelYataiSelector        = "yatai.ai/selector"
	KubeLabelYataiBentoRepository = "yatai.ai/bento-repository"
	KubeLabelYataiBento           = "yatai.ai/bento"
	KubeLabelYataiModelRepository = "yatai.ai/model-repository"
	KubeLabelYataiModel           = "yatai.ai/model"

	KubeHPAQPSMetric = "http_request"
	KubeHPAGPUMetric = "container_accelerator_duty_cycle"

	DefaultKubeNamespaceBentoDeployment            = "yatai"
	DefaultKubeNamespaceImageBuilders              = "yatai-builders"
	DefaultKubeNamespaceYataiSystem                = "yatai-system"
	DefaultKubeNamespaceYataiImageBuilderComponent = "yatai-image-builder"
	DefaultKubeNamespaceYataiDeploymentComponent   = "yatai-deployment"

	KubeLabelYataiBentoDeployment              = "yatai.ai/bento-deployment"
	KubeLabelYataiBentoDeploymentComponentType = "yatai.ai/bento-deployment-component-type"
	KubeLabelYataiBentoDeploymentComponentName = "yatai.ai/bento-deployment-component-name"
	KubeLabelYataiBentoDeploymentTargetType    = "yatai.ai/bento-deployment-target-type"
	KubeLabelYataiBentoDeploymentRunner        = "yatai.ai/bento-deployment-runner"
	KubeLabelBentoRepository                   = "yatai.ai/bento-repository"
	KubeLabelBentoVersion                      = "yatai.ai/bento-version"
	KubeLabelCreator                           = "yatai.ai/creator"
	// nolint: gosec
	KubeLabelYataiDeployToken    = "yatai.ai/deploy-token"
	KubeLabelIsBentoImageBuilder = "yatai.ai/is-bento-image-builder"
	KubeLabelIsModelSeeder       = "yatai.ai/is-model-seeder"
	KubeLabelBentoRequest        = "yatai.ai/bento-request"

	KubeLabelYataiOwnerReference = "yatai.ai/owner-reference"

	KubeLabelGPUAccelerator = "gpu-accelerator"

	KubeLabelHostName = "kubernetes.io/hostname"
	KubeLabelArch     = "kubernetes.io/arch"

	KubeLabelValueFalse = "false"
	KubeLabelValueTrue  = "true"

	KubeLabelYataiImageBuilderPod = "yatai.ai/yatai-image-builder-pod"
	KubeLabelBentoDeploymentPod   = "yatai.ai/bento-deployment-pod"

	KubeLabelManagedBy    = "app.kubernetes.io/managed-by"
	KubeLabelHelmHeritage = "heritage"
	KubeLabelHelmRelease  = "release"

	KubeAnnotationBentoRepository        = "yatai.ai/bento-repository"
	KubeAnnotationBentoVersion           = "yatai.ai/bento-version"
	KubeAnnotationYataiDeploymentId      = "yatai.ai/deployment-id"
	KubeAnnotationDockerRegistryInsecure = "yatai.ai/docker-registry-insecure"
	KubeAnnotationHelmReleaseName        = "meta.helm.sh/release-name"

	KubeAnnotationPrometheusScrape = "prometheus.io/scrape"
	KubeAnnotationPrometheusPort   = "prometheus.io/port"
	KubeAnnotationPrometheusPath   = "prometheus.io/path"

	KubeAnnotationARMSAutoEnable                  = "armsPilotAutoEnable"
	KubeAnnotationARMSAppName                     = "armsPilotCreateAppName"
	KubeAnnotationYataiImageBuilderSeparateModels = "yatai.ai/yatai-image-builder-separate-models"
	KubeAnnotationAWSAccessKeySecretName          = "yatai.ai/aws-access-key-secret-name"
	KubeAnnotationGCPAccessKeySecretName          = "yatai.ai/gcp-access-key-secret"
	KubeAnnotationIsMultiTenancy                  = "yatai.ai/is-multi-tenancy"

	KubeCreator = "yatai"

	KubeResourceGPUNvidia = "nvidia.com/gpu"

	KubeEventResourceKindPod        = "Pod"
	KubeEventResourceKindHPA        = "HorizontalPodAutoscaler"
	KubeEventResourceKindReplicaSet = "ReplicaSet"

	KubeTaintKeyDedicatedNodeGroup = "mcd.io/dedicated-node-group"
	KubeLabelDedicatedNodeGroup    = "mcd.io/dedicated-node-group"

	KubeImageCSIDriver          = "image.csi.k8s.io"
	KubeImageCSIDriverWarmMetal = "csi-image.warm-metal.tech"

	KubeConfigMapNameNetworkConfig = "network"

	KubeConfigMapKeyNetworkConfigDomainSuffix       = "domain-suffix"
	KubeConfigMapKeyNetworkConfigIngressClass       = "ingress-class"
	KubeConfigMapKeyNetworkConfigIngressAnnotations = "ingress-annotations"
	KubeConfigMapKeyNetworkConfigIngressPath        = "ingress-path"
	KubeConfigMapKeyNetworkConfigIngressPathType    = "ingress-path-type"

	KubeConfigMapNameYataiConfig = "yatai"

	KubeConfigMapKeyYataiConfigEndpoint           = "endpoint"
	KubeConfigMapKeyYataiConfigClusterName        = "cluster-name"
	KubeConfigMapKeyYataiConfigApiTokenSecretName = "api-token-secret-name"
	KubeConfigMapKeyYataiConfigApiTokenSecretKey  = "api-token-secret-key"

	// nolint: gosec
	KubeSecretNameRegcred = "yatai-regcred"

	KubeSecretNameYataiCommonEnv = "yatai-common-env"

	KubeSecretNameYataiImageBuilderSharedEnv = "yatai-image-builder-shared-env"
	KubeSecretNameYataiDeploymentSharedEnv   = "compoundai-deployment-shared-env"

	KubeSecretNameYataiImageBuilderEnv = "yatai-image-builder-env"
	KubeSecretNameYataiDeploymentEnv   = "compoundai-deployment-env"
)

var KubeListEverything = metav1.ListOptions{
	LabelSelector: labels.Everything().String(),
	FieldSelector: fields.Everything().String(),
}
