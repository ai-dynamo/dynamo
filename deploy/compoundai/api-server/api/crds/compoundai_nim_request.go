package crds

import (
	"time"

	corev1 "k8s.io/api/core/v1"
)

type CompoundNimRequestData struct {
	CompoundNimVersionTag string `json:"bentoTag"`
	DownloadURL           string `json:"downloadUrl,omitempty"`

	ImageBuildTimeout *time.Duration `json:"imageBuildTimeout,omitempty"`

	ImageBuilderExtraPodMetadata   *ExtraPodMetadata            `json:"imageBuilderExtraPodMetadata,omitempty"`
	ImageBuilderExtraPodSpec       *ExtraPodSpec                `json:"imageBuilderExtraPodSpec,omitempty"`
	ImageBuilderExtraContainerEnv  []corev1.EnvVar              `json:"imageBuilderExtraContainerEnv,omitempty"`
	ImageBuilderContainerResources *corev1.ResourceRequirements `json:"imageBuilderContainerResources,omitempty"`

	DockerConfigJSONSecretName string `json:"dockerConfigJsonSecretName,omitempty"`

	DownloaderContainerEnvFrom []corev1.EnvFromSource `json:"downloaderContainerEnvFrom,omitempty"`
}

type CompoundNimRequestConfigurationV1Alpha1 struct {
	Data    CompoundNimRequestData `json:"data,omitempty"`
	Version string                 `json:"version"`
}
