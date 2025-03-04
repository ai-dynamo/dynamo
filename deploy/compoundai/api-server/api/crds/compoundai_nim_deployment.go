package crds

import (
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
	autoscalingv2beta2 "k8s.io/api/autoscaling/v2beta2"
	corev1 "k8s.io/api/core/v1"
)

type Autoscaling struct {
	MinReplicas int32                                               `json:"minReplicas"`
	MaxReplicas int32                                               `json:"maxReplicas"`
	Metrics     []autoscalingv2beta2.MetricSpec                     `json:"metrics,omitempty"`
	Behavior    *autoscalingv2beta2.HorizontalPodAutoscalerBehavior `json:"behavior,omitempty"`
}

type CompoundNimVersionDeploymentIngressTLSSpec struct {
	SecretName string `json:"secretName,omitempty"`
}

type CompoundNimVersionDeploymentIngressSpec struct {
	Enabled     bool                                        `json:"enabled,omitempty"`
	Annotations map[string]string                           `json:"annotations,omitempty"`
	Labels      map[string]string                           `json:"labels,omitempty"`
	TLS         *CompoundNimVersionDeploymentIngressTLSSpec `json:"tls,omitempty"`
}

type MonitorExporterMountSpec struct {
	Path                string `json:"path,omitempty"`
	ReadOnly            bool   `json:"readOnly,omitempty"`
	corev1.VolumeSource `json:",inline"`
}

type MonitorExporterSpec struct {
	Enabled          bool                       `json:"enabled,omitempty"`
	Output           string                     `json:"output,omitempty"`
	Options          map[string]string          `json:"options,omitempty"`
	StructureOptions []corev1.EnvVar            `json:"structureOptions,omitempty"`
	Mounts           []MonitorExporterMountSpec `json:"mounts,omitempty"`
}

type CompoundNimDeploymentData struct {
	Annotations map[string]string `json:"annotations,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`

	CompoundNimVersion string `json:"compoundAINim"`

	Resources        schemas.Resources                  `json:"resources,omitempty"`
	Autoscaling      *Autoscaling                       `json:"autoscaling,omitempty"`
	Envs             []corev1.EnvVar                    `json:"envs,omitempty"`
	ExternalServices map[string]schemas.ExternalService `json:"externalServices,omitempty"`

	Ingress CompoundNimVersionDeploymentIngressSpec `json:"ingress,omitempty"`

	MonitorExporter *MonitorExporterSpec `json:"monitorExporter,omitempty"`

	ExtraPodMetadata *ExtraPodMetadata `json:"extraPodMetadata,omitempty"`
	ExtraPodSpec     *ExtraPodSpec     `json:"extraPodSpec,omitempty"`

	LivenessProbe  *corev1.Probe `json:"livenessProbe,omitempty"`
	ReadinessProbe *corev1.Probe `json:"readinessProbe,omitempty"`
}

type CompoundNimDeploymentConfigurationV1Alpha1 struct {
	Data    CompoundNimDeploymentData `json:"data"`
	Version string                    `json:"version"`
}
