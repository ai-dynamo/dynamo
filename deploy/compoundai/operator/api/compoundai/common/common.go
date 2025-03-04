// +k8s:deepcopy-gen=package
package common

import (
	corev1 "k8s.io/api/core/v1"
)

type ResourceItem struct {
	CPU    string            `json:"cpu,omitempty"`
	Memory string            `json:"memory,omitempty"`
	GPU    string            `json:"gpu,omitempty"`
	Custom map[string]string `json:"custom,omitempty"`
}

type Resources struct {
	Requests *ResourceItem `json:"requests,omitempty"`
	Limits   *ResourceItem `json:"limits,omitempty"`
}

type DeploymentTargetHPAConf struct {
	CPU         *int32  `json:"cpu,omitempty"`
	GPU         *int32  `json:"gpu,omitempty"`
	Memory      *string `json:"memory,omitempty"`
	QPS         *int64  `json:"qps,omitempty"`
	MinReplicas *int32  `json:"min_replicas,omitempty"`
	MaxReplicas *int32  `json:"max_replicas,omitempty"`
}

type LabelItemSchema struct {
	Key   string `json:"key"`
	Value string `json:"value"`
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

type ExtraPodMetadata struct {
	Annotations map[string]string `json:"annotations,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

type ExtraPodSpec struct {
	SchedulerName             string                            `json:"schedulerName,omitempty"`
	NodeSelector              map[string]string                 `json:"nodeSelector,omitempty"`
	Affinity                  *corev1.Affinity                  `json:"affinity,omitempty"`
	Tolerations               []corev1.Toleration               `json:"tolerations,omitempty"`
	TopologySpreadConstraints []corev1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
	Containers                []corev1.Container                `json:"containers,omitempty"`
	ServiceAccountName        string                            `json:"serviceAccountName,omitempty"`
	PriorityClassName         string                            `json:"priorityClassName,omitempty"`
}
