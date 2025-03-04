package crds

import corev1 "k8s.io/api/core/v1"

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
}

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
