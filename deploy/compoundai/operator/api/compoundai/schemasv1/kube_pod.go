package schemasv1

import (
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
)

type KubePodStatusSchema struct {
	Phase     apiv1.PodPhase `json:"phase"`
	Ready     bool           `json:"ready"`
	StartTime *metav1.Time   `json:"start_time"`
	IsOld     bool           `json:"is_old"`
	IsCanary  bool           `json:"is_canary"`
	HostIp    string         `json:"host_ip"`
}

type KubePodSchema struct {
	Name             string                     `json:"name"`
	Namespace        string                     `json:"namespace"`
	Annotations      map[string]string          `json:"annotations"`
	Labels           map[string]string          `json:"labels"`
	NodeName         string                     `json:"node_name"`
	RunnerName       *string                    `json:"runner_name"`
	DeploymentTarget *DeploymentTargetSchema    `json:"deployment_target"`
	CommitId         string                     `json:"commit_id"`
	Status           KubePodStatusSchema        `json:"status"`
	RawStatus        apiv1.PodStatus            `json:"raw_status"`
	PodStatus        modelschemas.KubePodStatus `json:"pod_status"`
	Warnings         []apiv1.Event              `json:"warnings"`
}
