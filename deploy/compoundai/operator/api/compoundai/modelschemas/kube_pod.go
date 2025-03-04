package modelschemas

import apiv1 "k8s.io/api/core/v1"

type KubePodActualStatus string

const (
	KubePodActualStatusPending     KubePodActualStatus = "Pending"
	KubePodActualStatusRunning     KubePodActualStatus = "Running"
	KubePodActualStatusSucceeded   KubePodActualStatus = "Succeeded"
	KubePodActualStatusFailed      KubePodActualStatus = "Failed"
	KubePodActualStatusUnknown     KubePodActualStatus = "Unknown"
	KubePodActualStatusTerminating KubePodActualStatus = "Terminating"
)

type KubePodStatus struct {
	Status          KubePodActualStatus    `json:"status"`
	Phase           apiv1.PodPhase         `json:"phase"`
	ContainerStates []apiv1.ContainerState `json:"container_states"`
}
