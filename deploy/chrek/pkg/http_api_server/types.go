// Package server provides HTTP server functionality for the checkpoint agent.
package httpApiServer

import "time"

// CheckpointRequest is the request body for checkpoint operations.
type CheckpointRequest struct {
	ContainerID   string `json:"container_id"`
	ContainerName string `json:"container_name,omitempty"` // K8s container name (for volume type lookup)
	CheckpointID  string `json:"checkpoint_id"`
	PodName       string `json:"pod_name,omitempty"`
	PodNamespace  string `json:"pod_namespace,omitempty"`
	DisableCUDA   bool   `json:"disable_cuda,omitempty"` // Disable CUDA plugin for non-GPU workloads
}

// CheckpointResponse is the response for checkpoint operations.
type CheckpointResponse struct {
	Success      bool   `json:"success"`
	CheckpointID string `json:"checkpoint_id,omitempty"`
	Message      string `json:"message,omitempty"`
	Error        string `json:"error,omitempty"`
}

// CheckpointInfo represents information about a checkpoint.
type CheckpointInfo struct {
	ID           string    `json:"id"`
	CreatedAt    time.Time `json:"created_at"`
	SourceNode   string    `json:"source_node"`
	ContainerID  string    `json:"container_id"`
	PodName      string    `json:"pod_name"`
	PodNamespace string    `json:"pod_namespace"`
	Image        string    `json:"image"`
}

// ListCheckpointsResponse is the response for list checkpoints.
type ListCheckpointsResponse struct {
	Checkpoints []CheckpointInfo `json:"checkpoints"`
}

// HealthResponse is the response for health check.
type HealthResponse struct {
	Status   string `json:"status"`
	NodeName string `json:"node_name"`
}
