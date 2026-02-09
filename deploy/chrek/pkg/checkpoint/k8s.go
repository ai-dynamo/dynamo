// k8s.go provides K8s identity metadata, containerd container discovery,
// and Kubernetes volume type detection for checkpoint operations.
package checkpoint

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
)

const (
	// K8sNamespace is the containerd namespace used by Kubernetes.
	K8sNamespace = "k8s.io"

	// ContainerdSocket is the default containerd socket path.
	ContainerdSocket = "/run/containerd/containerd.sock"
)

// K8sMetadata holds container and pod identity information from containerd and caller params.
type K8sMetadata struct {
	ContainerID  string `yaml:"containerId"`
	Image        string `yaml:"image"`
	PID          int    `yaml:"pid"`
	SourceNode   string `yaml:"sourceNode"`
	PodName      string `yaml:"podName"`
	PodNamespace string `yaml:"podNamespace"`
}

// NewK8sMetadata constructs K8sMetadata from checkpoint params and container info.
func NewK8sMetadata(params CheckpointParams, pid int, containerInfo *ContainerInfo) K8sMetadata {
	meta := K8sMetadata{
		ContainerID:  params.ContainerID,
		PID:          pid,
		SourceNode:   params.NodeName,
		PodName:      params.PodName,
		PodNamespace: params.PodNamespace,
	}
	if containerInfo != nil {
		meta.Image = containerInfo.Image
	}
	return meta
}

// DiscoveryClient wraps the containerd client for container discovery.
type DiscoveryClient struct {
	client *containerd.Client
	socket string
}

// NewDiscoveryClient creates a new discovery client connected to containerd.
func NewDiscoveryClient() (*DiscoveryClient, error) {
	socket := ContainerdSocket
	client, err := containerd.New(socket)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to containerd at %s: %w", socket, err)
	}

	return &DiscoveryClient{
		client: client,
		socket: socket,
	}, nil
}

// Close closes the containerd client connection.
func (c *DiscoveryClient) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}

// ResolveContainer resolves a container ID to its process information.
// This retrieves configuration from containerd RPCs (OCI spec, labels, image)
// and runtime paths from /proc (rootfs access path).
func (c *DiscoveryClient) ResolveContainer(ctx context.Context, containerID string) (*ContainerInfo, error) {
	ctx = namespaces.WithNamespace(ctx, K8sNamespace)

	container, err := c.client.LoadContainer(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to load container %s: %w", containerID, err)
	}

	task, err := container.Task(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get task for container %s: %w", containerID, err)
	}

	pid := task.Pid()

	image, err := container.Image(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get image for container %s: %w", containerID, err)
	}

	spec, err := container.Spec(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get spec for container %s: %w", containerID, err)
	}

	labels, err := container.Labels(ctx)
	if err != nil {
		// Labels are optional, don't fail
		labels = make(map[string]string)
	}

	// Construct the bundle path where containerd stores the container runtime files
	// Standard containerd layout: /run/containerd/io.containerd.runtime.v2.task/<namespace>/<container_id>/
	containerdRunRoot := os.Getenv("CONTAINERD_RUN_ROOT")
	if containerdRunRoot == "" {
		containerdRunRoot = "/run/containerd"
	}
	bundlePath := filepath.Join(containerdRunRoot, "io.containerd.runtime.v2.task", K8sNamespace, containerID)

	// Get the rootfs path from the OCI spec (usually "rootfs" relative to bundle)
	rootfsRelPath := "rootfs"
	if spec.Root != nil && spec.Root.Path != "" {
		rootfsRelPath = spec.Root.Path
	}

	var rootFS string
	if filepath.IsAbs(rootfsRelPath) {
		rootFS = rootfsRelPath
	} else {
		rootFS = filepath.Join(bundlePath, rootfsRelPath)
	}

	return &ContainerInfo{
		ContainerID: containerID,
		PID:         pid,
		RootFS:      rootFS,
		BundlePath:  bundlePath,
		Image:       image.Name(),
		Spec:        spec,
		Labels:      labels,
	}, nil
}

// ListContainers lists all containers in the K8s namespace.
func (c *DiscoveryClient) ListContainers(ctx context.Context) ([]string, error) {
	ctx = namespaces.WithNamespace(ctx, K8sNamespace)

	containers, err := c.client.Containers(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list containers: %w", err)
	}

	ids := make([]string, len(containers))
	for i, container := range containers {
		ids[i] = container.ID()
	}

	return ids, nil
}

// GetContainerLabels returns the labels for a container.
func (c *DiscoveryClient) GetContainerLabels(ctx context.Context, containerID string) (map[string]string, error) {
	ctx = namespaces.WithNamespace(ctx, K8sNamespace)

	container, err := c.client.LoadContainer(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to load container %s: %w", containerID, err)
	}

	labels, err := container.Labels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get labels for container %s: %w", containerID, err)
	}

	return labels, nil
}

// DetectVolumeTypeFromPath attempts to identify volume type from kubelet path patterns.
// This is a best-effort detection based on well-known kubelet directory conventions.
func DetectVolumeTypeFromPath(hostPath string) (volumeType, volumeName string) {
	volumeType = "unknown"
	volumeName = ""

	// Map of path patterns to volume types
	patterns := map[string]string{
		"/kubernetes.io~empty-dir/":             "emptyDir",
		"/kubernetes.io~configmap/":             "configMap",
		"/kubernetes.io~secret/":                "secret",
		"/kubernetes.io~projected/":             "projected",
		"/kubernetes.io~downward-api/":          "downwardAPI",
		"/kubernetes.io~persistentvolumeclaim/": "persistentVolumeClaim",
		"/kubernetes.io~hostpath/":              "hostPath",
	}

	for pattern, vType := range patterns {
		if strings.Contains(hostPath, pattern) {
			volumeType = vType
			// Extract volume name from path
			parts := strings.Split(hostPath, pattern)
			if len(parts) > 1 {
				volumeName = strings.Split(parts[1], "/")[0]
			}
			break
		}
	}

	return volumeType, volumeName
}
