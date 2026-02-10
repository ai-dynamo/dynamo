// k8s contains containerd discovery and Kubernetes path classification helpers.
package checkpoint

import (
	"context"
	"fmt"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	// K8sNamespace is the containerd namespace used by Kubernetes.
	K8sNamespace = "k8s.io"

	// ContainerdSocket is the default containerd socket path.
	ContainerdSocket = "/run/containerd/containerd.sock"
)

type SourcePodManifest struct {
	ContainerID  string `yaml:"containerId"`
	PID          int    `yaml:"pid"`
	SourceNode   string `yaml:"sourceNode"`
	PodName      string `yaml:"podName"`
	PodNamespace string `yaml:"podNamespace"`
}

func NewSourcePodManifest(params CheckpointRequest, pid int) SourcePodManifest {
	return SourcePodManifest{
		ContainerID:  params.ContainerID,
		PID:          pid,
		SourceNode:   params.NodeName,
		PodName:      params.PodName,
		PodNamespace: params.PodNamespace,
	}
}

type DiscoveryClient struct {
	client *containerd.Client
}

func NewDiscoveryClient() (*DiscoveryClient, error) {
	client, err := containerd.New(ContainerdSocket)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to containerd at %s: %w", ContainerdSocket, err)
	}
	return &DiscoveryClient{client: client}, nil
}

func (c *DiscoveryClient) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}

func (c *DiscoveryClient) ResolveContainer(ctx context.Context, containerID string) (int, *specs.Spec, error) {
	ctx = namespaces.WithNamespace(ctx, K8sNamespace)

	container, err := c.client.LoadContainer(ctx, containerID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to load container %s: %w", containerID, err)
	}

	task, err := container.Task(ctx, nil)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get task for container %s: %w", containerID, err)
	}

	pid := task.Pid()

	spec, err := container.Spec(ctx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get spec for container %s: %w", containerID, err)
	}

	return int(pid), spec, nil
}
