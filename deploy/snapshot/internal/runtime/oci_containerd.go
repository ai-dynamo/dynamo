package runtime

import (
	"context"
	"fmt"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// k8sNamespace is containerd's conventional namespace for kubelet-managed
// containers.
const k8sNamespace = "k8s.io"

type ContainerdRuntime struct {
	client *containerd.Client
}

func NewContainerdRuntime(socket string) (*ContainerdRuntime, error) {
	client, err := containerd.New(socket)
	if err != nil {
		return nil, fmt.Errorf("failed to dial containerd at %s: %w", socket, err)
	}
	return &ContainerdRuntime{client: client}, nil
}

func (r *ContainerdRuntime) Close() error {
	return r.client.Close()
}

func (r *ContainerdRuntime) ResolveContainer(ctx context.Context, containerID string) (int, *specs.Spec, error) {
	ctx = namespaces.WithNamespace(ctx, k8sNamespace)

	container, err := r.client.LoadContainer(ctx, containerID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to load container %s: %w", containerID, err)
	}

	task, err := container.Task(ctx, nil)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get task for container %s: %w", containerID, err)
	}

	spec, err := container.Spec(ctx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get spec for container %s: %w", containerID, err)
	}

	return int(task.Pid()), spec, nil
}

func (r *ContainerdRuntime) ResolveContainerByPod(ctx context.Context, podName, podNamespace, containerName string) (int, *specs.Spec, error) {
	container, err := r.findRunningContainerByPod(ctx, podName, podNamespace, containerName)
	if err != nil {
		return 0, nil, err
	}
	ctx = namespaces.WithNamespace(ctx, k8sNamespace)

	task, err := container.Task(ctx, nil)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get task for container %s (pod %s/%s): %w", container.ID(), podNamespace, podName, err)
	}
	spec, err := container.Spec(ctx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get spec for container %s (pod %s/%s): %w", container.ID(), podNamespace, podName, err)
	}
	return int(task.Pid()), spec, nil
}

func (r *ContainerdRuntime) ResolveContainerIDByPod(ctx context.Context, podName, podNamespace, containerName string) (string, error) {
	container, err := r.findRunningContainerByPod(ctx, podName, podNamespace, containerName)
	if err != nil {
		return "", err
	}
	return container.ID(), nil
}

func (r *ContainerdRuntime) findRunningContainerByPod(ctx context.Context, podName, podNamespace, containerName string) (containerd.Container, error) {
	ctx = namespaces.WithNamespace(ctx, k8sNamespace)

	filter := fmt.Sprintf("labels.\"io.kubernetes.pod.name\"==%s,labels.\"io.kubernetes.pod.namespace\"==%s,labels.\"io.kubernetes.container.name\"==%s",
		podName, podNamespace, containerName)

	containers, err := r.client.Containers(ctx, filter)
	if err != nil {
		return nil, fmt.Errorf("failed to list containers for pod %s/%s: %w", podNamespace, podName, err)
	}
	if len(containers) == 0 {
		return nil, fmt.Errorf("no container found for pod %s/%s container %s", podNamespace, podName, containerName)
	}

	// During container restarts, both the old and new container may be listed;
	// pick the first with a live task.
	for _, c := range containers {
		if _, err := c.Task(ctx, nil); err == nil {
			return c, nil
		}
	}
	return nil, fmt.Errorf("no running container found for pod %s/%s container %s (%d candidates)", podNamespace, podName, containerName, len(containers))
}
