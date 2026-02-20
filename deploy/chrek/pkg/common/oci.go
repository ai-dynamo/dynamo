// Package common provides low-level container, process, and device primitives
// shared across chrek packages.
package common

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	securejoin "github.com/cyphar/filepath-securejoin"
)

const (
	// k8sNamespace is the containerd namespace used by Kubernetes.
	k8sNamespace = "k8s.io"

	// ContainerdSocket is the default containerd socket path.
	ContainerdSocket = "/run/containerd/containerd.sock"
)

// ResolveContainer resolves a container by ID and returns its PID and OCI spec.
func ResolveContainer(ctx context.Context, client *containerd.Client, containerID string) (int, *specs.Spec, error) {
	ctx = namespaces.WithNamespace(ctx, k8sNamespace)

	container, err := client.LoadContainer(ctx, containerID)
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

// ResolveContainerByPod finds a container by pod name, namespace, and container name
// by listing containerd containers and matching CRI labels.
func ResolveContainerByPod(ctx context.Context, client *containerd.Client, podName, podNamespace, containerName string) (int, *specs.Spec, error) {
	ctx = namespaces.WithNamespace(ctx, k8sNamespace)

	filter := fmt.Sprintf("labels.\"io.kubernetes.pod.name\"==%s,labels.\"io.kubernetes.pod.namespace\"==%s,labels.\"io.kubernetes.container.name\"==%s",
		podName, podNamespace, containerName)

	containers, err := client.Containers(ctx, filter)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to list containers for pod %s/%s: %w", podNamespace, podName, err)
	}

	if len(containers) == 0 {
		return 0, nil, fmt.Errorf("no container found for pod %s/%s container %s", podNamespace, podName, containerName)
	}

	// During container restarts, containerd may transiently expose both the
	// old and new container with the same CRI labels. Pick the one with a
	// running task; fall back to the first container if none qualify.
	for _, c := range containers {
		task, err := c.Task(ctx, nil)
		if err != nil {
			continue
		}
		spec, err := c.Spec(ctx)
		if err != nil {
			continue
		}
		return int(task.Pid()), spec, nil
	}

	return 0, nil, fmt.Errorf("no running container found for pod %s/%s container %s (%d candidates)", podNamespace, podName, containerName, len(containers))
}

func collectOCIManagedPaths(ociSpec *specs.Spec, rootFS string) map[string]struct{} {
	set := map[string]struct{}{}
	if ociSpec == nil {
		return set
	}

	paths := make([]string, 0, len(ociSpec.Mounts))
	for _, mount := range ociSpec.Mounts {
		paths = append(paths, mount.Destination)
	}
	if ociSpec.Linux != nil {
		paths = append(paths, ociSpec.Linux.MaskedPaths...)
		paths = append(paths, ociSpec.Linux.ReadonlyPaths...)
	}
	for _, raw := range paths {
		if p := normalizeOCIPath(raw, rootFS); p != "" {
			set[p] = struct{}{}
		}
	}
	return set
}

// normalizeOCIPath resolves an OCI spec path relative to rootFS, following
// symlinks within the rootfs boundary (matching runc's addCriuDumpMount pattern).
func normalizeOCIPath(raw, rootFS string) string {
	p := filepath.Clean(strings.TrimSpace(raw))
	if p == "" || p == "." {
		return ""
	}
	if rootFS == "" {
		return p
	}
	if resolved, err := securejoin.SecureJoin(rootFS, p); err == nil {
		p = strings.TrimPrefix(resolved, filepath.Clean(rootFS))
	}
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return p
}
