// discovery provides container information resolution via containerd.
// This prefers containerd RPCs for configuration over /proc inspection,
// following the principle that configuration should come from the container runtime
// while runtime state (like namespace inodes) requires /proc.
package k8s

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	// K8sNamespace is the containerd namespace used by Kubernetes
	K8sNamespace = "k8s.io"
	// DefaultSocket is the default containerd socket path
	DefaultSocket = "/run/containerd/containerd.sock"
)

// ContainerInfo holds resolved container information from containerd.
// Configuration data comes from containerd RPCs, runtime state from /proc.
type ContainerInfo struct {
	ContainerID string
	PID         uint32
	RootFS      string // Actual rootfs path (bundle path + spec.Root.Path)
	BundlePath  string // Path to container bundle directory
	Image       string
	Spec        *specs.Spec // OCI spec from containerd (mounts, namespaces config)
	Labels      map[string]string
}

// MountInfo represents a mount from the OCI spec.
type MountInfo struct {
	Destination string   // Mount point inside container
	Source      string   // Source path on host
	Type        string   // Filesystem type (bind, tmpfs, etc.)
	Options     []string // Mount options
}

// NamespaceConfig represents namespace configuration from OCI spec.
type NamespaceConfig struct {
	Type string // Namespace type (network, pid, mount, etc.)
	Path string // Path to namespace (empty for new namespace)
}

// DiscoveryClient wraps the containerd client for container discovery.
type DiscoveryClient struct {
	client *containerd.Client
	socket string
}

// NewDiscoveryClient creates a new discovery client.
func NewDiscoveryClient(socket string) (*DiscoveryClient, error) {
	if socket == "" {
		socket = DefaultSocket
	}

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
	// Use the Kubernetes namespace for containerd
	ctx = namespaces.WithNamespace(ctx, K8sNamespace)

	// Load the container
	container, err := c.client.LoadContainer(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to load container %s: %w", containerID, err)
	}

	// Get the task (running process)
	task, err := container.Task(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get task for container %s: %w", containerID, err)
	}

	// Get the PID
	pid := task.Pid()

	// Get container image
	image, err := container.Image(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get image for container %s: %w", containerID, err)
	}

	// Get OCI spec from containerd - this contains mount config, namespace config, etc.
	// This is preferred over parsing /proc for configuration data.
	spec, err := container.Spec(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get spec for container %s: %w", containerID, err)
	}

	// Get container labels (includes K8s pod info)
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

	// Construct full rootfs path
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

// GetMounts returns the mount configuration from the OCI spec.
// This is preferred over parsing /proc/mountinfo for configuration,
// though /proc is still needed for runtime mount state.
func (info *ContainerInfo) GetMounts() []MountInfo {
	if info.Spec == nil || info.Spec.Mounts == nil {
		return nil
	}

	mounts := make([]MountInfo, len(info.Spec.Mounts))
	for i, m := range info.Spec.Mounts {
		mounts[i] = MountInfo{
			Destination: m.Destination,
			Source:      m.Source,
			Type:        m.Type,
			Options:     m.Options,
		}
	}
	return mounts
}

// GetNamespaces returns the namespace configuration from the OCI spec.
func (info *ContainerInfo) GetNamespaces() []NamespaceConfig {
	if info.Spec == nil || info.Spec.Linux == nil {
		return nil
	}

	namespaces := make([]NamespaceConfig, len(info.Spec.Linux.Namespaces))
	for i, ns := range info.Spec.Linux.Namespaces {
		namespaces[i] = NamespaceConfig{
			Type: string(ns.Type),
			Path: ns.Path,
		}
	}
	return namespaces
}

// GetMaskedPaths returns the masked paths from the OCI spec.
func (info *ContainerInfo) GetMaskedPaths() []string {
	if info.Spec == nil || info.Spec.Linux == nil {
		return nil
	}
	return info.Spec.Linux.MaskedPaths
}

// GetReadonlyPaths returns the readonly paths from the OCI spec.
func (info *ContainerInfo) GetReadonlyPaths() []string {
	if info.Spec == nil || info.Spec.Linux == nil {
		return nil
	}
	return info.Spec.Linux.ReadonlyPaths
}

// GetRootfsPath returns the rootfs path from the OCI spec.
// Note: For CRIU, use info.RootFS which is the /proc/<pid>/root path.
func (info *ContainerInfo) GetRootfsPath() string {
	if info.Spec == nil || info.Spec.Root == nil {
		return ""
	}
	return info.Spec.Root.Path
}

// IsRootReadonly returns whether the root filesystem is readonly.
func (info *ContainerInfo) IsRootReadonly() bool {
	if info.Spec == nil || info.Spec.Root == nil {
		return false
	}
	return info.Spec.Root.Readonly
}

// GetHostname returns the container's hostname from the OCI spec.
func (info *ContainerInfo) GetHostname() string {
	if info.Spec == nil {
		return ""
	}
	return info.Spec.Hostname
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
