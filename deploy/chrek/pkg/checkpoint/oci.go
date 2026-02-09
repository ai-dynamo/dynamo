// oci.go provides OCI runtime spec types, introspection methods, and
// cross-referencing for checkpoint metadata. The OCI spec (from containerd)
// provides mount configuration, masked/readonly paths, and bind mount
// destinations that enrich the raw /proc-based introspection data.
package checkpoint

import (
	specs "github.com/opencontainers/runtime-spec/specs-go"
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

// OCIMountInfo represents a mount from the OCI spec.
type OCIMountInfo struct {
	Destination string   // Mount point inside container
	Source      string   // Source path on host
	Type        string   // Filesystem type (bind, tmpfs, etc.)
	Options     []string // Mount options
}

// OCINamespaceConfig represents namespace configuration from the OCI spec.
type OCINamespaceConfig struct {
	Type string // Namespace type (network, pid, mount, etc.)
	Path string // Path to namespace (empty for new namespace)
}

// GetMounts returns the mount configuration from the OCI spec.
// This is preferred over parsing /proc/mountinfo for configuration,
// though /proc is still needed for runtime mount state.
func (info *ContainerInfo) GetMounts() []OCIMountInfo {
	if info.Spec == nil || info.Spec.Mounts == nil {
		return nil
	}

	mounts := make([]OCIMountInfo, len(info.Spec.Mounts))
	for i, m := range info.Spec.Mounts {
		mounts[i] = OCIMountInfo{
			Destination: m.Destination,
			Source:      m.Source,
			Type:        m.Type,
			Options:     m.Options,
		}
	}
	return mounts
}

// GetNamespaces returns the namespace configuration from the OCI spec.
func (info *ContainerInfo) GetNamespaces() []OCINamespaceConfig {
	if info.Spec == nil || info.Spec.Linux == nil {
		return nil
	}

	namespaces := make([]OCINamespaceConfig, len(info.Spec.Linux.Namespaces))
	for i, ns := range info.Spec.Linux.Namespaces {
		namespaces[i] = OCINamespaceConfig{
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

// ociState holds data extracted from the container's OCI runtime spec.
// This is used to enrich checkpoint metadata with configuration that only
// the container runtime knows (vs. /proc which shows runtime state).
type ociState struct {
	// MountsByDest maps container-side mount destinations to their OCI mount info.
	// Used to cross-reference /proc/mountinfo entries with OCI spec config.
	MountsByDest map[string]OCIMountInfo

	// BindMountDests are container paths where bind mounts are configured in the OCI spec.
	// These are excluded from rootfs diff capture to avoid conflicts during restore.
	BindMountDests []string

	// MaskedPaths are paths masked (made inaccessible) by the container runtime.
	MaskedPaths []string

	// ReadonlyPaths are paths mounted read-only by the container runtime.
	ReadonlyPaths []string
}

// extractOCIState extracts OCI-derived data from a resolved ContainerInfo.
// Returns nil if containerInfo is nil or has no OCI spec.
func extractOCIState(containerInfo *ContainerInfo) *ociState {
	if containerInfo == nil {
		return nil
	}

	oci := &ociState{
		MountsByDest:  make(map[string]OCIMountInfo),
		MaskedPaths:   containerInfo.GetMaskedPaths(),
		ReadonlyPaths: containerInfo.GetReadonlyPaths(),
	}

	for _, m := range containerInfo.GetMounts() {
		oci.MountsByDest[m.Destination] = m
		if m.Type == "bind" {
			oci.BindMountDests = append(oci.BindMountDests, m.Destination)
		}
	}

	return oci
}

// enrichMountWithOCI cross-references a MountMetadata with OCI spec data.
// If the mount's container path matches an OCI mount destination, the OCI
// source, type, and options are populated on the metadata.
func enrichMountWithOCI(meta *MountMetadata, ociMounts map[string]OCIMountInfo) {
	if ociMount, ok := ociMounts[meta.ContainerPath]; ok {
		meta.OCISource = ociMount.Source
		meta.OCIType = ociMount.Type
		meta.OCIOptions = ociMount.Options
	}
}
