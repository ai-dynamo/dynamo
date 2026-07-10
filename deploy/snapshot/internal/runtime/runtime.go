// Package runtime provides low-level host and container-runtime primitives for snapshot execution.
package runtime

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	securejoin "github.com/cyphar/filepath-securejoin"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// Default socket paths and runtime-type identifiers.
const (
	ContainerdSocket = "/run/containerd/containerd.sock"
	CRIOSocket       = "/var/run/crio/crio.sock"

	RuntimeContainerd = "containerd"
	RuntimeCRIO       = "crio"
)

// Runtime abstracts the container-identity APIs behind a two-backend switch.
// Resolve methods return non-nil *specs.Spec with PID > 0 on success, or an error.
type Runtime interface {
	ResolveContainer(ctx context.Context, id string) (int, *specs.Spec, error)
	ResolveContainerIDByPod(ctx context.Context, pod, ns, ctr string) (string, error)
	ResolveContainerByPod(ctx context.Context, pod, ns, ctr string) (int, *specs.Spec, error)
	Close() error
}

// criSchemes are the ContainerStatus.ContainerID prefixes kubelet emits across
// runtimes. Prefixes don't overlap, so stripping is order-independent.
var criSchemes = []string{"containerd://", "cri-o://", "crio://"}

// StripCRIScheme trims the kubelet-format scheme prefix from a
// ContainerStatus.ContainerID. Returns id unchanged if no known scheme matches.
func StripCRIScheme(id string) string {
	for _, scheme := range criSchemes {
		if s, ok := strings.CutPrefix(id, scheme); ok {
			return s
		}
	}
	return id
}

// defaultSocketFor returns the conventional socket path for a runtime type.
// Returns empty for unknown types; New validates before calling.
func defaultSocketFor(runtimeType string) string {
	switch runtimeType {
	case RuntimeCRIO:
		return CRIOSocket
	case RuntimeContainerd:
		return ContainerdSocket
	default:
		return ""
	}
}

// New constructs a Runtime backend for the given type and socket. Pass an
// empty socket to use the per-type default.
func New(runtimeType, socket string) (Runtime, error) {
	if socket == "" {
		socket = defaultSocketFor(runtimeType)
	}
	switch runtimeType {
	case RuntimeContainerd:
		return NewContainerdRuntime(socket)
	case RuntimeCRIO:
		return NewCRIORuntime(socket)
	default:
		return nil, fmt.Errorf("unsupported runtime %q (expected %q or %q)", runtimeType, RuntimeContainerd, RuntimeCRIO)
	}
}

// collectOCIManagedPaths returns the set of paths the OCI runtime considers
// "managed": mount destinations, masked paths, and readonly paths, normalized
// relative to the container rootfs.
func collectOCIManagedPaths(
	ociSpec *specs.Spec,
	rootFS string,
) (map[string]struct{}, error) {
	set := map[string]struct{}{}
	if ociSpec == nil {
		return set, nil
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
		p, err := normalizeOCIPath(raw, rootFS)
		if err != nil {
			return nil, err
		}
		set[p] = struct{}{}
	}
	return set, nil
}

// normalizeOCIPath resolves an OCI spec path relative to rootFS, following
// symlinks within the rootfs boundary (matching runc's addCriuDumpMount pattern).
func normalizeOCIPath(raw, rootFS string) (string, error) {
	p, err := normalizeAbsolutePath(raw, true)
	if err != nil {
		return "", fmt.Errorf("invalid OCI path %q: %w", raw, err)
	}
	if rootFS == "" {
		return p, nil
	}
	root := filepath.Clean(rootFS)
	resolved, err := securejoin.SecureJoin(root, p)
	if err != nil {
		return "", fmt.Errorf("resolve OCI path %q in rootfs: %w", raw, err)
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil {
		return "", fmt.Errorf("make resolved OCI path relative: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("resolved OCI path escapes rootfs")
	}
	if rel == "." {
		return "/", nil
	}
	return "/" + filepath.ToSlash(rel), nil
}

func normalizeAbsolutePath(raw string, allowRoot bool) (string, error) {
	if raw == "" {
		return "", fmt.Errorf("path is empty")
	}
	if len(raw) > maxRootfsPathLength {
		return "", fmt.Errorf("path is too long")
	}
	if strings.IndexByte(raw, 0) >= 0 {
		return "", fmt.Errorf("path contains NUL")
	}
	if !filepath.IsAbs(raw) {
		return "", fmt.Errorf("path is not absolute")
	}
	for _, component := range strings.Split(filepath.ToSlash(raw), "/") {
		if component == "." || component == ".." {
			return "", fmt.Errorf("path contains %q component", component)
		}
	}
	cleaned := filepath.Clean(raw)
	if cleaned == "/" && !allowRoot {
		return "", fmt.Errorf("path is root")
	}
	return filepath.ToSlash(cleaned), nil
}

// RootfsMountExclusions separates the logical OCI destinations from their
// source-rootfs-resolved forms while retaining a deterministic union.
type RootfsMountExclusions struct {
	Raw       []string
	Effective []string
	All       []string
}

// BuildRootfsMountExclusions builds exact capture exclusions for OCI bind
// mounts from their logical destinations and source mountpoints.
func BuildRootfsMountExclusions(
	ociSpec *specs.Spec,
	mounts []types.MountInfo,
	rootFS string,
) (RootfsMountExclusions, error) {
	var result RootfsMountExclusions
	if ociSpec == nil {
		return result, nil
	}

	rawSet := map[string]struct{}{}
	normalizedBindSet := map[string]struct{}{}
	for _, mount := range ociSpec.Mounts {
		if mount.Type != "bind" {
			continue
		}
		raw, err := normalizeAbsolutePath(mount.Destination, false)
		if err != nil {
			return result, fmt.Errorf(
				"invalid OCI bind destination %q: %w",
				mount.Destination,
				err,
			)
		}
		effective, err := normalizeOCIPath(raw, rootFS)
		if err != nil {
			return result, fmt.Errorf(
				"normalize OCI bind destination %q: %w",
				mount.Destination,
				err,
			)
		}
		if effective == "/" {
			return result, fmt.Errorf(
				"invalid OCI bind destination %q: resolves to root",
				mount.Destination,
			)
		}
		rawSet[raw] = struct{}{}
		normalizedBindSet[effective] = struct{}{}
	}

	effectiveSet := map[string]struct{}{}
	for _, mount := range mounts {
		if !mount.IsOCIManaged {
			continue
		}
		mountPoint, err := normalizeAbsolutePath(mount.MountPoint, true)
		if err != nil {
			return result, fmt.Errorf(
				"invalid OCI-managed mountpoint %q: %w",
				mount.MountPoint,
				err,
			)
		}
		if _, ok := normalizedBindSet[mountPoint]; ok {
			effectiveSet[mountPoint] = struct{}{}
		}
	}

	result.Raw = sortedPathSet(rawSet)
	result.Effective = sortedPathSet(effectiveSet)
	allSet := make(map[string]struct{}, len(rawSet)+len(effectiveSet))
	for path := range rawSet {
		allSet[path] = struct{}{}
	}
	for path := range effectiveSet {
		allSet[path] = struct{}{}
	}
	result.All = sortedPathSet(allSet)
	return result, nil
}

func sortedPathSet(set map[string]struct{}) []string {
	paths := make([]string, 0, len(set))
	for path := range set {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths
}
