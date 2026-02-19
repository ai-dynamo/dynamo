// mounts parses runtime mount state from /proc.
package checkpoint

import (
	"fmt"
	"path"
	"path/filepath"
	"strings"

	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

type MountInfo struct {
	MountID      string
	ParentID     string
	MountPoint   string
	Root         string
	FSType       string
	Source       string
	Options      string
	SuperOptions string
}

// MountPolicy is the classified mount plan for CRIU dump options.
type MountPolicy struct {
	Externalized []string
	Skipped      []string
}

// BuildMountPolicy classifies mounts into CRIU extMnt and skipMnt lists.
//
// Rule order and precedence (top to bottom):
//  1. Skip non-OCI proc/sys submounts and non-OCI runtime /run submounts.
//     These mounts are typically node/kernel/runtime specific and are the
//     highest-risk source of cross-node restore failures, so skip wins.
//  2. Externalize mounts owned by runtime/OCI:
//     - "/" (rootfs is recreated by runtime in OCI restore path)
//     - OCI mount destinations
//     - OCI masked/readonly paths
//  3. Externalize non-OCI bind-like mounts (mount root is not "/" or ".").
//     This captures runtime-injected file mounts (for example driver files)
//     so CRIU does not try to recreate them from checkpoint data.
//  4. Anything else is left unflagged and handled by CRIU default behavior.
//
// Precedence: skip > externalize. If a path is classified as skipped, it is
// removed from the externalized set.
func BuildMountPolicy(mountInfo []MountInfo, ociSpec *specs.Spec, rootFS string) *MountPolicy {
	ociManagedSet := collectOCIManagedDestinations(ociSpec, rootFS)

	externalizedSet := make(map[string]struct{}, len(mountInfo)+len(ociManagedSet))
	skippedSet := make(map[string]struct{}, len(mountInfo))

	for _, mount := range mountInfo {
		mp := normalizeMountPath(mount.MountPoint)
		if mp == "" {
			continue
		}

		source := path.Clean(strings.TrimSpace(mount.Source))
		root := path.Clean(strings.TrimSpace(mount.Root))
		isOCIManaged := false
		if _, ok := ociManagedSet[mp]; ok {
			isOCIManaged = true
		}
		if !isOCIManaged && strings.HasPrefix(mp, "/run/") {
			if _, ok := ociManagedSet["/var"+mp]; ok {
				isOCIManaged = true
			}
		}
		if !isOCIManaged && strings.HasPrefix(mp, "/var/run/") {
			if _, ok := ociManagedSet[strings.TrimPrefix(mp, "/var")]; ok {
				isOCIManaged = true
			}
		}

		// Runtime-owned /run mounts are usually ephemeral tmpfs/overlay mounts
		// or bind-like mounts sourced from host runtime directories.
		// We skip these unless OCI explicitly manages that destination.
		isRunRuntimeMount := strings.HasPrefix(mp, "/run/") &&
			(mount.FSType == "tmpfs" ||
				mount.FSType == "overlay" ||
				strings.HasPrefix(source, "/run/") ||
				strings.HasPrefix(source, "/var/run/") ||
				strings.HasPrefix(root, "/run/") ||
				strings.HasPrefix(root, "/var/run/"))

		if !isOCIManaged && (strings.HasPrefix(mp, "/proc/") || strings.HasPrefix(mp, "/sys/") || isRunRuntimeMount) {
			skippedSet[mp] = struct{}{}
			delete(externalizedSet, mp)
			continue
		}

		if mp == "/" || isOCIManaged || (root != "." && root != "/") {
			externalizedSet[mp] = struct{}{}
			continue
		}
	}

	// Ensure OCI-managed destinations are externalized, even when mountinfo does not
	// include a direct entry (e.g., runtime-managed masked/readonly paths).
	for mp := range ociManagedSet {
		if _, skipped := skippedSet[mp]; skipped {
			continue
		}
		externalizedSet[mp] = struct{}{}
	}

	externalized := make([]string, 0, len(externalizedSet))
	for mp := range externalizedSet {
		externalized = append(externalized, mp)
	}
	skipped := make([]string, 0, len(skippedSet))
	for mp := range skippedSet {
		skipped = append(skipped, mp)
	}

	return &MountPolicy{
		Externalized: externalized,
		Skipped:      skipped,
	}
}

// collectOCIManagedDestinations returns the canonical set of OCI-owned mount
// targets. This includes regular OCI mounts plus Linux masked/readonly paths.
// Those masked/readonly paths may not appear as direct mountinfo entries, but
// still need to be treated as runtime-owned and externalized.
func collectOCIManagedDestinations(ociSpec *specs.Spec, rootFS string) map[string]struct{} {
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
		if p := normalizeOCIDestinationPath(raw, rootFS); p != "" {
			set[p] = struct{}{}
		}
	}

	return set
}

// normalizeMountPath applies lexical normalization only.
// Mountinfo paths are already kernel truth for the container namespace.
func normalizeMountPath(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}

	p := path.Clean(raw)
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return path.Clean(p)
}

// normalizeOCIDestinationPath canonicalizes OCI destinations against container
// rootfs symlinks (for example /var/run -> /run) with lexical fallback.
func normalizeOCIDestinationPath(raw, rootFS string) string {
	p := normalizeMountPath(raw)
	if p == "" || rootFS == "" {
		return p
	}

	hostPath := filepath.Join(rootFS, strings.TrimPrefix(p, "/"))
	resolved, err := filepath.EvalSymlinks(hostPath)
	if err != nil {
		return p
	}

	rel, err := filepath.Rel(rootFS, resolved)
	if err != nil {
		return p
	}
	rel = filepath.ToSlash(rel)
	if rel == "." {
		return "/"
	}
	if strings.HasPrefix(rel, "../") || rel == ".." {
		return p
	}

	return normalizeMountPath("/" + rel)
}

func ReadMountInfoFromHostProcPath(pid int) ([]MountInfo, error) {
	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", HostProcPath, pid)
	parsedMounts, err := common.ParseMountInfoFile(mountinfoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo at %s: %w", mountinfoPath, err)
	}

	mounts := make([]MountInfo, 0, len(parsedMounts))
	for _, parsed := range parsedMounts {
		mounts = append(mounts, MountInfo{
			MountID:      parsed.MountID,
			ParentID:     parsed.ParentID,
			MountPoint:   parsed.Path,
			Root:         parsed.Root,
			FSType:       parsed.FSType,
			Source:       parsed.Source,
			Options:      parsed.Options,
			SuperOptions: parsed.SuperOpts,
		})
	}

	return mounts, nil
}
