// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nsmount

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/moby/sys/mountinfo"
)

// ValidateBasePathMount verifies that basePath is a real mountpoint without
// mutating what is mounted there.
func ValidateBasePathMount(basePath string) error {
	if !filepath.IsAbs(basePath) || filepath.Clean(basePath) != basePath {
		return fmt.Errorf("base path must be an absolute, clean path: %q", basePath)
	}
	info, err := os.Lstat(basePath)
	if err != nil {
		return fmt.Errorf("lstat base path %s: %w", basePath, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return fmt.Errorf("base path %s must be a real directory", basePath)
	}
	mounted, err := mountinfo.Mounted(basePath)
	if err != nil {
		return fmt.Errorf("inspect base path mount %s: %w", basePath, err)
	}
	if !mounted {
		return fmt.Errorf("base path %s is not a mountpoint", basePath)
	}
	return nil
}

// ResolveArtifact composes the artifact path from agent configuration and
// checks it is usable, returning the path the mount will be given.
//
// The path is composed rather than supplied: the storage-location annotation
// this replaced let a pod name any agent-visible directory, so the components
// are validated and the result must contain no symlink. Reading the manifest is
// left to the caller, which uses the ordinary types.ReadManifest.
func ResolveArtifact(basePath, artifactID, version string) (string, error) {
	artifactPath, err := ResolveArtifactPath(basePath, artifactID, version)
	if err != nil {
		return "", err
	}

	// EvalSymlinks both rejects a symlinked component and hands back the
	// canonical name, which is how mount points are recorded and therefore what
	// the nested-mount check has to compare against.
	resolved, err := filepath.EvalSymlinks(artifactPath)
	if err != nil {
		return "", fmt.Errorf("resolve artifact %s: %w", artifactPath, err)
	}
	if resolved != artifactPath {
		return "", fmt.Errorf("artifact %s resolves through a symlink to %s", artifactPath, resolved)
	}
	info, err := os.Stat(artifactPath)
	if err != nil {
		return "", fmt.Errorf("stat artifact %s: %w", artifactPath, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("artifact %s is not a directory", artifactPath)
	}
	if err := rejectNestedMounts(artifactPath); err != nil {
		return "", err
	}
	return artifactPath, nil
}

// rejectNestedMounts refuses an artifact that has anything mounted at or below
// it. The clone the helper makes is deliberately non-recursive, so a submount
// would be silently omitted from the restore rather than carried into the
// target; failing loudly is better than restoring a checkpoint that is quietly
// missing part of itself.
//
// A nested mount is a row in the mount table, not something discoverable by
// walking files, so this reads the table once. Walking the artifact instead
// would mean a statx per entry across a shared network filesystem, on the
// restore hot path, for a tree that can hold thousands of CRIU images.
//
// This is best-effort against a mount created after the check and before the
// clone, exactly as the previous implementation was: only host root can mount,
// and the same privilege would defeat any check the agent could make.
func rejectNestedMounts(artifactPath string) error {
	nested, err := mountinfo.GetMounts(mountinfo.PrefixFilter(artifactPath))
	if err != nil {
		return fmt.Errorf("inspect mounts under artifact %s: %w", artifactPath, err)
	}
	if len(nested) > 0 {
		return fmt.Errorf("artifact %s contains nested mount at %s", artifactPath, nested[0].Mountpoint)
	}
	return nil
}
