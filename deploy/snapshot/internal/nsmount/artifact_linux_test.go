// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// mountinfo.GetMountsFromReader, which the nested-mount cases use to state
// the filter rule against a fixed table, is Linux-only.
//go:build linux

package nsmount

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/moby/sys/mountinfo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func makeArtifactTree(t *testing.T) (basePath, artifactPath string) {
	t.Helper()
	basePath = t.TempDir()
	artifactPath = filepath.Join(basePath, "checkpoint-123", "versions", "1")
	require.NoError(t, os.MkdirAll(artifactPath, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(artifactPath, "manifest.yaml"), []byte("checkpointId: checkpoint-123\n"), 0o600))
	return basePath, artifactPath
}

func TestResolveArtifactReturnsTheComposedPath(t *testing.T) {
	t.Parallel()

	basePath, artifactPath := makeArtifactTree(t)
	got, err := ResolveArtifact(basePath, "checkpoint-123", "1")
	require.NoError(t, err)
	assert.Equal(t, artifactPath, got)
}

// TestResolveArtifactRejectsUnusablePaths covers what the deleted
// storage-location annotation used to make possible. The path is composed from
// agent configuration now, so these are the ways a bad one could still reach
// the mount.
func TestResolveArtifactRejectsUnusablePaths(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		setup func(t *testing.T) (basePath string)
	}{
		{
			// A symlink anywhere in the path would let the mount source be
			// somewhere other than the composed location.
			name: "symlinked checkpoint component",
			setup: func(t *testing.T) string {
				basePath := t.TempDir()
				outside, _ := makeArtifactTree(t)
				require.NoError(t, os.Symlink(filepath.Join(outside, "checkpoint-123"), filepath.Join(basePath, "checkpoint-123")))
				return basePath
			},
		},
		{
			name: "artifact itself is a symlink",
			setup: func(t *testing.T) string {
				basePath := t.TempDir()
				versions := filepath.Join(basePath, "checkpoint-123", "versions")
				require.NoError(t, os.MkdirAll(versions, 0o755))
				require.NoError(t, os.Symlink(t.TempDir(), filepath.Join(versions, "1")))
				return basePath
			},
		},
		{
			name: "artifact is a regular file",
			setup: func(t *testing.T) string {
				basePath := t.TempDir()
				versions := filepath.Join(basePath, "checkpoint-123", "versions")
				require.NoError(t, os.MkdirAll(versions, 0o755))
				require.NoError(t, os.WriteFile(filepath.Join(versions, "1"), []byte("not a directory"), 0o600))
				return basePath
			},
		},
		{
			name:  "artifact does not exist",
			setup: func(t *testing.T) string { return t.TempDir() },
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := ResolveArtifact(tc.setup(t), "checkpoint-123", "1")
			require.Error(t, err)
		})
	}
}

// TestNestedMountFilterSemantics pins the matching rule the nested-mount check
// depends on. rejectNestedMounts is a security control whose entire behaviour is
// PrefixFilter's definition of "below", so that definition is asserted here
// rather than assumed: a real mount table is not needed to state what should and
// should not count as nested.
func TestNestedMountFilterSemantics(t *testing.T) {
	t.Parallel()

	const artifact = "/checkpoints/abc/versions/1"
	// Field order matches /proc/<pid>/mountinfo: id, parent, dev, root,
	// mountpoint, options, separator, fstype, source, superblock options.
	const table = `21 20 0:20 / / rw,relatime - overlay overlay rw
22 21 0:21 / /checkpoints rw,relatime - nfs4 nfs rw
23 22 0:22 / /checkpoints/abc/versions/10 rw,relatime - tmpfs tmpfs rw
24 22 0:23 / /checkpoints/abcdef rw,relatime - tmpfs tmpfs rw
`

	tests := []struct {
		name  string
		extra string
		want  []string
	}{
		{
			name: "no mount at or below the artifact",
			want: nil,
		},
		{
			name:  "a mount inside the artifact is nested",
			extra: "25 22 0:24 / " + artifact + "/images rw,relatime - tmpfs tmpfs rw\n",
			want:  []string{artifact + "/images"},
		},
		{
			// The clone would take this mount rather than the directory the
			// agent published, so it is rejected too.
			name:  "a mount exactly at the artifact is nested",
			extra: "25 22 0:24 / " + artifact + " rw,relatime - tmpfs tmpfs rw\n",
			want:  []string{artifact},
		},
		{
			// /checkpoints/abc/versions/10 and /checkpoints/abcdef both share a
			// string prefix with the artifact path and must not match.
			name:  "sibling paths that share a string prefix are not nested",
			extra: "",
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mounts, err := mountinfo.GetMountsFromReader(
				strings.NewReader(table+tc.extra),
				mountinfo.PrefixFilter(artifact),
			)
			require.NoError(t, err)

			got := make([]string, 0, len(mounts))
			for _, m := range mounts {
				got = append(got, m.Mountpoint)
			}
			if tc.want == nil {
				assert.Empty(t, got)
				return
			}
			assert.Equal(t, tc.want, got)
		})
	}
}
