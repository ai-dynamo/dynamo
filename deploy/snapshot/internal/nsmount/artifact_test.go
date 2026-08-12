// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nsmount

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveArtifactPath(t *testing.T) {
	t.Parallel()

	got, err := ResolveArtifactPath("/checkpoints", "checkpoint-123", "2")
	require.NoError(t, err)
	assert.Equal(t, filepath.Join("/checkpoints", "checkpoint-123", "versions", "2"), got)

	got, err = ResolveArtifactPath("/checkpoints", "checkpoint-123", "")
	require.NoError(t, err)
	assert.Equal(t, filepath.Join("/checkpoints", "checkpoint-123", "versions", "1"), got)
}

func TestResolveArtifactPathRejectsUnsafeCoordinates(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name, basePath, checkpointID, version string
	}{
		{name: "relative base", basePath: "checkpoints", checkpointID: "checkpoint-123", version: "1"},
		{name: "unclean base", basePath: "/checkpoints/../etc", checkpointID: "checkpoint-123", version: "1"},
		{name: "checkpoint traversal", basePath: "/checkpoints", checkpointID: "..", version: "1"},
		{name: "checkpoint separator", basePath: "/checkpoints", checkpointID: "a/b", version: "1"},
		{name: "version traversal", basePath: "/checkpoints", checkpointID: "checkpoint-123", version: ".."},
		{name: "version separator", basePath: "/checkpoints", checkpointID: "checkpoint-123", version: "1/2"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := ResolveArtifactPath(tc.basePath, tc.checkpointID, tc.version)
			require.Error(t, err)
		})
	}
}
