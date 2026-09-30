/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func writeManifestV2Payload(t *testing.T, buildDir string, payload []byte) {
	t.Helper()
	require.NoError(t, os.MkdirAll(buildDir, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildManifestV2CapnpFile), payload, 0o600))
}

func TestAcquireBuildTracksOnlyManifestContent(t *testing.T) {
	t.Parallel()

	t.Log("Write the compiler manifest beside invalid publication JSON and unrelated payloads")
	registryDir := t.TempDir()
	buildDir := filepath.Join(registryDir, "build-id")
	manifest := newManifestV2ContractFixture(t)
	payload, err := manifest.Message().Marshal()
	require.NoError(t, err)
	writeManifestV2Payload(t, buildDir, payload)
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildManifestJSONFile), []byte(`not valid json`), 0o600))
	for _, path := range []string{"extra-metadata.json", "part-0/chip-0.gas", "weights.bin"} {
		fullPath := filepath.Join(buildDir, filepath.FromSlash(path))
		require.NoError(t, os.MkdirAll(filepath.Dir(fullPath), 0o700))
		require.NoError(t, os.WriteFile(fullPath, []byte(`{}`), 0o600))
	}

	t.Log("Acquire an already validated build exclusively from its binary manifest")
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	ref := (&url.URL{Scheme: BuildSchemeFile, Path: buildDir}).String()
	first, err := registry.AcquireBuild(t.Context(), ref)
	require.NoError(t, err)
	require.Equal(t, ref, first.path)
	require.EqualValues(t, 4, first.ioFPGACount)
	require.EqualValues(t, 2, first.ioFanoutFactor)

	t.Log("Resolve the same snapshot through a registry-relative build ID")
	relativeRegistry, err := NewModelRegistry(registryDir, nil)
	require.NoError(t, err)
	relative, err := relativeRegistry.AcquireBuild(t.Context(), "build-id")
	require.NoError(t, err)
	require.Equal(t, first, relative)

	t.Log("Ignore additions, removals, renames and content changes outside the manifest")
	require.NoError(t, os.Rename(filepath.Join(buildDir, "part-0/chip-0.gas"), filepath.Join(buildDir, "part-0/chip-1.gas")))
	require.NoError(t, os.Remove(filepath.Join(buildDir, "extra-metadata.json")))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "weights.bin"), []byte("changed weights"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "new-file"), nil, 0o600))
	second, err := registry.AcquireBuild(t.Context(), ref)
	require.NoError(t, err)
	require.Equal(t, first, second)

	t.Log("Change valid manifest bytes and observe a new digest and decoded build")
	deployment, err := manifest.Deployment()
	require.NoError(t, err)
	runtimeIO, err := deployment.RuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetFanoutFactor(1)
	updated, err := manifest.Message().Marshal()
	require.NoError(t, err)
	writeManifestV2Payload(t, buildDir, updated)
	third, err := registry.AcquireBuild(t.Context(), ref)
	require.NoError(t, err)
	require.NotEqual(t, second.contentID, third.contentID)
	require.EqualValues(t, 1, third.ioFanoutFactor)
	require.EqualValues(t, 2, first.ioFanoutFactor)
}
