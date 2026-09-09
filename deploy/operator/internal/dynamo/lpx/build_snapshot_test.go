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

func TestAcquireBuildSnapshotTracksLocalContent(t *testing.T) {
	t.Parallel()

	t.Log("Write revision-2 compiler metadata beside invalid publication JSON and unrelated payloads")
	registryDir := t.TempDir()
	buildDir := filepath.Join(registryDir, "build-id")
	payload := manifestV2Payload(t)
	writeManifestV2Payload(t, buildDir, payload)
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildManifestJSONFile), []byte(`not valid json`), 0o600))
	for _, path := range []string{"extra-metadata.json", "part-0/chip-0.gas", "weights.bin"} {
		fullPath := filepath.Join(buildDir, filepath.FromSlash(path))
		require.NoError(t, os.MkdirAll(filepath.Dir(fullPath), 0o700))
		require.NoError(t, os.WriteFile(fullPath, []byte(`{}`), 0o600))
	}

	t.Log("Inventory every file beneath the build root, including metadata, artifacts and weights")
	paths, err := localBuildFilePaths(buildDir)
	require.NoError(t, err)
	require.ElementsMatch(t, []string{
		"extra-metadata.json",
		gbuildManifestJSONFile,
		gbuildManifestV2CapnpFile,
		"part-0/chip-0.gas",
		"weights.bin",
	}, paths)

	t.Log("Acquire the absolute file URL without a configured registry prefix and retain the exact binary manifest")
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	ref := (&url.URL{Scheme: BuildSchemeFile, Path: buildDir}).String()
	first, err := registry.AcquireBuildSnapshot(t.Context(), ref)
	require.NoError(t, err)
	require.Equal(t, payload, first.manifestBytes)

	t.Log("Normalize exclusively from the binary revision-2 contract")
	normalized, err := normalizeBuildSnapshot(first)
	require.NoError(t, err)
	build := normalized.build
	require.Equal(t, 8, build.BatchSize)
	require.Equal(t, int64(8), build.runtimeSettings["batch_size"])
	require.EqualValues(t, 4, build.IOFPGACount)
	require.EqualValues(t, 2, build.IOFanoutFactor)

	t.Log("Resolve and normalize the relative build ID through a configured local registry")
	relativeRegistry, err := NewModelRegistry(registryDir, nil)
	require.NoError(t, err)
	relativeBuild, err := normalizeRegistryFixtureBuild(t.Context(), relativeRegistry, "build-id")
	require.NoError(t, err)
	require.Equal(t, 8, relativeBuild.BatchSize)

	t.Log("Changing only a payload path changes snapshot identity")
	require.NoError(t, os.Rename(filepath.Join(buildDir, "part-0/chip-0.gas"), filepath.Join(buildDir, "part-0/chip-1.gas")))
	second, err := registry.AcquireBuildSnapshot(t.Context(), ref)
	require.NoError(t, err)
	require.NotEqual(t, first.contentID, second.contentID)

	t.Log("Changing only the manifest bytes changes identity again without changing the inventory")
	writeManifestV2Payload(t, buildDir, append(payload, 0))
	third, err := registry.AcquireBuildSnapshot(t.Context(), ref)
	require.NoError(t, err)
	require.NotEqual(t, second.contentID, third.contentID)

	t.Log("Acquire a stable build whose manifest bytes are malformed")
	writeManifestV2Payload(t, buildDir, []byte("invalid"))
	snapshot, err := registry.AcquireBuildSnapshot(t.Context(), ref)
	require.NoError(t, err)

	t.Log("Reject malformed compiler input after successful acquisition")
	_, err = normalizeBuildSnapshot(snapshot)
	require.ErrorContains(t, err, "parsing manifest.v2.capnp.bin")
}

func TestAcquireBuildSnapshotRejectsLegacyCompilerMetadata(t *testing.T) {
	t.Parallel()

	t.Log("Write a build containing only unsupported revision-1 and JSON compiler metadata")
	buildDir := t.TempDir()
	for _, path := range []string{
		"manifest.capnp.bin",
		"allocation_metadata.json",
		"compile_summary.json",
		"gas_program_layout.json",
		gbuildManifestJSONFile,
	} {
		require.NoError(t, os.WriteFile(filepath.Join(buildDir, path), []byte(`{}`), 0o600))
	}
	registry, err := NewModelRegistry(buildDir, nil)
	require.NoError(t, err)

	t.Log("Require revision-2 binary compiler metadata without fallback")
	_, err = registry.AcquireBuildSnapshot(t.Context(), buildDir)
	require.ErrorIs(t, err, ErrBuildSnapshotInconsistent)
	require.ErrorContains(t, err, "is missing manifest.v2.capnp.bin")
}

func TestNormalizeBuildFilePathsCanonicalizesAndRejectsAmbiguity(t *testing.T) {
	t.Parallel()

	t.Log("Canonicalize and sort an unambiguous build inventory")
	inventory, err := normalizeBuildFilePaths([]string{"z/file.gas", gbuildManifestJSONFile, "./a/file.gas"})
	require.NoError(t, err)
	require.Equal(t, []string{"a/file.gas", gbuildManifestJSONFile, "z/file.gas"}, inventory)

	for name, paths := range map[string][]string{
		"normalized duplicate": {"a/file.gas", "./a/file.gas"},
		"parent traversal":     {"../outside.gas"},
		"absolute path":        {filepath.Join(string(filepath.Separator), "outside.gas")},
		"empty path":           {"  "},
	} {
		t.Run(name, func(t *testing.T) {
			t.Log("Reject the selected ambiguous or unsafe inventory path")
			_, err := normalizeBuildFilePaths(paths)
			require.Error(t, err)
		})
	}
}
