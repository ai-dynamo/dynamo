/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestBuildRuntimePath(t *testing.T) {
	t.Parallel()

	t.Log("Define snapshot remapping and final local or GCS runtime-path contracts")
	tests := []struct {
		name             string
		buildPath        string
		runtimeRef       string
		modelStoragePath string
		want             string
		wantErr          string
	}{
		{name: "absolute path", buildPath: "/snapshot/../build", want: "/build"},
		{name: "file URL", buildPath: "file:///snapshot/../build", want: "/build"},
		{name: "GCS URL", buildPath: "gs://bucket/model/build", runtimeRef: "model-build", modelStoragePath: "/models", want: "/models/gcs/bucket/model/build"},
		{name: "relative runtime", buildPath: "file:///snapshot", runtimeRef: "model-build", modelStoragePath: "/models", want: "/models/model-build"},
		{name: "cleaned runtime", buildPath: "file:///snapshot", runtimeRef: " a/../b ", modelStoragePath: "/custom/models", want: "/custom/models/b"},
		{name: "literal runtime escaping", buildPath: "file:///snapshot", runtimeRef: "model%20build", modelStoragePath: "/models", want: "/models/model%20build"},
		{name: "absolute snapshot", buildPath: "/snapshot", runtimeRef: "model-build", modelStoragePath: "/models", want: "/snapshot"},
		{name: "empty runtime", buildPath: "file:///snapshot", modelStoragePath: "/models", want: "/snapshot"},
		{name: "malformed runtime", buildPath: "file:///snapshot", runtimeRef: "%", modelStoragePath: "/models", want: "/snapshot"},
		{name: "runtime URL", buildPath: "file:///snapshot", runtimeRef: "gs://bucket/build", modelStoragePath: "/models", want: "/snapshot"},
		{name: "absolute runtime", buildPath: "file:///snapshot", runtimeRef: "/model-build", modelStoragePath: "/models", want: "/snapshot"},
		{name: "dot runtime", buildPath: "file:///snapshot", runtimeRef: ".", modelStoragePath: "/models", want: "/snapshot"},
		{name: "parent runtime", buildPath: "file:///snapshot", runtimeRef: "..", modelStoragePath: "/models", want: "/snapshot"},
		{name: "escaping runtime", buildPath: "file:///snapshot", runtimeRef: "a/../../b", modelStoragePath: "/models", want: "/snapshot"},
		{name: "malformed snapshot", buildPath: "%", runtimeRef: "model-build", modelStoragePath: "/models", wantErr: "unable to parse ref"},
		{name: "remapped path without storage", buildPath: "file:///snapshot", runtimeRef: "model-build", wantErr: "unexpected file host"},
		{name: "relative path", buildPath: "relative/build", wantErr: "ref \"relative/build\" must be an absolute path or URL"},
		{name: "GCS without storage", buildPath: "gs://bucket/model", wantErr: "model storage path is empty"},
		{name: "GCS without host", buildPath: "gs:///model", modelStoragePath: "/models", wantErr: "invalid ref \"gs:///model\": missing host"},
		{name: "GCS without object", buildPath: "gs://bucket", modelStoragePath: "/models", wantErr: "invalid GCS build path \"gs://bucket\": missing object path"},
		{name: "GCS empty segment", buildPath: "gs://bucket/a//b", modelStoragePath: "/models", wantErr: "bad path segment \"\""},
		{name: "GCS dot segment", buildPath: "gs://bucket/a/./b", modelStoragePath: "/models", wantErr: "bad path segment \".\""},
		{name: "GCS parent segment", buildPath: "gs://bucket/a/../b", modelStoragePath: "/models", wantErr: "bad path segment \"..\""},
		{name: "unsupported scheme", buildPath: "https://bucket/model", wantErr: "unsupported build path scheme \"https\""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the build reference into its runtime filesystem path")
			projection := &ModelProjection{configuredBuild: Build{path: test.buildPath}, runtimeBuildRef: test.runtimeRef}
			got, err := projection.buildRuntimePath(test.modelStoragePath)

			t.Log("Verify the resolved path or exact error contract")
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.want, got)
		})
	}
}

func TestGCSModelRegistrySnapshotUsesDeploymentFromModelExpress(t *testing.T) {
	t.Parallel()

	t.Log("Serve a valid revision-1 compiler manifest through Model Express")
	payload := deploymentPayload(t)
	shortCtx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	for _, testCase := range []struct {
		registryURL string
		ref         string
		ctx         context.Context
	}{
		{registryURL: "gs://bucket/registry", ref: "model/build", ctx: t.Context()},
		{ref: "gs://bucket/registry/model/build", ctx: t.Context()},
		{registryURL: "gs://bucket/registry", ref: "model/build", ctx: shortCtx},
	} {
		t.Run(testCase.ref, func(t *testing.T) {
			t.Log("Create an independent client and registry for this reference form")
			client := &fakeModelServiceClient{
				files: map[string]*fakeModelFileStream{
					gbuildDeploymentV1CapnpFile: {chunks: modelFileChunks(gbuildDeploymentV1CapnpFile, string(payload))},
				},
			}
			registry, err := NewModelRegistry(testCase.registryURL, client)
			require.NoError(t, err)

			t.Log("Read and validate the manifest once, retaining the current registry locator")
			started := time.Now()
			build, err := normalizeRegistryFixtureBuild(testCase.ctx, registry, testCase.ref)
			require.NoError(t, err)
			require.Equal(t, "gs://bucket/registry/model/build", build.path)
			require.Equal(t, 1, build.lpuUnits.Len())
			require.Empty(t, client.listRequests)
			require.Len(t, client.filesRequests, 1)
			require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, client.filesRequests[0].GetFileSelector().GetPaths())

			t.Log("Bound the read to one acquisition deadline and release its context")
			require.Len(t, client.metadataContexts, 1)
			deadline, ok := client.metadataContexts[0].Deadline()
			require.True(t, ok)
			if parentDeadline, hasDeadline := testCase.ctx.Deadline(); hasDeadline {
				require.Equal(t, parentDeadline, deadline)
			} else {
				require.WithinRange(t, deadline, started.Add(30*time.Second), time.Now().Add(30*time.Second))
			}
			for _, rpcCtx := range client.metadataContexts {
				rpcDeadline, hasDeadline := rpcCtx.Deadline()
				require.True(t, hasDeadline)
				require.Equal(t, deadline, rpcDeadline)
				require.ErrorIs(t, rpcCtx.Err(), context.Canceled)
			}
			require.NoError(t, testCase.ctx.Err())
		})
	}
}

func TestAcquireBuildRejectsInvalidGCSContracts(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name    string
		files   []string
		payload string
		wantErr error
		message string
	}{
		{
			name: "corrupt deployment", files: []string{gbuildDeploymentV1CapnpFile},
			payload: "invalid compiler bytes", wantErr: errInvalidBuildManifest, message: "parsing",
		},
		{
			name: "empty manifest exists", files: []string{gbuildDeploymentV1CapnpFile},
			wantErr: errInvalidBuildManifest, message: "parsing",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Expose candidate streams through strict Model Express path selection")
			client := &fakeModelServiceClient{files: map[string]*fakeModelFileStream{}}
			for _, name := range test.files {
				client.files[name] = &fakeModelFileStream{chunks: modelFileChunks(name, test.payload)}
			}

			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)

			t.Log("Read the deployment contract and reject malformed bytes")
			snapshot, err := registry.AcquireBuild(t.Context(), "model/build")
			require.ErrorIs(t, err, test.wantErr)
			require.ErrorContains(t, err, test.message)
			require.Nil(t, snapshot)
			require.Empty(t, client.listRequests)
			require.Len(t, client.filesRequests, 1)
		})
	}
}

func TestAcquireBuildPreservesGCSReadFailures(t *testing.T) {
	t.Parallel()

	for _, readErr := range []error{
		status.Error(codes.NotFound, "manifest missing"),
		status.Error(codes.Unavailable, "object store unavailable"),
		status.Error(codes.PermissionDenied, "access denied"),
		status.Error(codes.DeadlineExceeded, "read deadline exceeded"),
		status.Error(codes.Canceled, "read canceled"),
		errors.New("transport failure"),
	} {
		t.Run(readErr.Error(), func(t *testing.T) {
			t.Log("Fail reading the deployment contract")
			client := &fakeModelServiceClient{files: map[string]*fakeModelFileStream{
				gbuildDeploymentV1CapnpFile: {err: readErr},
			}}
			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)

			t.Log("Preserve the deployment read error")
			snapshot, err := registry.AcquireBuild(t.Context(), "model/build")
			require.ErrorIs(t, err, readErr)
			require.NotErrorIs(t, err, errInvalidBuildManifest)
			require.Nil(t, snapshot)
			require.Empty(t, client.listRequests)
		})
	}
}

func TestAcquireBuildTracksOnlyManifestContent(t *testing.T) {
	t.Parallel()

	t.Log("Write the compiler manifest beside invalid publication JSON and unrelated payloads")
	registryDir := t.TempDir()
	buildDir := filepath.Join(registryDir, "build-id")
	manifest := newDeploymentV1ContractFixture(t)
	payload, err := manifest.Message().Marshal()
	require.NoError(t, err)
	writeDeploymentPayload(t, buildDir, payload)
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
	deployment, err := manifest.Config()
	require.NoError(t, err)
	runtimeIO, err := deployment.RuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetFanoutFactor(1)
	updated, err := manifest.Message().Marshal()
	require.NoError(t, err)
	writeDeploymentPayload(t, buildDir, updated)
	third, err := registry.AcquireBuild(t.Context(), ref)
	require.NoError(t, err)
	require.NotEqual(t, second.contentID, third.contentID)
	require.EqualValues(t, 1, third.ioFanoutFactor)
	require.EqualValues(t, 2, first.ioFanoutFactor)
}

func TestAcquireBuildRejectsLegacyCompilerMetadata(t *testing.T) {
	t.Parallel()

	t.Log("Write a build containing only the retired manifest.capnp.bin and JSON compiler metadata")
	buildDir := t.TempDir()
	for _, path := range []string{
		"manifest.capnp.bin", "allocation_metadata.json", "compile_summary.json",
		"gas_program_layout.json", gbuildManifestJSONFile,
	} {
		require.NoError(t, os.WriteFile(filepath.Join(buildDir, path), []byte(`{}`), 0o600))
	}
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)

	t.Log("Require a supported binary compiler contract without JSON fallback")
	_, err = registry.AcquireBuild(t.Context(), buildDir)
	require.ErrorIs(t, err, os.ErrNotExist)
	require.ErrorContains(t, err, "deployment.v1.capnp.bin")
}

func TestAcquireBuildRejectsMalformedManifest(t *testing.T) {
	t.Parallel()

	t.Log("Write malformed compiler bytes at the required manifest path")
	buildDir := t.TempDir()
	writeDeploymentPayload(t, buildDir, []byte("invalid"))
	registry := &defaultModelRegistry{}

	t.Log("Reject malformed input during acquisition without publishing a snapshot")
	snapshot, err := registry.AcquireBuild(t.Context(), buildDir)
	require.ErrorIs(t, err, errInvalidBuildManifest)
	require.ErrorContains(t, err, "parsing deployment.v1.capnp.bin")
	require.Nil(t, snapshot)
}
