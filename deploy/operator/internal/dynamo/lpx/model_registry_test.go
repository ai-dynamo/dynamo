/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"
)

func TestRegistryBuildURL(t *testing.T) {
	t.Parallel()

	t.Log("Reject unsupported registry schemes at construction")
	_, err := NewModelRegistry("https://unsupported.invalid", nil)
	require.EqualError(t, err, `unsupported model registry URL scheme "https"`)

	t.Log("Define absolute and registry-relative model build references")
	tests := []struct {
		name           string
		registryURL    string
		id             string
		want           string
		wantErrMessage string
	}{
		{name: "direct gcs url", id: "gs://bucket/model/build", want: "gs://bucket/model/build"},
		{name: "direct file url", id: "file:///models/build", want: "file:///models/build"},
		{name: "relative id with gcs registry", registryURL: "gs://bucket/registry", id: "model/build", want: "gs://bucket/registry/model/build"},
		{name: "relative id with file registry", registryURL: "file:///registry", id: "model/build", want: "file:///registry/model/build"},
		{name: "absolute path", id: "/models/build", want: "file:///models/build"},
		{name: "relative id escaping registry", registryURL: "gs://bucket/registry", id: "../model/build", wantErrMessage: "build ID \"../model/build\" must remain within the configured model registry"},
		{name: "relative id without registry", id: "model/build", wantErrMessage: "model registry URL is not configured"},
		{name: "empty id", wantErrMessage: "empty ref"},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Construct the selected model registry")
			registry, err := NewModelRegistry(tt.registryURL, nil)
			require.NoError(t, err)

			t.Log("Resolve the selected build reference")
			buildURL, err := registry.BuildURL(tt.id)
			if tt.wantErrMessage != "" {
				require.EqualError(t, err, tt.wantErrMessage)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, buildURL.String())
		})
	}
}

func TestLocalModelRegistrySnapshot_DanglingGbuildCapnpSymlinkFailsClosed(t *testing.T) {
	t.Parallel()

	t.Log("Create a dangling symlink for the required binary manifest")
	registryDir := t.TempDir()
	buildDir := filepath.Join(registryDir, "build-id")
	require.NoError(t, os.MkdirAll(buildDir, 0o700))
	require.NoError(t, os.Symlink("missing-manifest.v2.capnp.bin", filepath.Join(buildDir, gbuildManifestV2CapnpFile)))

	t.Log("Normalize the local registry build")
	registry, err := NewModelRegistry(registryDir, nil)
	require.NoError(t, err)
	_, err = normalizeRegistryFixtureBuild(t.Context(), registry, "build-id")

	t.Log("Fail closed while preserving the missing manifest identity")
	require.ErrorContains(t, err, gbuildManifestV2CapnpFile)
}

func TestLocalModelRegistrySnapshot_ReportsMetadataDirectories(t *testing.T) {
	t.Parallel()

	t.Log("Create a directory where compiler metadata must be a regular file")
	registryDir := t.TempDir()
	buildDir := filepath.Join(registryDir, "build-id")
	require.NoError(t, os.MkdirAll(buildDir, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildManifestJSONFile), []byte(`{}`), 0o600))
	require.NoError(t, os.Mkdir(filepath.Join(buildDir, gbuildManifestV2CapnpFile), 0o755))

	t.Log("Normalize the invalid local registry build")
	registry, err := NewModelRegistry(registryDir, nil)
	require.NoError(t, err)
	_, err = normalizeRegistryFixtureBuild(t.Context(), registry, "build-id")

	t.Log("Report the invalid metadata path")
	require.Error(t, err)
	require.ErrorContains(t, err, gbuildManifestV2CapnpFile)
}

func TestReadBuildFileBoundsLocalFileAtAcquisition(t *testing.T) {
	t.Parallel()

	t.Log("Create a five-byte local compiler metadata file")
	buildDir := t.TempDir()
	const metadataFile = "metadata.bin"
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, metadataFile), []byte("12345"), 0o600))
	buildURL := &url.URL{Scheme: BuildSchemeFile, Path: buildDir}
	registry := &ModelRegistry{}

	t.Log("Read the file at the inclusive acquisition limit")
	data, err := registry.readBuildFileBounded(t.Context(), buildURL, metadataFile, 5)
	require.NoError(t, err)
	require.Equal(t, []byte("12345"), data)

	t.Log("Reject the same file above a four-byte limit")
	_, err = registry.readBuildFileBounded(t.Context(), buildURL, metadataFile, 4)
	require.ErrorIs(t, err, errBuildFileTooLarge)
}

func TestReadBuildFileBoundsGCSAccumulatedChunks(t *testing.T) {
	t.Parallel()

	t.Log("Create a GCS metadata stream whose chunks exceed the aggregate limit")
	const metadataFile = "metadata.bin"
	client := &fakeModelServiceClient{
		fileStreams: []*fakeModelFileStream{
			{chunks: modelFileChunks(metadataFile, "12", "345")},
		},
	}
	registry := &ModelRegistry{mxClient: client}
	buildURL := &url.URL{Scheme: BuildSchemeGCS, Host: "bucket", Path: "/model/build"}

	t.Log("Reject the accumulated stream before accepting oversized metadata")
	_, err := registry.readBuildFileBounded(t.Context(), buildURL, metadataFile, 4)
	require.ErrorIs(t, err, errBuildFileTooLarge)
	require.ErrorContains(t, err, "larger than 4 bytes")

	t.Log("Release the rejected metadata stream without canceling the caller")
	require.ErrorIs(t, client.filesContext.Err(), context.Canceled)
	require.NoError(t, t.Context().Err())
}

func TestGCSModelRegistrySnapshot_RequiresModelExpressClient(t *testing.T) {
	t.Parallel()

	t.Log("Construct a GCS registry without a Model Express client")
	registry, err := NewModelRegistry("gs://bucket/registry", nil)
	if err != nil {
		t.Fatalf("NewModelRegistry() error = %v", err)
	}

	t.Log("Attempt to normalize a remote build")
	_, err = normalizeRegistryFixtureBuild(context.Background(), registry, "model/build")
	if err == nil {
		t.Fatal("snapshot normalization error = nil, want error")
	}

	t.Log("Require Model Express for GCS registry reads")
	if !strings.Contains(err.Error(), "Model Express client is required for GCS model registry reads") {
		t.Fatalf("snapshot normalization error = %v, want Model Express required error", err)
	}
}

func TestGCSModelRegistrySnapshot_StreamFailures(t *testing.T) {
	t.Parallel()

	t.Log("Define Model Express RPC and chunk-stream failures")
	rpcErr := errors.New("rpc unavailable")
	recvErr := errors.New("stream interrupted")

	tests := []struct {
		name            string
		filesErr        error
		recvErr         error
		chunkPath       string
		chunkData       string
		chunkOffset     uint64
		chunkSize       uint64
		lastChunk       bool
		wantErr         error
		wantErrContains string
	}{
		{
			name:     "stream rpc error",
			filesErr: rpcErr,
			wantErr:  rpcErr,
		},
		{
			name:    "receive error",
			recvErr: recvErr,
			wantErr: recvErr,
		},
		{
			name:            "empty stream",
			wantErrContains: "returned no chunks",
		},
		{
			name: "unexpected relative path", chunkPath: "other.json", chunkData: "{}", chunkSize: 2, lastChunk: true,
			wantErrContains: `unexpected file "other.json"`,
		},
		{
			name: "out of order chunk", chunkPath: gbuildManifestV2CapnpFile, chunkData: "{}",
			chunkOffset: 1, chunkSize: 2, lastChunk: true,
			wantErrContains: "out-of-order chunk",
		},
		{
			name: "missing final chunk", chunkPath: gbuildManifestV2CapnpFile, chunkData: `{"topologies":[]}`,
			wantErrContains: "ended before final chunk",
		},
		{
			name: "final size mismatch", chunkPath: gbuildManifestV2CapnpFile, chunkData: "{}",
			chunkSize: 3, lastChunk: true,
			wantErrContains: "incomplete",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Construct the selected Model Express stream failure")
			client := &fakeModelServiceClient{filesErr: tt.filesErr}
			if tt.filesErr == nil {
				stream := &fakeModelFileStream{err: tt.recvErr}
				if tt.chunkPath != "" {
					stream.chunks = []*modelpb.FileChunk{{
						RelativePath: tt.chunkPath, Data: []byte(tt.chunkData), Offset: tt.chunkOffset,
						TotalSize: tt.chunkSize, IsLastChunk: tt.lastChunk,
					}}
				}
				client.fileStreams = []*fakeModelFileStream{stream}
			}
			client.list = &modelpb.ModelFileList{
				Files: []*modelpb.ModelFileInfo{{RelativePath: gbuildManifestV2CapnpFile}},
			}

			registry, err := NewModelRegistry("gs://bucket/registry", client)
			if err != nil {
				t.Fatalf("NewModelRegistry() error = %v", err)
			}

			t.Log("Normalize through the selected failing stream")
			_, err = normalizeRegistryFixtureBuild(context.Background(), registry, "model/build")
			if err == nil {
				t.Fatal("snapshot normalization error = nil, want error")
			}

			t.Log("Preserve the selected transport or protocol failure")
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Fatalf("snapshot normalization error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErrContains != "" && !strings.Contains(err.Error(), tt.wantErrContains) {
				t.Fatalf("snapshot normalization error = %v, want substring %q", err, tt.wantErrContains)
			}
		})
	}
}

func TestDefinitiveBuildMemberAbsenceIncludesLocalENOTDIR(t *testing.T) {
	t.Parallel()

	t.Log("Classify local ENOTDIR as definitive build-member absence")
	err := &os.PathError{Op: "open", Path: "/build/manifest.v2.capnp.bin", Err: syscall.ENOTDIR}
	require.True(t, isDefinitiveBuildMemberAbsence(err))

	t.Log("Keep transient read failures outside definitive absence")
	require.False(t, isDefinitiveBuildMemberAbsence(errors.New("temporary read failure")))
}

func TestRegistryEnsureDownloadedSkipsLocalBuilds(t *testing.T) {
	t.Log("Construct a registry without Model Express")
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)

	t.Log("Accept an already-local build")
	ready, err := registry.EnsureDownloaded(context.Background(), mustParseURL(t, "file:///models/build"))
	require.NoError(t, err)
	require.True(t, ready)
}

func TestRegistryEnsureDownloadedReturnsFirstStatus(t *testing.T) {
	t.Log("Define terminal and in-progress first Model Express statuses")
	tests := []struct {
		name    string
		status  modelpb.ModelStatus
		message string
	}{
		{name: "downloaded", status: modelpb.ModelStatus_DOWNLOADED},
		{name: "downloading", status: modelpb.ModelStatus_DOWNLOADING},
		{name: "error", status: modelpb.ModelStatus_ERROR, message: "download failed"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Construct the selected first-status Model Express stream")
			client := &fakeModelServiceClient{
				stream: &fakeModelDownloadStream{
					update: &modelpb.ModelStatusUpdate{
						Status:  tt.status,
						Message: ptr.To(tt.message),
					},
				},
			}
			registry := &ModelRegistry{mxClient: client}
			buildURL := mustParseURL(t, "gs://bucket/models/build")

			t.Log("Classify the first Model Express download status")
			ready, err := registry.EnsureDownloaded(context.Background(), buildURL)
			require.NotNil(t, client.request)
			require.Equal(t, buildURL.String(), client.request.ModelName)
			require.Equal(t, modelpb.ModelProvider_GCS, client.request.Provider)
			if tt.status == modelpb.ModelStatus_ERROR {
				require.ErrorContains(t, err, tt.message)
				require.False(t, ready)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.status == modelpb.ModelStatus_DOWNLOADED, ready)
		})
	}
}

func TestRegistryEnsureDownloadedPropagatesTransportFailures(t *testing.T) {
	for _, scenario := range []struct {
		name string
		rpc  bool
		err  error
	}{
		{name: "RPC failure", rpc: true, err: errors.New("rpc unavailable")},
		{name: "receive failure", err: errors.New("stream interrupted")},
		{name: "empty stream", err: io.EOF},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Construct the selected transport failure")
			client := &fakeModelServiceClient{stream: &fakeModelDownloadStream{err: scenario.err}}
			if scenario.rpc {
				client.err = scenario.err
			}
			registry := &ModelRegistry{mxClient: client}

			t.Log("Reject the failed or empty stream before classifying readiness")
			ready, err := registry.EnsureDownloaded(t.Context(), mustParseURL(t, "gs://bucket/models/build"))
			if scenario.err == io.EOF {
				require.ErrorContains(t, err, "returned no status")
			} else {
				require.ErrorIs(t, err, scenario.err)
			}
			require.False(t, ready)
		})
	}
}
