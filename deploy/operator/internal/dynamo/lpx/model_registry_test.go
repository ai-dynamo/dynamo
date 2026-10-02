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
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
		{name: "gcs url with gcs registry", registryURL: "gs://bucket/registry", id: "gs://other-bucket/build", wantErrMessage: `build ID "gs://other-bucket/build" must be relative when model registry URL is configured`},
		{name: "file url with gcs registry", registryURL: "gs://bucket/registry", id: "file:///models/build", wantErrMessage: `build ID "file:///models/build" must be relative when model registry URL is configured`},
		{name: "absolute path with gcs registry", registryURL: "gs://bucket/registry", id: "/models/build", wantErrMessage: `build ID "/models/build" must be relative when model registry URL is configured`},
		{name: "gcs url with file registry", registryURL: "file:///registry", id: "gs://bucket/build", wantErrMessage: `build ID "gs://bucket/build" must be relative when model registry URL is configured`},
		{name: "file url with file registry", registryURL: "file:///registry", id: "file:///models/build", wantErrMessage: `build ID "file:///models/build" must be relative when model registry URL is configured`},
		{name: "absolute path with file registry", registryURL: "file:///registry", id: "/models/build", wantErrMessage: `build ID "/models/build" must be relative when model registry URL is configured`},
		{name: "absolute url within registry", registryURL: "gs://bucket/registry", id: "gs://bucket/registry/build", wantErrMessage: `build ID "gs://bucket/registry/build" must be relative when model registry URL is configured`},
		{name: "padded absolute path", registryURL: "file:///registry", id: " /models/build ", wantErrMessage: `build ID " /models/build " must be relative when model registry URL is configured`},
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

func TestAcquireLocalBuildSnapshotRejectsUnreadableManifest(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name  string
		setup func(t *testing.T, manifestPath string)
	}{
		{name: "missing", setup: func(*testing.T, string) {}},
		{name: "dangling symlink", setup: func(t *testing.T, manifestPath string) {
			require.NoError(t, os.Symlink("missing-manifest.v2.capnp.bin", manifestPath))
		}},
		{name: "directory", setup: func(t *testing.T, manifestPath string) {
			require.NoError(t, os.Mkdir(manifestPath, 0o755))
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Create a registry build whose manifest cannot be read as a regular file")
			registryDir := t.TempDir()
			buildDir := filepath.Join(registryDir, "build-id")
			require.NoError(t, os.MkdirAll(buildDir, 0o700))
			test.setup(t, filepath.Join(buildDir, gbuildManifestV2CapnpFile))
			registry, err := NewModelRegistry(registryDir, nil)
			require.NoError(t, err)

			t.Log("Fail closed as an acquisition error that names the manifest")
			snapshot, err := registry.AcquireBuild(t.Context(), "build-id")
			require.ErrorContains(t, err, gbuildManifestV2CapnpFile)
			require.NotErrorIs(t, err, errInvalidBuildManifest)
			require.Nil(t, snapshot)
		})
	}
}

func TestAcquireLocalBuildSnapshotCancellation(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name    string
		timeout time.Duration
		wantErr error
	}{
		{name: "canceled", timeout: time.Hour, wantErr: context.Canceled},
		{name: "expired", timeout: -time.Second, wantErr: context.DeadlineExceeded},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Create a valid local build with an already canceled or expired acquisition context")
			buildDir := t.TempDir()
			writeManifestV2Payload(t, buildDir, manifestV2Payload(t))
			ctx, cancel := context.WithTimeout(t.Context(), tt.timeout)
			defer cancel()
			if tt.wantErr == context.Canceled {
				cancel()
			}

			t.Log("Reject the acquisition without returning a snapshot")
			registry := &defaultModelRegistry{}
			snapshot, err := registry.AcquireBuild(ctx, buildDir)
			require.ErrorIs(t, err, tt.wantErr)
			require.Nil(t, snapshot)
		})
	}
}

func TestAcquireLocalBuildSnapshotRejectsFIFOManifest(t *testing.T) {
	t.Parallel()

	const buildDirEnv = "DYNAMO_TEST_FIFO_MANIFEST_BUILD_DIR"
	if buildDir := os.Getenv(buildDirEnv); buildDir != "" {
		t.Log("Reject the FIFO manifest while the acquisition context is still active")
		ctx, cancel := context.WithTimeout(t.Context(), time.Second)
		defer cancel()
		snapshot, err := (&defaultModelRegistry{}).AcquireBuild(ctx, buildDir)
		require.ErrorContains(t, err, "not a regular file")
		require.ErrorContains(t, err, gbuildManifestV2CapnpFile)
		require.Nil(t, snapshot)
		require.NoError(t, ctx.Err())
		return
	}

	t.Log("Create a valid build directory containing a FIFO manifest with no writer")
	buildDir := t.TempDir()
	require.NoError(t, syscall.Mkfifo(filepath.Join(buildDir, gbuildManifestV2CapnpFile), 0o600))

	t.Log("Bound the subprocess so a regressed FIFO open cannot hang or leak a goroutine")
	executable, err := os.Executable()
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, "-test.run=^"+t.Name()+"$", "-test.v")
	cmd.Env = append(os.Environ(), buildDirEnv+"="+buildDir)
	output, err := cmd.CombinedOutput()
	require.NoError(t, ctx.Err(), "FIFO manifest acquisition blocked past its deadline: %s", output)
	require.NoError(t, err, "%s", output)
}

func TestReadBuildFileBoundsLocalFileAtAcquisition(t *testing.T) {
	t.Parallel()

	t.Log("Create a five-byte local compiler metadata file")
	buildDir := t.TempDir()
	const metadataFile = "metadata.bin"
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, metadataFile), []byte("12345"), 0o600))
	buildURL := &url.URL{Scheme: BuildSchemeFile, Path: buildDir}
	registry := &defaultModelRegistry{}

	t.Log("Read the file at the inclusive acquisition limit")
	data, err := registry.readBuildFileBounded(t.Context(), buildURL, metadataFile, 5)
	require.NoError(t, err)
	require.Equal(t, []byte("12345"), data)

	t.Log("Read the same metadata through a symlink at the inclusive acquisition limit")
	const metadataLink = "metadata-link.bin"
	require.NoError(t, os.Symlink(metadataFile, filepath.Join(buildDir, metadataLink)))
	data, err = registry.readBuildFileBounded(t.Context(), buildURL, metadataLink, 5)
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
	registry := &defaultModelRegistry{mxClient: client}
	buildURL := &url.URL{Scheme: BuildSchemeGCS, Host: "bucket", Path: "/model/build"}

	t.Log("Reject the accumulated stream before accepting oversized metadata")
	_, err := registry.readBuildFileBounded(t.Context(), buildURL, metadataFile, 4)
	require.ErrorIs(t, err, errBuildFileTooLarge)
	require.ErrorContains(t, err, "larger than 4 bytes")

	t.Log("Release the rejected metadata stream without canceling the caller")
	require.ErrorIs(t, client.metadataContexts[0].Err(), context.Canceled)
	require.NoError(t, t.Context().Err())
}

func TestGCSBuildSnapshotReadFailures(t *testing.T) {
	t.Parallel()

	t.Log("Define Model Express availability, RPC, and chunk-stream failures")
	rpcErr := errors.New("rpc unavailable")
	notFoundErr := status.Error(codes.NotFound, "manifest missing")
	unavailableErr := status.Error(codes.Unavailable, "temporary object-store failure")
	chunk := func(path, data string, offset, size uint64, last bool) []*modelpb.FileChunk {
		return []*modelpb.FileChunk{{RelativePath: path, Data: []byte(data), Offset: offset, TotalSize: size, IsLastChunk: last}}
	}

	for _, test := range []struct {
		name            string
		client          *fakeModelServiceClient
		wantErr         error
		wantErrContains string
	}{
		{
			name:            "missing Model Express client",
			wantErrContains: "Model Express client is required for GCS model registry reads",
		},
		{
			name:    "stream rpc error",
			client:  &fakeModelServiceClient{filesErr: rpcErr},
			wantErr: rpcErr,
		},
		{
			name:    "not found",
			client:  &fakeModelServiceClient{fileStreams: []*fakeModelFileStream{{err: notFoundErr}}},
			wantErr: notFoundErr,
		},
		{
			name:    "unavailable",
			client:  &fakeModelServiceClient{fileStreams: []*fakeModelFileStream{{err: unavailableErr}}},
			wantErr: unavailableErr,
		},
		{
			name:            "empty stream",
			client:          &fakeModelServiceClient{fileStreams: []*fakeModelFileStream{{}}},
			wantErrContains: "returned no chunks",
		},
		{
			name:            "unexpected relative path",
			client:          &fakeModelServiceClient{fileStreams: []*fakeModelFileStream{{chunks: chunk("other.json", "{}", 0, 2, true)}}},
			wantErrContains: `unexpected file "other.json"`,
		},
		{
			name:            "out of order chunk",
			client:          &fakeModelServiceClient{fileStreams: []*fakeModelFileStream{{chunks: chunk(gbuildManifestV2CapnpFile, "{}", 1, 2, true)}}},
			wantErrContains: "out-of-order chunk",
		},
		{
			name:            "missing final chunk",
			client:          &fakeModelServiceClient{fileStreams: []*fakeModelFileStream{{chunks: chunk(gbuildManifestV2CapnpFile, "{}", 0, 0, false)}}},
			wantErrContains: "ended before final chunk",
		},
		{
			name:            "final size mismatch",
			client:          &fakeModelServiceClient{fileStreams: []*fakeModelFileStream{{chunks: chunk(gbuildManifestV2CapnpFile, "{}", 0, 3, true)}}},
			wantErrContains: "incomplete",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Acquire the manifest through the selected failing Model Express client")
			var client modelpb.ModelServiceClient
			if test.client != nil {
				client = test.client
			}
			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)
			snapshot, err := registry.AcquireBuild(t.Context(), "model/build")

			t.Log("Preserve the transport or protocol failure separately from invalid compiler input")
			require.Error(t, err)
			require.Nil(t, snapshot)
			require.NotErrorIs(t, err, errInvalidBuildManifest)
			if test.wantErr != nil {
				require.ErrorIs(t, err, test.wantErr)
			}
			if test.wantErrContains != "" {
				require.ErrorContains(t, err, test.wantErrContains)
			}
			if test.client != nil {
				require.Empty(t, test.client.listRequests)
				require.Len(t, test.client.filesRequests, 1)
			}
		})
	}
}

func TestRegistryEnsureDownloaded(t *testing.T) {
	t.Log("Define local, first-status, and transport outcomes")
	rpcErr := errors.New("rpc unavailable")
	recvErr := errors.New("stream interrupted")
	statusStream := func(status modelpb.ModelStatus, message string) *fakeModelServiceClient {
		return &fakeModelServiceClient{stream: &fakeModelDownloadStream{
			update: &modelpb.ModelStatusUpdate{Status: status, Message: ptr.To(message)},
		}}
	}
	for _, test := range []struct {
		name      string
		buildURL  string
		client    *fakeModelServiceClient
		wantReady bool
		wantErr   error
		wantMsg   string
	}{
		{name: "local build", buildURL: "file:///models/build", wantReady: true},
		{name: "unsupported scheme", buildURL: "https://models/build", wantMsg: `unsupported build download scheme "https"`},
		{name: "missing Model Express client", buildURL: "gs://bucket/models/build", wantMsg: "Model Express client is required"},
		{name: "downloaded", client: statusStream(modelpb.ModelStatus_DOWNLOADED, ""), wantReady: true},
		{name: "downloading", client: statusStream(modelpb.ModelStatus_DOWNLOADING, "")},
		{name: "error", client: statusStream(modelpb.ModelStatus_ERROR, "download failed"), wantMsg: "download failed"},
		{name: "RPC failure", client: &fakeModelServiceClient{err: rpcErr}, wantErr: rpcErr},
		{name: "receive failure", client: &fakeModelServiceClient{stream: &fakeModelDownloadStream{err: recvErr}}, wantErr: recvErr},
		{name: "empty stream", client: &fakeModelServiceClient{stream: &fakeModelDownloadStream{err: io.EOF}}, wantMsg: "returned no status"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Classify readiness from the selected build location and Model Express stream")
			buildURL := mustParseURL(t, defaultTestString(test.buildURL, "gs://bucket/models/build"))
			registry := &defaultModelRegistry{}
			if test.client != nil {
				registry.mxClient = test.client
			}
			ready, err := registry.EnsureDownloaded(t.Context(), buildURL)
			require.Equal(t, test.wantReady, ready)
			switch {
			case test.wantErr != nil:
				require.ErrorIs(t, err, test.wantErr)
			case test.wantMsg != "":
				require.ErrorContains(t, err, test.wantMsg)
			default:
				require.NoError(t, err)
			}
			if test.client != nil {
				require.Equal(t, buildURL.String(), test.client.request.ModelName)
				require.Equal(t, modelpb.ModelProvider_GCS, test.client.request.Provider)
			}
		})
	}
}
