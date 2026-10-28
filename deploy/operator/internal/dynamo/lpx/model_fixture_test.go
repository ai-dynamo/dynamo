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
	"testing"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const registryTestTopology = "test-topology"

const gbuildManifestJSONFile = "manifest.json"

func normalizeRegistryFixtureBuild(ctx context.Context, registry ModelRegistry, id string) (*Build, error) {
	snapshot, err := registry.AcquireBuild(ctx, id)
	if err != nil {
		return nil, err
	}
	return snapshot, nil
}

type fakeModelServiceClient struct {
	request       *modelpb.ModelDownloadRequest
	stream        grpc.ServerStreamingClient[modelpb.ModelStatusUpdate]
	err           error
	filesRequests []*modelpb.ModelFilesRequest
	filesContext  context.Context
	fileStreams   []*fakeModelFileStream
	files         map[string]*fakeModelFileStream
	filesErr      error
	listRequests  []*modelpb.ModelFilesRequest

	metadataContexts []context.Context
}

func (c *fakeModelServiceClient) EnsureModelDownloaded(
	_ context.Context,
	request *modelpb.ModelDownloadRequest,
	_ ...grpc.CallOption,
) (grpc.ServerStreamingClient[modelpb.ModelStatusUpdate], error) {
	c.request = request
	if c.err != nil {
		return nil, c.err
	}
	return c.stream, nil
}

func (c *fakeModelServiceClient) StreamModelFiles(
	ctx context.Context,
	request *modelpb.ModelFilesRequest,
	_ ...grpc.CallOption,
) (grpc.ServerStreamingClient[modelpb.FileChunk], error) {
	c.filesRequests = append(c.filesRequests, request)
	c.metadataContexts = append(c.metadataContexts, ctx)
	c.filesContext = ctx
	if c.filesErr != nil {
		return nil, c.filesErr
	}
	if c.files != nil {
		// Each acquisition must request one candidate; missing paths fail before streaming.
		paths := request.GetFileSelector().GetPaths()
		if len(paths) != 1 {
			return nil, status.Error(codes.InvalidArgument, "expected exactly one selected path")
		}
		stream, exists := c.files[paths[0]]
		if !exists {
			return nil, status.Errorf(codes.NotFound, "Selected file not found in model directory: %s", paths[0])
		}
		return stream, nil
	}
	stream := c.fileStreams[0]
	c.fileStreams = c.fileStreams[1:]
	return stream, nil
}

func (c *fakeModelServiceClient) DeleteModel(
	context.Context,
	*modelpb.DeleteModelRequest,
	...grpc.CallOption,
) (*modelpb.DeleteModelResponse, error) {
	return nil, errors.New("DeleteModel is not used by LPX")
}

func (c *fakeModelServiceClient) ListModelFiles(
	_ context.Context,
	request *modelpb.ModelFilesRequest,
	_ ...grpc.CallOption,
) (*modelpb.ModelFileList, error) {
	c.listRequests = append(c.listRequests, request)
	return nil, status.Error(codes.Unimplemented, "unexpected ListModelFiles call")
}

type fakeModelDownloadStream struct {
	grpc.ClientStream
	update *modelpb.ModelStatusUpdate
	err    error
}

func (s *fakeModelDownloadStream) Recv() (*modelpb.ModelStatusUpdate, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.update, nil
}

type fakeModelFileStream struct {
	grpc.ClientStream
	chunks []*modelpb.FileChunk
	err    error
}

func (s *fakeModelFileStream) Recv() (*modelpb.FileChunk, error) {
	if len(s.chunks) > 0 {
		chunk := s.chunks[0]
		s.chunks = s.chunks[1:]
		return chunk, nil
	}
	if s.err != nil {
		return nil, s.err
	}
	return nil, io.EOF
}

func modelFileChunks(relativePath string, parts ...string) []*modelpb.FileChunk {
	totalSize := 0
	for _, part := range parts {
		totalSize += len(part)
	}

	var offset uint64
	chunks := make([]*modelpb.FileChunk, 0, len(parts))
	for i, part := range parts {
		chunks = append(chunks, &modelpb.FileChunk{
			RelativePath: relativePath,
			Data:         []byte(part),
			Offset:       offset,
			TotalSize:    uint64(totalSize),
			IsLastChunk:  i == len(parts)-1,
		})
		offset += uint64(len(part))
	}
	return chunks
}

func mustParseURL(t *testing.T, rawURL string) url.URL {
	t.Helper()

	buildURL, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("url.Parse(%q) error = %v", rawURL, err)
	}
	return *buildURL
}

func defaultTestString(value string, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
