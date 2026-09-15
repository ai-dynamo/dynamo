/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"io"
	"net/url"
	"testing"

	"capnproto.org/go/capnp/v3"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

const registryTestTopology = "URSA_V2_1__Q8__8C__G_106__KP_FEC__GHZ_1_0__NO_FPGA"

const gbuildManifestJSONFile = "manifest.json"

func normalizeRegistryFixtureBuild(ctx context.Context, registry *ModelRegistry, id string) (*Build, error) {
	snapshot, err := registry.AcquireBuildSnapshot(ctx, id)
	if err != nil {
		return nil, err
	}
	normalized, err := normalizeBuildSnapshot(snapshot)
	if err != nil {
		return nil, err
	}
	return normalized.build, nil
}

type fakeModelServiceClient struct {
	request       *modelpb.ModelDownloadRequest
	stream        grpc.ServerStreamingClient[modelpb.ModelStatusUpdate]
	err           error
	filesRequests []*modelpb.ModelFilesRequest
	filesContext  context.Context
	fileStreams   []*fakeModelFileStream
	filesErr      error
	listRequests  []*modelpb.ModelFilesRequest
	list          *modelpb.ModelFileList
	listErr       error

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
	stream := c.fileStreams[0]
	c.fileStreams = c.fileStreams[1:]
	return stream, nil
}

func (c *fakeModelServiceClient) ListModelFiles(
	ctx context.Context,
	request *modelpb.ModelFilesRequest,
	_ ...grpc.CallOption,
) (*modelpb.ModelFileList, error) {
	c.listRequests = append(c.listRequests, request)
	c.metadataContexts = append(c.metadataContexts, ctx)
	if c.listErr != nil {
		return nil, c.listErr
	}
	if c.list != nil {
		return c.list, nil
	}
	return &modelpb.ModelFileList{}, nil
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

func newManifestV2ContractFixture(t *testing.T) manifestcapnpv2.Manifest {
	t.Helper()

	_, seg := capnp.NewSingleSegmentMessage(nil)
	manifest, err := manifestcapnpv2.NewRootManifest(seg)
	require.NoError(t, err)
	manifest.SetContractRevision(manifestcapnpv2.CurrentContractRevision)
	deployment, err := manifest.NewDeployment()
	require.NoError(t, err)
	deployment.SetCompilationMode(manifestcapnpv2.CompilationMode_lpx)
	deployment.SetNumLpuNodes(1)
	program, err := deployment.NewProgram()
	require.NoError(t, err)
	program.SetBatchSize(8)
	runtimeIO, err := deployment.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(runtimeIOProtocolMultiEndpoint)
	runtimeIO.SetReserved1(2)
	runtimeIO.SetIoFpgaCount(4)
	runtimeIO.SetFanoutFactor(2)
	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	partitions, err := artifacts.NewPartitions(1)
	require.NoError(t, err)
	setManifestV2LPUArtifact(t, partitions.At(0), 0)

	return manifest
}

func setManifestV2LPUArtifact(t *testing.T, partition manifestcapnpv2.PartitionInfo, id uint32) {
	t.Helper()

	ref, err := partition.NewPartition()
	require.NoError(t, err)
	ref.SetPartitionId(id)
	ref.SetDeviceType(manifestcapnpv2.DeviceType_lpu)
	detail, err := partition.Detail().NewLpu()
	require.NoError(t, err)
	require.NoError(t, detail.SetPath("part-0"))
	require.NoError(t, detail.SetTopology(registryTestTopology))
	detail.SetNumChips(8)
	detail.SetDevicesPerNode(8)
}
