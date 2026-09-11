/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"testing"
	"time"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func manifestV2Payload(t *testing.T) []byte {
	t.Helper()
	payload, err := newManifestV2ContractFixture(t).Message().Marshal()
	require.NoError(t, err)
	return payload
}

func TestAcquireBuildSnapshotFencesCompilerMetadata(t *testing.T) {
	t.Parallel()

	t.Log("Create two different revision-2 compiler metadata reads for one build")
	const buildID = "model/build"
	client := &fakeModelServiceClient{
		list: &modelpb.ModelFileList{Files: []*modelpb.ModelFileInfo{
			{RelativePath: gbuildManifestV2CapnpFile},
		}},
		fileStreams: []*fakeModelFileStream{
			{chunks: modelFileChunks(gbuildManifestV2CapnpFile, "first")},
			{chunks: modelFileChunks(gbuildManifestV2CapnpFile, "second")},
		},
	}
	registry, err := NewModelRegistry("gs://bucket/registry", client)
	require.NoError(t, err)

	t.Log("Reject compiler metadata that changes during acquisition")
	_, err = registry.AcquireBuildSnapshot(t.Context(), buildID)
	require.ErrorIs(t, err, ErrBuildSnapshotInconsistent)
	require.ErrorContains(t, err, "compiler metadata manifest.v2.capnp.bin changed while acquiring")
}

func TestAcquireBuildSnapshotClassifiesSelectedMemberDisappearance(t *testing.T) {
	t.Parallel()

	t.Log("Define definitive and transient failures for both manifest reads")
	payload := manifestV2Payload(t)
	for _, testCase := range []struct {
		name        string
		failingRead int
		readErr     error
	}{
		{name: "gRPC NotFound on first read", failingRead: 1, readErr: status.Error(codes.NotFound, "object disappeared")},
		{name: "gRPC NotFound on duplicate read", failingRead: 2, readErr: status.Error(codes.NotFound, "object disappeared")},
		{name: "transient gRPC failure on first read", failingRead: 1, readErr: status.Error(codes.Unavailable, "temporary object-store failure")},
		{name: "transient gRPC failure on duplicate read", failingRead: 2, readErr: status.Error(codes.Unavailable, "temporary object-store failure")},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Log("Construct the selected Model Express read sequence")
			streams := []*fakeModelFileStream{{err: testCase.readErr}}
			if testCase.failingRead == 2 {
				streams = append([]*fakeModelFileStream{
					{chunks: modelFileChunks(gbuildManifestV2CapnpFile, string(payload))},
				}, streams...)
			}
			client := &fakeModelServiceClient{
				list:        &modelpb.ModelFileList{Files: []*modelpb.ModelFileInfo{{RelativePath: gbuildManifestV2CapnpFile}}},
				fileStreams: streams,
			}
			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)

			t.Log("Preserve the transport error while classifying definitive absence as inconsistency")
			_, err = registry.AcquireBuildSnapshot(t.Context(), "model/build")
			require.Error(t, err)
			require.Equal(t, status.Code(testCase.readErr) == codes.NotFound, errors.Is(err, ErrBuildSnapshotInconsistent))
			require.ErrorIs(t, err, testCase.readErr)
		})
	}
}

func TestGCSModelRegistrySnapshotUsesManifestV2FromModelExpress(t *testing.T) {
	t.Parallel()

	t.Log("Serve a valid revision-2 compiler manifest through Model Express")
	payload := manifestV2Payload(t)
	shortCtx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	for _, testCase := range []struct {
		ref string
		ctx context.Context
	}{
		{ref: "model/build", ctx: t.Context()},
		{ref: "gs://bucket/registry/model/build", ctx: t.Context()},
		{ref: "model/build", ctx: shortCtx},
	} {
		t.Run(testCase.ref, func(t *testing.T) {
			t.Log("Create an independent client and registry for this reference form")
			client := &fakeModelServiceClient{
				fileStreams: []*fakeModelFileStream{
					{chunks: modelFileChunks(gbuildManifestV2CapnpFile, string(payload))},
					{chunks: modelFileChunks(gbuildManifestV2CapnpFile, string(payload))},
				},
				list: &modelpb.ModelFileList{Files: []*modelpb.ModelFileInfo{{RelativePath: gbuildManifestV2CapnpFile}}},
			}
			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)

			t.Log("Normalize the manifest and retain the current registry locator")
			started := time.Now()
			build, err := normalizeRegistryFixtureBuild(testCase.ctx, registry, testCase.ref)
			require.NoError(t, err)
			require.Equal(t, "gs://bucket/registry/model/build", build.Path)
			require.Equal(t, 8, build.BatchSize)
			require.Len(t, build.Partitions, 1)
			require.Equal(t, "part-0", build.Partitions[0].PartPath)
			require.Len(t, client.listRequests, 2)
			require.Len(t, client.filesRequests, 2)
			for _, request := range client.filesRequests {
				require.Equal(t, []string{gbuildManifestV2CapnpFile}, request.GetFileSelector().GetPaths())
			}

			t.Log("Share a finite deadline across all metadata RPCs and release only their contexts")
			require.Len(t, client.metadataContexts, 4)
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
