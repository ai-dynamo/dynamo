/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"testing"
	"time"

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

func TestAcquireBuildSnapshotPreservesReadFailures(t *testing.T) {
	t.Parallel()

	t.Log("Define missing and temporarily unavailable manifest reads")
	for _, test := range []struct {
		name    string
		readErr error
	}{
		{name: "not found", readErr: status.Error(codes.NotFound, "manifest missing")},
		{name: "unavailable", readErr: status.Error(codes.Unavailable, "temporary object-store failure")},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Read the requested manifest without a preceding inventory")
			client := &fakeModelServiceClient{
				fileStreams: []*fakeModelFileStream{{err: test.readErr}},
			}
			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)
			snapshot, err := registry.AcquireBuildSnapshot(t.Context(), "model/build")

			t.Log("Preserve the transport failure separately from invalid compiler input")
			require.ErrorIs(t, err, test.readErr)
			require.NotErrorIs(t, err, errInvalidBuildManifest)
			require.Nil(t, snapshot)
			require.Empty(t, client.listRequests)
			require.Len(t, client.filesRequests, 1)
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
				fileStreams: []*fakeModelFileStream{
					{chunks: modelFileChunks(gbuildManifestV2CapnpFile, string(payload))},
				},
			}
			registry, err := NewModelRegistry(testCase.registryURL, client)
			require.NoError(t, err)

			t.Log("Read and validate the manifest once, retaining the current registry locator")
			started := time.Now()
			build, err := normalizeRegistryFixtureBuild(testCase.ctx, registry, testCase.ref)
			require.NoError(t, err)
			require.Equal(t, "gs://bucket/registry/model/build", build.Path)
			require.Len(t, build.Partitions, 1)
			require.Equal(t, "part-0", build.Partitions[0].PartPath)
			require.Empty(t, client.listRequests)
			require.Len(t, client.filesRequests, 1)
			for _, request := range client.filesRequests {
				require.Equal(t, []string{gbuildManifestV2CapnpFile}, request.GetFileSelector().GetPaths())
			}

			t.Log("Bound the single metadata RPC and release its context")
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
