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
)

func manifestV2Payload(t *testing.T) []byte {
	t.Helper()
	payload, err := newManifestV2ContractFixture(t).Message().Marshal()
	require.NoError(t, err)
	return payload
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
			snapshot, err := registry.AcquireBuild(testCase.ctx, testCase.ref)
			require.NoError(t, err)
			build := snapshot
			require.Equal(t, "gs://bucket/registry/model/build", build.path)
			require.Len(t, build.partitions, 1)
			require.Equal(t, "part-0", build.partitions[0].partPath)
			require.Empty(t, client.listRequests)
			require.Len(t, client.filesRequests, 1)
			require.Equal(t, []string{gbuildManifestV2CapnpFile}, client.filesRequests[0].GetFileSelector().GetPaths())

			t.Log("Bound the single metadata RPC and release its context")
			require.Len(t, client.metadataContexts, 1)
			rpcCtx := client.metadataContexts[0]
			deadline, ok := rpcCtx.Deadline()
			require.True(t, ok)
			if parentDeadline, hasDeadline := testCase.ctx.Deadline(); hasDeadline {
				require.Equal(t, parentDeadline, deadline)
			} else {
				require.WithinRange(t, deadline, started.Add(30*time.Second), time.Now().Add(30*time.Second))
			}
			require.ErrorIs(t, rpcCtx.Err(), context.Canceled)
			require.NoError(t, testCase.ctx.Err())
		})
	}
}
