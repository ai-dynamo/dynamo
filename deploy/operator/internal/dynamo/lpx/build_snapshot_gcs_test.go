/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"os"
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
			require.Len(t, client.listRequests, 2)
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
			require.Len(t, client.listRequests, 2)
			require.Equal(t, []string{gbuildManifestV2CapnpFile}, client.listRequests[0].GetFileSelector().GetPaths())
			require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, client.listRequests[1].GetFileSelector().GetPaths())
			require.Len(t, client.filesRequests, 1)
			for _, request := range client.filesRequests {
				require.Equal(t, []string{gbuildManifestV2CapnpFile}, request.GetFileSelector().GetPaths())
			}

			t.Log("Bound discovery and the manifest read to one acquisition deadline and release their contexts")
			require.Len(t, client.metadataContexts, 3)
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

func TestAcquireBuildSnapshotRejectsMissingOrAmbiguousGCSContracts(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name    string
		files   []string
		wantErr error
		message string
	}{
		{name: "neither format", wantErr: os.ErrNotExist, message: "missing build contract"},
		{
			name: "both formats", files: []string{gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile},
			wantErr: errInvalidBuildManifest, message: "ambiguous",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Expose the selected build files through strict Model Express path selection")
			client := &fakeModelServiceClient{list: &modelpb.ModelFileList{}}
			for _, name := range test.files {
				client.list.Files = append(client.list.Files, &modelpb.ModelFileInfo{RelativePath: name})
			}
			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)

			t.Log("Discover both alternatives and reject the invalid build before reading any bytes")
			snapshot, err := registry.AcquireBuildSnapshot(t.Context(), "model/build")
			require.ErrorIs(t, err, test.wantErr)
			require.ErrorContains(t, err, test.message)
			require.Nil(t, snapshot)
			require.Len(t, client.listRequests, 2)
			require.Empty(t, client.filesRequests)
		})
	}
}

func TestAcquireBuildSnapshotPreservesGCSDiscoveryFailures(t *testing.T) {
	t.Parallel()

	for _, discoveryErr := range []error{
		status.Error(codes.Unavailable, "object store unavailable"),
		status.Error(codes.PermissionDenied, "access denied"),
		status.Error(codes.DeadlineExceeded, "discovery deadline exceeded"),
		status.Error(codes.Canceled, "discovery canceled"),
		errors.New("transport failure"),
	} {
		for _, path := range []string{gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile} {
			t.Run(discoveryErr.Error()+"/"+path, func(t *testing.T) {
				t.Log("Fail discovery of one alternative while the other format is available")
				client := &fakeModelServiceClient{
					list: &modelpb.ModelFileList{Files: []*modelpb.ModelFileInfo{
						{RelativePath: gbuildManifestV2CapnpFile},
						{RelativePath: gbuildDeploymentV1CapnpFile},
					}},
					listErrors: map[string]error{path: discoveryErr},
				}
				registry, err := NewModelRegistry("gs://bucket/registry", client)
				require.NoError(t, err)

				t.Log("Preserve the discovery error instead of accepting an incompletely discovered build")
				snapshot, err := registry.AcquireBuildSnapshot(t.Context(), "model/build")
				require.ErrorIs(t, err, discoveryErr)
				require.NotErrorIs(t, err, errInvalidBuildManifest)
				require.NotErrorIs(t, err, os.ErrNotExist)
				require.Nil(t, snapshot)
				require.Empty(t, client.filesRequests)
			})
		}
	}
}
