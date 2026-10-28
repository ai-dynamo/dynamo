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

func deploymentV1Payload(t *testing.T) []byte {
	t.Helper()
	payload, err := newDeploymentV1ContractFixture(t).Message().Marshal()
	require.NoError(t, err)
	return payload
}

func TestGCSModelRegistrySnapshotUsesDeploymentV1FromModelExpress(t *testing.T) {
	t.Parallel()

	t.Log("Serve a valid deployment contract through Model Express")
	payload := deploymentV1Payload(t)
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
			require.Len(t, build.partitions, 1)
			require.Len(t, build.lpuUnits, 1)
			require.Empty(t, client.listRequests)
			require.Len(t, client.filesRequests, 1)
			require.Equal(t, []string{gbuildDeploymentV1CapnpFile}, client.filesRequests[0].GetFileSelector().GetPaths())

			t.Log("Bound the contract read to the acquisition deadline and release its context")
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
		name          string
		files         []string
		payload       string
		missingAtRecv bool
		wantErr       error
		message       string
	}{
		{name: "no contract", wantErr: os.ErrNotExist, message: "missing build contract"},
		{name: "manifest.v2 only", files: []string{"manifest.v2.capnp.bin"}, payload: "compiler bytes", wantErr: os.ErrNotExist, message: "missing build contract"},
		{name: "missing at first receive", missingAtRecv: true, wantErr: os.ErrNotExist, message: "missing build contract"},
		{
			name: "empty contract exists", files: []string{gbuildDeploymentV1CapnpFile},
			wantErr: errInvalidBuildManifest, message: "parsing",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Expose candidate streams through strict Model Express path selection")
			client := &fakeModelServiceClient{files: map[string]*fakeModelFileStream{}}
			for _, name := range test.files {
				client.files[name] = &fakeModelFileStream{chunks: modelFileChunks(name, test.payload)}
			}
			if test.missingAtRecv {
				client.files[gbuildDeploymentV1CapnpFile] = &fakeModelFileStream{err: status.Error(codes.NotFound, "manifest missing")}
			}
			registry, err := NewModelRegistry("gs://bucket/registry", client)
			require.NoError(t, err)

			t.Log("Read only the deployment contract and distinguish its absence from malformed bytes")
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

			t.Log("Preserve the read error instead of reporting a missing or invalid contract")
			snapshot, err := registry.AcquireBuild(t.Context(), "model/build")
			require.ErrorIs(t, err, readErr)
			require.NotErrorIs(t, err, errInvalidBuildManifest)
			require.NotErrorIs(t, err, os.ErrNotExist)
			require.Nil(t, snapshot)
			require.Empty(t, client.listRequests)
		})
	}
}

func TestAcquireBuildDoesNotTreatInterruptedStreamAsAbsent(t *testing.T) {
	t.Parallel()

	t.Log("Return NotFound after receiving part of an existing contract")
	readErr := status.Error(codes.NotFound, "stream interrupted")
	client := &fakeModelServiceClient{files: map[string]*fakeModelFileStream{
		gbuildDeploymentV1CapnpFile: {
			chunks: []*modelpb.FileChunk{{RelativePath: gbuildDeploymentV1CapnpFile, Data: []byte("partial")}},
			err:    readErr,
		},
	}}
	registry, err := NewModelRegistry("gs://bucket/registry", client)
	require.NoError(t, err)

	t.Log("Preserve the interrupted read instead of reporting a missing contract")
	build, err := registry.AcquireBuild(t.Context(), "model/build")
	require.Nil(t, build)
	require.ErrorIs(t, err, readErr)
	require.NotErrorIs(t, err, os.ErrNotExist)
	require.Len(t, client.filesRequests, 1)
}
