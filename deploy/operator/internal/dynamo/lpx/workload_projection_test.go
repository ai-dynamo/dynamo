/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

func TestHybridSettingsRejectRuntimeOverrides(t *testing.T) {
	for _, fixture := range []testV3CapnpFixture{newV2CompilerFixture(), newV3CompilerFixture()} {
		t.Log("Acquire a hybrid build for each supported family")
		fixture.compilationMode = manifestcapnp.CompilationMode_lpx
		snapshot := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)))

		t.Log("Allow empty settings and XT scheduler controls, but not runtime overrides")
		for _, test := range []struct {
			settings       string
			xtOnly, reject bool
		}{
			{settings: ""},
			{settings: "null"},
			{settings: "{}"},
			{settings: `{"prop_sync":true}`, xtOnly: true},
			{settings: `{"prop_sync_multiple_load_sets":true}`, xtOnly: true},
			{settings: `{"batch_size":4}`, reject: true},
			{settings: `{"setup":{"agent_setup_timeout":"600s"}}`, reject: true},
			{settings: `{"cpu_embeddings":true}`, reject: true},
			{settings: `{"batch_folding":false,"num_batch_split_divisions":1}`, reject: true},
			{settings: `{"prop_sync":false,"batch_size":4}`, reject: true},
		} {
			t.Run(string(snapshot.build.Family)+"/"+test.settings, func(t *testing.T) {
				t.Log("Reject unsupported settings before producing a workload projection")
				_, err := appendModelProjections(nil, ModelProjectionInput{
					Pipeline: PipelineLPX, Models: []string{"default"}, BuildSnapshot: snapshot,
					ModelSettings: json.RawMessage(test.settings),
				})
				if test.reject || (test.xtOnly && snapshot.build.Family != BuildFamilyXT) {
					require.ErrorContains(t, err, "do not support runtime settings overrides")
				} else {
					require.NoError(t, err)
				}
			})
		}
	}
}

func TestWorkloadDigestIsIndependentOfBuildLocator(t *testing.T) {
	t.Parallel()

	t.Log("Acquire equal compiler contents under distinct local build paths")
	first := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	second := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	require.NotEqual(t, first.build.Path, second.build.Path)
	require.Equal(t, first.contentID, second.contentID)

	t.Log("Project either immutable snapshot through the same intent")
	firstProjection := projectTestBuild(t, first, PipelineSingle, `{"prop_sync":true}`)
	secondProjection := projectTestBuild(t, second, PipelineSingle, `{"prop_sync":true}`)

	t.Log("Publish the canonical compiler snapshot identity independently of build locator")
	require.Equal(t, "scheduling.lpu.nvidia.com/compiler-snapshot-digest", lpxv1alpha1.CompilerSnapshotDigestAnnotation)
	require.Equal(t, first.contentID, firstProjection.CompilerSnapshotDigest())
	require.Equal(t, second.contentID, secondProjection.CompilerSnapshotDigest())

	t.Log("Produce the same workload projection digest independent of build locator")
	require.Equal(t, firstProjection.Digest(), secondProjection.Digest())

	t.Log("Keep compiler identity separate from downstream workload projection identity")
	specDecodeProjection := projectTestBuild(t, first, PipelineSpecDecode, `{"prop_sync":true}`)
	require.Equal(t, firstProjection.CompilerSnapshotDigest(), specDecodeProjection.CompilerSnapshotDigest())
	require.NotEqual(t, firstProjection.Digest(), specDecodeProjection.Digest())
}

func TestCanonicalModelSettingsRetainRaw(t *testing.T) {
	t.Log("Decode canonical model settings while retaining their exact raw bytes")
	raw, object := canonicalModelSettings(json.RawMessage(
		`{"exponent":1e3,"float":1.0,"integer":8192,"nested":{"array":[128001,1.5,9223372036854775808]}}`,
	))
	require.Equal(t, json.RawMessage(`{"exponent":1e3,"float":1.0,"integer":8192,"nested":{"array":[128001,1.5,9223372036854775808]}}`), raw)
	require.Equal(t, json.Number("8192"), object["integer"])

	t.Log("Normalize explicit null settings to a present empty object")
	raw, object = canonicalModelSettings(json.RawMessage("null"))
	require.Nil(t, raw)
	require.NotNil(t, object)
	require.Empty(t, object)
}

func TestProjectModelSettingsPlacementBoundary(t *testing.T) {
	t.Log("Prepare both physical families with conductor-owned runtime settings")
	builds := []NormalizedBuildSnapshot{
		normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t))),
		normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV3CompilerFixture(t))),
	}
	for _, build := range builds {
		for _, pipeline := range []Pipeline{PipelineSingle, PipelineSpecDecode} {
			for _, test := range []struct {
				settings, wantErr string
				xtOnly            bool
			}{
				{settings: `{"sequence_length":4096}`, wantErr: "model settings.sequence_length is a runtime setting; configure it in the conductor podTemplate"},
				{settings: `{"setup":{"agent_setup_timeout":"30s"}}`, wantErr: "model settings.setup is a runtime setting"},
				{settings: `{"z":null,"a":true}`, wantErr: "model settings.a is a runtime setting"},
				{settings: `{"cpu_embeddings":"false"}`, wantErr: "model settings.cpu_embeddings must be a boolean"},
				{settings: `{"prop_sync":null}`, wantErr: "model settings.prop_sync must be a boolean"},
				{settings: `{"prop_sync_multiple_load_sets":1}`, wantErr: "model settings.prop_sync_multiple_load_sets must be a boolean"},
				{settings: `{"cpu_embeddings":true}`, xtOnly: true},
				{settings: `{"cpu_embeddings":false}`, xtOnly: true},
				{settings: `{"prop_sync_multiple_load_sets":true}`, xtOnly: true},
				{settings: `{"prop_sync_multiple_load_sets":false}`, xtOnly: true},
				{settings: `{"prop_sync":false}`},
			} {
				t.Run(string(build.build.Family)+"/"+string(pipeline)+"/"+test.settings, func(t *testing.T) {
					t.Log("Accept only placement controls consumed by the selected physical family")
					_, err := appendModelProjections(nil, ModelProjectionInput{
						Pipeline: pipeline, Models: []string{"default"}, BuildSnapshot: build,
						ModelSettings: json.RawMessage(test.settings),
					})
					if test.xtOnly && build.build.Family == BuildFamilyHX {
						require.ErrorContains(t, err, "is not used for HX placement; configure runtime options in the conductor podTemplate")
					} else if test.wantErr != "" {
						require.ErrorContains(t, err, test.wantErr)
					} else {
						require.NoError(t, err)
					}
				})
			}
		}
	}
}
