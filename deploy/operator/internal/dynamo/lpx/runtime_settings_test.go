/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"maps"
	"slices"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
)

func TestResolveBuildSettingsSelectedPropSyncChainWithCPUEmbeddings(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name                                  string
		cpuSupported, standalone, cpuDisabled bool
		firstPartition                        int
	}{
		{"standalone CPU embeddings", true, true, false, 1},
		{"CPU embeddings unsupported", false, true, false, 0},
		{"partition zero is runnable", true, false, false, 0},
		{"explicitly disabled CPU embeddings", true, true, true, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Construct a selected chain with independent embedding placement requirements")
			build := selectedPropSyncBuildWithTopologies(t,
				[]string{registryTestTopology, registryTestTopology, registryTestTopology}, [][]int{{0, 1, 2}})
			build.SupportsCPUEmbeddings, build.StandaloneTokenEmbeddings = test.cpuSupported, test.standalone
			settings := map[string]any{}
			if test.cpuDisabled {
				settings["cpu_embeddings"] = false
			}
			partitions := slices.Clone(build.Partitions)
			configured := *build

			t.Log("Resolve the selected chain and omit partition zero only when both flags permit it")
			require.NoError(t, resolveBuildSettings(&configured, settings))
			require.Equal(t, partitions[test.firstPartition:], configured.Partitions)
			require.Empty(t, configured.SelectedPropSyncChains)
			require.Equal(t, settings, configured.runtimeSettings)

			t.Log("Keep the original partition IDs, paths and selected chain unchanged")
			require.Equal(t, partitions, build.Partitions)
			require.Equal(t, [][]int{{0, 1, 2}}, build.SelectedPropSyncChains)
		})
	}
}

func TestResolveBuildSettingsRejectsInvalidSelectedPropSyncChain(t *testing.T) {
	t.Parallel()

	t.Log("Define malformed or incompatible selected prop-sync chains")
	tests := []struct {
		name    string
		chains  [][]int
		wantErr string
	}{
		{
			name:    "multiple chains",
			chains:  [][]int{{0, 1}, {2, 3}},
			wantErr: "LPU-only runtime requires exactly one selected prop-sync chain, got 2",
		},
		{
			name:    "missing partition",
			chains:  [][]int{{3, 4}},
			wantErr: "references missing partition id 4",
		},
		{
			name:    "duplicate partition",
			chains:  [][]int{{1, 2, 1}},
			wantErr: "contains duplicate partition id 1",
		},
		{
			name:    "noncontiguous partitions",
			chains:  [][]int{{1, 3}},
			wantErr: "is not contiguous at partition id 3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Configure malformed or incompatible LPU-only selected-chain metadata")
			build := selectedPropSyncBuild(t, tt.chains)
			build.runtimeSettings = map[string]any{"cpu_embeddings": false}
			configured := *build

			t.Log("Reject the build without consuming its selected-chain marker")
			err := resolveBuildSettings(&configured, nil)
			require.ErrorContains(t, err, tt.wantErr)
			require.Equal(t, tt.chains, configured.SelectedPropSyncChains)
		})
	}
}

func TestResolveBuildSettingsDeepMergesDefaultsAndOverrides(t *testing.T) {
	t.Parallel()

	t.Log("Construct paired runtime defaults and model overrides")
	defaults, overrides := map[string]any{}, map[string]any{}
	for name, values := range map[string][2]any{
		"batch_size":                   {int64(1), int64(2)},
		"sequence_length":              {int64(131072), int64(4096)},
		"input_size":                   {int64(1), int64(2)},
		"num_kv_caches":                {int64(1), int64(2)},
		"cpu_embeddings":               {true, false},
		"dkvc":                         {true, false},
		"num_dkvc_blocks":              {int64(256), int64(64)},
		"prop_sync":                    {true, false},
		"prop_sync_back_pressure":      {false, true},
		"prop_sync_multiple_load_sets": {false, true},
		"has_init_tx":                  {false, true},
		"extended_topology":            {true, false},
		"c2c_error_scan":               {false, true},
		"num_batch_split_divisions":    {int64(2), int64(0)},
		"batch_folding":                {true, false},
		"num_layers":                   {int64(24), int64(25)},
		"vocab_size":                   {int64(201088), int64(201089)},
		"tokenizer_path":               {"/build/tokenizer", "/dgd/tokenizer"},
		"stop_tokens":                  {[]uint32{200002, 199999, 200012}, []any{int64(42)}},
		"swa": {
			map[string]any{
				"chunked": false, "num_swa_dkvc_blocks": int64(1), "swa_ctx_len": int64(128),
				"swa_num_users": int64(8), "swa_padding_len": int64(0),
			},
			map[string]any{
				"chunked": true, "num_swa_dkvc_blocks": int64(2), "swa_ctx_len": int64(256),
				"swa_num_users": int64(4), "swa_padding_len": int64(16),
			},
		},
	} {
		defaults[name], overrides[name] = values[0], values[1]
	}
	overrides["custom_setting"] = "preserved"
	overrides["custom_array"] = []any{map[string]any{"nested": []any{int64(1)}}}
	build := &Build{CompilationMode: BuildCompilationModeLPUOnly, runtimeSettings: defaults}

	t.Log("Resolve overrides and verify every explicit model value wins")
	require.NoError(t, resolveBuildSettings(build, overrides))
	expectedSettings := maps.Clone(overrides)
	delete(expectedSettings, "batch_folding")
	delete(expectedSettings, "num_batch_split_divisions")
	require.Equal(t, expectedSettings, build.runtimeSettings, "every supported explicit DGD value must win")

	t.Log("Verify merge ownership and copy-on-write behavior")
	overrideTokens := overrides["stop_tokens"].([]any)
	configuredTokens := build.runtimeSettings["stop_tokens"].([]any)
	require.Same(t, &overrideTokens[0], &configuredTokens[0], "caller-owned override slices are consumed")
	empty, err := mergeRuntimeSettingOverride(nil, []any(nil), "settings.empty")
	require.NoError(t, err)
	require.NotNil(t, empty.([]any), "typed-nil arrays retain normalized empty-slice shape")
	build.runtimeSettings["custom_setting"] = "changed"
	require.Equal(t, "preserved", overrides["custom_setting"], "the settings map header remains independent")
	build.runtimeSettings["swa"].(map[string]any)["swa_ctx_len"] = int64(512)
	require.Equal(t, int64(128), defaults["swa"].(map[string]any)["swa_ctx_len"])
	require.Equal(t, uint32(200002), defaults["stop_tokens"].([]uint32)[0])
	require.Equal(t, int64(256), overrides["swa"].(map[string]any)["swa_ctx_len"], "overridden maps remain copy-on-write")

	t.Log("Complete a partial nested override from immutable defaults")
	sharedDefault := []uint32{1}
	partialDefaults := map[string]any{
		"shared": sharedDefault,
		"swa": map[string]any{
			"chunked":             false,
			"num_swa_dkvc_blocks": int64(1),
			"swa_ctx_len":         int64(128),
			"swa_num_users":       int64(8),
			"swa_padding_len":     int64(0),
		},
	}
	partial := &Build{CompilationMode: BuildCompilationModeLPUOnly, runtimeSettings: partialDefaults}
	require.NoError(t, resolveBuildSettings(partial, map[string]any{"swa": map[string]any{"chunked": true}}))
	require.Equal(t, map[string]any{
		"chunked":             true,
		"num_swa_dkvc_blocks": int64(1),
		"swa_ctx_len":         int64(128),
		"swa_num_users":       int64(8),
		"swa_padding_len":     int64(0),
	}, partial.runtimeSettings["swa"])

	t.Log("Verify untouched defaults remain shared while nested maps remain isolated")
	configuredDefault := partial.runtimeSettings["shared"].([]uint32)
	require.Same(t, &sharedDefault[0], &configuredDefault[0], "untouched defaults remain immutable shared evidence")
	partial.runtimeSettings["swa"].(map[string]any)["swa_ctx_len"] = int64(256)
	require.Equal(t, int64(128), partialDefaults["swa"].(map[string]any)["swa_ctx_len"])
}

func TestProjectModelLegacyNovaSettings(t *testing.T) {
	t.Parallel()

	t.Log("Capture the Nova and hybrid settings paths before runtime-specific overrides")
	singleIntent := ModelProjectionInput{
		Models: []string{"default"}, Pipeline: PipelineSingle,
		BuildSnapshot: normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV3CompilerFixture(t))),
	}
	hybridFixture := newV2CompilerFixture()
	hybridFixture.compilationMode = manifestcapnp.CompilationMode_lpx
	hybridIntent := ModelProjectionInput{
		Models: []string{"default"}, Pipeline: PipelineLPX,
		BuildSnapshot: normalizeTestSnapshot(t, acquireTestSnapshot(t, writeCompilerFixture(t, hybridFixture))),
	}
	singleBaselineBatch, err := appendModelProjections(nil, singleIntent)
	require.NoError(t, err)
	singleBaseline := singleBaselineBatch[0]
	hybridBaselineBatch, err := appendModelProjections(nil, hybridIntent)
	require.NoError(t, err)
	hybridBaseline := hybridBaselineBatch[0]

	t.Log("Define accepted legacy defaults and unsupported values")
	tests := []struct {
		settings, wantErr string
		hybrid            bool
	}{
		{`{"batch_folding":false}`, "", false},
		{`{"num_batch_split_divisions":0}`, "", false},
		{`{"num_batch_split_divisions":1}`, "", false},
		{`{"num_batch_split_divisions":1.0}`, "", false},
		{`{"num_batch_split_divisions":10e-1}`, "", false},
		{`{"num_batch_split_divisions":-0.0}`, "", false},
		{`{"batch_folding":true}`, `iop.batch_folding=true is not supported`, false},
		{`{"batch_folding":"false"}`, "iop.batch_folding must be a boolean", false},
		{`{"num_batch_split_divisions":-1}`, `iop.num_batch_split_divisions=-1 is not supported`, false},
		{`{"num_batch_split_divisions":2}`, `iop.num_batch_split_divisions=2 is not supported`, false},
		{`{"num_batch_split_divisions":-1.0}`, `iop.num_batch_split_divisions=-1.0 is not supported`, false},
		{`{"num_batch_split_divisions":0.5}`, "iop.num_batch_split_divisions must be an integer", false},
		{`{"num_batch_split_divisions":0.99999999999999999999}`, "iop.num_batch_split_divisions must be an integer", false},
		{`{"num_batch_split_divisions":1.00000000000000000001}`, "iop.num_batch_split_divisions must be an integer", false},
		{`{"num_batch_split_divisions":1e-1000}`, "iop.num_batch_split_divisions must be an integer", false},
		{`{"num_batch_split_divisions":"1"}`, "iop.num_batch_split_divisions must be an integer", false},
		{`{"batch_folding":false,"num_batch_split_divisions":1.0}`, "", true},
		{`{"batch_folding":true}`, "", true},
		{`{"num_batch_split_divisions":0.99999999999999999999}`, "", true},
		{`{"cpu_embeddings":"runtime-owned"}`, "", true},
	}

	for _, test := range tests {
		intent, baseline := singleIntent, singleBaseline
		if test.hybrid {
			intent, baseline = hybridIntent, hybridBaseline
		}
		t.Run(string(intent.Pipeline)+"/"+test.settings, func(t *testing.T) {
			t.Log("Validate model-provided JSON settings before rendering")
			intent.ModelSettings = []byte(test.settings)
			projectionBatch, err := appendModelProjections(nil, intent)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			projection := projectionBatch[0]

			t.Log("Preserve physical partitions, runtime partitions, and prop-sync connectors")
			require.Equal(t, baseline.partitions, projection.partitions)
			require.Equal(t, baseline.configuredBuild.Partitions, projection.configuredBuild.Partitions)
			require.Equal(t, baseline.connectors, projection.connectors)

			t.Log("Omit legacy Nova settings from runtime output without validating unused hybrid values")
			if !test.hybrid {
				require.NotContains(t, projection.configuredBuild.runtimeSettings, "batch_folding")
				require.NotContains(t, projection.configuredBuild.runtimeSettings, "num_batch_split_divisions")
			}
			configMap, err := renderLPUConfigMap("test", "test-dgd", "/models", []*ModelProjection{projection})
			require.NoError(t, err)
			require.NotContains(t, configMap.Data["model_config.toml"], "batch_folding")
			require.NotContains(t, configMap.Data["model_config.toml"], "num_batch_split_divisions")
		})
	}
}

func TestResolveBuildSettingsRejectsNestedNullOverride(t *testing.T) {
	t.Parallel()

	t.Log("Reject a null value in a nested known runtime setting")
	build := &Build{runtimeSettings: map[string]any{
		"swa": map[string]any{"chunked": false},
	}}

	err := resolveBuildSettings(build, map[string]any{
		"swa": map[string]any{"chunked": nil},
	})
	require.ErrorContains(t, err, "settings.swa.chunked must not be null")

	t.Log("Reject a null value nested inside a custom array")
	build = &Build{}
	err = resolveBuildSettings(build, map[string]any{
		"custom": []any{map[string]any{"value": nil}},
	})
	require.ErrorContains(t, err, "settings.custom[0].value must not be null")

	t.Log("Report deterministic field order when multiple settings are null")
	build = &Build{}
	err = resolveBuildSettings(build, map[string]any{"z": nil, "a": nil})
	require.ErrorContains(t, err, "settings.a must not be null")
}

func selectedPropSyncBuild(t *testing.T, chains [][]int) *Build {
	t.Helper()

	topology := registryTestTopology
	return selectedPropSyncBuildWithTopologies(t, []string{topology, topology, topology, topology}, chains)
}

func selectedPropSyncBuildWithTopologies(t *testing.T, topologies []string, chains [][]int) *Build {
	t.Helper()

	partitions := make([]BuildPartition, len(topologies))
	for i, raw := range topologies {
		topology, err := parse(raw)
		require.NoError(t, err)
		partitions[i] = BuildPartition{SourcePartitionID: i, PartPath: fmt.Sprintf("part-%d", i), Topology: topology}
	}
	return &Build{
		Path:                      "file:///path/to/build-id",
		CompilationMode:           BuildCompilationModeLPUOnly,
		Partitions:                partitions,
		StandaloneTokenEmbeddings: true,
		SupportsCPUEmbeddings:     true,
		SelectedPropSyncChains:    chains,
	}
}
