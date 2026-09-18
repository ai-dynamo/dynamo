/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestRenderRuntimeConfigMapSizeLimit(t *testing.T) {
	t.Log("Accept exactly 1 MiB of UTF-8 bytes across data values, excluding keys")
	data := map[string]string{
		"text":   strings.Repeat("é", corev1.MaxSecretSize/2-1),
		"suffix": "é",
	}
	_, err := renderRuntimeConfigMap("test-namespace", "test-lpu", data)
	require.NoError(t, err)

	t.Log("Reject one extra byte with a named error and no usable ConfigMap")
	data["suffix"] += "x"
	configMap, err := renderRuntimeConfigMap("test-namespace", "test-lpu", data)
	require.Nil(t, configMap)
	require.ErrorContains(t, err, `rendered LPX ConfigMap "test-lpu-`)
	require.ErrorContains(t, err, "data is 1048577 bytes; maximum is 1048576")
}

func TestLPXRuntimeConfigNamesMatchPodIdentity(t *testing.T) {
	for _, root := range []string{"short", strings.Repeat("a", MaxPodCliqueSetNameLength), strings.Repeat("a", validation.LabelValueMaxLength)} {
		t.Run(root, func(t *testing.T) {
			t.Log("Render immutable runtime tables for PCS and Pod-label identity bounds")
			data := map[string]string{"runtime": "config"}
			lpu, err := renderRuntimeConfigMap("test-namespace", root+"-lpu", data)
			require.NoError(t, err)
			decode, err := renderRuntimeConfigMap("test-namespace", root+"-decode", data)
			require.NoError(t, err)

			t.Log("Resolve the same LPU table from Pod identity and preserve each role suffix")
			require.Equal(t, LPUConfigMapName(root, LPUConfigMapHash(lpu)), lpu.Name)
			require.Equal(t, root+"-lpu-"+LPUConfigMapHash(lpu)[:16], lpu.Name)
			require.Equal(t, root+"-decode-"+LPUConfigMapHash(decode)[:16], decode.Name)
			require.Empty(t, validation.IsDNS1123Subdomain(lpu.Name))
			require.Empty(t, validation.IsDNS1123Subdomain(decode.Name))
		})
	}
}

func TestLPURuntimeBuildRef(t *testing.T) {
	t.Parallel()

	t.Log("Define runtime build-reference selection contracts")
	tests := []struct {
		name     string
		snapshot string
		runtime  string
		want     string
	}{
		{name: "relative runtime", snapshot: "file:///snapshot", runtime: "model-build", want: "file:///models/model-build"},
		{name: "cleaned runtime", snapshot: "file:///snapshot", runtime: " a/../b ", want: "file:///models/b"},
		{name: "GCS snapshot", snapshot: "gs://bucket/snapshot", runtime: "model-build", want: "gs://bucket/snapshot"},
		{name: "malformed snapshot", snapshot: "%", runtime: "model-build", want: "%"},
		{name: "empty runtime", snapshot: "file:///snapshot", want: "file:///snapshot"},
		{name: "malformed runtime", snapshot: "file:///snapshot", runtime: "%", want: "file:///snapshot"},
		{name: "runtime URL", snapshot: "file:///snapshot", runtime: "gs://bucket/build", want: "file:///snapshot"},
		{name: "absolute runtime", snapshot: "file:///snapshot", runtime: "/model-build", want: "file:///snapshot"},
		{name: "dot runtime", snapshot: "file:///snapshot", runtime: ".", want: "file:///snapshot"},
		{name: "parent runtime", snapshot: "file:///snapshot", runtime: "..", want: "file:///snapshot"},
		{name: "escaping runtime", snapshot: "file:///snapshot", runtime: "a/../../b", want: "file:///snapshot"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Select the runtime build reference")
			got := lpuRuntimeBuildRef(&ModelProjection{
				runtimeBuildRef: test.runtime,
				configuredBuild: Build{Path: test.snapshot},
			}, "/models")

			t.Log("Preserve valid remapping and every fallback byte exactly")
			require.Equal(t, test.want, got)
		})
	}
}

func TestRenderLPUConfigMapPreservesV2HybridGasDir(t *testing.T) {
	t.Parallel()

	t.Log("Construct a V2 hybrid projection with an image-owned runtime build")
	projection := &ModelProjection{
		model:           "default",
		pipeline:        PipelineLPX,
		runtimeBuildRef: "model-build",
		configuredBuild: Build{
			Path:            "file:///snapshot-build",
			Family:          BuildFamilyXT,
			runtimeSettings: map[string]any{},
		},
	}

	t.Log("Render the hybrid ConfigMap from the selected pipeline")
	configMap, err := renderLPUConfigMap(
		"test-namespace",
		"test-dgd",
		"/models",
		[]*ModelProjection{projection},
	)

	t.Log("Keep gas_dir rooted at the selected image-owned build")
	require.NoError(t, err)
	require.NotContains(t, configMap.Data, "model_config.toml")
	require.NotContains(t, configMap.Data, "datacenter.toml")
	require.Equal(t, "/models/model-build", configMap.Data["gas_dir"])
	require.True(t, *configMap.Immutable)
	require.Equal(t, LPUConfigMapName("test-dgd", LPUConfigMapHash(configMap)), configMap.Name)

	t.Log("Ignore model settings that are not consumed by the direct runtime")
	projection.configuredBuild.runtimeSettings["oversized"] = strings.Repeat("x", corev1.MaxSecretSize)
	projection.configuredBuild.runtimeSettings["scheduler"] = "not-a-Nova-scheduler"
	unchanged, err := renderLPUConfigMap("test-namespace", "test-dgd", "/models", []*ModelProjection{projection})
	require.NoError(t, err)
	require.Equal(t, configMap, unchanged)

	t.Log("Render the preserved runtime with an invalid snapshot build reference")
	projection.configuredBuild.Path = "relative-build"
	_, err = renderLPUConfigMap(
		"test-namespace",
		"test-dgd",
		"/models",
		[]*ModelProjection{projection},
	)

	t.Log("Return the model-path error before projecting gas_dir")
	require.ErrorContains(t, err, "resolve gas_dir: parse build path \"relative-build\": ref \"relative-build\" must be an absolute path or URL")
}

func TestResolvedPartitionDataOmitsXTModelColumnsBeforeMaterialization(t *testing.T) {
	t.Parallel()

	t.Log("Construct an XT Single projection with two physical runtime partitions")
	projection := &ModelProjection{
		model:    "default",
		pipeline: PipelineSingle,
		configuredBuild: Build{
			Family: BuildFamilyXT,
			Partitions: []BuildPartition{
				{SourcePartitionID: 7, PartPath: "part-7", Topology: Topology{ChipCount: 16, Raw: "topology-7"}},
				{SourcePartitionID: 9, PartPath: "part-9", Topology: Topology{ChipCount: 8, Raw: "topology-9"}},
			},
		},
	}

	t.Log("Project only the five columns consumed by the XT Single runtime")
	data := resolvedPartitionData([]*ModelProjection{projection})

	t.Log("Verify omitted model columns never enter the final map and retained bytes remain exact")
	require.Equal(t, map[string]string{
		"nodes_per_partition":    "2\n1",
		"partition_ids":          "7\n9",
		"partition_node_offsets": "0\n2",
		"partition_paths":        "part-7\npart-9",
		"topologies":             "topology-7\ntopology-9",
	}, data)
}

func TestV3SingleConfigUsesManifestTopologyAndArbitraryModelSettings(t *testing.T) {
	t.Parallel()

	t.Log("Project the V3 manifest with arbitrary deployment-time model settings")
	snapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: PipelineSingle, Models: []string{"default"}, RuntimeBuildRef: "model-build", BuildSnapshot: normalizeTestSnapshot(t, snapshot),
		ModelSettings: json.RawMessage(`{
			"batch_size":4,
			"custom_runtime_knob":"enabled",
			"scheduler":{"max_inflight_tasks":3},
			"tokenizer_path":"/nfs/models/gemma4-31b",
			"stop_tokens":[1]
		}`),
	})
	require.NoError(t, err)
	projection := projectionBatch[0]

	t.Log("Render the complete LPU ConfigMap")
	configMap, err := renderLPUConfigMap(
		"test",
		"test-dgd",
		"/models",
		[]*ModelProjection{projection},
	)
	require.NoError(t, err)

	t.Log("Verify the generated files need no manual topology or model-setting patches")
	require.Equal(t, "part-1", configMap.Data["partition_paths"])
	require.Equal(t, v3OpaqueTopology, configMap.Data["topologies"])
	require.NotContains(t, configMap.Data["model_config.toml"], "sequence_length")
	require.NotContains(t, configMap.Data, "datacenter.toml")
	require.Contains(t, configMap.Data["model_config.toml"], "model_path = '/models/model-build'")
	require.Contains(t, configMap.Data["model_config.toml"], "batch_size = 4")
	require.Contains(t, configMap.Data["model_config.toml"], "custom_runtime_knob = 'enabled'")
	require.Contains(t, configMap.Data["model_config.toml"], "max_inflight_tasks = 3")
	require.Contains(t, configMap.Data["model_config.toml"], "tokenizer_path = '/nfs/models/gemma4-31b'")
	require.Contains(t, configMap.Data["model_config.toml"], "stop_tokens = [1]")
}

func TestNestedLPUModelConfigPreservesExplicitOverrides(t *testing.T) {
	t.Log("Construct a model with deployment overrides and a remapped runtime path")
	projection := &ModelProjection{
		model: "default", pipeline: PipelineSingle, runtimeBuildRef: "model-build",
		configuredBuild: Build{
			Path: "file:///snapshot-build", Family: BuildFamilyHX,
			runtimeSettings: map[string]any{"tokenizer_path": "/custom/tokenizer", "batch_size": 4},
		},
	}

	t.Log("Leave manifest defaults to the runtime while retaining authored overrides")
	config, err := nestedLPUModelConfig(projection, "/models")
	require.NoError(t, err)
	require.Equal(t, map[string]any{
		"model_path": "/models/model-build", "tokenizer_path": "/custom/tokenizer", "batch_size": 4,
	}, config["iop"])
	require.Equal(t, map[string]any{"resolved_partitions_dir": "/configs"}, config["setup"])
	require.NotContains(t, config, "scheduler")
}

func TestV2SingleConfigRejectsExtraPrograms(t *testing.T) {
	t.Log("Render a V2 single-model configuration with unsupported extra programs")
	_, err := lpuModelConfig([]*ModelProjection{{
		model:    "default",
		pipeline: PipelineSingle,
		configuredBuild: Build{
			Family:          BuildFamilyXT,
			runtimeSettings: map[string]any{"extra_programs": []any{"unsupported"}},
		},
	}}, "/models")

	t.Log("Verify the unsupported runtime setting is rejected")
	require.ErrorContains(t, err, "settings.extra_programs is not supported")
}
