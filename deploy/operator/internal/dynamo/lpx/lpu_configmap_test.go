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
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestLPXAuxiliaryNamesAreBounded(t *testing.T) {
	t.Log("Use a source name already at the Kubernetes length limit")
	root := strings.Repeat("a", validation.DNS1123SubdomainMaxLength)

	t.Log("Render the LPU and decode ConfigMap names")
	lpuName := LPUConfigMapName(root, "0123456789abcdef")
	decodeName := boundedAuxiliaryName(root, "-decode-0123456789abcdef")

	t.Log("Keep names valid and distinct while leaving short names unchanged")
	require.Len(t, lpuName, validation.DNS1123SubdomainMaxLength)
	require.Len(t, decodeName, validation.DNS1123SubdomainMaxLength)
	require.Empty(t, validation.IsDNS1123Subdomain(lpuName))
	require.Empty(t, validation.IsDNS1123Subdomain(decodeName))
	require.NotEqual(t, lpuName, decodeName)
	require.Equal(t, "short-lpu-0123456789abcdef", LPUConfigMapName("short", "0123456789abcdef"))
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
		nil,
	)

	t.Log("Keep gas_dir rooted at the selected image-owned build")
	require.NoError(t, err)
	require.Contains(t, configMap.Data["model_config.toml"], "model_path = '/models/model-build'")
	require.Equal(t, "/models/model-build", configMap.Data["gas_dir"])
	require.True(t, *configMap.Immutable)
	require.Equal(t, LPUConfigMapName("test-dgd", LPUConfigMapHash(configMap)), configMap.Name)

	t.Log("Render the preserved runtime with an invalid snapshot build reference")
	projection.configuredBuild.Path = "relative-build"
	_, err = renderLPUConfigMap(
		"test-namespace",
		"test-dgd",
		"/models",
		[]*ModelProjection{projection},
		nil,
	)

	t.Log("Return the model-path error before projecting gas_dir")
	require.ErrorContains(t, err, "render model_config.toml: model \"default\" model_path: parse build path \"relative-build\": ref \"relative-build\" must be an absolute path or URL")
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
		[]ExpectedAgent{{TemplateName: "rack", Replicas: 1}},
	)
	require.NoError(t, err)

	t.Log("Verify the generated files need no manual topology or model-setting patches")
	require.Equal(t, "part-1", configMap.Data["partition_paths"])
	require.Equal(t, v3OpaqueTopology, configMap.Data["topologies"])
	require.Contains(t, configMap.Data["model_config.toml"], "sequence_length = 8192")
	require.Contains(t, configMap.Data["model_config.toml"], "model_path = '/models/model-build'")
	require.Contains(t, configMap.Data["model_config.toml"], "batch_size = 4")
	require.Contains(t, configMap.Data["model_config.toml"], "custom_runtime_knob = 'enabled'")
	require.Contains(t, configMap.Data["model_config.toml"], "max_inflight_tasks = 3")
	require.Contains(t, configMap.Data["model_config.toml"], "tokenizer_path = '/nfs/models/gemma4-31b'")
	require.Contains(t, configMap.Data["model_config.toml"], "stop_tokens = [1]")
}

func TestNestedLPUModelConfigJoinsRuntimeAssetsToRemappedRoot(t *testing.T) {
	t.Log("Construct a projection with build-relative runtime assets")
	projection := &ModelProjection{
		model: "default", pipeline: PipelineLPX, runtimeBuildRef: "model-build",
		configuredBuild: Build{
			Path: "file:///snapshot-build", SupportsCPUEmbeddings: true,
			RuntimeTokenizerPath: ".", RuntimeTokenEmbeddingsPath: "runtime/text_embeddings.npz",
			runtimeSettings: map[string]any{"cpu_embeddings": true},
		},
	}

	t.Log("Render model configuration under the remapped runtime root")
	config, err := nestedLPUModelConfig(projection, "/models")

	t.Log("Verify default runtime asset paths are joined to the remapped build")
	require.NoError(t, err)
	iop := config["iop"].(map[string]any)
	require.Equal(t, "/models/model-build", iop["model_path"])
	require.Equal(t, "/models/model-build", iop["tokenizer_path"])
	require.Equal(t, "/models/model-build/runtime/text_embeddings.npz", iop["embedding_path"])

	t.Log("Render an explicit absolute embedding-path override")
	projection.configuredBuild.runtimeSettings["embedding_path"] = "/custom/embeddings"
	config, err = nestedLPUModelConfig(projection, "/models")

	t.Log("Verify the absolute override remains unchanged")
	require.NoError(t, err)
	require.Equal(t, "/custom/embeddings", config["iop"].(map[string]any)["embedding_path"])
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
