/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
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
			Path:   "file:///snapshot-build",
			Family: BuildFamilyXT,
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

func TestConductorConfigMapContainsOnlyPartitions(t *testing.T) {
	t.Parallel()

	for _, family := range []string{"v2", "v3"} {
		t.Run(family, func(t *testing.T) {
			t.Log("Project a compiler build with a template-owned runtime configuration")
			var path string
			if family == "v2" {
				path = writeV2CompilerFixture(t)
			} else {
				path = writeV3CompilerFixture(t)
			}
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineSingle, Models: []string{"default"},
				BuildSnapshot: normalizeTestSnapshot(t, acquireTestSnapshot(t, path)),
			})
			require.NoError(t, err)

			t.Log("Retain only the resolved partition files in the immutable ConfigMap")
			configMap, err := renderLPUConfigMap("test", "test-dgd", "/models", projections)
			require.NoError(t, err)
			require.Equal(t, resolvedPartitionData(projections), configMap.Data)
			require.NotContains(t, configMap.Data, "model_config.toml")
			require.NotContains(t, configMap.Data, "datacenter.toml")
			require.NotEmpty(t, configMap.Data["partition_paths"])
			require.NotEmpty(t, configMap.Data["topologies"])
		})
	}
}
