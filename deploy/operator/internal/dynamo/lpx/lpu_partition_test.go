/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestLPUAgentRuntimePartition(t *testing.T) {
	t.Log("Cover model-local Grove indexes and incomplete or ambiguous runtime mappings")
	tests := []struct {
		name      string
		index     string
		model     string
		data      map[string]string
		want      int
		wantError string
	}{
		{name: "second rank", index: "1", model: "model", want: 0},
		{name: "next partition", index: "2", model: "model", want: 1},
		{name: "other model resets indexes", index: "1", model: "other", want: 2},
		{name: "missing index", model: "model", wantError: "invalid Grove Pod index"},
		{name: "noncanonical index", index: "01", model: "model", wantError: "invalid Grove Pod index"},
		{name: "negative index", index: "-1", model: "model", wantError: "invalid Grove Pod index"},
		{name: "overflow index", index: "4294967296", model: "model", wantError: "invalid Grove Pod index"},
		{name: "missing model", index: "0", wantError: "has no LPU model"},
		{name: "absent model", index: "0", model: "absent", wantError: "matches no runtime partition"},
		{name: "out of range", index: "3", model: "model", wantError: "matches no runtime partition"},
		{name: "missing column", index: "0", model: "model", data: map[string]string{"partition_node_offsets": ""}, wantError: "inconsistent runtime partition columns"},
		{name: "bad count", index: "0", model: "model", data: map[string]string{"nodes_per_partition": "bad\n1\n3"}, wantError: "invalid runtime partition row"},
		{name: "zero count", index: "0", model: "model", data: map[string]string{"nodes_per_partition": "0\n1\n3"}, wantError: "invalid runtime partition row"},
		{name: "bad offset", index: "0", model: "model", data: map[string]string{"partition_node_offsets": "-1\n2\n0"}, wantError: "invalid runtime partition row"},
		{name: "overlapping ranges", index: "1", model: "model", data: map[string]string{"partition_node_offsets": "0\n1\n0"}, wantError: "matches multiple runtime partitions"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build a runtime table with variable partition sizes and model-local offsets")
			config := &corev1.ConfigMap{Data: map[string]string{
				"nodes_per_partition": "2\n1\n3", "partition_node_offsets": "0\n2\n0",
				"partition_models": "model\nmodel\nother",
			}}
			for key, value := range test.data {
				config.Data[key] = value
			}
			pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{grovecommon.LabelPodCliquePodIndex: test.index},
				Annotations: map[string]string{
					lpxv1alpha1.PodModelAnnotation:       test.model,
					lpxv1alpha1.PodPartitionIDAnnotation: "stale", lpxv1alpha1.PodRankInPartitionAnnotation: "not-authority",
				},
			}}
			originalPod, originalConfig := pod.DeepCopy(), config.DeepCopy()

			t.Log("Resolve the canonical row without mutating producer-owned data")
			got, err := LPUAgentRuntimePartition(pod, config)
			require.Equal(t, originalPod, pod)
			require.Equal(t, originalConfig, config)
			if test.wantError != "" {
				require.ErrorContains(t, err, test.wantError)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.want, got)
		})
	}
}

func TestLPUAgentRuntimePartitionUsesRenderedHybridTables(t *testing.T) {
	for _, test := range []struct {
		name      string
		family    BuildFamily
		collapsed bool
	}{
		{name: "XT", family: BuildFamilyXT},
		{name: "HX", family: BuildFamilyHX},
		{name: "XT collapsed PropSync chain", family: BuildFamilyXT, collapsed: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Render a hybrid runtime table with a two-Agent partition followed by one Agent")
			partitions := []BuildPartition{
				{Topology: Topology{ChipCount: 16}, HXExtent: []int64{1, 1, 1, 2}},
				{Topology: Topology{ChipCount: 8}, HXExtent: []int64{1, 1, 1, 1}},
			}
			projection := &ModelProjection{
				model: "model", pipeline: PipelineLPX, partitions: partitions,
				configuredBuild: Build{Family: test.family, Partitions: partitions},
			}
			if test.collapsed {
				projection.configuredBuild.Partitions = []BuildPartition{{Topology: Topology{ChipCount: 24}}}
			}
			config := &corev1.ConfigMap{Data: resolvedPartitionData([]*ModelProjection{projection})}

			t.Log("Resolve every rendered Grove index to the same runtime grouping")
			for _, row := range []struct {
				index string
				want  int
			}{{"0", 0}, {"1", 0}, {"2", 1}} {
				pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
					Labels:      map[string]string{grovecommon.LabelPodCliquePodIndex: row.index},
					Annotations: map[string]string{lpxv1alpha1.PodModelAnnotation: "model"},
				}}
				got, err := LPUAgentRuntimePartition(pod, config)
				require.NoError(t, err)
				if test.collapsed {
					row.want = 0
				}
				require.Equal(t, row.want, got)
			}
		})
	}
}
