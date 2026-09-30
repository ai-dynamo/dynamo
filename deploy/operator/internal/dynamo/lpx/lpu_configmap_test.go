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
	_, _, err := renderRuntimeConfigMap("test-lpu", data)
	require.NoError(t, err)

	t.Log("Reject one extra byte with a named error and no usable ConfigMap")
	data["suffix"] += "x"
	configMap, _, err := renderRuntimeConfigMap("test-lpu", data)
	require.Nil(t, configMap)
	require.ErrorContains(t, err, `rendered LPX ConfigMap "test-lpu-`)
	require.ErrorContains(t, err, "data is 1048577 bytes; maximum is 1048576")
}

func TestLPXRuntimeConfigNamesMatchPodIdentity(t *testing.T) {
	for _, root := range []string{"short", strings.Repeat("a", MaxPodCliqueSetNameLength), strings.Repeat("a", validation.LabelValueMaxLength)} {
		t.Run(root, func(t *testing.T) {
			t.Log("Render immutable runtime tables for PCS and Pod-label identity bounds")
			data := map[string]string{"runtime": "config"}
			lpu, lpuHash, err := renderRuntimeConfigMap(root+"-lpu", data)
			require.NoError(t, err)
			decode, decodeHash, err := renderRuntimeConfigMap(root+"-decode", data)
			require.NoError(t, err)

			t.Log("Name each table from its content hash and preserve each role suffix")
			require.Equal(t, root+"-lpu-"+lpuHash[:16], lpu.Name)
			require.Equal(t, root+"-decode-"+decodeHash[:16], decode.Name)
			require.Empty(t, validation.IsDNS1123Subdomain(lpu.Name))
			require.Empty(t, validation.IsDNS1123Subdomain(decode.Name))
		})
	}
}

func TestResolvedPartitionDataOmitsXTModelColumnsBeforeMaterialization(t *testing.T) {
	t.Parallel()

	t.Log("Construct an XT Single projection with two physical runtime partitions")
	projection := &Model{name: "default", component: &component{
		pipeline: PipelineSingle,
		configuredBuild: Build{
			family: xtFamily,
			partitions: []buildPartition{
				{sourcePartitionID: 7, partPath: "part-7", numChips: 16, devicesPerNode: 8},
				{sourcePartitionID: 9, partPath: "part-9", numChips: 8, devicesPerNode: 8},
			},
		},
	}}

	t.Log("Project the four placement and artifact columns for the XT Single runtime")
	data := resolvedPartitionData([]*Model{projection})

	t.Log("Verify omitted model columns never enter the final map and retained bytes remain exact")
	require.Equal(t, map[string]string{
		"nodes_per_partition":    "2\n1",
		"partition_ids":          "7\n9",
		"partition_node_offsets": "0\n2",
		"partition_paths":        "part-7\npart-9",
	}, data)
}

func TestLPUConfigVolumeRejectsAuthoredSourceMismatch(t *testing.T) {
	for _, test := range []struct {
		name   string
		source corev1.VolumeSource
	}{
		{name: "other ConfigMap", source: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
			LocalObjectReference: corev1.LocalObjectReference{Name: "other-config"},
		}}},
		{name: "other volume source", source: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Keep an authored runtime mount with a conflicting generated-volume source")
			spec := corev1.PodSpec{
				Containers: []corev1.Container{{Name: "main", VolumeMounts: []corev1.VolumeMount{{Name: "config", MountPath: "/runtime/partitions", ReadOnly: true}}}},
				Volumes:    []corev1.Volume{{Name: "config", VolumeSource: test.source}},
			}
			before := spec.DeepCopy()

			t.Log("Reject the source mismatch before changing the HX template")
			err := withLPUConfigVolume(&spec, "generated-config", false)
			require.ErrorContains(t, err, `volume "config" is reserved for ConfigMap "generated-config"`)
			require.Equal(t, *before, spec)
		})
	}
}
