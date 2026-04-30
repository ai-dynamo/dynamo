// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpoint

import (
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
)

func expectedPodInfoItemPaths() []string {
	return []string{
		"pod_name",
		"pod_uid",
		"pod_namespace",
		commonconsts.PodInfoFileDynNamespace,
		commonconsts.PodInfoFileDynNamespaceWorkerSuffix,
		commonconsts.PodInfoFileDynComponent,
		commonconsts.PodInfoFileDynParentDGDName,
		commonconsts.PodInfoFileDynParentDGDNamespace,
	}
}

func assertCanonicalPodInfoVolume(t *testing.T, volume corev1.Volume) {
	t.Helper()
	assert.Equal(t, commonconsts.PodInfoVolumeName, volume.Name)
	if assert.NotNil(t, volume.VolumeSource.DownwardAPI, "expected DownwardAPI source") {
		paths := make([]string, 0, len(volume.VolumeSource.DownwardAPI.Items))
		for _, item := range volume.VolumeSource.DownwardAPI.Items {
			paths = append(paths, item.Path)
		}
		assert.ElementsMatch(t, expectedPodInfoItemPaths(), paths)
	}
}

func TestEnsurePodInfoVolumeAppendsWhenMissing(t *testing.T) {
	spec := &corev1.PodSpec{Volumes: []corev1.Volume{{Name: "other"}}}

	EnsurePodInfoVolume(spec)

	if assert.Len(t, spec.Volumes, 2) {
		assert.Equal(t, "other", spec.Volumes[0].Name)
		assertCanonicalPodInfoVolume(t, spec.Volumes[1])
	}
}

func TestEnsurePodInfoVolumeReplacesWrongSourceType(t *testing.T) {
	spec := &corev1.PodSpec{Volumes: []corev1.Volume{
		{Name: commonconsts.PodInfoVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
	}}

	EnsurePodInfoVolume(spec)

	if assert.Len(t, spec.Volumes, 1) {
		assertCanonicalPodInfoVolume(t, spec.Volumes[0])
	}
}

func TestEnsurePodInfoVolumeRepairsPartialItems(t *testing.T) {
	spec := &corev1.PodSpec{Volumes: []corev1.Volume{{
		Name: commonconsts.PodInfoVolumeName,
		VolumeSource: corev1.VolumeSource{DownwardAPI: &corev1.DownwardAPIVolumeSource{
			Items: []corev1.DownwardAPIVolumeFile{
				{Path: "pod_name", FieldRef: &corev1.ObjectFieldSelector{FieldPath: commonconsts.PodInfoFieldPodName}},
			},
		}},
	}}}

	EnsurePodInfoVolume(spec)

	if assert.Len(t, spec.Volumes, 1) {
		assertCanonicalPodInfoVolume(t, spec.Volumes[0])
	}
}

func TestEnsurePodInfoVolumeIsIdempotent(t *testing.T) {
	spec := &corev1.PodSpec{}

	EnsurePodInfoVolume(spec)
	EnsurePodInfoVolume(spec)

	if assert.Len(t, spec.Volumes, 1) {
		assertCanonicalPodInfoVolume(t, spec.Volumes[0])
	}
}
