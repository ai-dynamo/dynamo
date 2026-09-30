/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controllercommon "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const lpuConfigVolumeName = "config"

func renderRuntimeConfigMap(namePrefix string, data map[string]string) (*corev1.ConfigMap, string, error) {
	// Name immutable configuration from the content hash used by Pod templates.
	configMap := &corev1.ConfigMap{
		Immutable: ptr.To(true),
		Data:      data,
	}
	// The graph's extra-resource annotation hashes each resource's spec hash.
	// Apply the same second hash here so LPX can render that annotation directly.
	contentHash, _ := controllercommon.GetSpecHash(configMap)
	configHash := fmt.Sprintf("%x", sha256.Sum256([]byte(contentHash)))
	configMap.Name = fmt.Sprintf("%s-%.16s", namePrefix, configHash)

	// Reject oversized configuration before any caller can publish it.
	totalSize := 0
	for _, value := range data {
		totalSize += len(value)
	}
	if totalSize > corev1.MaxSecretSize {
		return nil, "", fmt.Errorf(
			"rendered LPX ConfigMap %q data is %d bytes; maximum is %d",
			configMap.Name,
			totalSize,
			corev1.MaxSecretSize,
		)
	}
	return configMap, configHash, nil
}

func lpuModelStoragePath(spec corev1.PodSpec) (string, error) {
	container := common.FindContainerByName(spec.Containers, commonconsts.MainContainerName)
	mountIndex := slices.IndexFunc(container.VolumeMounts, func(mount corev1.VolumeMount) bool {
		return mount.Name == v1alpha1.ModelStorageVolumeName
	})
	if mountIndex < 0 {
		return "", fmt.Errorf(
			"selected LPX main container requires model storage volume mount %q",
			v1alpha1.ModelStorageVolumeName,
		)
	}
	mount := container.VolumeMounts[mountIndex]
	if strings.TrimSpace(mount.MountPath) == "" {
		return "", fmt.Errorf("model storage volume %q has no mount path", mount.Name)
	}
	volumeIndex := slices.IndexFunc(spec.Volumes, func(volume corev1.Volume) bool { return volume.Name == mount.Name })
	if volumeIndex < 0 {
		return "", fmt.Errorf("selected LPX podTemplate has no model storage volume %q", mount.Name)
	}
	return mount.MountPath, nil
}

func resolvedPartitionData(projections []*ModelProjection) map[string]string {
	var nodes, indices, ids, models, offsets, paths []string
	for _, projection := range projections {
		// Render runtime partitions, including XT's collapsed prop-sync chains.
		offset := 0
		for index, partition := range projection.configuredBuild.Partitions {
			nodeCount := partition.effectiveNodeCount()
			nodes = append(nodes, strconv.Itoa(nodeCount))
			indices = append(indices, strconv.Itoa(index))
			ids = append(ids, strconv.FormatUint(uint64(uint32(partition.SourcePartitionID)), 10))
			models = append(models, projection.model)
			offsets = append(offsets, strconv.Itoa(offset))
			paths = append(paths, partition.PartPath)
			offset += nodeCount
		}
	}
	data := map[string]string{
		"nodes_per_partition":    strings.Join(nodes, "\n"),
		"partition_ids":          strings.Join(ids, "\n"),
		"partition_node_offsets": strings.Join(offsets, "\n"),
		"partition_paths":        strings.Join(paths, "\n"),
	}

	// Omit model-identity columns that the XT Single runtime never consumes.
	if projections[0].configuredBuild.Family != BuildFamilyXT || projections[0].pipeline != PipelineSingle {
		data["partition_indices"] = strings.Join(indices, "\n")
		data["partition_models"] = strings.Join(models, "\n")
	}
	return data
}

func withLPUConfigVolume(spec *corev1.PodSpec, configMapName string, allowOverrides bool) error {
	found := false
	for _, volume := range spec.Volumes {
		if volume.Name != lpuConfigVolumeName {
			continue
		}
		if !allowOverrides && (found ||
			volume.ConfigMap == nil ||
			volume.ConfigMap.Name != configMapName ||
			len(volume.ConfigMap.Items) != 0 ||
			volume.ConfigMap.DefaultMode != nil ||
			volume.ConfigMap.Optional != nil) {
			return fmt.Errorf("selected LPX podTemplate volume %q is reserved for ConfigMap %q", lpuConfigVolumeName, configMapName)
		}
		found = true
	}
	if !found {
		spec.Volumes = append(spec.Volumes, corev1.Volume{
			Name: lpuConfigVolumeName,
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
			}},
		})
	}
	return nil
}
