/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"net/url"
	"path/filepath"
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
	configHash := LPUConfigMapHash(configMap)
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

// LPUConfigMapHash returns the hash stamped on LPU runtime Pod templates.
// configMap must be non-nil, with valid native metadata from rendering or Kubernetes.
// Its concrete content is always serializable; the input is not mutated.
func LPUConfigMapHash(configMap *corev1.ConfigMap) string {
	contentHash, _ := controllercommon.GetSpecHash(configMap)

	// The graph's extra-resource annotation hashes each resource's spec hash.
	// Apply the same second hash here so LPX can render that annotation directly.
	return fmt.Sprintf("%x", sha256.Sum256([]byte(contentHash)))
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

func lpuRuntimeBuildRef(projection *ModelProjection, modelStoragePath string) string {
	buildRef := projection.configuredBuild.Path
	snapshotRef, snapshotErr := url.Parse(buildRef)
	runtimeRef := strings.TrimSpace(projection.runtimeBuildRef)
	runtimeURL, runtimeErr := url.Parse(runtimeRef)
	if snapshotErr != nil || runtimeErr != nil || snapshotRef.Scheme != BuildSchemeFile ||
		runtimeRef == "" || runtimeURL.Scheme != "" || filepath.IsAbs(runtimeRef) {
		return buildRef
	}
	cleaned := filepath.Clean(runtimeRef)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return buildRef
	}
	return (&url.URL{Scheme: BuildSchemeFile, Path: filepath.Join(modelStoragePath, cleaned)}).String()
}

func resolvedPartitionData(projections []*ModelProjection) map[string]string {
	keys := [...]string{"nodes_per_partition", "partition_indices", "partition_ids", "partition_models",
		"partition_node_offsets", "partition_paths", "topologies"}

	var columns [len(keys)]strings.Builder

	// Keep partitions grouped by model in canonical projection order.
	for _, projection := range projections {
		// Render runtime partitions, including XT's collapsed prop-sync chains.
		offset := int64(0)
		for index, partition := range projection.configuredBuild.Partitions {
			endpointCount := int64(partition.effectiveNodeCount())
			row := [len(keys)]string{strconv.FormatInt(endpointCount, 10), strconv.Itoa(index),
				strconv.FormatUint(uint64(uint32(partition.SourcePartitionID)), 10), projection.model,
				strconv.FormatInt(offset, 10), partition.PartPath, partition.Topology.Raw}

			// Preserve compiler order across the parallel runtime columns.
			for column := range row {
				columns[column].WriteString(row[column])
				columns[column].WriteByte('\n')
			}
			offset += endpointCount
		}
	}
	data := make(map[string]string, len(keys))

	// Model columns disambiguate rows only when several models share the table.
	includeModelColumns := len(projections) > 1
	for column, key := range keys {
		if !includeModelColumns && (key == "partition_indices" || key == "partition_models") {
			continue
		}
		data[key] = strings.TrimSuffix(columns[column].String(), "\n")
	}
	return data
}

// ensureLPUConfigVolume adds the generated configuration only when the nonnil
// PodSpec has no authored config volume. Both LPU families preserve overrides.
func ensureLPUConfigVolume(spec *corev1.PodSpec, configMapName string) {
	if slices.ContainsFunc(spec.Volumes, func(volume corev1.Volume) bool { return volume.Name == lpuConfigVolumeName }) {
		return
	}

	spec.Volumes = append(spec.Volumes, corev1.Volume{
		Name: lpuConfigVolumeName,
		VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
			LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
		}},
	})
}
