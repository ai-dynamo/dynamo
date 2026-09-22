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

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controllercommon "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const lpuConfigVolumeName = "config"

// LPUConfigMapName names the immutable runtime table using its Pod-template content hash.
// The root is the workload resource prefix, including its group when independent workloads share a PCS.
func LPUConfigMapName(root, configHash string) string {
	return fmt.Sprintf("%s-lpu-%.16s", root, configHash)
}

// LPUAgentConfigMapName returns the runtime table referenced by a non-nil Agent Pod.
// The cached Pod projection must preserve ConfigMap volume names. Reading the Pod's
// reference keeps old runtime revisions addressable across workload template updates.
func LPUAgentConfigMapName(pod *corev1.Pod) (string, error) {
	for _, volume := range pod.Spec.Volumes {
		if volume.Name == lpuConfigVolumeName && volume.ConfigMap != nil && volume.ConfigMap.Name != "" {
			return volume.ConfigMap.Name, nil
		}
	}
	return "", fmt.Errorf("LPU-GPU eviction trigger pod %s/%s has no runtime ConfigMap volume", pod.Namespace, pod.Name)
}

func renderRuntimeConfigMap(namePrefix string, data map[string]string) (*corev1.ConfigMap, error) {
	// Name immutable configuration from the content hash used by Pod templates.
	configMap := &corev1.ConfigMap{
		Immutable: ptr.To(true),
		Data:      data,
	}
	configMap.Name = fmt.Sprintf("%s-%.16s", namePrefix, LPUConfigMapHash(configMap))

	// Reject oversized configuration before any caller can publish it.
	totalSize := 0
	for _, value := range data {
		totalSize += len(value)
	}
	if totalSize > corev1.MaxSecretSize {
		return nil, fmt.Errorf(
			"rendered LPX ConfigMap %q data is %d bytes; maximum is %d",
			configMap.Name,
			totalSize,
			corev1.MaxSecretSize,
		)
	}
	return configMap, nil
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
		return mount.Name == commonconsts.ModelStorageVolumeName
	})
	if mountIndex < 0 {
		return "", fmt.Errorf(
			"selected LPX main container requires model storage volume mount %q",
			commonconsts.ModelStorageVolumeName,
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

	// Omit model-identity columns that the XT Single runtime never consumes.
	includeModelColumns := projections[0].configuredBuild.Family != BuildFamilyXT ||
		projections[0].pipeline != PipelineSingle
	var columns [len(keys)]strings.Builder

	// Accumulate each projection's runtime partitions into the surviving columns.
	for _, projection := range projections {
		// Render configured V2 runtime metadata without reapplying scheduler-shape validation to collapsed partitions.
		partitions := projection.partitions
		v2Runtime := projection.configuredBuild.Family == BuildFamilyXT &&
			len(projection.configuredBuild.Partitions) != 0
		if v2Runtime {
			partitions = projection.configuredBuild.Partitions
		}

		offset := int64(0)
		for index, partition := range partitions {
			var nodes string
			var endpointCount int64
			if v2Runtime {
				nodeCount := partition.effectiveNodeCount()
				nodes, endpointCount = strconv.Itoa(nodeCount), int64(nodeCount)
			} else {
				endpointCount = partition.HXExtent[1] * partition.HXExtent[2] * partition.HXExtent[3]
				nodes = strconv.FormatInt(endpointCount, 10)
			}
			// Project one row and populate optional model identity only when consumed.
			row := [len(keys)]string{nodes, "",
				strconv.FormatUint(uint64(uint32(partition.SourcePartitionID)), 10), "",
				strconv.FormatInt(offset, 10), partition.PartPath, partition.Topology.Raw}
			if includeModelColumns {
				row[1] = strconv.Itoa(index)
				row[3] = projection.model
			}

			// Write only columns that survive into the ConfigMap.
			for column := range row {
				if !includeModelColumns && (column == 1 || column == 3) {
					continue
				}
				columns[column].WriteString(row[column])
				columns[column].WriteByte('\n')
			}
			offset += endpointCount
		}
	}
	data := make(map[string]string, len(keys))

	// Materialize only the runtime-visible columns.
	for column := range keys {
		if !includeModelColumns && (column == 1 || column == 3) {
			continue
		}
		data[keys[column]] = strings.TrimSuffix(columns[column].String(), "\n")
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
