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
)

const lpuConfigVolumeName = "config"

func renderRuntimeConfigMap(namePrefix string, data map[string]string) (*corev1.ConfigMap, string, error) {
	// Name immutable configuration from the content hash used by Pod templates.
	configMap := &corev1.ConfigMap{
		Immutable: new(true),
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

// resolvedPartitionData renders the shared partition columns. Each row is one
// runtime partition; Agent pods find theirs by matching GROVE_PCLQ_POD_INDEX
// against the node counts and offsets of their model's rows.
//
// Deployed Hydra and Quasar images read these columns through lpx-runtime's
// load_partition_metadata. Remove them once both read the per-pod records from
// agentPartitionRecords instead.
func resolvedPartitionData(projections []*ModelProjection) map[string]string {
	keys := [...]string{"nodes_per_partition", "partition_indices", "partition_ids", "partition_models",
		"partition_node_offsets", "partition_paths"}

	var columns [len(keys)]strings.Builder

	// Keep partitions grouped by model in canonical projection order.
	for _, projection := range projections {
		// Render runtime partitions, including XT's collapsed prop-sync chains.
		offset := int64(0)
		for index, partition := range projection.configuredBuild.partitions {
			endpointCount := int64(partition.effectiveNodeCount())
			row := [len(keys)]string{strconv.FormatInt(endpointCount, 10), strconv.Itoa(index),
				strconv.FormatUint(uint64(uint32(partition.sourcePartitionID)), 10), projection.model,
				strconv.FormatInt(offset, 10), partition.partPath}

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

// agentPartitionRecords renders one record per Agent pod, keyed by
// agentConfigKeyPrefix(templateName) plus the pod's clique index. Each value holds
// newline-terminated key=value lines describing the row that resolvedPartitionData
// renders for the pod, so a runtime reads its partition without scanning rows:
//
//	partition_row=<row in the shared columns>
//	partition_index=<partition index within the model>
//	partition_id=<compiler partition id>
//	partition_path=<partition path under the build>
//	node_count=<Agent pods in the partition>
//	node_offset=<clique index of the partition's first pod>
//	rank=<pod index minus node_offset>
//
// projections and agents are parallel, as in MaterializationPlan.Agents. Every
// Agent clique's replicas must equal its model's runtime node count.
//
// Each record is at most about 120 bytes plus the partition path, so a
// 1024-pod model with 64-byte paths adds about 185 KiB to the ConfigMap's 1 MiB budget.
func agentPartitionRecords(projections []*ModelProjection, agents []ExpectedAgent) (map[string]string, error) {
	records := make(map[string]string)
	row := 0
	for index, projection := range projections {
		agent := agents[index]
		prefix := agentConfigKeyPrefix(agent.TemplateName)

		// Assign every pod of the model's clique to the row covering its index.
		offset := 0
		for partitionIndex, partition := range projection.configuredBuild.partitions {
			nodeCount := partition.effectiveNodeCount()
			for rank := range nodeCount {
				records[prefix+strconv.Itoa(offset+rank)] = fmt.Sprintf(
					"partition_row=%d\npartition_index=%d\npartition_id=%d\npartition_path=%s\nnode_count=%d\nnode_offset=%d\nrank=%d\n",
					row, partitionIndex, uint32(partition.sourcePartitionID), partition.partPath, nodeCount, offset, rank,
				)
			}
			offset += nodeCount
			row++
		}
		if offset != agent.Replicas {
			return nil, fmt.Errorf(
				"model %q runtime partitions cover %d Agent pods, but clique %q has %d replicas",
				projection.model, offset, agent.TemplateName, agent.Replicas,
			)
		}
	}
	return records, nil
}

// agentConfigKeyPrefix returns the ConfigMap key prefix of an Agent clique's
// per-pod records; the pod's clique index completes the key.
func agentConfigKeyPrefix(templateName string) string {
	return "agent-" + templateName + "-"
}

// ensureLPUConfigVolume adds the generated configuration only when the nonnil
// PodSpec has no authored config volume. Both LPU families preserve overrides.
func ensureLPUConfigVolume(spec *corev1.PodSpec, configMapName string) {
	if slices.ContainsFunc(spec.Volumes, func(volume corev1.Volume) bool { return volume.Name == lpuConfigVolumeName }) {
		return
	}

	spec.Volumes = append(spec.Volumes, corev1.Volume{
		Name: lpuConfigVolumeName,
		ConfigMap: &corev1.ConfigMapVolumeSource{
			Name: configMapName,
		},
	})
}
