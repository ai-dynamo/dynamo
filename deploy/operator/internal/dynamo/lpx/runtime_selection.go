/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// runtimePartitionIDs preserves the operator's final runtime order after local
// filtering, CPU embedding placement and pipeline-specific prop-sync handling.
func runtimePartitionIDs(projection *ModelProjection) string {
	ids := make([]string, len(projection.configuredBuild.Partitions))
	for index, partition := range projection.configuredBuild.Partitions {
		ids[index] = strconv.FormatUint(uint64(uint32(partition.SourcePartitionID)), 10)
	}
	return strings.Join(ids, ",")
}

func applyRuntimeSelection(container *corev1.Container, projection *ModelProjection, modelStoragePath string) error {
	path, err := buildRuntimePath(lpuRuntimeBuildRef(projection, modelStoragePath), modelStoragePath)
	if err != nil {
		return err
	}
	bindings := []corev1.EnvVar{{Name: "LPX_MODEL_PATH", Value: path}}
	if projection.remoteSelectionRequired {
		bindings = append(bindings, corev1.EnvVar{Name: "LPX_REMOTE_PARTITION_IDS", Value: runtimePartitionIDs(projection)})
	}
	removeRuntimeEnv(container, "LPX_REMOTE_PARTITION_IDS")
	applyOwnedRuntimeEnv(container, bindings)
	return nil
}

func applyNovaSelections(container *corev1.Container, projections []*ModelProjection) {
	removeRuntimeEnv(container, "NOVA_REMOTE_PARTITION_IDS", "NOVA_DRAFT_REMOTE_PARTITION_IDS", "NOVA_TARGET_REMOTE_PARTITION_IDS")
	names := []string{"NOVA_REMOTE_PARTITION_IDS"}
	models := []*ModelProjection{projections[0]}
	if projections[0].pipeline == PipelineSpecDecode {
		names = []string{"NOVA_DRAFT_REMOTE_PARTITION_IDS", "NOVA_TARGET_REMOTE_PARTITION_IDS"}
		models = append(models, projections[len(projections)-1])
	}
	var bindings []corev1.EnvVar
	for index, projection := range models {
		if projection.remoteSelectionRequired {
			bindings = append(bindings, corev1.EnvVar{Name: names[index], Value: runtimePartitionIDs(projection)})
		}
	}
	applyOwnedRuntimeEnv(container, bindings)
}

func removeRuntimeEnv(container *corev1.Container, names ...string) {
	container.Env = slices.DeleteFunc(container.Env, func(variable corev1.EnvVar) bool {
		return slices.Contains(names, variable.Name)
	})
}

// Prepend operator bindings so authored Kubernetes environment references see
// authoritative values, including an explicitly empty remote selection.
func applyOwnedRuntimeEnv(container *corev1.Container, bindings []corev1.EnvVar) {
	env := make([]corev1.EnvVar, 0, len(bindings)+len(container.Env))
	env = append(env, bindings...)
	for _, variable := range container.Env {
		if !slices.ContainsFunc(bindings, func(binding corev1.EnvVar) bool { return binding.Name == variable.Name }) {
			env = append(env, variable)
		}
	}
	container.Env = env
}
