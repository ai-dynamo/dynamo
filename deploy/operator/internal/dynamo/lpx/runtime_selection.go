/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

const runtimeContractEnv = "LPX_RUNTIME_CONTRACT"
const manifestRuntimeContract = "manifest-v1"

// manifestRuntime is an explicit image/template compatibility declaration. Missing
// declarations retain the legacy runtime until the workload is migrated as a unit.
func manifestRuntime(spec corev1.PodSpec) (bool, error) {
	selected := false
	for _, container := range spec.Containers {
		if container.Name != "main" {
			continue
		}
		for _, variable := range container.Env {
			if variable.Name != runtimeContractEnv {
				continue
			}
			if variable.ValueFrom != nil || variable.Value != manifestRuntimeContract {
				return false, fmt.Errorf("%s must be the literal %q", runtimeContractEnv, manifestRuntimeContract)
			}
			selected = true
		}
	}
	// Opted-in templates must not retain paths that only the compatibility writer supplies.
	if selected {
		for _, container := range spec.Containers {
			for _, mount := range container.VolumeMounts {
				if mount.Name == lpuConfigVolumeName {
					return false, fmt.Errorf("manifest-v1 templates must remove the %s mount", lpuConfigVolumeName)
				}
			}
			for _, variable := range container.Env {
				if variable.Name == "LPU_CONFIG_DIR" || strings.HasSuffix(variable.Name, "RESOLVED_PARTITIONS_DIR") {
					return false, fmt.Errorf("manifest-v1 templates must remove %s", variable.Name)
				}
			}
		}
	}
	return selected, nil
}

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
	applyOwnedRuntimeEnv(container, []corev1.EnvVar{
		{Name: "LPX_MODEL_PATH", Value: path},
		{Name: "LPX_REMOTE_PARTITION_IDS", Value: runtimePartitionIDs(projection)},
	})
	return nil
}

func applyNovaSelections(container *corev1.Container, projections []*ModelProjection) {
	bindings := []corev1.EnvVar{{Name: "NOVA_REMOTE_PARTITION_IDS", Value: runtimePartitionIDs(projections[0])}}
	if projections[0].pipeline == PipelineSpecDecode {
		bindings[0].Name = "NOVA_DRAFT_REMOTE_PARTITION_IDS"
		bindings = append(bindings, corev1.EnvVar{Name: "NOVA_TARGET_REMOTE_PARTITION_IDS", Value: runtimePartitionIDs(projections[len(projections)-1])})
	}
	applyOwnedRuntimeEnv(container, bindings)
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
