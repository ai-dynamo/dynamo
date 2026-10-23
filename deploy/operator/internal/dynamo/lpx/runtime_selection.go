/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"

	corev1 "k8s.io/api/core/v1"
)

// applyRuntimeSelection publishes the model and GPU-local selection to every role.
// Runtime manifests own remote order, host counts, and prop-sync grouping.
func applyRuntimeSelection(container *corev1.Container, projection *ModelProjection, modelStoragePath string) error {
	path, err := buildRuntimePath(projection.configuredBuild.path, projection.runtimeBuildRef, modelStoragePath)
	if err != nil {
		return err
	}
	applyOwnedEnv(container, []corev1.EnvVar{{Name: "LPX_MODEL_PATH", Value: path}})
	if projection.pipeline == PipelineHybrid {
		applyLocalPartitionIDs(container, projection)
	}
	return nil
}

// applyOwnedEnv prepends operator-owned bindings to container's environment and
// removes authored variables of the same names. Kubernetes expands references in
// order, so authored values see the authoritative bindings, including an
// explicitly empty local selection.
func applyOwnedEnv(container *corev1.Container, owned []corev1.EnvVar) {
	env := make([]corev1.EnvVar, 0, len(owned)+len(container.Env))
	env = append(env, owned...)
	for _, variable := range container.Env {
		if !slices.ContainsFunc(owned, func(binding corev1.EnvVar) bool { return binding.Name == variable.Name }) {
			env = append(env, variable)
		}
	}
	container.Env = env
}
