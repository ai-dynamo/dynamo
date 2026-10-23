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
	applyOwnedRuntimeEnv(container, []corev1.EnvVar{{Name: "LPX_MODEL_PATH", Value: path}})
	if projection.pipeline == PipelineHybrid {
		applyLocalPartitionIDs(container, projection)
	}
	return nil
}

// Prepend operator bindings so authored Kubernetes environment references see
// authoritative values, including an explicitly empty local selection.
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
