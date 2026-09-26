/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"path/filepath"

	corev1 "k8s.io/api/core/v1"
)

const gbuildManifestPathEnv = "GBUILD_MANIFEST_PATH"

// MinimumCyborgReplicas returns one complete client group for a resolved hybrid workload.
func (w *Workload) MinimumCyborgReplicas() (int32, error) {
	build := &w.modelProjections[0].configuredBuild
	replicas := int64(build.IOFPGACount) * int64(build.IOFanoutFactor)
	if replicas > math.MaxInt32 {
		return 0, fmt.Errorf("minimum Cyborg replicas %d exceeds the PodClique replica limit %d", replicas, math.MaxInt32)
	}
	return int32(replicas), nil
}

// ValidateCyborgReplicas ensures that the specified Cyborg replicas are in a valid configuration.
func (w *Workload) ValidateCyborgReplicas(replicas int32) error {
	build := &w.modelProjections[0].configuredBuild
	return validateCyborgReplicas(build, replicas)
}

// applyCyborgManifestPath projects an authoritative manifest location into one Cyborg container.
func applyCyborgManifestPath(container *corev1.Container, projection *ModelProjection, modelStoragePath string) error {
	buildRoot, err := buildRuntimePath(lpuRuntimeBuildRef(projection, modelStoragePath), modelStoragePath)
	if err != nil {
		return fmt.Errorf("resolve GBuild manifest path: %w", err)
	}

	// Kubernetes expands environment references in order; publish the manifest before authored bindings.
	env := make([]corev1.EnvVar, 0, len(container.Env)+1)
	env = append(env, corev1.EnvVar{
		Name:  gbuildManifestPathEnv,
		Value: filepath.Join(buildRoot, gbuildManifestV2CapnpFile),
	})
	for _, variable := range container.Env {
		if variable.Name != gbuildManifestPathEnv {
			env = append(env, variable)
		}
	}
	container.Env = env
	return nil
}

// validateCyborgReplicas requires a nonnil normalized build and validates its Cyborg replica domain.
func validateCyborgReplicas(build *Build, replicas int32) error {
	// Require every physical endpoint and client-owned transaction in this replica domain.
	ioFPGACount := build.IOFPGACount
	ioFanoutFactor := build.IOFanoutFactor
	if replicas%ioFPGACount != 0 {
		return fmt.Errorf("decode service Cyborg replicas %d must be divisible by ioFpgaCount %d", replicas, ioFPGACount)
	}
	if replicas/ioFPGACount%ioFanoutFactor != 0 {
		return fmt.Errorf(
			"decode service Cyborg replicas %d must provide fanoutFactor %d clients for each of %d I/O FPGA endpoints",
			replicas,
			ioFanoutFactor,
			ioFPGACount,
		)
	}

	return nil
}
