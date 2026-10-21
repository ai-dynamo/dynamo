/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// modelPathEnvVar carries a single model's build path to the LPX runtimes that read it.
const modelPathEnvVar = "LPX_MODEL_PATH"

// configureNodeLocalConductorRuntime consumes a fresh conductor PodSpec with a validated main container.
func configureNodeLocalConductorRuntime(
	conductorPodSpec *corev1.PodSpec,
	allocation string,
) {
	// Bind placement data without interpreting the template's executable or arguments.
	conductor := common.FindContainerByName(conductorPodSpec.Containers, commonconsts.MainContainerName)

	// Kubernetes expands environment references in order; publish allocation before authored bindings.
	env := make([]corev1.EnvVar, 0, len(conductor.Env)+1)
	env = append(env, corev1.EnvVar{Name: allocationEnvVar, Value: allocation})
	for _, variable := range conductor.Env {
		if variable.Name != allocationEnvVar {
			env = append(env, variable)
		}
	}
	conductor.Env = env
}

// applyModelPaths binds nonempty canonical projections into a fresh runtime container.
func applyModelPaths(container *corev1.Container, projections []*ModelProjection, modelStoragePath string) error {
	type modelPathBinding struct {
		name       string
		projection *ModelProjection
	}
	bindings := []modelPathBinding{{modelPathEnvVar, projections[0]}}
	if projections[0].pipeline == PipelineSpecDecode {
		bindings[0].name = "LPX_DRAFT_MODEL_PATH"
		bindings = append(bindings, modelPathBinding{"LPX_TARGET_MODEL_PATH", projections[len(projections)-1]})
	}

	// Resolve all paths before publishing authoritative values ahead of authored references.
	env := make([]corev1.EnvVar, 0, len(container.Env)+len(bindings))
	for _, binding := range bindings {
		projection := binding.projection
		path, err := buildRuntimePath(projection.configuredBuild.path, projection.runtimeBuildRef, modelStoragePath)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", binding.name, err)
		}
		env = append(env, corev1.EnvVar{Name: binding.name, Value: path})
	}
	for _, variable := range container.Env {
		if !slices.ContainsFunc(bindings, func(binding modelPathBinding) bool { return binding.name == variable.Name }) {
			env = append(env, variable)
		}
	}
	container.Env = env
	return nil
}

// configureServingAgent binds a fresh Agent PodSpec that serves a single-model
// workload through its runtime sidecar. The sidecar and main receive the model
// path when they expand it. agentPodSpec must contain main and the runtime init
// container; projections describe one LPU-only model.
func configureServingAgent(agentPodSpec *corev1.PodSpec, projections []*ModelProjection, modelStoragePath string) error {
	// The sidecar receives the model path as a conductor's sidecar does.
	if err := configureRuntimeSidecar(agentPodSpec, projections, modelStoragePath); err != nil {
		return err
	}

	// The engine in main reads the build that the sidecar tokenizes from.
	main := common.FindContainerByName(agentPodSpec.Containers, commonconsts.MainContainerName)
	if !referencesEnvVar(main, modelPathEnvVar) {
		return nil
	}
	return applyModelPaths(main, projections, modelStoragePath)
}

// configureRuntimeSidecar publishes the model path to the runtime init container
// of a fresh serving PodSpec when that container expands it, which requires its
// model-storage mount at modelStoragePath. podSpec must contain the runtime init
// container; projections describe one single-model workload.
func configureRuntimeSidecar(podSpec *corev1.PodSpec, projections []*ModelProjection, modelStoragePath string) error {
	sidecar := common.FindContainerByName(podSpec.InitContainers, commonconsts.RuntimeContainerName)
	if !referencesEnvVar(sidecar, modelPathEnvVar) {
		return nil
	}

	// The published path resolves only beneath the serving role's model-storage mount.
	if !slices.ContainsFunc(sidecar.VolumeMounts, func(mount corev1.VolumeMount) bool {
		return mount.Name == v1alpha1.ModelStorageVolumeName && mount.MountPath == modelStoragePath
	}) {
		return fmt.Errorf(
			"%q init container references $(%s), so it requires volume mount %q at the model-storage mount path %q",
			commonconsts.RuntimeContainerName, modelPathEnvVar, v1alpha1.ModelStorageVolumeName, modelStoragePath,
		)
	}
	return applyModelPaths(sidecar, projections, modelStoragePath)
}

// referencesEnvVar reports whether Kubernetes expands $(name) in container's
// command, arguments or environment values, where $$ escapes a dollar.
// container must be non-nil.
func referencesEnvVar(container *corev1.Container, name string) bool {
	values := slices.Concat(container.Command, container.Args)
	for _, variable := range container.Env {
		values = append(values, variable.Value)
	}

	// An odd run of dollars before the reference leaves one unescaped.
	reference := "(" + name + ")"
	return slices.ContainsFunc(values, func(value string) bool {
		for {
			index := strings.Index(value, reference)
			if index < 0 {
				return false
			}
			if dollars := index - len(strings.TrimRight(value[:index], "$")); dollars%2 == 1 {
				return true
			}
			value = value[index+len(reference):]
		}
	})
}
