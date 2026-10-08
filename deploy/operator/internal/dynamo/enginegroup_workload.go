/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// EngineGroupNameForComponent returns the stable name of one indexed world, independent of workload revisions.
// worldIndex must be nonnegative and identifies a world, not a capacity slot within it.
func EngineGroupNameForComponent(dgdName, componentName string, worldIndex int32) string {
	name := strings.ToLower(dgdName + "-" + componentName + "-" + strconv.FormatInt(int64(worldIndex), 10))
	if len(name) <= 63 {
		return name
	}

	// Keep names usable as label values without losing uniqueness after truncation.
	digest := sha256.Sum256([]byte(name))
	return fmt.Sprintf("%s-%x", strings.TrimRight(name[:50], "-."), digest[:6])
}

// ResolveComponentEngineGroupProfile validates the implemented Grove growth profile before creation.
// component and component.EngineGroup must not be nil. No input is mutated.
func ResolveComponentEngineGroupProfile(component *v1beta1.DynamoComponentDeploymentSharedSpec) (ResolvedSGLangElasticEPProfile, error) {
	// Unsupported lifecycle combinations must never fall through to ordinary DGD rendering.
	if !IsWorkerComponent(string(component.ComponentType)) || ptr.Deref(component.Replicas, 1) != 1 {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("Engine Group components require one independent worker world")
	}
	if component.EngineGroup.InitialSize < 1 {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("Engine Group initialSize must be positive")
	}
	if component.Multinode != nil || len(component.Roles) != 0 || component.ScalingAdapter != nil || component.Experimental != nil {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("Engine Group growth does not support multinode, roles, scalingAdapter, or experimental lifecycle features")
	}
	if component.ProviderOverride != nil {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("Engine Group workload overrides cannot bypass the resolved launch profile")
	}
	if component.PodTemplate == nil {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("Engine Group capacity requires an explicit podTemplate")
	}
	if policy := component.PodTemplate.Spec.RestartPolicy; policy != "" && policy != corev1.RestartPolicyNever {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("Engine Group capacity requires restartPolicy Never")
	}
	main := GetMainContainer(component)
	if main == nil || len(main.Command) != 3 || !isSupportedProfilePythonExecutable(main.Command[0]) ||
		main.Command[1] != "-m" || main.Command[2] != SGLangElasticEPBootstrapModule {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("Engine Group growth requires the SGLang template-invariant bootstrap entrypoint")
	}

	// The first profile resolves scalar GPUs owned exclusively by the main container.
	gpuCount, err := resolveEngineGroupGPUAllocation(&component.PodTemplate.Spec, main)
	if err != nil {
		return ResolvedSGLangElasticEPProfile{}, err
	}

	// Resolve the same geometry used by the runtime, binding it to the immutable declaration.
	podTemplate, err := json.Marshal(component.PodTemplate)
	if err != nil {
		return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("encode Engine Group pod template: %w", err)
	}
	digest := sha256.Sum256(podTemplate)
	resolved, err := ResolveSGLangElasticEPProfile(SGLangProfileGeometrySource{
		Command: main.Command, Args: main.Args, InitialReplicas: component.EngineGroup.InitialSize,
		MainContainerGPUs: gpuCount, DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest: fmt.Sprintf("pod-template:%x", digest),
	})
	if err != nil {
		return ResolvedSGLangElasticEPProfile{}, err
	}

	// Declared policy may narrow, but never exceed, the growth-only engine bounds.
	if policy := component.EngineGroup.Policy; policy != nil {
		if policy.MinSize != nil && *policy.MinSize != resolved.InitialReplicas {
			return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("policy.minSize must equal the growth-only initial size")
		}
		if policy.MaxSize != nil && (*policy.MaxSize < component.EngineGroup.InitialSize || *policy.MaxSize > resolved.MaximumReplicas) {
			return ResolvedSGLangElasticEPProfile{}, fmt.Errorf("policy.maxSize must be between initialSize and the engine maximum")
		}
	}
	return resolved, nil
}

// resolveEngineGroupGPUAllocation proves scalar GPU ownership for the implemented profile.
// podSpec and main must not be nil; main must be the pod's main container.
func resolveEngineGroupGPUAllocation(podSpec *corev1.PodSpec, main *corev1.Container) (int64, error) {
	// Dynamic claims cannot establish geometry before allocation, and sidecars must not own GPUs.
	if len(podSpec.ResourceClaims) != 0 || len(main.Resources.Claims) != 0 {
		return 0, fmt.Errorf("Engine Group growth does not yet support DRA GPU allocation")
	}
	containers := append([]corev1.Container(nil), podSpec.InitContainers...)
	containers = append(containers, podSpec.Containers...)
	for _, container := range containers {
		if container.Name == consts.MainContainerName {
			continue
		}
		gpuLimit := container.Resources.Limits[corev1.ResourceName(consts.KubeResourceGPUNvidia)]
		if !gpuLimit.IsZero() || len(container.Resources.Claims) != 0 {
			return 0, fmt.Errorf("Engine Group GPUs must belong exclusively to the main container")
		}
	}

	// Fractional or absent limits cannot define a fixed accelerator allocation.
	gpuLimit := main.Resources.Limits[corev1.ResourceName(consts.KubeResourceGPUNvidia)]
	gpuCount, exact := gpuLimit.AsInt64()
	if !exact || gpuCount < 1 {
		return 0, fmt.Errorf("Engine Group main container requires a positive whole-GPU limit")
	}
	return gpuCount, nil
}
