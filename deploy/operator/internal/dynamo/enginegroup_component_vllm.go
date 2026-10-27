/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// ResolveVLLMComponentEngineGroup resolves the immutable native declaration, not the bootstrap Pod.
// component and component.EngineGroup must not be nil. parentEnv may be nil.
// Effective environment uses the same component-over-parent precedence as rendering. No input is mutated.
func ResolveVLLMComponentEngineGroup(component *api.DynamoComponentDeploymentSharedSpec, parentEnv []corev1.EnvVar) (VLLMProfileGeometrySource, ResolvedVLLMElasticEPProfile, error) {
	// Resolve inherited environment before validating visibility, ports and native geometry.
	component = component.DeepCopy()
	if main := GetMainContainer(component); main != nil {
		main.Env = MergeEnvs(parentEnv, main.Env)
	}
	// The first integration owns one world with one dedicated, scalar-GPU main container per allocation.
	if !IsWorkerComponent(string(component.ComponentType)) || ptr.Deref(component.Replicas, 1) != 1 ||
		component.EngineGroup.InitialSize < 1 || component.PodTemplate == nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM Engine Groups require one worker world, an explicit podTemplate and positive initialSize")
	}
	if component.Multinode != nil || len(component.Roles) != 0 || component.ScalingAdapter != nil ||
		component.Experimental != nil || component.ProviderOverride != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM Engine Groups do not support multinode, roles, scalingAdapter, experimental lifecycle features or provider overrides")
	}
	main := GetMainContainer(component)
	if main == nil || main.Image == "" {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM Engine Groups require an explicit main image")
	}
	if err := validateEngineGroupPodTemplate(component.PodTemplate); err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}
	if err := requireEngineGroupRetentionGate(component.PodTemplate); err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}
	for _, setting := range []struct{ name, value string }{
		{vllmMultiprocessingEnvVar, "1"}, {"VLLM_USE_RUST_FRONTEND", "0"}, {vllmElasticEPDrainEnvVar, "1"},
	} {
		if err := requireEngineGroupLiteralEnv(main, setting.name, setting.value); err != nil {
			return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
		}
	}
	if IsVLLMWorkerLaunch(VLLMProfileGeometrySource{Command: main.Command, Args: main.Args}) {
		if err := requireEngineGroupLiteralEnv(main, "DYNAMO_ENGINE_GROUP", "1"); err != nil {
			return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
		}
		if err := requireEngineGroupFieldEnv(main, "DYNAMO_POD_UID", "metadata.uid"); err != nil {
			return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
		}
	}
	gpus, err := resolveVLLMComponentGPUs(&component.PodTemplate.Spec, main)
	if err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}
	source := VLLMProfileGeometrySource{
		Command: main.Command, Args: main.Args, Env: main.Env, EnvFrom: main.EnvFrom,
		InitialReplicas: component.EngineGroup.InitialSize, MainContainerGPUs: gpus, DedicatedMainGPUAllocation: true,
	}

	// Fingerprint immutable launch geometry without changing the rendered command.
	source.WorkloadRevisionDigest, err = vllmWorkloadRevision(&component.PodTemplate.Spec, main)
	if err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}
	profile, err := ResolveVLLMLaunchProfile(source)
	if err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}

	// Policy bounds narrow the supported integration, never the engine's reserved maximum.
	if policy := component.EngineGroup.Policy; policy != nil {
		if policy.MinSize != nil && *policy.MinSize != profile.InitialReplicas {
			return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM policy.minSize must equal initialSize")
		}
		if policy.MaxSize != nil && (*policy.MaxSize < profile.InitialReplicas || *policy.MaxSize > profile.MaximumReplicas) {
			return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM policy.maxSize must be between initialSize and the reserved DP maximum")
		}
	}
	return source, profile, nil
}

// vllmWorkloadRevision excludes only the creation seed from the immutable workload identity.
// spec and main must not be nil. Neither input is mutated.
func vllmWorkloadRevision(spec *corev1.PodSpec, main *corev1.Container) (string, error) {
	// Project the private copy so creation-time sizing does not alter fixed geometry.
	spec = spec.DeepCopy()
	for i := range spec.Containers {
		if spec.Containers[i].Name == consts.MainContainerName {
			argv := append(append([]string(nil), main.Command...), main.Args...)
			projected := make([]string, 0, len(argv))
			for index := 0; index < len(argv); index++ {
				name, _, inline := strings.Cut(argv[index], "=")
				if strings.ReplaceAll(name, "_", "-") == "--elastic-ep-initial-dp-size" {
					if !inline {
						index++
					}
					continue
				}
				projected = append(projected, argv[index])
			}
			spec.Containers[i].Command, spec.Containers[i].Args = projected, nil
		}
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		return "", fmt.Errorf("encode vLLM workload declaration: %w", err)
	}
	return fmt.Sprintf("pod-spec:%x", sha256.Sum256(encoded)), nil
}

func resolveVLLMComponentGPUs(spec *corev1.PodSpec, main *corev1.Container) (int64, error) {
	// A container restart, shared host port or opaque GPU claim needs a different launch contract.
	if (spec.RestartPolicy != "" && spec.RestartPolicy != corev1.RestartPolicyNever) || spec.HostNetwork ||
		len(spec.ResourceClaims) != 0 || len(main.Resources.Claims) != 0 {
		return 0, fmt.Errorf("vLLM Engine Groups require restartPolicy Never, Pod networking and scalar GPU limits")
	}
	for _, variable := range main.Env {
		if variable.Name == "CUDA_VISIBLE_DEVICES" || variable.Name == "NVIDIA_VISIBLE_DEVICES" {
			return 0, fmt.Errorf("vLLM GPU visibility must come from the Kubernetes allocation")
		}
	}
	containers := append(append([]corev1.Container(nil), spec.InitContainers...), spec.Containers...)
	for _, container := range containers {
		if container.Name == consts.MainContainerName {
			continue
		}
		limit := container.Resources.Limits[consts.KubeResourceGPUNvidia]
		request := container.Resources.Requests[consts.KubeResourceGPUNvidia]
		if !limit.IsZero() || !request.IsZero() || len(container.Resources.Claims) != 0 {
			return 0, fmt.Errorf("vLLM GPUs must belong exclusively to the main container")
		}
	}

	// The complete TP subgroup must fit the exact, indivisible Pod allocation.
	limit := main.Resources.Limits[consts.KubeResourceGPUNvidia]
	gpus, exact := limit.AsInt64()
	request, requested := main.Resources.Requests[consts.KubeResourceGPUNvidia]
	if !exact || gpus < 1 || (requested && request.Cmp(limit) != 0) {
		return 0, fmt.Errorf("vLLM main requires a positive whole-GPU limit and an equal GPU request when present")
	}
	return gpus, nil
}

// requireEngineGroupLiteralEnv validates an effective, explicitly declared environment value.
// main must not be nil.
func requireEngineGroupLiteralEnv(main *corev1.Container, name, value string) error {
	for _, variable := range main.Env {
		if variable.Name == name {
			if variable.ValueFrom != nil || variable.Value != value {
				return fmt.Errorf("Engine Group main environment %s must explicitly be %q", name, value)
			}
			return nil
		}
	}
	return fmt.Errorf("Engine Group main environment %s=%s must be explicit in the Pod template", name, value)
}

// requireEngineGroupRetentionGate keeps exact release explicit in the user's allocation template.
// template must not be nil.
func requireEngineGroupRetentionGate(template *corev1.PodTemplateSpec) error {
	count := 0
	for _, gate := range template.Spec.ReadinessGates {
		if gate.ConditionType == corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate) {
			count++
		}
	}
	if count != 1 {
		return fmt.Errorf("Engine Group podTemplate must declare exactly one %s readiness gate", consts.EngineGroupRetainedReadinessGate)
	}
	return nil
}
