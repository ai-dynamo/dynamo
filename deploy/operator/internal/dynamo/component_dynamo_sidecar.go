// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"fmt"
	"slices"
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

// ValidateComponentSidecarConfiguration validates runtime init-container declarations
// in the component PodTemplate and non-LPX role PodTemplates for admission and rendering.
// LPX admission validates roles through validateComponentRoleSpec; LPX rendering
// passes the selected role template as component.PodTemplate.
// Components without a runtime init container have no sidecar-specific requirements.
// component and fldPath must not be nil.
func ValidateComponentSidecarConfiguration(component *v1beta1.DynamoComponentDeploymentSharedSpec, fldPath *field.Path, backendFramework string) field.ErrorList {
	var allErrs field.ErrorList
	var hasSidecar bool
	detail := fmt.Sprintf("component %q with a runtime init container", component.ComponentName)

	// Validate each authored container at its original API path, including invalid role layouts.
	templates := make([]*corev1.PodTemplateSpec, 1, 1+len(component.Roles))
	templates[0] = component.PodTemplate
	paths := make([]*field.Path, 1, 1+len(component.Roles))
	paths[0] = fldPath.Child("podTemplate")
	// Avoid duplicate LPX admission errors; validateComponentRoleSpec owns its role checks.
	if !component.IsLPX() {
		for i := range component.Roles {
			templates = append(templates, component.Roles[i].PodTemplate)
			paths = append(paths, fldPath.Child("roles").Index(i).Child("podTemplate"))
		}
	}
	for i, template := range templates {
		if template == nil {
			continue
		}
		index := slices.IndexFunc(template.Spec.InitContainers, func(c corev1.Container) bool { return c.Name == commonconsts.RuntimeContainerName })
		if index < 0 {
			continue
		}
		hasSidecar = true
		runtime := &template.Spec.InitContainers[index]
		runtimePath := paths[i].Child("spec", "initContainers").Index(index)
		if !IsWorkerComponent(string(component.ComponentType)) && !component.IsLPX() {
			allErrs = append(allErrs, field.Forbidden(runtimePath.Child("name"), "is supported only for worker, prefill, decode, and LPX components"))
		}
		if runtime.RestartPolicy == nil || *runtime.RestartPolicy != corev1.ContainerRestartPolicyAlways {
			allErrs = append(allErrs, field.Invalid(runtimePath.Child("restartPolicy"), ptr.Deref(runtime.RestartPolicy, ""), "must be Always for "+detail))
		}
	}

	// Components without a runtime declaration have no further sidecar requirements.
	if !hasSidecar {
		return allErrs
	}

	// Every native engine needs an explicit image, even a compute-only worker.
	for i, template := range templates {
		if template == nil {
			continue
		}
		index := slices.IndexFunc(template.Spec.Containers, func(c corev1.Container) bool { return c.Name == commonconsts.MainContainerName })
		containerPath := paths[i].Child("spec", "containers")
		if index < 0 {
			allErrs = append(allErrs, field.Required(containerPath, "main engine container is required for "+detail))
		} else if template.Spec.Containers[index].Image == "" {
			allErrs = append(allErrs, field.Required(containerPath.Index(index).Child("image"), "engine image is required for "+detail))
		}
	}

	// Multinode sidecars require complete role templates; users own engine launches and placement.
	if component.Multinode != nil {
		switch BackendFramework(backendFramework) {
		case BackendFrameworkSGLang, BackendFrameworkVLLM:
			if !HasCompleteMultinodeRolePodTemplates(component) {
				allErrs = append(allErrs, field.Required(fldPath.Child("roles"), "multinode sidecars require complete leader and worker podTemplates"))
			} else if GetComponentDynamoSidecar(component) == nil {
				allErrs = append(allErrs, field.Required(fldPath.Child("roles"), "the leader template must declare runtime to select sidecar mode"))
			}
		default:
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("multinode"), "Dynamo sidecar multinode support currently requires backendFramework: sglang or vllm"))
		}
	}

	// Existing incompatibilities apply to the component regardless of template source.
	if component.Experimental != nil {
		unsupportedDetail := "is not currently supported for " + detail + "; support is planned for a future release"
		if component.Experimental.Checkpoint != nil && component.Experimental.Checkpoint.Enabled {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("experimental", "checkpoint", "enabled"), unsupportedDetail))
		}
		if component.Experimental.GPUMemoryService != nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("experimental", "gpuMemoryService"), unsupportedDetail))
		}
		if component.Experimental.Failover != nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("experimental", "failover"), unsupportedDetail))
		}
	}
	return allErrs
}

// dynamoSidecarBaseContainer configures the runtime independently of engine health,
// model loading, inference canaries, and engine-owned NIXL telemetry.
func dynamoSidecarBaseContainer(context ComponentContext) corev1.Container {
	// Preserve the sidecar image entrypoint unless the user provides a command.
	container := (&BaseComponentDefaults{}).getCommonContainer(context)
	container.Command = nil
	container.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
	container.Ports = []corev1.ContainerPort{{
		Name: commonconsts.DynamoSystemPortName, ContainerPort: int32(commonconsts.DynamoSystemPort), Protocol: corev1.ProtocolTCP,
	}}
	container.Env = append(container.Env,
		corev1.EnvVar{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(commonconsts.DynamoSystemPort)},
	)

	// Rollout isolation belongs to the worker runtime even though the engine is main.
	if context.WorkerHashSuffix != "" {
		container.Env = append(container.Env, corev1.EnvVar{Name: commonconsts.DynamoNamespaceWorkerSuffixEnvVar, Value: context.WorkerHashSuffix})
	}

	// Startup covers only the independent HTTP listener, not engine model loading.
	container.StartupProbe = &corev1.Probe{
		HTTPGet:       &corev1.HTTPGetAction{Path: "/live", Port: intstr.FromString(commonconsts.DynamoSystemPortName)},
		PeriodSeconds: 2, TimeoutSeconds: 1, FailureThreshold: 30,
	}
	container.LivenessProbe = &corev1.Probe{
		HTTPGet:       &corev1.HTTPGetAction{Path: "/live", Port: intstr.FromString(commonconsts.DynamoSystemPortName)},
		PeriodSeconds: 5, TimeoutSeconds: 4, FailureThreshold: 3,
	}
	container.ReadinessProbe = &corev1.Probe{
		HTTPGet:       &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromString(commonconsts.DynamoSystemPortName)},
		PeriodSeconds: 5, TimeoutSeconds: 4, FailureThreshold: 3,
	}
	return container
}

// findRuntimeInitContainer returns the declared sidecar, or nil when absent.
// podSpec must not be nil.
func findRuntimeInitContainer(podSpec *corev1.PodSpec) *corev1.Container {
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == commonconsts.RuntimeContainerName {
			return &podSpec.InitContainers[i]
		}
	}
	return nil
}

// mergeDynamoSidecarDefaults merges defaults into the selected runtime init container.
// runtime must point to the declared sidecar and must not be nil.
func mergeDynamoSidecarDefaults(runtime *corev1.Container, context ComponentContext) error {
	// User configuration overrides defaults, including entire probe handlers.
	base := dynamoSidecarBaseContainer(context)
	if err := mergeContainerByName(&base, runtime); err != nil {
		return fmt.Errorf("merge runtime init container %q: %w", runtime.Name, err)
	}
	*runtime = base
	return nil
}
