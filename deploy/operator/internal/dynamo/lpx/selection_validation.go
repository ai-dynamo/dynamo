/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"strings"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

// ValidateSelectedIntent validates the snapshot-independent selected-LPX
// requirements beyond admission's structural checks before controller downloads
// and selected-workload projection.
// dgd must be non-nil.
func ValidateSelectedIntent(dgd *dynamov1beta1.DynamoGraphDeployment) field.ErrorList {
	_, allErrs := validateSelectedIntent(dgd)
	return allErrs
}

func validateSelectedIntent(dgd *dynamov1beta1.DynamoGraphDeployment) ([]*dynamov1beta1.DynamoComponentDeploymentSharedSpec, field.ErrorList) {
	specPath := field.NewPath("spec")
	if !dgd.HasLPXComponent() {
		return nil, field.ErrorList{field.Forbidden(
			specPath.Child("components"),
			"DynamoGraphDeployment has no LPX component",
		)}
	}
	allErrs := field.ErrorList{}
	if dgd.Spec.TopologyConstraint != nil {
		allErrs = append(allErrs, field.Forbidden(
			specPath.Child("topologyConstraint"),
			"selected LPX does not support Grove topologyConstraint",
		))
	}
	if dgd.Spec.ProviderOverride != nil {
		allErrs = append(allErrs, field.Forbidden(
			specPath.Child("providerOverride"),
			"selected LPX does not support Grove topology overrides on the deployment",
		))
	}
	components := Components(dgd)
	allErrs = append(allErrs, validateLPXComposition(dgd, len(components))...)
	allErrs = append(allErrs, ValidateAgentContainerNames(dgd)...)

	// Keep errors tied to authored component and role indices rather than runtime order.
	for index := range dgd.Spec.Components {
		component := &dgd.Spec.Components[index]
		if !component.IsLPX() {
			continue
		}
		componentPath := field.NewPath("spec", "components").Index(index)
		allErrs = append(allErrs, validateSelectedLPXComponent(component, componentPath)...)
	}
	return components, allErrs
}

// validateLPXComposition bounds the shared runtime before model expansion or acquisition.
func validateLPXComposition(dgd *dynamov1beta1.DynamoGraphDeployment, componentCount int) field.ErrorList {
	componentsPath := field.NewPath("spec", "components")
	if componentCount < 1 || componentCount > 2 {
		return field.ErrorList{field.Forbidden(componentsPath, "requires one complete LPX component or a shared draft and target pair")}
	}
	allErrs := field.ErrorList{}
	for index := range dgd.Spec.Components {
		component := &dgd.Spec.Components[index]
		if !component.IsLPX() {
			continue
		}
		componentPath := componentsPath.Index(index)
		conductor := component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor)
		if componentCount != 2 {
			continue
		}
		// Draft fanout is per model instance; the target owns one shared serving runtime.
		replicas := ptr.Deref(component.Replicas, 1)
		if conductor == nil {
			if replicas < 1 || replicas > maxSpecDecodeNumDrafts {
				allErrs = append(allErrs, field.Invalid(componentPath.Child("replicas"), replicas, "draft replicas must be between 1 and 8"))
			}
			if component.ModelRef != nil {
				allErrs = append(allErrs, field.Forbidden(componentPath.Child("modelRef"), "the shared target owns the serving endpoint"))
			}
			// Grove persists its immutable default of 1 even for drafts.
			if ptr.Deref(component.MinAvailable, 1) != 1 {
				allErrs = append(allErrs, field.Forbidden(componentPath.Child("minAvailable"), "draft minAvailable must be omitted or 1; the shared target owns minimum availability"))
			}
		} else if replicas != 1 {
			allErrs = append(allErrs, field.Invalid(componentPath.Child("replicas"), replicas, "shared target replicas must be one"))
		}
	}
	allErrs = append(allErrs, ValidateConductorRoles(&dgd.Spec, componentsPath, componentsPath.Index)...)
	return allErrs
}

// validateSelectedLPXComponent validates inputs required by the existing runtime.
// Shared role admission owns structure; composition validation bounds materialization.
func validateSelectedLPXComponent(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec, componentPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if component.Experimental != nil && component.Experimental.Checkpoint != nil && component.Experimental.Checkpoint.Enabled {
		allErrs = append(allErrs, field.Forbidden(componentPath.Child("experimental", "checkpoint"),
			fmt.Sprintf("selected LPX component %q does not support checkpointing", component.ComponentName)))
	}
	if component.TopologyConstraint != nil {
		allErrs = append(allErrs, field.Forbidden(componentPath.Child("topologyConstraint"),
			fmt.Sprintf("selected LPX component %q does not support Grove topologyConstraint", component.ComponentName)))
	}
	if component.ProviderOverride != nil {
		allErrs = append(allErrs, field.Forbidden(componentPath.Child("providerOverride"), "LPX component does not support Grove topology overrides"))
	}
	if component.IsMultinode() {
		allErrs = append(allErrs, field.Forbidden(componentPath.Child("multinode"), "LPX component cannot use multinode"))
	}
	config := component.LPX
	configPath := componentPath.Child("lpx")
	if strings.TrimSpace(config.BuildID) == "" {
		allErrs = append(allErrs, field.Required(configPath.Child("buildId"), "LPX component requires a buildId"))
	}
	for index, role := range component.Roles {
		rolePath := componentPath.Child("roles").Index(index)
		if role.PodTemplate == nil {
			continue
		}
		allErrs = append(allErrs, validateSelectedRuntimeContainer(role.PodTemplate, "LPX "+role.Name, rolePath)...)
		allErrs = append(allErrs, validateLPXRolePlacement(&role.PodTemplate.Spec, rolePath.Child("podTemplate", "spec"), role.Name == dynamov1beta1.ComponentRoleLPXAgent)...)
	}
	return allErrs
}

// ValidateConductorRoles requires one explicit serving template across LPX components.
// Draft components remain Agent-only. Arguments must be non-nil; spec is not mutated.
// componentsPath and componentPath locate the collection and each component in the source API.
func ValidateConductorRoles(spec *dynamov1beta1.DynamoGraphDeploymentSpec, componentsPath *field.Path, componentPath func(int) *field.Path) field.ErrorList {
	// Inspect authored roles so admission and preflight enforce the same snapshot-independent shape.
	allErrs := field.ErrorList{}
	hasLPX, conductorCount := false, 0
	for componentIndex := range spec.Components {
		component := &spec.Components[componentIndex]
		if !component.IsLPX() {
			continue
		}
		hasLPX = true
		for roleIndex, role := range component.Roles {
			if role.Name != dynamov1beta1.ComponentRoleLPXConductor {
				continue
			}
			conductorCount++
			if role.PodTemplate == nil {
				rolePath := componentPath(componentIndex).Child("roles").Index(roleIndex)
				allErrs = append(allErrs, field.Required(rolePath.Child("podTemplate"), "LPX conductor requires an explicit podTemplate"))
			}
		}
	}
	if hasLPX && conductorCount != 1 {
		allErrs = append(allErrs, field.Forbidden(componentsPath, "LPX components must declare exactly one conductor role"))
	}
	return allErrs
}

// ValidateAgentContainerNames reserves the Agent identity in authored LPX agent
// templates. Conductor identities depend on the immutable build's execution mode.
// dgd must be non-nil and is not mutated.
func ValidateAgentContainerNames(dgd *dynamov1beta1.DynamoGraphDeployment) field.ErrorList {
	// Every LPX agent template renames main independently of the selected build.
	allErrs := field.ErrorList{}
	for componentIndex := range dgd.Spec.Components {
		component := &dgd.Spec.Components[componentIndex]
		if !component.IsLPX() {
			continue
		}

		// Keep collision errors on each authored agent template.
		for roleIndex, role := range component.Roles {
			if role.Name != dynamov1beta1.ComponentRoleLPXAgent || role.PodTemplate == nil {
				continue
			}
			podSpecPath := field.NewPath("spec", "components").Index(componentIndex).Child("roles").Index(roleIndex).Child("podTemplate", "spec")
			allErrs = append(allErrs, validateRolePodSpecContainerNames(&role.PodTemplate.Spec, podSpecPath, lpuAgentContainerName)...)
		}
	}
	return allErrs
}

// validateRolePodSpecContainerNames checks both lists that share the Pod's name space.
// spec and fldPath must be non-nil.
func validateRolePodSpecContainerNames(spec *corev1.PodSpec, fldPath *field.Path, reservedName string) field.ErrorList {
	// Both lists must avoid the materialized role container name.
	allErrs := field.ErrorList{}
	for _, group := range []struct {
		name       string
		containers []corev1.Container
	}{
		{"containers", spec.Containers},
		{"initContainers", spec.InitContainers},
	} {
		for containerIndex, container := range group.containers {
			if container.Name == reservedName {
				allErrs = append(allErrs, field.Forbidden(
					fldPath.Child(group.name).Index(containerIndex).Child("name"),
					fmt.Sprintf("LPX reserves %q for the materialized role container", container.Name),
				))
			}
		}
	}
	return allErrs
}

// validateLPXRolePlacement protects controller-owned addressing and scheduling.
// Required node affinity remains a supported input to LPX Agent placement.
func validateLPXRolePlacement(spec *corev1.PodSpec, fldPath *field.Path, agent bool) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.SchedulerName != "" && spec.SchedulerName != corev1.DefaultSchedulerName {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("schedulerName"), "LPX owns role scheduler selection"))
	}
	for _, entry := range []struct{ name, value string }{
		{"hostname", spec.Hostname}, {"subdomain", spec.Subdomain}, {"nodeName", spec.NodeName},
	} {
		if entry.value != "" {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child(entry.name), "LPX owns role addressing and placement"))
		}
	}
	if len(spec.TopologySpreadConstraints) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("topologySpreadConstraints"), "LPX owns role placement"))
	}
	if !agent {
		return allErrs
	}
	if len(spec.NodeSelector) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("nodeSelector"), "LPX exclusively owns Agent node selection"))
	}
	if a := spec.Affinity; a != nil &&
		((a.NodeAffinity != nil && len(a.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution) != 0) ||
			(a.PodAffinity != nil && (len(a.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 || len(a.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution) != 0)) ||
			(a.PodAntiAffinity != nil && (len(a.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 || len(a.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution) != 0))) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("affinity"), "node-local LPX supports only required nodeAffinity"))
	}
	if len(spec.SchedulingGates) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("schedulingGates"), "Grove and LPX own Agent scheduling gates"))
	}
	if len(spec.ResourceClaims) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("resourceClaims"), "node-local LPX Agents cannot use ResourceClaims"))
	}
	return allErrs
}

// validateSelectedRuntimeContainer validates a present template at its authored role path.
func validateSelectedRuntimeContainer(
	template *corev1.PodTemplateSpec,
	role string,
	rolePath *field.Path,
) field.ErrorList {
	containersPath := rolePath.Child("podTemplate", "spec", "containers")
	for index := range template.Spec.Containers {
		container := &template.Spec.Containers[index]
		if container.Name != commonconsts.MainContainerName {
			continue
		}
		return nil
	}
	return field.ErrorList{
		field.Required(
			containersPath,
			fmt.Sprintf("%s component requires a %q runtime container", role, commonconsts.MainContainerName),
		),
	}
}
