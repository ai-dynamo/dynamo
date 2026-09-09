/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path"
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
	conductorCount := 0
	for index := range dgd.Spec.Components {
		component := &dgd.Spec.Components[index]
		if !component.IsLPX() {
			continue
		}
		componentPath := componentsPath.Index(index)
		conductor := component.ComponentRole(dynamov1beta1.ComponentRoleLeader)
		if conductor != nil {
			conductorCount++
		}
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
		} else if replicas != 1 {
			allErrs = append(allErrs, field.Invalid(componentPath.Child("replicas"), replicas, "shared target replicas must be one"))
		}
	}
	if componentCount == 2 && conductorCount != 1 {
		allErrs = append(allErrs, field.Forbidden(componentsPath, "LPX components must declare exactly one leader role"))
	}
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
		if role.PodTemplate == nil {
			continue
		}
		rolePath := componentPath.Child("roles").Index(index)
		main, mainPath, errs := validateSelectedRuntimeContainer(role.PodTemplate, "LPX "+role.Name, rolePath)
		allErrs = append(allErrs, errs...)
		if main != nil && role.Name == dynamov1beta1.ComponentRoleWorker {
			allErrs = append(allErrs, validateAllocationInjectionTargetFields(main, mainPath)...)
		}
		allErrs = append(allErrs, validateLPXRolePlacement(&role.PodTemplate.Spec, rolePath.Child("podTemplate", "spec"), role.Name == dynamov1beta1.ComponentRoleWorker)...)
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
) (*corev1.Container, *field.Path, field.ErrorList) {
	containersPath := rolePath.Child("podTemplate", "spec", "containers")
	for index := range template.Spec.Containers {
		container := &template.Spec.Containers[index]
		if container.Name != commonconsts.MainContainerName {
			continue
		}
		containerPath := containersPath.Index(index)
		return container, containerPath, nil
	}
	return nil, nil, field.ErrorList{
		field.Required(
			containersPath,
			fmt.Sprintf("%s component requires a %q runtime container", role, commonconsts.MainContainerName),
		),
	}
}

// validateAllocationInjectionTargetFields requires nonnil container and containerPath.
func validateAllocationInjectionTargetFields(container *corev1.Container, containerPath *field.Path) field.ErrorList {
	// Classify Command and Args once before reporting violations in contract order.
	var hasAllocation, terminatesArguments, hasShell [2]bool
	for vector, values := range [2][]string{container.Command, container.Args} {
		for _, value := range values {
			hasAllocation[vector] = hasAllocation[vector] || value == "--allocation" ||
				strings.HasPrefix(value, "--allocation=")
			terminatesArguments[vector] = terminatesArguments[vector] || value == "--"
			if !hasShell[vector] {
				switch path.Base(value) {
				case "ash", "bash", "dash", "ksh", "sh", "zsh":
					hasShell[vector] = true
				}
			}
		}
	}

	allErrs := field.ErrorList{}
	if hasAllocation[0] {
		allErrs = append(allErrs, field.Forbidden(containerPath.Child("command"), "selected LPX main container must not set --allocation; Dynamo renders the immutable Agent clique list"))
	}
	if hasAllocation[1] {
		allErrs = append(allErrs, field.Forbidden(containerPath.Child("args"), "selected LPX main container must not set --allocation; Dynamo renders the immutable Agent clique list"))
	}
	if terminatesArguments[0] {
		allErrs = append(allErrs, field.Forbidden(containerPath.Child("command"), "selected LPX main container must not terminate arguments before Dynamo appends --allocation"))
	}
	if terminatesArguments[1] {
		allErrs = append(allErrs, field.Forbidden(containerPath.Child("args"), "selected LPX main container must not terminate arguments before Dynamo appends --allocation"))
	}
	if hasShell[0] {
		allErrs = append(allErrs, field.Forbidden(containerPath.Child("command"), "selected LPX main container cannot use a shell because Dynamo appends --allocation"))
	}
	if hasShell[1] {
		allErrs = append(allErrs, field.Forbidden(containerPath.Child("args"), "selected LPX main container cannot use a shell because Dynamo appends --allocation"))
	}
	return allErrs
}
