/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	runtimefeatures "github.com/ai-dynamo/dynamo/deploy/operator/internal/features/runtime"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type plannerRuntimeContract struct {
	name                   string
	image                  string
	runtimeVersionOverride string
}

// validateRolePodTemplatesPlannerRuntime rejects a DGD shape that an embedded
// pre-1.6 Planner cannot discover. Engine runtime versions are intentionally
// irrelevant: only Planner consumes the operator-projected component facts.
func validateRolePodTemplatesPlannerRuntime(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	componentsPath := fldPath.Child("components")
	rolePaths := make([]*field.Path, 0, 1)
	for i := range spec.Components {
		if dynamo.HasRolePodTemplates(&spec.Components[i]) {
			rolePaths = append(rolePaths, componentsPath.Index(i).Child("roles"))
		}
	}
	if len(rolePaths) == 0 {
		return nil
	}

	planners := make([]plannerRuntimeContract, 0, 1)
	for i := range spec.Components {
		component := &spec.Components[i]
		if component.ComponentType != nvidiacomv1beta1.ComponentTypePlanner {
			continue
		}
		image, _ := runtimeVersionImageAndPath(component, componentsPath.Index(i))
		planners = append(planners, plannerRuntimeContract{
			name:                   component.ComponentName,
			image:                  image,
			runtimeVersionOverride: component.RuntimeVersionOverride,
		})
	}

	detail := rolePodTemplatesPlannerRuntimeErrorDetail(planners)
	if detail == "" {
		return nil
	}

	allErrs := field.ErrorList{}
	for _, rolePath := range rolePaths {
		allErrs = append(allErrs, field.Forbidden(rolePath, detail))
	}
	return allErrs
}

// validateRolePodTemplatesPlannerRuntimeV1Alpha1 preserves source-version
// field paths for v1alpha1 requests converted through the v1beta1 webhook.
func validateRolePodTemplatesPlannerRuntimeV1Alpha1(
	spec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	servicesPath := fldPath.Child("services")
	rolePaths := make([]*field.Path, 0, 1)
	for _, serviceName := range sortedV1Alpha1ServiceNames(spec.Services) {
		if hasRolePodTemplatesV1Alpha1(spec.Services[serviceName]) {
			rolePaths = append(rolePaths, servicesPath.Key(serviceName).Child("roles"))
		}
	}
	if len(rolePaths) == 0 {
		return nil
	}

	planners := make([]plannerRuntimeContract, 0, 1)
	for _, serviceName := range sortedV1Alpha1ServiceNames(spec.Services) {
		service := spec.Services[serviceName]
		if service.ComponentType != consts.ComponentTypePlanner {
			continue
		}
		image, _ := runtimeVersionImageAndPathV1Alpha1(service, servicesPath.Key(serviceName))
		planners = append(planners, plannerRuntimeContract{
			name:                   serviceName,
			image:                  image,
			runtimeVersionOverride: service.RuntimeVersionOverride,
		})
	}

	detail := rolePodTemplatesPlannerRuntimeErrorDetail(planners)
	if detail == "" {
		return nil
	}

	allErrs := field.ErrorList{}
	for _, rolePath := range rolePaths {
		allErrs = append(allErrs, field.Forbidden(rolePath, detail))
	}
	return allErrs
}

func rolePodTemplatesPlannerRuntimeErrorDetail(planners []plannerRuntimeContract) string {
	if len(planners) == 0 {
		return ""
	}

	minimum := runtimefeatures.PlannerDGDComponentStatus.MinRuntimeVersion.String()
	for _, planner := range planners {
		version, err := runtimeversion.Resolve(planner.image, planner.runtimeVersionOverride)
		if err != nil {
			return fmt.Sprintf(
				"role-specific PodTemplates require every planner component to use Dynamo runtime %s or later; planner component %q has no resolvable runtime version: %v",
				minimum,
				planner.name,
				err,
			)
		}
		if !runtimefeatures.PlannerDGDComponentStatus.Enabled(&version) {
			return fmt.Sprintf(
				"role-specific PodTemplates require every planner component to use Dynamo runtime %s or later; planner component %q resolves to %s",
				minimum,
				planner.name,
				version.String(),
			)
		}
	}
	return ""
}
