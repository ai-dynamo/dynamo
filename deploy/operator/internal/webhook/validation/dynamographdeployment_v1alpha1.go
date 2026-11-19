/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package validation

import (
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	disaggregatedset "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/disaggregatedset"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoGraphDeploymentV1alpha1 validates newDGD; oldDGD is nil on create.
// newDGD must be non-nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentV1alpha1(
	newDGD, oldDGD *nvidiacomv1alpha1.DynamoGraphDeployment,
) field.ErrorList {
	if !hasV1Alpha1CompatibilityFields(newDGD) &&
		!v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		return nil
	}
	var oldSpec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec
	if oldDGD != nil {
		oldSpec = &oldDGD.Spec
	}
	allErrs := v.validateDynamoGraphDeploymentSpecV1alpha1(
		&newDGD.Spec,
		oldSpec,
		field.NewPath("spec"),
		newDGD.Name,
		newDGD.Namespace,
	)
	return append(allErrs, v.validateDynamoGraphDeploymentServicesV1alpha1(newDGD, oldDGD, field.NewPath("spec", "services"))...)
}

// validateDynamoGraphDeploymentSpecV1alpha1 validates newSpec. newSpec and fldPath must be non-nil; oldSpec is nil on create.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSpecV1alpha1(
	newSpec, oldSpec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
	dgdName string,
	dgdNamespace string,
) field.ErrorList {
	allErrs := field.ErrorList{}
	pvcsPath := fldPath.Child("pvcs")
	for i := range newSpec.PVCs {
		allErrs = append(allErrs, v.validatePVCV1alpha1(&newSpec.PVCs[i], pvcsPath.Index(i))...)
	}

	servicesPath := fldPath.Child("services")
	for _, serviceName := range sortedV1Alpha1ServiceNames(newSpec.Services) {
		service := newSpec.Services[serviceName]
		servicePath := servicesPath.Key(serviceName)
		if oldSpec == nil || oldSpec.Services[serviceName] == nil {
			if messages := k8svalidation.IsDNS1123Label(strings.ToLower(serviceName)); len(messages) > 0 {
				allErrs = append(allErrs, field.Invalid(servicePath, serviceName, "component name is invalid: "+strings.Join(messages, "; ")))
			}
		}
		dynamoNamespace := nvidiacomv1alpha1.ComputeDynamoNamespace(service.GlobalDynamoNamespace, dgdNamespace, dgdName)
		allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecV1alpha1(
			service,
			servicePath,
			dynamoNamespace,
		)...)
	}
	if v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		allErrs = append(allErrs, validateRolePodTemplatesPlannerRuntimeV1Alpha1(newSpec, fldPath)...)
	}
	return allErrs
}

// validateDynamoGraphDeploymentSpecUpdateV1alpha1 validates source-version runtime fields on update.
// newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSpecUpdateV1alpha1(
	newSpec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
	oldSpec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	servicesPath := fldPath.Child("services")
	for _, serviceName := range sortedV1Alpha1ServiceNames(newSpec.Services) {
		newService := newSpec.Services[serviceName]
		oldService, exists := oldSpec.Services[serviceName]
		if !exists {
			continue
		}
		allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecUpdateV1alpha1(
			newService,
			oldService,
			servicesPath.Key(serviceName),
		)...)
	}
	return allErrs
}

// validatePVCV1alpha1 validates pvc. pvc and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validatePVCV1alpha1(
	pvc *nvidiacomv1alpha1.PVC,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if pvc.Name == nil || *pvc.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "is required"))
	}
	return allErrs
}

// validateDynamoGraphDeploymentServicesV1alpha1 checks source-only Ingress backend identities.
// newDGD and fldPath must be non-nil; oldDGD is nil on create.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentServicesV1alpha1(newDGD, oldDGD *nvidiacomv1alpha1.DynamoGraphDeployment, fldPath *field.Path) field.ErrorList {
	renderedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := newDGD.ConvertTo(renderedDGD); err != nil {
		return field.ErrorList{field.Forbidden(fldPath, "cannot convert source services for rendering: "+err.Error())}
	}
	grovePathway, _ := grovePathwayForDynamoGraphDeployment(features.MustGateFrom(v.ctx).Enabled(features.Grove), renderedDGD)
	if grovePathway {
		return nil
	}

	// The storage traversal reports lowering failures; compatibility checks consume only real child objects.
	workerHash := "00000000"
	deployments, err := dynamo.GenerateDynamoComponentsDeployments(renderedDGD, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: workerHash})
	if err != nil {
		return nil
	}
	var oldDeployments map[string]*nvidiacomv1beta1.DynamoComponentDeployment
	var oldRenderedDGD *nvidiacomv1beta1.DynamoGraphDeployment
	if oldDGD != nil {
		oldRenderedDGD = &nvidiacomv1beta1.DynamoGraphDeployment{}
		if err := oldDGD.ConvertTo(oldRenderedDGD); err == nil {
			oldRenderedDGD = oldDynamoGraphDeploymentWithObservedProvider(renderedDGD, oldRenderedDGD)
			oldGrove, _ := grovePathwayForDynamoGraphDeployment(features.MustGateFrom(v.ctx).Enabled(features.Grove), oldRenderedDGD)
			if !oldGrove {
				oldDeployments, _ = dynamo.GenerateDynamoComponentsDeployments(oldRenderedDGD, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: workerHash})
			}
		}
	}
	var allErrs field.ErrorList
	ds := renderedDGD.Annotations[consts.KubeAnnotationWorkloadProvider] == consts.WorkloadProviderDisaggregatedSet || (renderedDGD.Annotations[consts.KubeAnnotationWorkloadProvider] == "" && strings.EqualFold(renderedDGD.Annotations[consts.KubeAnnotationEnableDisaggregatedSet], "true"))
	for _, name := range sortedV1Alpha1ServiceNames(newDGD.Spec.Services) {
		dcd := deployments[name]
		if dcd.Spec.ManagedByExternalController() || (ds && disaggregatedset.Owns(&dcd.Spec.DynamoComponentDeploymentSharedSpec)) {
			continue
		}
		alpha, err := alphaDynamoComponentDeploymentForValidation(dcd)
		if err != nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Key(name), "cannot project rendered component to v1alpha1: "+err.Error()))
			continue
		}
		var oldAlpha *nvidiacomv1alpha1.DynamoComponentDeployment
		if oldDCD := oldDeployments[name]; oldDCD != nil {
			oldDS := oldRenderedDGD.Annotations[consts.KubeAnnotationWorkloadProvider] == consts.WorkloadProviderDisaggregatedSet || (oldRenderedDGD.Annotations[consts.KubeAnnotationWorkloadProvider] == "" && strings.EqualFold(oldRenderedDGD.Annotations[consts.KubeAnnotationEnableDisaggregatedSet], "true"))
			if !(oldDS && disaggregatedset.Owns(&oldDCD.Spec.DynamoComponentDeploymentSharedSpec)) {
				oldAlpha, _ = alphaDynamoComponentDeploymentForValidation(oldDCD)
			}
		}
		errors := v.validateDynamoComponentDeploymentMetadataV1alpha1(alpha, oldAlpha, fldPath.Key(name).Child("ingress"))
		if dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
			for _, err := range errors {
				err.Detail += "; worker hash is shown as zeros"
			}
		}
		allErrs = append(allErrs, errors...)
	}
	return allErrs
}
