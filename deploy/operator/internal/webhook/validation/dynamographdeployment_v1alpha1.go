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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoGraphDeploymentV1alpha1 validates dgd using renderedDGD for child names.
// dgd and renderedDGD must not be nil; both old objects are nil on create.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentV1alpha1(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	oldDGD *nvidiacomv1alpha1.DynamoGraphDeployment,
	renderedDGD, oldRenderedDGD *nvidiacomv1beta1.DynamoGraphDeployment,
) field.ErrorList {
	if !hasV1Alpha1CompatibilityFields(dgd) &&
		!v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		return nil
	}
	allErrs := v.validateDynamoGraphDeploymentSpecV1alpha1(
		&dgd.Spec,
		field.NewPath("spec"),
		dgd.Name,
		dgd.Namespace,
	)
	grovePathway, _ := grovePathwayForDynamoGraphDeployment(features.MustGateFrom(v.ctx).Enabled(features.Grove), renderedDGD)
	// Only DCDs render preserved alpha Ingress resources.
	if grovePathway {
		return allErrs
	}
	disaggregatedSet := renderedDGD.Annotations[consts.KubeAnnotationWorkloadProvider] == consts.WorkloadProviderDisaggregatedSet ||
		(renderedDGD.Annotations[consts.KubeAnnotationWorkloadProvider] == "" && strings.EqualFold(renderedDGD.Annotations[consts.KubeAnnotationEnableDisaggregatedSet], "true"))
	workerHash := strings.Repeat("0", dynamo.WorkerHashLength)
	for _, name := range sortedV1Alpha1ServiceNames(dgd.Spec.Services) {
		service := dgd.Spec.Services[name]
		component := renderedDGD.GetComponentByName(name)
		if service == nil || component == nil || component.ManagedByExternalController() ||
			(disaggregatedSet && component.IsMultinode() && dynamo.IsWorkerComponent(string(component.ComponentType))) {
			continue
		}
		var oldIngress *nvidiacomv1alpha1.IngressSpec
		var oldName string
		if oldDGD != nil && oldDGD.Spec.Services[name] != nil {
			oldIngress = oldDGD.Spec.Services[name].Ingress
			oldName = dynamo.GetDCDResourceName(oldRenderedDGD, name, workerHash)
		}
		backendName := dynamo.GetDCDResourceName(renderedDGD, name, workerHash)
		ingressErrs := alphaIngressBackendNameErrors(service.Ingress, oldIngress, backendName, oldName, field.NewPath("spec", "services").Key(name).Child("ingress"))
		if dynamo.IsWorkerComponent(string(component.ComponentType)) {
			for _, err := range ingressErrs {
				err.Detail += "; worker hash is shown as zeros"
			}
		}
		allErrs = append(allErrs, ingressErrs...)
	}
	return allErrs
}

// validateDynamoGraphDeploymentSpecV1alpha1 validates spec. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSpecV1alpha1(
	spec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
	dgdName string,
	dgdNamespace string,
) field.ErrorList {
	allErrs := field.ErrorList{}
	pvcsPath := fldPath.Child("pvcs")
	for i := range spec.PVCs {
		allErrs = append(allErrs, v.validatePVCV1alpha1(&spec.PVCs[i], pvcsPath.Index(i))...)
	}

	servicesPath := fldPath.Child("services")
	for _, serviceName := range sortedV1Alpha1ServiceNames(spec.Services) {
		service := spec.Services[serviceName]
		servicePath := servicesPath.Key(serviceName)
		dynamoNamespace := nvidiacomv1alpha1.ComputeDynamoNamespace(service.GlobalDynamoNamespace, dgdNamespace, dgdName)
		allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecV1alpha1(
			service,
			servicePath,
			dynamoNamespace,
		)...)
	}
	if v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		allErrs = append(allErrs, validateRolePodTemplatesPlannerRuntimeV1Alpha1(spec, fldPath)...)
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
