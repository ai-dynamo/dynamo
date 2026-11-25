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
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoGraphDeploymentV1alpha1 supplements storage validation with Alpha-only
// PVC/component fields and env-list/schema distinctions lost by conversion.
// Shared semantics remain in the storage-version owner. dgd must not be nil.
// oldDGD is nil on create, otherwise it is the previous source-version object.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentV1alpha1(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	oldDGD *nvidiacomv1alpha1.DynamoGraphDeployment,
) field.ErrorList {
	if !hasV1Alpha1CompatibilityFields(dgd) &&
		v.runtimeVersionSource != runtimeVersionSourceV1Alpha1 {
		return nil
	}
	var oldSpec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec
	if oldDGD != nil {
		oldSpec = &oldDGD.Spec
	}
	return v.validateDynamoGraphDeploymentSpecV1alpha1(
		&dgd.Spec,
		field.NewPath("spec"),
		dgd.Name,
		dgd.Namespace,
		oldSpec,
		compatibility.OrderedEnvironmentVariables.Enabled(dgd.Annotations),
	)
}

// validateDynamoGraphDeploymentSpecV1alpha1 visits Alpha-only PVC declarations and
// component settings, plus source env-list/schema gaps unavailable in storage validation.
// spec and fldPath must not be nil.
// oldSpec is nil on create, otherwise it contains the previous source-version lists.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSpecV1alpha1(
	spec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
	dgdName string,
	dgdNamespace string,
	oldSpec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
	orderedEnvironment bool,
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
		var oldEnv []corev1.EnvVar
		if oldSpec != nil && oldSpec.Services[serviceName] != nil {
			oldEnv = oldSpec.Services[serviceName].Envs
		}
		allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecV1alpha1(
			service,
			servicePath,
			dynamoNamespace,
			oldEnv,
			orderedEnvironment,
		)...)
	}
	if v.runtimeVersionSource == runtimeVersionSourceV1Alpha1 {
		var oldEnv []corev1.EnvVar
		if oldSpec != nil {
			oldEnv = oldSpec.Envs
		}
		allErrs = append(allErrs, v.validateEnvVars(spec.Envs, oldEnv, fldPath.Child("envs"), orderedEnvironment)...)
	}
	return allErrs
}

// validateDynamoGraphDeploymentSpecUpdateV1alpha1 freezes Alpha-only resource annotations,
// labels, subcomponent type, and Ingress on updates submitted through v1alpha1.
// The storage-version traversal already checks every represented component change;
// ignored compatibility settings remain editable. Do not compare whole versioned specs here.
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
		if !exists || !newService.IsLPX() || features.MustGateFrom(v.ctx).Enabled(features.LPX) {
			continue
		}

		// These source-only inputs affect rendered LPX resources or their revision.
		servicePath := servicesPath.Key(serviceName)
		if !apiequality.Semantic.DeepEqual(newService.Annotations, oldService.Annotations) {
			allErrs = append(allErrs, field.Forbidden(
				servicePath.Child("annotations"),
				"LPX components require lpx.enabled=true",
			))
		}
		if !apiequality.Semantic.DeepEqual(newService.Labels, oldService.Labels) {
			allErrs = append(allErrs, field.Forbidden(
				servicePath.Child("labels"),
				"LPX components require lpx.enabled=true",
			))
		}
		if newService.SubComponentType != oldService.SubComponentType {
			allErrs = append(allErrs, field.Forbidden(
				servicePath.Child("subComponentType"),
				"LPX components require lpx.enabled=true",
			))
		}
		if !apiequality.Semantic.DeepEqual(newService.Ingress, oldService.Ingress) {
			allErrs = append(allErrs, field.Forbidden(
				servicePath.Child("ingress"),
				"LPX components require lpx.enabled=true",
			))
		}
	}
	return allErrs
}

// validatePVCV1alpha1 checks the name of a source-only graph PVC declaration.
// The Alpha schema requires presence but permits an empty name. pvc and fldPath must not be nil.
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
