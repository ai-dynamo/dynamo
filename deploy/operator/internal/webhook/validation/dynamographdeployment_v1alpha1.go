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
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoGraphDeploymentV1alpha1 validates dgd. dgd must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentV1alpha1(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) field.ErrorList {
	if !hasV1Alpha1CompatibilityFields(dgd) {
		return nil
	}
	return v.validateDynamoGraphDeploymentSpecV1alpha1(
		&dgd.Spec,
		field.NewPath("spec"),
		dgd.Name,
		dgd.Namespace,
	)
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
