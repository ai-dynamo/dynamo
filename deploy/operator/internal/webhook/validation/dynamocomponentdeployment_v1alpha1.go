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
	componentdeployment "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/component"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoComponentDeploymentV1alpha1 validates newDCD; oldDCD is nil on create.
// newDCD must be non-nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentV1alpha1(
	newDCD, oldDCD *nvidiacomv1alpha1.DynamoComponentDeployment,
	orderedEnvironment bool,
) field.ErrorList {
	if !hasDynamoComponentDeploymentV1alpha1CompatibilityFields(newDCD) &&
		!v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		return nil
	}
	var oldEnv []corev1.EnvVar
	if oldDCD != nil && v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		oldEnv = oldDCD.Spec.Envs
	}
	allErrs := v.validateDynamoComponentDeploymentSpecV1alpha1(
		&newDCD.Spec,
		field.NewPath("spec"),
		newDCD.GetDynamoNamespace(),
		oldEnv,
		orderedEnvironment,
	)

	// The source-only Ingress backend combines metadata and compatibility spec fields.
	ingress := newDCD.Spec.Ingress
	if ingress == nil || !componentdeployment.IngressEnabled(ingress.Enabled, ingress.IngressControllerClassName) {
		return allErrs
	}
	rendered := dynamo.GenerateComponentIngress(v.ctx, newDCD.Name, newDCD.Namespace, dynamo.IngressSpec{IngressControllerClassName: ingress.IngressControllerClassName})
	backend := &rendered.Spec.Rules[0].HTTP.Paths[0].Backend
	var oldBackend *networkingv1.IngressBackend
	if oldDCD != nil && oldDCD.Spec.Ingress != nil && componentdeployment.IngressEnabled(oldDCD.Spec.Ingress.Enabled, oldDCD.Spec.Ingress.IngressControllerClassName) {
		oldIngress := dynamo.GenerateComponentIngress(v.ctx, oldDCD.Name, oldDCD.Namespace, dynamo.IngressSpec{IngressControllerClassName: oldDCD.Spec.Ingress.IngressControllerClassName})
		oldBackend = &oldIngress.Spec.Rules[0].HTTP.Paths[0].Backend
	}
	return append(allErrs, v.validateIngressBackend(backend, oldBackend, field.NewPath("spec", "ingress"))...)
}

// validateDynamoComponentDeploymentSpecV1alpha1 validates spec. spec and fldPath must not be nil.
// oldEnv is nil on create, otherwise it is the previous source-version envs list.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentSpecV1alpha1(
	spec *nvidiacomv1alpha1.DynamoComponentDeploymentSpec,
	fldPath *field.Path,
	dynamoNamespace string,
	oldEnv []corev1.EnvVar,
	orderedEnvironment bool,
) field.ErrorList {
	return v.validateDynamoComponentDeploymentSharedSpecV1alpha1(
		&spec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		dynamoNamespace,
		oldEnv,
		orderedEnvironment,
	)
}
