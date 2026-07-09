/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"context"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoComponentDeploymentValidator validates v1beta1 DynamoComponentDeployment resources.
type DynamoComponentDeploymentValidator struct{}

// NewDynamoComponentDeploymentValidator creates a validator for v1beta1 DynamoComponentDeployment.
func NewDynamoComponentDeploymentValidator() *DynamoComponentDeploymentValidator {
	return &DynamoComponentDeploymentValidator{}
}

// dynamoComponentDeploymentValidation carries DCD-specific request state.
// API values and derived traversal state remain explicit validator arguments.
type dynamoComponentDeploymentValidation struct {
	sharedValidation
}

// Validate performs stateless validation on the v1beta1 DynamoComponentDeployment.
// ctx and dcd must not be nil.
func (v *DynamoComponentDeploymentValidator) Validate(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (admission.Warnings, error) {
	validation := &dynamoComponentDeploymentValidation{
		sharedValidation: sharedValidation{ctx: ctx},
	}

	allErrs := validation.validateDynamoComponentDeployment(dcd)
	alpha, err := alphaDynamoComponentDeploymentForValidation(dcd)
	if err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoComponentDeployment fields: %w", err)
	}
	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentV1alpha1(alpha)...)

	return validation.warnings, invalidDynamoComponentDeploymentError(dcd, allErrs)
}

// ValidateUpdate performs complete validation of an updated v1beta1 DCD and
// compares its state with the previous object.
// ctx, oldDCD, and newDCD must not be nil.
func (v *DynamoComponentDeploymentValidator) ValidateUpdate(
	ctx context.Context,
	oldDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	newDCD *nvidiacomv1beta1.DynamoComponentDeployment,
) (admission.Warnings, error) {
	validation := &dynamoComponentDeploymentValidation{
		sharedValidation: sharedValidation{ctx: ctx},
	}

	allErrs := validation.validateDynamoComponentDeployment(newDCD)
	alpha, err := alphaDynamoComponentDeploymentForValidation(newDCD)
	if err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoComponentDeployment fields: %w", err)
	}
	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentV1alpha1(alpha)...)
	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentUpdate(newDCD, oldDCD)...)
	return validation.warnings, invalidDynamoComponentDeploymentError(newDCD, allErrs)
}

// validateDynamoComponentDeployment validates dcd. dcd must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeployment(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) field.ErrorList {
	return v.validateDynamoComponentDeploymentSpec(&dcd.Spec, field.NewPath("spec"))
}

// validateDynamoComponentDeploymentSpec validates spec. spec and fldPath must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentSpec(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	// Standalone DCDs use neither Grove nor live InferencePool discovery.
	const (
		grovePathway                      = false
		validateInferencePoolAvailability = false
	)
	return v.validateDynamoComponentDeploymentSharedSpec(
		&spec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		grovePathway,
		validateInferencePoolAvailability,
	)
}

// validateDynamoComponentDeploymentUpdate validates an update. newDCD and oldDCD must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentUpdate(
	newDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	oldDCD *nvidiacomv1beta1.DynamoComponentDeployment,
) field.ErrorList {
	return v.validateDynamoComponentDeploymentSpecUpdate(
		&newDCD.Spec,
		&oldDCD.Spec,
		field.NewPath("spec"),
	)
}

// validateDynamoComponentDeploymentSpecUpdate validates a spec update.
// newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentSpecUpdate(
	newSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	oldSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	// Standalone DCD updates preserve direct replica modification.
	const canModifyReplicas = true

	allErrs := field.ErrorList{}
	if newSpec.BackendFramework != oldSpec.BackendFramework {
		v.warn("Changing spec.backendFramework may cause unexpected behavior")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("backendFramework"),
			newSpec.BackendFramework,
			"is immutable and cannot be changed after creation",
		))
	}

	allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecUpdate(
		&newSpec.DynamoComponentDeploymentSharedSpec,
		&oldSpec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		canModifyReplicas,
		nvidiacomv1beta1.DynamoComponentDeploymentGVK.GroupKind(),
	)...)
	return allErrs
}
