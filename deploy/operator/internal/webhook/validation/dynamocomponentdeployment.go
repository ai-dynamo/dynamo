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
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
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
	return v.validate(ctx, dcd, runtimeVersionSourceV1Beta1)
}

func (v *DynamoComponentDeploymentValidator) validate(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	runtimeVersionSource runtimeVersionValidationSource,
) (admission.Warnings, error) {
	validation := &dynamoComponentDeploymentValidation{
		sharedValidation: sharedValidation{
			ctx:                                ctx,
			runtimeVersionSource:               runtimeVersionSource,
			allowMissingRuntimeVersionOverride: true,
		},
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
// ctx, oldDCD, and newDCD must not be nil. runtimeVersionSource identifies the request's source API.
func (v *DynamoComponentDeploymentValidator) ValidateUpdate(
	ctx context.Context,
	oldDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	newDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	canModifyReplicas bool,
	runtimeVersionSource runtimeVersionValidationSource,
) (admission.Warnings, error) {
	validation := &dynamoComponentDeploymentValidation{
		sharedValidation: sharedValidation{
			ctx:                                ctx,
			runtimeVersionSource:               runtimeVersionSourceDisabled,
			allowMissingRuntimeVersionOverride: true,
		},
	}

	allErrs := validation.validateDynamoComponentDeployment(newDCD)
	newAlpha, err := alphaDynamoComponentDeploymentForValidation(newDCD)
	if err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoComponentDeployment fields: %w", err)
	}
	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentV1alpha1(newAlpha)...)

	// Re-enable source-version runtime validation for the old/new ratchet.
	validation.runtimeVersionSource = runtimeVersionSource
	if validation.validatesRuntimeVersionFor(runtimeVersionSourceV1Alpha1) {
		oldAlpha, err := alphaDynamoComponentDeploymentForValidation(oldDCD)
		if err != nil {
			return nil, fmt.Errorf("cannot validate old preserved v1alpha1 DynamoComponentDeployment fields: %w", err)
		}
		allErrs = append(allErrs, validation.validateDynamoComponentDeploymentSharedSpecUpdateV1alpha1(
			&newAlpha.Spec.DynamoComponentDeploymentSharedSpec,
			&oldAlpha.Spec.DynamoComponentDeploymentSharedSpec,
			field.NewPath("spec"),
		)...)
	}

	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentUpdate(
		newDCD,
		oldDCD,
		canModifyReplicas,
	)...)
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
	allErrs := validateElasticEPRequiresCommand(spec.BackendFramework, &spec.DynamoComponentDeploymentSharedSpec, fldPath)
	allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpec(
		&spec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		grovePathway,
		validateInferencePoolAvailability,
	)...)
	return allErrs
}

// validateDynamoComponentDeploymentUpdate validates an update. newDCD and oldDCD must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentUpdate(
	newDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	oldDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	canModifyReplicas bool,
) field.ErrorList {
	allErrs := v.validateObjectMetaUpdate(
		&newDCD.ObjectMeta,
		&oldDCD.ObjectMeta,
		field.NewPath("metadata"),
	)
	allErrs = append(allErrs, v.validateDynamoComponentDeploymentSpecUpdate(
		&newDCD.Spec,
		&oldDCD.Spec,
		field.NewPath("spec"),
		canModifyReplicas,
	)...)
	return allErrs
}

// validateObjectMetaUpdate validates a DCD objectMeta update. newMeta,
// oldMeta, and fldPath must not be nil.
func (v *dynamoComponentDeploymentValidation) validateObjectMetaUpdate(
	newMeta *metav1.ObjectMeta,
	oldMeta *metav1.ObjectMeta,
	fldPath *field.Path,
) field.ErrorList {
	oldOwner := dgdControllerOwnerReference(oldMeta.OwnerReferences)
	if oldOwner == nil {
		return nil
	}
	for index := range newMeta.OwnerReferences {
		if apiequality.Semantic.DeepEqual(&newMeta.OwnerReferences[index], oldOwner) {
			return nil
		}
	}
	return field.ErrorList{field.Forbidden(
		fldPath.Child("ownerReferences"),
		"DynamoGraphDeployment controller owner reference is immutable",
	)}
}

func dgdControllerOwnerReference(ownerReferences []metav1.OwnerReference) *metav1.OwnerReference {
	for index := range ownerReferences {
		owner := &ownerReferences[index]
		if ptr.Deref(owner.Controller, false) &&
			owner.Kind == nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind &&
			strings.HasPrefix(owner.APIVersion, nvidiacomv1beta1.GroupVersion.Group+"/") {
			return owner
		}
	}
	return nil
}

// validateDynamoComponentDeploymentSpecUpdate validates a spec update.
// newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentSpecUpdate(
	newSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	oldSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	fldPath *field.Path,
	canModifyReplicas bool,
) field.ErrorList {
	const validateGPUMemoryServiceNewState = false // ValidateUpdate already runs the stateless new-state traversal.

	allErrs := field.ErrorList{}
	if newSpec.BackendFramework != oldSpec.BackendFramework {
		v.warn("Changing spec.backendFramework may cause unexpected behavior")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("backendFramework"),
			newSpec.BackendFramework,
			"is immutable and cannot be changed after creation",
		))
	}

	sharedCanModifyReplicas := canModifyReplicas
	if !canModifyReplicas && ptr.Deref(newSpec.Replicas, int32(1)) != ptr.Deref(oldSpec.Replicas, int32(1)) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("replicas"),
			"transactional DGD-owned worker replicas are operator-owned; update the related DynamoGraphDeploymentScalingAdapter request instead",
		))
		sharedCanModifyReplicas = true
	}
	allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecUpdate(
		&newSpec.DynamoComponentDeploymentSharedSpec,
		&oldSpec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		sharedCanModifyReplicas,
		nvidiacomv1beta1.DynamoComponentDeploymentGVK.GroupKind(),
		validateGPUMemoryServiceNewState,
	)...)
	return allErrs
}
