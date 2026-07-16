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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoGraphDeploymentRequestValidator validates v1beta1 DynamoGraphDeploymentRequest resources.
type DynamoGraphDeploymentRequestValidator struct{}

// NewDynamoGraphDeploymentRequestValidator creates a DynamoGraphDeploymentRequest validator.
func NewDynamoGraphDeploymentRequestValidator() *DynamoGraphDeploymentRequestValidator {
	return &DynamoGraphDeploymentRequestValidator{}
}

// dynamoGraphDeploymentRequestValidation carries DGDR-specific request state.
// API values, paths, and accumulated errors remain explicit validator arguments.
type dynamoGraphDeploymentRequestValidation struct {
	sharedValidation
}

// Validate performs stateless validation on request. ctx and request must not be nil.
func (v *DynamoGraphDeploymentRequestValidator) Validate(
	ctx context.Context,
	request *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentRequestValidation{sharedValidation: sharedValidation{ctx: ctx}}
	allErrs := validation.validateDynamoGraphDeploymentRequest(request)
	return validation.warnings, invalidDynamoGraphDeploymentRequestError(request, allErrs)
}

// ValidateUpdate validates newRequest against oldRequest. ctx, oldRequest, and newRequest must not be nil.
func (v *DynamoGraphDeploymentRequestValidator) ValidateUpdate(
	ctx context.Context,
	oldRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	newRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentRequestValidation{sharedValidation: sharedValidation{ctx: ctx}}
	allErrs := validation.validateDynamoGraphDeploymentRequest(newRequest)
	allErrs = append(allErrs, validation.validateDynamoGraphDeploymentRequestUpdate(newRequest, oldRequest)...)
	return validation.warnings, invalidDynamoGraphDeploymentRequestError(newRequest, allErrs)
}

// validateDynamoGraphDeploymentRequest validates request. request must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequest(
	request *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, v.validateObjectMeta(&request.ObjectMeta, field.NewPath("metadata"), false)...)
	allErrs = append(allErrs, v.validateDynamoGraphDeploymentRequestSpec(&request.Spec, field.NewPath("spec"))...)
	return allErrs
}

// validateDynamoGraphDeploymentRequestSpec validates spec. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestSpec(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}

	if !features.MustGateFrom(v.ctx).Enabled(features.GPUDiscovery) && !hasManualDGDRHardware(spec.Hardware) {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("hardware"),
			"GPU hardware configuration is required when GPU discovery is disabled",
		))
	}

	if spec.SearchStrategy == nvidiacomv1beta1.SearchStrategyThorough &&
		spec.Backend == nvidiacomv1beta1.BackendTypeAuto {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("searchStrategy"),
			spec.SearchStrategy,
			fmt.Sprintf("is incompatible with spec.backend %q; set spec.backend to a specific backend (sglang, trtllm, or vllm)", spec.Backend),
		))
	}

	return allErrs
}

// validateDynamoGraphDeploymentRequestUpdate validates an update. newRequest and oldRequest must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestUpdate(
	newRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	oldRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) field.ErrorList {
	return v.validateDynamoGraphDeploymentRequestSpecUpdate(
		&newRequest.Spec,
		&oldRequest.Spec,
		field.NewPath("spec"),
		oldRequest.Status.Phase,
	)
}

// validateDynamoGraphDeploymentRequestSpecUpdate validates a spec update.
// newSpec, oldSpec, and fldPath must not be nil; oldPhase comes from the owning old resource status.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestSpecUpdate(
	newSpec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	oldSpec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
	oldPhase nvidiacomv1beta1.DGDRPhase,
) field.ErrorList {
	if !isImmutableDGDRPhase(oldPhase) || apiequality.Semantic.DeepEqual(newSpec, oldSpec) {
		return nil
	}
	return field.ErrorList{field.Forbidden(
		fldPath,
		fmt.Sprintf("updates are forbidden while the resource is in phase %q; delete and recreate the resource to change its spec", oldPhase),
	)}
}
