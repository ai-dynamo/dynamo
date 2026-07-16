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
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoModelValidator validates DynamoModel resources.
type DynamoModelValidator struct{}

// NewDynamoModelValidator creates a DynamoModel validator.
func NewDynamoModelValidator() *DynamoModelValidator {
	return &DynamoModelValidator{}
}

// dynamoModelValidation carries DynamoModel request accumulation.
// API values, paths, and accumulated errors remain explicit validator arguments.
type dynamoModelValidation struct {
	warnings admission.Warnings
}

func (v *dynamoModelValidation) warn(message string) {
	v.warnings = append(v.warnings, message)
}

// Validate performs stateless validation on model. model must not be nil.
func (v *DynamoModelValidator) Validate(
	model *nvidiacomv1alpha1.DynamoModel,
) (admission.Warnings, error) {
	validation := &dynamoModelValidation{}
	allErrs := validation.validateDynamoModel(model)
	return validation.warnings, invalidDynamoModelError(model, allErrs)
}

// ValidateUpdate validates newModel against oldModel. oldModel and newModel must not be nil.
func (v *DynamoModelValidator) ValidateUpdate(
	oldModel *nvidiacomv1alpha1.DynamoModel,
	newModel *nvidiacomv1alpha1.DynamoModel,
) (admission.Warnings, error) {
	validation := &dynamoModelValidation{}
	allErrs := validation.validateDynamoModel(newModel)
	allErrs = append(allErrs, validation.validateDynamoModelUpdate(newModel, oldModel)...)
	return validation.warnings, invalidDynamoModelError(newModel, allErrs)
}

// validateDynamoModel validates model. model must not be nil.
func (v *dynamoModelValidation) validateDynamoModel(
	model *nvidiacomv1alpha1.DynamoModel,
) field.ErrorList {
	return v.validateDynamoModelSpec(&model.Spec, field.NewPath("spec"))
}

// validateDynamoModelSpec validates spec. spec and fldPath must not be nil.
func (v *dynamoModelValidation) validateDynamoModelSpec(
	spec *nvidiacomv1alpha1.DynamoModelSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.ModelName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("modelName"), "must not be empty"))
	}
	if spec.BaseModelName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("baseModelName"), "must not be empty"))
	}

	if spec.ModelType == "lora" {
		if spec.Source == nil {
			allErrs = append(allErrs, field.Required(
				fldPath.Child("source"),
				`is required when spec.modelType is "lora"`,
			))
		} else {
			allErrs = append(allErrs, v.validateModelSource(spec.Source, fldPath.Child("source"))...)
		}
	}
	return allErrs
}

// validateModelSource validates source. source and fldPath must not be nil.
func (v *dynamoModelValidation) validateModelSource(
	source *nvidiacomv1alpha1.ModelSource,
	fldPath *field.Path,
) field.ErrorList {
	uriPath := fldPath.Child("uri")
	if source.URI == "" {
		return field.ErrorList{field.Required(uriPath, `must be specified when spec.modelType is "lora"`)}
	}
	if !hasSupportedModelSourceScheme(source.URI) {
		return field.ErrorList{field.Invalid(
			uriPath,
			source.URI,
			`must start with "s3://", "hf://", or "file:///"`,
		)}
	}
	return nil
}

// validateDynamoModelUpdate validates an update. newModel and oldModel must not be nil.
func (v *dynamoModelValidation) validateDynamoModelUpdate(
	newModel *nvidiacomv1alpha1.DynamoModel,
	oldModel *nvidiacomv1alpha1.DynamoModel,
) field.ErrorList {
	return v.validateDynamoModelSpecUpdate(&newModel.Spec, &oldModel.Spec, field.NewPath("spec"))
}

// validateDynamoModelSpecUpdate validates a spec update. newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoModelValidation) validateDynamoModelSpecUpdate(
	newSpec *nvidiacomv1alpha1.DynamoModelSpec,
	oldSpec *nvidiacomv1alpha1.DynamoModelSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if newSpec.BaseModelName != oldSpec.BaseModelName {
		v.warn("Changing spec.baseModelName will break endpoint discovery")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("baseModelName"),
			newSpec.BaseModelName,
			"is immutable and cannot be changed after creation",
		))
	}
	if newSpec.ModelType != oldSpec.ModelType {
		v.warn("Changing spec.modelType may cause unexpected behavior")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("modelType"),
			newSpec.ModelType,
			"is immutable and cannot be changed after creation",
		))
	}
	return allErrs
}
