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
	"fmt"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoModelValidator validates DynamoModel resources.
// This validator can be used by both webhooks and controllers for consistent validation.
type DynamoModelValidator struct {
	model *nvidiacomv1alpha1.DynamoModel
}

// NewDynamoModelValidator creates a new validator for DynamoModel.
func NewDynamoModelValidator(model *nvidiacomv1alpha1.DynamoModel) *DynamoModelValidator {
	return &DynamoModelValidator{
		model: model,
	}
}

// Validate performs stateless validation on the DynamoModel.
// Returns warnings and error.
func (v *DynamoModelValidator) Validate() (admission.Warnings, error) {
	// Validate modelName is not empty
	if v.model.Spec.ModelName == "" {
		return nil, fmt.Errorf("spec.modelName is required")
	}

	// Validate baseModelName is not empty
	if v.model.Spec.BaseModelName == "" {
		return nil, fmt.Errorf("spec.baseModelName is required")
	}

	// Validate LoRA model requirements
	if v.model.Spec.ModelType == "lora" {
		if v.model.Spec.Source == nil {
			return nil, fmt.Errorf("spec.source is required when modelType is 'lora'")
		}

		if v.model.Spec.Source.URI == "" {
			return nil, fmt.Errorf("spec.source.uri must be specified when modelType is 'lora'")
		}

		// Validate URI format
		if err := v.validateSourceURI(v.model.Spec.Source.URI); err != nil {
			return nil, err
		}
	}

	return nil, nil
}

// ValidateUpdate performs stateful validation comparing old and new DynamoModel.
// Returns warnings and error.
func (v *DynamoModelValidator) ValidateUpdate(old *nvidiacomv1alpha1.DynamoModel) (admission.Warnings, error) {
	var warnings admission.Warnings

	// modelType is immutable
	if v.model.Spec.ModelType != old.Spec.ModelType {
		warnings = append(warnings, "Changing spec.modelType may cause unexpected behavior")
		return warnings, fmt.Errorf("spec.modelType is immutable and cannot be changed after creation")
	}

	// baseModelName is immutable
	if v.model.Spec.BaseModelName != old.Spec.BaseModelName {
		warnings = append(warnings, "Changing spec.baseModelName will break endpoint discovery")
		return warnings, fmt.Errorf("spec.baseModelName is immutable and cannot be changed after creation")
	}

	return nil, nil
}

// validateSourceURI validates the model source URI format.
func (v *DynamoModelValidator) validateSourceURI(uri string) error {
	if uri == "" {
		return fmt.Errorf("source URI cannot be empty")
	}

	// Check for supported schemes
	if !strings.HasPrefix(uri, "s3://") && !strings.HasPrefix(uri, "hf://") {
		return fmt.Errorf("source URI must start with 's3://' or 'hf://', got: %s", uri)
	}

	return nil
}
