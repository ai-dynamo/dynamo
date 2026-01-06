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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoComponentDeploymentValidator validates DynamoComponentDeployment resources.
// This validator can be used by both webhooks and controllers for consistent validation.
type DynamoComponentDeploymentValidator struct {
	deployment *nvidiacomv1alpha1.DynamoComponentDeployment
}

// NewDynamoComponentDeploymentValidator creates a new validator for DynamoComponentDeployment.
func NewDynamoComponentDeploymentValidator(deployment *nvidiacomv1alpha1.DynamoComponentDeployment) *DynamoComponentDeploymentValidator {
	return &DynamoComponentDeploymentValidator{
		deployment: deployment,
	}
}

// Validate performs stateless validation on the DynamoComponentDeployment.
// Returns warnings and error.
func (v *DynamoComponentDeploymentValidator) Validate() (admission.Warnings, error) {
	// Validate shared spec fields using SharedSpecValidator
	sharedValidator := NewSharedSpecValidator(&v.deployment.Spec.DynamoComponentDeploymentSharedSpec, "spec")

	// DCD-specific validation would go here (currently none)

	return sharedValidator.Validate()
}

// ValidateUpdate performs stateful validation comparing old and new DynamoComponentDeployment.
// Returns warnings and error.
func (v *DynamoComponentDeploymentValidator) ValidateUpdate(old *nvidiacomv1alpha1.DynamoComponentDeployment) (admission.Warnings, error) {
	// Validate that BackendFramework is not changed (immutable)
	if v.deployment.Spec.BackendFramework != old.Spec.BackendFramework {
		warning := "Changing spec.backendFramework may cause unexpected behavior"
		return admission.Warnings{warning}, fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation")
	}

	return nil, nil
}
