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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoComponentDeploymentV1Beta1Validator validates v1beta1-only DCD invariants.
type DynamoComponentDeploymentV1Beta1Validator struct {
	deployment *nvidiacomv1beta1.DynamoComponentDeployment
}

// NewDynamoComponentDeploymentV1Beta1Validator creates a v1beta1 DCD validator.
func NewDynamoComponentDeploymentV1Beta1Validator(deployment *nvidiacomv1beta1.DynamoComponentDeployment) *DynamoComponentDeploymentV1Beta1Validator {
	return &DynamoComponentDeploymentV1Beta1Validator{
		deployment: deployment,
	}
}

// Validate performs stateless validation that must see the original v1beta1 object.
func (v *DynamoComponentDeploymentV1Beta1Validator) Validate(_ context.Context) (admission.Warnings, error) {
	validator := NewV1Beta1SharedSpecValidator(&v.deployment.Spec.DynamoComponentDeploymentSharedSpec, "spec")
	return nil, validator.Validate()
}
