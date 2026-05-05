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
	"errors"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoGraphDeploymentV1Beta1Validator validates v1beta1-only DGD invariants.
type DynamoGraphDeploymentV1Beta1Validator struct {
	deployment *nvidiacomv1beta1.DynamoGraphDeployment
}

// NewDynamoGraphDeploymentV1Beta1Validator creates a v1beta1 DGD validator.
func NewDynamoGraphDeploymentV1Beta1Validator(deployment *nvidiacomv1beta1.DynamoGraphDeployment) *DynamoGraphDeploymentV1Beta1Validator {
	return &DynamoGraphDeploymentV1Beta1Validator{
		deployment: deployment,
	}
}

// Validate performs stateless validation that must see the original v1beta1 object.
func (v *DynamoGraphDeploymentV1Beta1Validator) Validate(_ context.Context) (admission.Warnings, error) {
	var errs []error

	// Validate beta-only component shapes before conversion hides them.
	for i := range v.deployment.Spec.Components {
		component := &v.deployment.Spec.Components[i]
		validator := NewV1Beta1SharedSpecValidator(component, betaDGDComponentFieldPath(component, i))
		if err := validator.Validate(); err != nil {
			errs = append(errs, err)
		}
	}

	return nil, errors.Join(errs...)
}

func betaDGDComponentFieldPath(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, index int) string {
	if component.ComponentName == "" {
		return fmt.Sprintf("spec.components[%d]", index)
	}
	return fmt.Sprintf("spec.components[%s]", component.ComponentName)
}
