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

// validateDynamoComponentDeploymentV1alpha1 validates dcd. dcd must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentV1alpha1(
	dcd *nvidiacomv1alpha1.DynamoComponentDeployment,
) field.ErrorList {
	if !hasDynamoComponentDeploymentV1alpha1CompatibilityFields(dcd) {
		return nil
	}
	return v.validateDynamoComponentDeploymentSpecV1alpha1(
		&dcd.Spec,
		field.NewPath("spec"),
		dcd.GetDynamoNamespace(),
	)
}

// validateDynamoComponentDeploymentSpecV1alpha1 validates spec. spec and fldPath must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentSpecV1alpha1(
	spec *nvidiacomv1alpha1.DynamoComponentDeploymentSpec,
	fldPath *field.Path,
	dynamoNamespace string,
) field.ErrorList {
	return v.validateDynamoComponentDeploymentSharedSpecV1alpha1(
		&spec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		dynamoNamespace,
	)
}
