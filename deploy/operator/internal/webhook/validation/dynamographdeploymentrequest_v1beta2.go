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
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoGraphDeploymentRequestSpecV1beta2 validates spec. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestSpecV1beta2(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := v.validateNativeSearchInput(&spec.NativeSearchInput, fldPath)
	return append(allErrs, recommendationInputErrors(&spec.NativeSearchInput, fldPath)...)
}

// validateDynamoGraphDeploymentRequestSpecUpdateV1beta2 validates a spec update.
// newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestSpecUpdateV1beta2(
	newSpec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	oldSpec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	// A schema upgrade must not prevent metadata updates to persisted requests.
	// Structural rules still apply; any spec edit requires the entire input to be valid.
	if apiequality.Semantic.DeepEqual(newSpec, oldSpec) {
		return v.validateNativeSearchInput(&newSpec.NativeSearchInput, fldPath)
	}
	return v.validateDynamoGraphDeploymentRequestSpecV1beta2(newSpec, fldPath)
}
