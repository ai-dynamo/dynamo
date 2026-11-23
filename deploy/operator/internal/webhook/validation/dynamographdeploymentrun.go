/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// dynamoGraphDeploymentRunValidation shares native input rules with Request admission.
type dynamoGraphDeploymentRunValidation struct {
	sharedValidation
}

// validateDynamoGraphDeploymentRunSpec validates an already resolved snapshot without defaulting it.
// spec and fldPath must not be nil. The CRD owns snapshot completeness and immutability.
func (v *dynamoGraphDeploymentRunValidation) validateDynamoGraphDeploymentRunSpec(spec *nvidiacomv1beta2.DynamoGraphDeploymentRunSpec, fldPath *field.Path) field.ErrorList {
	return v.validateNativeSearchInput(&spec.NativeSearchInput, fldPath, true)
}

// validateDynamoGraphDeploymentRunSpecUpdate validates an update to the immutable snapshot.
// newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoGraphDeploymentRunValidation) validateDynamoGraphDeploymentRunSpecUpdate(newSpec, oldSpec *nvidiacomv1beta2.DynamoGraphDeploymentRunSpec, fldPath *field.Path) field.ErrorList {
	return v.validateNativeSearchInputUpdate(&newSpec.NativeSearchInput, &oldSpec.NativeSearchInput, fldPath)
}
