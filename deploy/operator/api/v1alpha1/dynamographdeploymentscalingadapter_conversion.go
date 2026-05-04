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

// Conversion scaffolding between v1alpha1 and v1beta1 DynamoGraphDeploymentScalingAdapter.
// See dynamographdeployment_conversion.go for the design rationale.

package v1alpha1

import (
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// errDGDSAConversionNotImplemented is returned by the v1alpha1 <-> v1beta1
// DGDSA conversion stubs until real mapping logic is added.
var errDGDSAConversionNotImplemented = fmt.Errorf(
	"DynamoGraphDeploymentScalingAdapter v1alpha1 <-> v1beta1 conversion is not yet implemented; " +
		"v1beta1 is marked unserved, use v1alpha1")

// ConvertTo converts this DynamoGraphDeploymentScalingAdapter (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeploymentScalingAdapter) ConvertTo(dstRaw conversion.Hub) error {
	if _, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter); !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", dstRaw)
	}
	return errDGDSAConversionNotImplemented
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeploymentScalingAdapter (v1alpha1).
func (dst *DynamoGraphDeploymentScalingAdapter) ConvertFrom(srcRaw conversion.Hub) error {
	if _, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter); !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", srcRaw)
	}
	return errDGDSAConversionNotImplemented
}
