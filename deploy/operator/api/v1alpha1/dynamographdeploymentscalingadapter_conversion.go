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

// Conversion between v1alpha1 and v1beta1 DynamoGraphDeploymentScalingAdapter.
//
// v1beta1 is the hub (see api/v1beta1/dynamographdeploymentscalingadapter_conversion.go).
// The DGDSA shapes differ only in the embedded DGDRef carrier:
//
//   - v1alpha1: spec.dgdRef is a DynamoGraphDeploymentServiceRef{Name, ServiceName}
//   - v1beta1:  spec.dgdRef is a DynamoGraphDeploymentComponentRef{Name, ComponentName}
//
// Replicas, status, and metadata are structurally identical, so this is a
// lossless straight copy with a single field rename. v1beta1 is served, and
// conversion wiring routes cross-version requests through this path.

package v1alpha1

import (
	"fmt"

	conversion "k8s.io/apimachinery/pkg/conversion"
	crconversion "sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// ConvertTo converts this DynamoGraphDeploymentScalingAdapter (v1alpha1) into
// the hub version (v1beta1).
func (src *DynamoGraphDeploymentScalingAdapter) ConvertTo(dstRaw crconversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", dstRaw)
	}
	return Convert_v1alpha1_DynamoGraphDeploymentScalingAdapter_To_v1beta1_DynamoGraphDeploymentScalingAdapter(src, dst, nil)
}

// ConvertFrom converts from the hub (v1beta1) DynamoGraphDeploymentScalingAdapter
// into this v1alpha1 instance.
func (dst *DynamoGraphDeploymentScalingAdapter) ConvertFrom(srcRaw crconversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", srcRaw)
	}
	return Convert_v1beta1_DynamoGraphDeploymentScalingAdapter_To_v1alpha1_DynamoGraphDeploymentScalingAdapter(src, dst, nil)
}

// Convert_v1alpha1_DynamoGraphDeploymentServiceRef_To_v1beta1_DynamoGraphDeploymentComponentRef
// handles the service-to-component terminology change that conversion-gen
// cannot infer from field names.
func Convert_v1alpha1_DynamoGraphDeploymentServiceRef_To_v1beta1_DynamoGraphDeploymentComponentRef(src *DynamoGraphDeploymentServiceRef, dst *v1beta1.DynamoGraphDeploymentComponentRef, _ conversion.Scope) error {
	dst.Name = src.Name
	dst.ComponentName = src.ServiceName
	return nil
}

// Convert_v1beta1_DynamoGraphDeploymentComponentRef_To_v1alpha1_DynamoGraphDeploymentServiceRef
// handles the component-to-service terminology change that conversion-gen
// cannot infer from field names.
func Convert_v1beta1_DynamoGraphDeploymentComponentRef_To_v1alpha1_DynamoGraphDeploymentServiceRef(src *v1beta1.DynamoGraphDeploymentComponentRef, dst *DynamoGraphDeploymentServiceRef, _ conversion.Scope) error {
	dst.Name = src.Name
	dst.ServiceName = src.ComponentName
	return nil
}
