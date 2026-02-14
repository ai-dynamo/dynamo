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

package v1alpha1

import (
	"fmt"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
)

// ConvertTo converts this DynamoGraphDeploymentScalingAdapter (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeploymentScalingAdapter) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Replicas = src.Spec.Replicas
	dst.Spec.DGDRef = v1beta1.DynamoGraphDeploymentServiceRef{
		Name:        src.Spec.DGDRef.Name,
		ServiceName: src.Spec.DGDRef.ServiceName,
	}

	// Status
	dst.Status.Replicas = src.Status.Replicas
	dst.Status.Selector = src.Status.Selector
	dst.Status.LastScaleTime = src.Status.LastScaleTime

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeploymentScalingAdapter (v1alpha1).
func (dst *DynamoGraphDeploymentScalingAdapter) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentScalingAdapter)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentScalingAdapter but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Replicas = src.Spec.Replicas
	dst.Spec.DGDRef = DynamoGraphDeploymentServiceRef{
		Name:        src.Spec.DGDRef.Name,
		ServiceName: src.Spec.DGDRef.ServiceName,
	}

	// Status
	dst.Status.Replicas = src.Status.Replicas
	dst.Status.Selector = src.Status.Selector
	dst.Status.LastScaleTime = src.Status.LastScaleTime

	return nil
}
