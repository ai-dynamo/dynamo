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

// ConvertTo converts this DynamoModel (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoModel) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoModel)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoModel but got %T", dstRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.ModelName = src.Spec.ModelName
	dst.Spec.BaseModelName = src.Spec.BaseModelName
	dst.Spec.ModelType = src.Spec.ModelType
	if src.Spec.Source != nil {
		dst.Spec.Source = &v1beta1.ModelSource{
			URI: src.Spec.Source.URI,
		}
	}

	// Status
	dst.Status.ReadyEndpoints = src.Status.ReadyEndpoints
	dst.Status.TotalEndpoints = src.Status.TotalEndpoints
	dst.Status.Conditions = src.Status.Conditions
	if src.Status.Endpoints != nil {
		dst.Status.Endpoints = make([]v1beta1.EndpointInfo, len(src.Status.Endpoints))
		for i, ep := range src.Status.Endpoints {
			dst.Status.Endpoints[i] = v1beta1.EndpointInfo{
				Address: ep.Address,
				PodName: ep.PodName,
				Ready:   ep.Ready,
			}
		}
	}

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoModel (v1alpha1).
func (dst *DynamoModel) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoModel)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoModel but got %T", srcRaw)
	}

	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.ModelName = src.Spec.ModelName
	dst.Spec.BaseModelName = src.Spec.BaseModelName
	dst.Spec.ModelType = src.Spec.ModelType
	if src.Spec.Source != nil {
		dst.Spec.Source = &ModelSource{
			URI: src.Spec.Source.URI,
		}
	}

	// Status
	dst.Status.ReadyEndpoints = src.Status.ReadyEndpoints
	dst.Status.TotalEndpoints = src.Status.TotalEndpoints
	dst.Status.Conditions = src.Status.Conditions
	if src.Status.Endpoints != nil {
		dst.Status.Endpoints = make([]EndpointInfo, len(src.Status.Endpoints))
		for i, ep := range src.Status.Endpoints {
			dst.Status.Endpoints[i] = EndpointInfo{
				Address: ep.Address,
				PodName: ep.PodName,
				Ready:   ep.Ready,
			}
		}
	}

	return nil
}
