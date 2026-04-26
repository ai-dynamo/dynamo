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

// Conversion between v1alpha1 and v1beta1 DynamoComponentDeployment.
// See dynamographdeployment_conversion.go for the design rationale.

package v1alpha1

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// ConvertTo converts this DynamoComponentDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoComponentDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	dst.Spec.BackendFramework = src.Spec.BackendFramework

	carrier := newDCDCarrier(&dst.ObjectMeta)
	if err := convertSharedSpecTo(&src.Spec.DynamoComponentDeploymentSharedSpec,
		&dst.Spec.DynamoComponentDeploymentSharedSpec, carrier); err != nil {
		return err
	}

	convertDCDStatusTo(&src.Status, &dst.Status)
	return nil
}

// ConvertFrom converts from the hub (v1beta1) DynamoComponentDeployment into
// this v1alpha1 instance.
func (dst *DynamoComponentDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", srcRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	dst.Spec.BackendFramework = src.Spec.BackendFramework

	carrier := newDCDCarrier(&dst.ObjectMeta)
	if err := convertSharedSpecFrom(&src.Spec.DynamoComponentDeploymentSharedSpec,
		&dst.Spec.DynamoComponentDeploymentSharedSpec, carrier); err != nil {
		return err
	}

	convertDCDStatusFrom(&src.Status, &dst.Status)
	scrubDCDAnnotations(&dst.ObjectMeta)
	return nil
}

func convertDCDStatusTo(src *DynamoComponentDeploymentStatus, dst *v1beta1.DynamoComponentDeploymentStatus) {
	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Service != nil {
		dst.Service = convertReplicaStatusTo(src.Service)
	}
	// PodSelector is dropped in v1beta1 (the field was never populated by the
	// controller). No annotation is needed: the round-trip invariant is on
	// v1beta1 inputs, which do not carry PodSelector.
}

func convertDCDStatusFrom(src *v1beta1.DynamoComponentDeploymentStatus, dst *DynamoComponentDeploymentStatus) {
	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Service != nil {
		dst.Service = convertReplicaStatusFrom(src.Service)
	}
}

// scrubDCDAnnotations removes any lingering "nvidia.com/dcd-*" keys that
// convertSharedSpecFrom did not consume.
func scrubDCDAnnotations(obj *metav1.ObjectMeta) {
	scrubAnnotationsByPrefix(obj, annDCDPrefix)
}
