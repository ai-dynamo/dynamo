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
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// invalidDynamoComponentDeploymentError converts allErrs for dcd into an API error.
// dcd must not be nil.
func invalidDynamoComponentDeploymentError(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	allErrs field.ErrorList,
) error {
	if len(allErrs) == 0 {
		return nil
	}
	return k8serrors.NewInvalid(nvidiacomv1beta1.DynamoComponentDeploymentGVK.GroupKind(), dcd.Name, allErrs)
}

// alphaDynamoComponentDeploymentForValidation reconstructs the compatibility view.
// dcd must not be nil.
func alphaDynamoComponentDeploymentForValidation(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (*nvidiacomv1alpha1.DynamoComponentDeployment, error) {
	alpha := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	if err := alpha.ConvertFrom(dcd); err != nil {
		return nil, fmt.Errorf("failed to reconstruct compatibility view: %w", err)
	}
	return alpha, nil
}

func hasDynamoComponentDeploymentV1alpha1CompatibilityFields(
	dcd *nvidiacomv1alpha1.DynamoComponentDeployment,
) bool {
	spec := &dcd.Spec.DynamoComponentDeploymentSharedSpec
	hasDeprecatedAutoscaling := false
	//nolint:staticcheck // SA1019: Intentionally checking a deprecated field preserved by conversion.
	if spec.Autoscaling != nil {
		hasDeprecatedAutoscaling = true
	}
	return len(spec.Annotations) > 0 ||
		spec.DynamoNamespace != nil ||
		hasDeprecatedAutoscaling ||
		spec.Ingress != nil ||
		len(spec.VolumeMounts) > 0 ||
		spec.FrontendSidecar != nil ||
		spec.Failover != nil
}

// castToDynamoComponentDeployment converts the v1alpha1 spoke to the v1beta1
// hub used by the DCD validator, or returns a v1beta1 object unchanged.
func castToDynamoComponentDeployment(obj runtime.Object) (*nvidiacomv1beta1.DynamoComponentDeployment, error) {
	switch deployment := obj.(type) {
	case *nvidiacomv1beta1.DynamoComponentDeployment:
		return deployment, nil
	case *nvidiacomv1alpha1.DynamoComponentDeployment:
		converted := &nvidiacomv1beta1.DynamoComponentDeployment{}
		if err := deployment.ConvertTo(converted); err != nil {
			return nil, fmt.Errorf("convert v1alpha1 DynamoComponentDeployment to v1beta1: %w", err)
		}
		return converted, nil
	default:
		return nil, fmt.Errorf("expected v1alpha1 or v1beta1 DynamoComponentDeployment but got %T", obj)
	}
}

func dynamoComponentDeploymentMetadata(obj runtime.Object) (metav1.Object, error) {
	switch deployment := obj.(type) {
	case *nvidiacomv1beta1.DynamoComponentDeployment:
		return deployment, nil
	case *nvidiacomv1alpha1.DynamoComponentDeployment:
		return deployment, nil
	default:
		return nil, fmt.Errorf("expected DynamoComponentDeployment but got %T", obj)
	}
}
