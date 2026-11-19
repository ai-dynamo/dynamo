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
	"context"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
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
		spec.EPPConfig != nil ||
		spec.FrontendSidecar != nil ||
		spec.Failover != nil
}

// controlledByDynamoGraphDeployment identifies DGD controller ownership, not caller authorization.
// dcd must be non-nil.
func controlledByDynamoGraphDeployment(dcd metav1.Object) bool {
	owner := metav1.GetControllerOf(dcd)
	if owner == nil || owner.Kind != nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind {
		return false
	}
	gv, err := schema.ParseGroupVersion(owner.APIVersion)
	return err == nil && gv.Group == nvidiacomv1beta1.GroupVersion.Group
}

// operatorOwnedDCDNameExemption permits legacy child identities only for authenticated operator writes.
// ctx and dcd must be non-nil. An empty operatorPrincipal disables the exemption.
func operatorOwnedDCDNameExemption(ctx context.Context, dcd metav1.Object, operatorPrincipal string) bool {
	if operatorPrincipal == "" || !controlledByDynamoGraphDeployment(dcd) {
		return false
	}
	request, err := admission.RequestFromContext(ctx)
	return err == nil && request.UserInfo.Username == operatorPrincipal
}
