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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
		spec.EPPConfig != nil ||
		spec.FrontendSidecar != nil ||
		spec.Failover != nil
}

// controlledByDynamoGraphDeployment reports whether dcd's controller owner
// reference names a DynamoGraphDeployment. dcd must not be nil.
func controlledByDynamoGraphDeployment(dcd *nvidiacomv1beta1.DynamoComponentDeployment) bool {
	controller := metav1.GetControllerOf(dcd)
	if controller == nil || controller.Kind != nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind {
		return false
	}
	groupVersion, err := schema.ParseGroupVersion(controller.APIVersion)
	return err == nil && groupVersion.Group == nvidiacomv1beta1.GroupVersion.Group
}

// standaloneDCDNameViolations reports each failing naming rule for a standalone DCD.
// Service names are checked only when the DCD gets a Service. dcd must not be nil.
func standaloneDCDNameViolations(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	discoveryBackend configv1alpha1.DiscoveryBackend,
) []generatedNameViolation {
	// The controller renders a Service only for a frontend or under Kubernetes discovery.
	hasService := dcd.IsFrontendComponent() ||
		commoncontroller.IsK8sDiscoveryEnabled(discoveryBackend, dynamo.GetDCDKubeAnnotations(dcd))
	return generatedNameViolations(
		field.NewPath("metadata", "name"),
		dcd.Name,
		hasService,
		"choose a shorter or letter-leading DynamoComponentDeployment name",
	)
}
