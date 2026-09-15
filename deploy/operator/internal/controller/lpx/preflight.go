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

package lpx

import (
	"context"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

// reconcileLPXSafetyPreflight classifies and fences scheduler-owned state
// after the LPX controller validates source identity and the Grove provider.
func (r *graphReconciler) reconcileLPXSafetyPreflight(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	requests []lpxv1alpha1.LPUPipelineRequest,
) (*lpxMaterializing, lpxClassification, error) {
	classification, requests, err := r.reconcileLPXKnownIntentFence(ctx, deployment, source, requests)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to fence prior LPX intent: %w", err)
	}
	if classification != nil {
		return nil, classification, nil
	}

	// The serving component owns every Grove name, including the shared draft cliques.
	serving := dynamolpx.ServingComponent(source)
	for index := range source.Spec.Components {
		component := &source.Spec.Components[index]
		if component != serving {
			continue
		}
		combinedLength := len(dynamo.PCSNameForLPX(deployment)) + dynamo.LPXComponentNameBudget(component.ComponentName)
		if combinedLength > consts.MaxCombinedGroveResourceNameLength {
			err := field.Invalid(field.NewPath("spec", "components").Index(index).Child("name"), component.ComponentName,
				fmt.Sprintf("combined Grove resource name length %d exceeds the %d-character limit; shorten the deployment or component name",
					combinedLength, consts.MaxCombinedGroveResourceNameLength))
			return nil, &lpxRejected{reason: err.Error()}, nil
		}
		break
	}

	if len(requests) == 0 {
		if err := dynamolpx.ValidateSelectedIntent(source).ToAggregate(); err != nil {
			return nil, &lpxRejected{reason: err.Error()}, nil
		}
		return nil, nil, nil
	}

	return r.reconcileSelectedLPXSafetyPreflight(ctx, deployment, source)
}

// reconcileSelectedLPXSafetyPreflight resolves and validates one selected LPX
// workload after its caller establishes the required download order.
func (r *graphReconciler) reconcileSelectedLPXSafetyPreflight(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
) (*lpxMaterializing, lpxClassification, error) {
	// Resolve one immutable workload projection for selection and rendering.
	selectedLPX, rejected, prepareErr := r.prepareLPXMaterializing(ctx, deployment, source)
	if prepareErr != nil {
		return nil, nil, fmt.Errorf("failed to resolve the LPX workload: %w", prepareErr)
	}
	if rejected != nil {
		return nil, rejected, nil
	}

	// Grove accepts the desired spec before individual engine requests are reconciled.
	return selectedLPX, nil, nil
}
