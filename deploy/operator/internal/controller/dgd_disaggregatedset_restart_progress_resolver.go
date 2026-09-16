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

package controller

import (
	"context"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// disaggregatedSetRestartProgressResolver owns the read-only DisaggregatedSet
// observations used to determine which components have not completed a
// requested restart.
type disaggregatedSetRestartProgressResolver struct {
	reader    client.Reader
	readiness *disaggregatedSetReadinessResolver
	component *componentRestartProgressResolver
}

const disaggregatedSetRestartResourceNotFoundReason = "resource not found"

func newDisaggregatedSetRestartProgressResolver(
	reader client.Reader,
	readiness *disaggregatedSetReadinessResolver,
	component *componentRestartProgressResolver,
) *disaggregatedSetRestartProgressResolver {
	return &disaggregatedSetRestartProgressResolver{
		reader:    reader,
		readiness: readiness,
		component: component,
	}
}

func (r *disaggregatedSetRestartProgressResolver) Resolve(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inProgress []string,
) []string {
	logger := log.FromContext(ctx)
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		logger.V(1).Info("failed to select DisaggregatedSet components for restart progress", "reason", reason)
		return inProgress
	}

	ds := newDisaggregatedSetObject()
	dsErr := r.reader.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, ds)
	if dsErr != nil && !apierrors.IsNotFound(dsErr) {
		logger.V(1).Info("failed to get DisaggregatedSet for restart progress", "error", dsErr)
	}

	dsReady := false
	dsReason := disaggregatedSetRestartResourceNotFoundReason
	if dsErr == nil {
		readiness, err := r.readiness.Resolve(ctx, ds, selection)
		if err != nil {
			dsReason = err.Error()
		} else {
			dsReady = readiness.Ready
			dsReason = readiness.Reason
		}
	}

	updatedInProgress := make([]string, 0, len(inProgress))
	for _, componentName := range inProgress {
		if _, selected := selection.componentToRole[componentName]; !selected {
			isFullyUpdated, reason := r.component.checkComponentFullyUpdated(ctx, dgd, componentName)
			if !isFullyUpdated {
				logger.V(1).Info("component not fully updated", "componentName", componentName, "reason", reason)
				updatedInProgress = append(updatedInProgress, componentName)
			}
			continue
		}

		if dsErr != nil {
			reason := disaggregatedSetRestartResourceNotFoundReason
			if !apierrors.IsNotFound(dsErr) {
				reason = dsErr.Error()
			}
			logger.V(1).Info("DisaggregatedSet component not fully updated", "componentName", componentName, "reason", reason)
			updatedInProgress = append(updatedInProgress, componentName)
			continue
		}

		if !dsReady {
			logger.V(1).Info("DisaggregatedSet component not fully updated", "componentName", componentName, "reason", dsReason)
			updatedInProgress = append(updatedInProgress, componentName)
		}
	}
	return updatedInProgress
}
