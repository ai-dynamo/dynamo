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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groveRestartProgressResolver owns the read-only Grove observations used to
// determine which components have not completed a requested restart.
type groveRestartProgressResolver struct {
	reader client.Reader
}

func newGroveRestartProgressResolver(reader client.Reader) *groveRestartProgressResolver {
	return &groveRestartProgressResolver{reader: reader}
}

func (r *groveRestartProgressResolver) Resolve(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inProgress []string,
) []string {
	logger := log.FromContext(ctx)

	pcs := &grovev1alpha1.PodCliqueSet{}
	pcsName := dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components)
	if err := r.reader.Get(ctx, types.NamespacedName{Name: pcsName, Namespace: dgd.Namespace}, pcs); err != nil {
		logger.Error(err, "failed to get PodCliqueSet")
		return inProgress
	}

	if pcs.Status.ObservedGeneration == nil {
		logger.Info("PodCliqueSet observedGeneration is nil", "name", dgd.Name)
		return inProgress
	}
	if *pcs.Status.ObservedGeneration < pcs.Generation {
		logger.Info(
			"PodCliqueSet not yet reconciled",
			"name", dgd.Name,
			"generation", pcs.Generation,
			"observedGeneration", *pcs.Status.ObservedGeneration,
		)
		return inProgress
	}

	updatedInProgress := make([]string, 0, len(inProgress))
	for _, componentName := range inProgress {
		component := dgd.GetComponentByName(componentName)
		if component == nil {
			logger.V(1).Info("component not found in DGD", "componentName", componentName)
			continue
		}
		resourceName := dynamo.GroveComponentResourceName(dgd, componentName)

		var (
			isReady bool
			reason  string
		)
		// Any component represented by a PodCliqueScalingGroup must use the
		// PCSG readiness path. Read failures conservatively keep the component
		// in progress; authoritative readiness returns the error separately.
		if component.UsesPCSG() {
			isReady, reason, _, _, _ = dynamo.CheckPCSGReady(
				ctx,
				r.reader,
				resourceName,
				dgd.Namespace,
				logger,
			)
		} else {
			isReady, reason, _, _, _ = dynamo.CheckPodCliqueReady(
				ctx,
				r.reader,
				resourceName,
				dgd.Namespace,
				logger,
			)
		}
		if !isReady {
			logger.V(1).Info(
				"component not ready",
				"componentName", componentName,
				"resourceName", resourceName,
				"reason", reason,
			)
			updatedInProgress = append(updatedInProgress, componentName)
		}
	}

	return updatedInProgress
}
