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
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	schedulergrovev1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdPlacementScoreReconciler projects the scheduler's placement observation for
// the DGD's PodGangs into the program-owned DGD status. It never writes the
// status subresource itself.
type dgdPlacementScoreReconciler struct {
	reader client.Reader
}

func newDGDPlacementScoreReconciler(
	reader client.Reader,
) *dgdPlacementScoreReconciler {
	return &dgdPlacementScoreReconciler{reader: reader}
}

// Reconcile replaces status.placement with the current observation. Every
// outcome that is not a usable score clears any previously reported score, so a
// stale value cannot survive a failed or incomplete observation.
func (r *dgdPlacementScoreReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	result *workloadProgramResult,
) {
	score, state := r.observe(ctx, dgd)
	result.Status.Placement = &nvidiacomv1beta1.PlacementStatus{
		Score: score,
		State: state,
	}
}

// observe resolves the placement score for the DGD's current PodCliqueSet
// generation. Absence is not an error: a PodCliqueSet or PodGang the cache has
// not observed yet is an indeterminate score, and the Grove watches requeue the
// DGD once the scheduler publishes it.
func (r *dgdPlacementScoreReconciler) observe(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (*float64, nvidiacomv1beta1.PlacementScoreState) {
	logger := log.FromContext(ctx)
	pcsName := dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components)

	// The PodCliqueSet carries the generation hash that scopes the observation.
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.reader.Get(ctx, types.NamespacedName{Name: pcsName, Namespace: dgd.Namespace}, pcs); err != nil {
		logger.V(1).Info(
			"no PodCliqueSet observed for placement score",
			"podCliqueSet", pcsName,
			"error", err,
		)
		return nil, nvidiacomv1beta1.PlacementScoreStateUnknown
	}

	// Scope the selection to the current generation so a PodGang draining from a
	// previous spec cannot lower the reported score.
	selectorLabels := map[string]string{
		grovecommon.LabelPartOfKey:    pcsName,
		grovecommon.LabelComponentKey: grovecommon.LabelComponentNamePodGang,
	}
	if hash := pcs.Status.CurrentGenerationHash; hash != nil && *hash != "" {
		selectorLabels[grovecommon.LabelPodCliqueSetGenerationHash] = *hash
	}

	gangs := &schedulergrovev1alpha1.PodGangList{}
	if err := r.reader.List(
		ctx,
		gangs,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels(selectorLabels),
	); err != nil {
		logger.V(1).Info(
			"failed to list PodGangs for placement score",
			"podCliqueSet", pcsName,
			"error", err,
		)
		return nil, nvidiacomv1beta1.PlacementScoreStateUnknown
	}

	return dynamo.AggregatePlacementScore(gangs.Items)
}
