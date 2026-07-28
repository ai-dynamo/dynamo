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
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groveStatusResolver owns the provider-specific observations used to derive
// restart progress and DGD status contributions. It never writes DGD status.
type groveStatusResolver struct {
	reader client.Reader
}

func newGroveStatusResolver(reader client.Reader) *groveStatusResolver {
	return &groveStatusResolver{reader: reader}
}

// getUpdatedInProgress returns the subset of components whose Grove workload
// has not completed the requested restart.
func (r *groveStatusResolver) getUpdatedInProgress(
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
		if component.GetNumberOfNodes() > 1 || component.IsInterPodGMSEnabled() {
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

// projectTopologyCondition maps Grove's PodCliqueSet topology condition into
// the program's accumulated DGD status. The outer reconciler remains the only
// status-subresource writer.
func (r *groveStatusResolver) projectTopologyCondition(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	result *workloadProgramResult,
) {
	if result == nil || !dgd.HasAnyTopologyConstraint() {
		return
	}
	status := &result.Status
	logger := log.FromContext(ctx)

	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.reader.Get(ctx, types.NamespacedName{
		Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
		Namespace: dgd.Namespace,
	}, pcs); err != nil {
		if !apierrors.IsNotFound(err) {
			logger.V(1).Info("failed to read PCS for topology condition projection", "error", err)
		}
		return
	}

	var groveTopologyCondition *metav1.Condition
	for i := range pcs.Status.Conditions {
		if pcs.Status.Conditions[i].Type == groveconstants.ConditionTopologyLevelsUnavailable {
			groveTopologyCondition = &pcs.Status.Conditions[i]
			break
		}
	}

	var condition metav1.Condition
	if groveTopologyCondition == nil {
		condition = metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable,
			Status:  metav1.ConditionUnknown,
			Reason:  nvidiacomv1beta1.ConditionReasonTopologyConditionPending,
			Message: "Waiting for topology condition from the scheduling framework",
		}
	} else if groveTopologyCondition.Status == metav1.ConditionTrue {
		reason := nvidiacomv1beta1.ConditionReasonTopologyLevelsUnavailable
		if groveTopologyCondition.Reason == groveconstants.ConditionReasonClusterTopologyNotFound {
			reason = nvidiacomv1beta1.ConditionReasonTopologyDefinitionNotFound
		}
		condition = metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable,
			Status:  metav1.ConditionFalse,
			Reason:  reason,
			Message: groveTopologyCondition.Message,
		}
		previous := meta.FindStatusCondition(status.Conditions, nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable)
		if previous == nil ||
			previous.Status != metav1.ConditionFalse ||
			previous.Reason != reason ||
			previous.Message != groveTopologyCondition.Message {
			logger.Info(
				"Topology constraints no longer enforced",
				"reason", reason,
				"message", groveTopologyCondition.Message,
			)
			result.Eventf(
				corev1.EventTypeWarning,
				reason,
				"Topology constraints no longer enforced: %s",
				groveTopologyCondition.Message,
			)
		}
	} else {
		condition = metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable,
			Status:  metav1.ConditionTrue,
			Reason:  nvidiacomv1beta1.ConditionReasonAllTopologyLevelsAvailable,
			Message: "All required topology levels are available in the cluster topology",
		}
	}

	meta.SetStatusCondition(&status.Conditions, condition)
}
