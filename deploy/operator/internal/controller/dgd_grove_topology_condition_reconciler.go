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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
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

// dgdGroveTopologyConditionReconciler projects Grove's topology observation
// into the program-owned DGD status and queues any resulting status event. It
// never writes the status subresource itself.
type dgdGroveTopologyConditionReconciler struct {
	reader client.Reader
}

func newDGDGroveTopologyConditionReconciler(
	reader client.Reader,
) *dgdGroveTopologyConditionReconciler {
	return &dgdGroveTopologyConditionReconciler{reader: reader}
}

func (r *dgdGroveTopologyConditionReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	result *workloadProgramResult,
) {
	// Project status only when typed or provider-native topology is configured.
	if result == nil || (!dgd.HasAnyTopologyConstraint() && !provideroverride.HasGroveTopologyOverrides(dgd)) {
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
