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
	"fmt"
	"strconv"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// reconcileDGDPowerBudget materializes the immutable annotation policy once.
// waitForObservation is true after a create or status bind; callers must stop
// before reconciling workloads so the next pass observes the durable DGPB.
func reconcileDGDPowerBudget(
	ctx context.Context,
	kubeClient client.Client,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (waitForObservation bool, err error) {
	spec, enrolled, err := dgdPowerBudgetSpec(dgd.Annotations)
	if err != nil || !enrolled {
		return false, err
	}

	key := types.NamespacedName{Namespace: dgd.Namespace, Name: dgd.Name}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(ctx, key, dgpb); err != nil {
		if !apierrors.IsNotFound(err) {
			return false, fmt.Errorf("read DynamoGraphPowerBudget %s: %w", key, err)
		}
		dgpb = &nvidiacomv1beta1.DynamoGraphPowerBudget{
			ObjectMeta: metav1.ObjectMeta{Namespace: dgd.Namespace, Name: dgd.Name},
			Spec:       spec,
		}
		if err := controllerutil.SetControllerReference(dgd, dgpb, kubeClient.Scheme()); err != nil {
			return false, fmt.Errorf("bind DynamoGraphPowerBudget %s owner: %w", key, err)
		}
		if err := kubeClient.Create(ctx, dgpb); err != nil {
			if apierrors.IsAlreadyExists(err) {
				return true, nil
			}
			return false, fmt.Errorf("create DynamoGraphPowerBudget %s: %w", key, err)
		}
		dgpb.Status.DGDUID = string(dgd.UID)
		dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing
		if err := kubeClient.Status().Update(ctx, dgpb); err != nil {
			return false, fmt.Errorf("initialize DynamoGraphPowerBudget %s status: %w", key, err)
		}
		return true, nil
	}

	owner := metav1.GetControllerOf(dgpb)
	if owner == nil || owner.APIVersion != nvidiacomv1beta1.GroupVersion.String() ||
		owner.Kind != nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind || owner.Name != dgd.Name || owner.UID != dgd.UID {
		return false, fmt.Errorf("DynamoGraphPowerBudget %s is not controlled by DGD UID %q", key, dgd.UID)
	}
	if dgpb.Spec != spec {
		return false, fmt.Errorf(
			"DynamoGraphPowerBudget %s spec does not match immutable DGD annotation policy",
			key,
		)
	}
	if dgpb.Status.DGDUID != "" && dgpb.Status.DGDUID != string(dgd.UID) {
		return false, fmt.Errorf(
			"DynamoGraphPowerBudget %s status is bound to DGD UID %q, want %q",
			key,
			dgpb.Status.DGDUID,
			dgd.UID,
		)
	}
	if dgpb.Status.DGDUID == "" {
		dgpb.Status.DGDUID = string(dgd.UID)
		if dgpb.Status.Phase == "" {
			dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing
		}
		if err := kubeClient.Status().Update(ctx, dgpb); err != nil {
			return false, fmt.Errorf("bind DynamoGraphPowerBudget %s status: %w", key, err)
		}
		return true, nil
	}
	return false, nil
}

// reconcileTransactionalPowerBootstrap persists the create-only DGDSA seed
// vector and enforces the durable DGPB fence before workload programs run.
func (r *DynamoGraphDeploymentReconciler) reconcileTransactionalPowerBootstrap(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (waitForObservation bool, err error) {
	if dgd.Annotations[nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation] !=
		nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence {
		return false, nil
	}

	waitForObservation, err = newDGDScalingAdaptersReconciler(r.Client, r.Recorder).
		ReconcileTransactionalReplicas(ctx, dgd)
	if err != nil {
		return false, fmt.Errorf("reconcile create-time DGDSA seed vector: %w", err)
	}
	return waitForObservation, nil
}

func dgdPowerBudgetSpec(annotations map[string]string) (nvidiacomv1beta1.DynamoGraphPowerBudgetSpec, bool, error) {
	keys := []string{
		nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation,
		nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation,
		nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation,
	}
	enrolled := false
	for _, key := range keys {
		if _, exists := annotations[key]; exists {
			enrolled = true
			break
		}
	}
	if !enrolled {
		return nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{}, false, nil
	}
	for _, key := range keys {
		if _, exists := annotations[key]; !exists {
			return nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{}, true, fmt.Errorf("power-control annotation %q is required", key)
		}
	}
	if mode := annotations[nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation]; mode != nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence {
		return nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{}, true, fmt.Errorf("unsupported power-control mode %q", mode)
	}
	budget, err := parsePositivePowerAnnotation(annotations[nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation], 64)
	if err != nil {
		return nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{}, true, fmt.Errorf("invalid %s: %w", nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation, err)
	}
	minEndpoint, err := parsePositivePowerAnnotation(annotations[nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation], 32)
	if err != nil {
		return nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{}, true, fmt.Errorf("invalid %s: %w", nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation, err)
	}
	return nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
		BudgetWatts: budget,
		Policy: nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{
			MinEndpoint: int32(minEndpoint),
		},
	}, true, nil
}

func parsePositivePowerAnnotation(value string, bitSize int) (int64, error) {
	if value == "" || strings.Trim(value, "0123456789") != "" {
		return 0, fmt.Errorf("must be a positive base-10 integer")
	}
	parsed, err := strconv.ParseInt(value, 10, bitSize)
	if err != nil || parsed < 1 {
		return 0, fmt.Errorf("must be a positive base-10 integer")
	}
	return parsed, nil
}
