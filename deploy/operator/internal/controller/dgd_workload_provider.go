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
	"errors"
	"fmt"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type workloadProvider string

const (
	workloadProviderComponent        workloadProvider = consts.WorkloadProviderComponent
	workloadProviderGrove            workloadProvider = consts.WorkloadProviderGrove
	workloadProviderDisaggregatedSet workloadProvider = consts.WorkloadProviderDisaggregatedSet
)

var (
	errConflictingWorkloadProviders = errors.New("conflicting workload providers")
	errUnsupportedWorkloadProvider  = errors.New("unsupported workload provider")
)

// ensureWorkloadProvider persists a missing provider before workload reconciliation.
func (r *DynamoGraphDeploymentReconciler) ensureWorkloadProvider(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (workloadProvider, error) {
	// Reuse a materialized provider without consulting mutable routing inputs.
	if value, exists := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; exists {
		return parseWorkloadProvider(value)
	}

	// Adopt the workload family already owned by a legacy unannotated DGD.
	provider, found, err := providerFromOwnedWorkloads(ctx, r.Client, dgd)
	if err != nil {
		return "", err
	}
	if !found {
		provider = providerFromCurrentIntent(r.RuntimeConfig.Gate, dgd)
	}

	// Persist the adopted or creation-time selection with optimistic concurrency control.
	base := dgd.DeepCopy()

	// Materialize the selected provider on the object passed to the API client.
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.KubeAnnotationWorkloadProvider] = string(provider)

	// Persist the selection and retain the API server's returned object state.
	patch := client.MergeFromWithOptions(base, client.MergeFromWithOptimisticLock{})
	if err := r.Patch(ctx, dgd, patch); err != nil {
		dgd.Annotations = base.Annotations
		return "", fmt.Errorf("persist workload provider %q: %w", provider, err)
	}

	return provider, nil
}

func providerFromOwnedWorkloads(
	ctx context.Context,
	reader client.Reader,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (workloadProvider, bool, error) {
	// Observe both workload families before deciding whether either is authoritative.
	hasComponents, err := hasOwnedComponentWorkloads(ctx, reader, dgd)
	if err != nil {
		return "", false, err
	}
	hasGrove, err := hasOwnedGroveWorkloads(ctx, reader, dgd)
	if err != nil {
		return "", false, err
	}
	hasDisaggregatedSet, err := hasOwnedDisaggregatedSet(ctx, reader, dgd)
	if err != nil {
		return "", false, err
	}

	// Adopt one unambiguous family and fail closed when multiple families exist.
	switch {
	case hasComponents && hasGrove || hasComponents && hasDisaggregatedSet || hasGrove && hasDisaggregatedSet:
		return "", false, fmt.Errorf(
			"%w: DynamoGraphDeployment %s/%s owns workloads from multiple providers",
			errConflictingWorkloadProviders,
			dgd.Namespace,
			dgd.Name,
		)
	case hasGrove:
		return workloadProviderGrove, true, nil
	case hasComponents:
		return workloadProviderComponent, true, nil
	case hasDisaggregatedSet:
		return workloadProviderDisaggregatedSet, true, nil
	default:
		return "", false, nil
	}
}

func hasOwnedDisaggregatedSet(
	ctx context.Context,
	reader client.Reader,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	// A DGD owns one deterministic DisaggregatedSet. Treat an unavailable
	// optional API as an empty observable workload family during adoption.
	ds := newDisaggregatedSetObject()
	key := client.ObjectKey{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}
	if err := reader.Get(ctx, key, ds); err != nil {
		if apierrors.IsNotFound(err) || meta.IsNoMatchError(err) {
			return false, nil
		}
		return false, fmt.Errorf("get owned DisaggregatedSet: %w", err)
	}
	return metav1.IsControlledBy(ds, dgd), nil
}

func hasOwnedComponentWorkloads(
	ctx context.Context,
	reader client.Reader,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	// Scan the namespace cache because mutable labels cannot prove legacy ownership.
	list := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := reader.List(ctx, list, client.InNamespace(dgd.Namespace)); err != nil {
		return false, fmt.Errorf("list owned DynamoComponentDeployments: %w", err)
	}

	// Accept only the controller reference as authoritative ownership evidence.
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], dgd) {
			return true, nil
		}
	}
	return false, nil
}

func hasOwnedGroveWorkloads(
	ctx context.Context,
	reader client.Reader,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	// Treat an unavailable optional Grove API as an empty observable workload family.
	list := &grovev1alpha1.PodCliqueSetList{}
	if err := reader.List(ctx, list, client.InNamespace(dgd.Namespace)); err != nil {
		if meta.IsNoMatchError(err) {
			return false, nil
		}
		return false, fmt.Errorf("list owned PodCliqueSets: %w", err)
	}

	// Accept only the controller reference as authoritative ownership evidence.
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], dgd) {
			return true, nil
		}
	}
	return false, nil
}

func providerFromCurrentIntent(
	gate features.Gate,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) workloadProvider {
	// Grove is the default when its gate is enabled unless the DGD opts out.
	if gate.Enabled(features.Grove) &&
		strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse {
		return workloadProviderGrove
	}
	if strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableDisaggregatedSet]) == consts.KubeLabelValueTrue {
		return workloadProviderDisaggregatedSet
	}
	return workloadProviderComponent
}

func parseWorkloadProvider(value string) (workloadProvider, error) {
	// Accept only the workload programs implemented by this controller.
	switch workloadProvider(value) {
	case workloadProviderComponent, workloadProviderGrove, workloadProviderDisaggregatedSet:
		return workloadProvider(value), nil
	default:
		return "", fmt.Errorf("%w: %q", errUnsupportedWorkloadProvider, value)
	}
}
