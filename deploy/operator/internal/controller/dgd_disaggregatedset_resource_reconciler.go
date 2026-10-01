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
	"maps"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
)

type disaggregatedSetResourceReconciler struct {
	client client.Client
}

func newDisaggregatedSetResourceReconciler(k8sClient client.Client) *disaggregatedSetResourceReconciler {
	return &disaggregatedSetResourceReconciler{client: k8sClient}
}

func (r *disaggregatedSetResourceReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *unstructured.Unstructured,
) (*unstructured.Unstructured, bool, error) {
	current := newDisaggregatedSetObject()
	key := types.NamespacedName{Name: desired.GetName(), Namespace: desired.GetNamespace()}
	err := r.client.Get(ctx, key, current)
	if apierrors.IsNotFound(err) {
		if err := r.client.Create(ctx, desired); err != nil {
			return nil, false, fmt.Errorf("failed to create DisaggregatedSet %s: %w", key, err)
		}
		return desired, true, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("failed to get DisaggregatedSet %s: %w", key, err)
	}
	if !isControlledByBetaDGD(current, dgd) {
		return nil, false, fmt.Errorf(
			"refusing to reconcile DisaggregatedSet %s because it is not controlled by DynamoGraphDeployment %s/%s",
			key,
			dgd.Namespace,
			dgd.Name,
		)
	}

	original := current.DeepCopy()
	labels := maps.Clone(current.GetLabels())
	if labels == nil {
		labels = map[string]string{}
	}
	maps.Copy(labels, desired.GetLabels())
	current.SetLabels(labels)
	annotations := maps.Clone(current.GetAnnotations())
	if annotations == nil && len(desired.GetAnnotations()) > 0 {
		annotations = map[string]string{}
	}
	maps.Copy(annotations, desired.GetAnnotations())
	current.SetAnnotations(annotations)
	setDGDControllerOwnerReference(dgd, current)
	current.Object["spec"] = desired.Object["spec"]
	if disaggregatedSetDesiredStateEqual(original, current) {
		return current, false, nil
	}

	if err := r.client.Patch(ctx, current, client.MergeFrom(original), client.DryRunAll); err != nil {
		return nil, false, fmt.Errorf("failed to dry-run patch DisaggregatedSet %s: %w", key, err)
	}
	if disaggregatedSetDesiredStateEqual(original, current) {
		return current, false, nil
	}
	if err := r.client.Patch(ctx, current, client.MergeFrom(original)); err != nil {
		return nil, false, fmt.Errorf("failed to patch DisaggregatedSet %s: %w", key, err)
	}
	return current, true, nil
}

func disaggregatedSetDesiredStateEqual(a, b *unstructured.Unstructured) bool {
	if a == nil || b == nil {
		return a == b
	}
	return equality.Semantic.DeepEqual(a.Object["spec"], b.Object["spec"]) &&
		equality.Semantic.DeepEqual(a.GetLabels(), b.GetLabels()) &&
		equality.Semantic.DeepEqual(a.GetAnnotations(), b.GetAnnotations()) &&
		equality.Semantic.DeepEqual(a.GetOwnerReferences(), b.GetOwnerReferences())
}

func (r *disaggregatedSetResourceReconciler) RestartAnnotations(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) (map[string]string, error) {
	ds := newDisaggregatedSetObject()
	key := types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}
	if err := r.client.Get(ctx, key, ds); err != nil {
		if apierrors.IsNotFound(err) {
			return map[string]string{}, nil
		}
		return nil, fmt.Errorf("failed to get DisaggregatedSet %s: %w", key, err)
	}
	return restartAnnotationsFromDisaggregatedSet(ds, selection)
}

func restartAnnotationsFromDisaggregatedSet(
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
) (map[string]string, error) {
	restartAnnotations := make(map[string]string)
	if ds == nil {
		return restartAnnotations, nil
	}
	spec, found, err := unstructured.NestedMap(ds.Object, "spec")
	if err != nil {
		return nil, fmt.Errorf("failed to read DisaggregatedSet spec: %w", err)
	}
	if !found {
		return restartAnnotations, nil
	}
	typedSpec := disaggregatedsetv1.DisaggregatedSetSpec{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(spec, &typedSpec); err != nil {
		return nil, fmt.Errorf("failed to decode DisaggregatedSet spec: %w", err)
	}
	roleToComponent := make(map[string]string, len(selection.componentToRole))
	for componentName, roleName := range selection.componentToRole {
		roleToComponent[roleName] = componentName
	}
	for i := range typedSpec.Roles {
		role := &typedSpec.Roles[i]
		componentName, selected := roleToComponent[role.Name]
		if !selected {
			continue
		}
		if role.Spec.LeaderWorkerTemplate.LeaderTemplate != nil {
			if timestamp := role.Spec.LeaderWorkerTemplate.LeaderTemplate.Annotations[consts.RestartAnnotation]; timestamp != "" {
				restartAnnotations[componentName] = timestamp
				continue
			}
		}
		if timestamp := role.Spec.LeaderWorkerTemplate.WorkerTemplate.Annotations[consts.RestartAnnotation]; timestamp != "" {
			restartAnnotations[componentName] = timestamp
		}
	}
	return restartAnnotations, nil
}
