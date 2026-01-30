/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
)

// DynamoGraphDeploymentScalingAdapterReconciler reconciles a DynamoGraphDeploymentScalingAdapter object
type DynamoGraphDeploymentScalingAdapterReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Config   commonController.Config
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentscalingadapters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentscalingadapters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;update;patch

// Reconcile implements the reconciliation loop for DynamoGraphDeploymentScalingAdapter
func (r *DynamoGraphDeploymentScalingAdapterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// 1. Fetch the DynamoGraphDeploymentScalingAdapter
	adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
	if err := r.Get(ctx, req.NamespacedName, adapter); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Skip reconciliation if being deleted
	if !adapter.GetDeletionTimestamp().IsZero() {
		logger.V(1).Info("Adapter is being deleted, skipping reconciliation")
		return ctrl.Result{}, nil
	}

	// 2. Fetch the referenced DGD
	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	dgdKey := types.NamespacedName{
		Name:      adapter.Spec.DGDRef.Name,
		Namespace: adapter.Namespace,
	}
	if err := r.Get(ctx, dgdKey, dgd); err != nil {
		if errors.IsNotFound(err) {
			logger.Error(err, "Referenced DGD not found", "dgd", dgdKey)
			// DGD doesn't exist, can't proceed
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, err
	}

	// 3. Find the target service in DGD's spec.services map
	component, exists := dgd.Spec.Services[adapter.Spec.DGDRef.ServiceName]
	if !exists || component == nil {
		logger.Error(nil, "Service not found in DGD",
			"service", adapter.Spec.DGDRef.ServiceName,
			"dgd", dgd.Name,
			"availableServices", getServiceKeys(dgd.Spec.Services))
		return ctrl.Result{}, fmt.Errorf("service %s not found in DGD", adapter.Spec.DGDRef.ServiceName)
	}

	// Get current replicas from DGD (default to 1 if not set)
	currentReplicas := int32(1)
	if component.Replicas != nil {
		currentReplicas = *component.Replicas
	}

	// 4. Update DGD if replicas changed (DGDSA is the source of truth)
	if currentReplicas != adapter.Spec.Replicas {
		// Update the service's replicas in DGD
		component.Replicas = &adapter.Spec.Replicas
		dgd.Spec.Services[adapter.Spec.DGDRef.ServiceName] = component

		if err := r.Update(ctx, dgd); err != nil {
			logger.Error(err, "Failed to update DGD")
			r.Recorder.Eventf(adapter, corev1.EventTypeWarning, "UpdateFailed",
				"Failed to update DGD %s: %v", dgd.Name, err)
			return ctrl.Result{}, err
		}

		logger.Info("Scaled service",
			"dgd", dgd.Name,
			"service", adapter.Spec.DGDRef.ServiceName,
			"from", currentReplicas,
			"to", adapter.Spec.Replicas)

		r.Recorder.Eventf(adapter, corev1.EventTypeNormal, "Scaled",
			"Scaled service %s from %d to %d replicas", adapter.Spec.DGDRef.ServiceName, currentReplicas, adapter.Spec.Replicas)

		// Record scaling event
		now := metav1.Now()
		adapter.Status.LastScaleTime = &now
	}

	// 5. Update adapter status
	adapter.Status.Replicas = adapter.Spec.Replicas
	adapter.Status.Selector = r.buildPodSelector(dgd, adapter.Spec.DGDRef.ServiceName)

	if err := r.Status().Update(ctx, adapter); err != nil {
		logger.Error(err, "Failed to update adapter status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// buildPodSelector constructs a label selector for the pods managed by this service
func (r *DynamoGraphDeploymentScalingAdapterReconciler) buildPodSelector(dgd *nvidiacomv1alpha1.DynamoGraphDeployment, serviceName string) string {
	// Pods are labeled with:
	// - nvidia.com/dynamo-graph-deployment-name = dgd.Name
	// - nvidia.com/dynamo-component = serviceName (the key from spec.services map)
	return fmt.Sprintf("%s=%s,%s=%s",
		consts.KubeLabelDynamoGraphDeploymentName, dgd.Name,
		consts.KubeLabelDynamoComponent, serviceName)
}

// SetupWithManager sets up the controller with the Manager
func (r *DynamoGraphDeploymentScalingAdapterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}, builder.WithPredicates(
			predicate.GenerationChangedPredicate{},
		)).
		Named(consts.ResourceTypeDynamoGraphDeploymentScalingAdapter).
		// Watch DGDs to sync status when DGD service replicas change
		Watches(
			&nvidiacomv1alpha1.DynamoGraphDeployment{},
			handler.EnqueueRequestsFromMapFunc(r.findAdaptersForDGD),
			builder.WithPredicates(predicate.Funcs{
				CreateFunc: func(ce event.CreateEvent) bool { return false },
				DeleteFunc: func(de event.DeleteEvent) bool { return true },
				UpdateFunc: func(ue event.UpdateEvent) bool {
					// Only trigger on spec changes (not status)
					oldDGD, okOld := ue.ObjectOld.(*nvidiacomv1alpha1.DynamoGraphDeployment)
					newDGD, okNew := ue.ObjectNew.(*nvidiacomv1alpha1.DynamoGraphDeployment)
					if !okOld || !okNew {
						return false
					}
					// Trigger if services map changed
					return !servicesEqual(oldDGD.Spec.Services, newDGD.Spec.Services)
				},
				GenericFunc: func(ge event.GenericEvent) bool { return false },
			}),
		).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config)).
		Complete(observability.NewObservedReconciler(r, consts.ResourceTypeDynamoGraphDeploymentScalingAdapter))
}

// findAdaptersForDGD maps DGD changes to adapter reconcile requests
// Uses label selector to efficiently query only adapters for this specific DGD
func (r *DynamoGraphDeploymentScalingAdapterReconciler) findAdaptersForDGD(ctx context.Context, obj client.Object) []reconcile.Request {
	dgd, ok := obj.(*nvidiacomv1alpha1.DynamoGraphDeployment)
	if !ok {
		return nil
	}

	// Use label selector to filter at API level (more efficient than in-memory filtering)
	adapterList := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterList{}
	if err := r.List(ctx, adapterList,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list adapters for DGD", "dgd", dgd.Name)
		return nil
	}

	// All returned adapters are guaranteed to belong to this DGD
	requests := make([]reconcile.Request, 0, len(adapterList.Items))
	for i := range adapterList.Items {
		requests = append(requests, reconcile.Request{
			NamespacedName: types.NamespacedName{
				Name:      adapterList.Items[i].Name,
				Namespace: adapterList.Items[i].Namespace,
			},
		})
	}

	return requests
}
