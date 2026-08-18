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
	"reflect"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
)

// DynamoGraphDeploymentScalingAdapterReconciler reconciles a DynamoGraphDeploymentScalingAdapter object
type DynamoGraphDeploymentScalingAdapterReconciler struct {
	client.Client
	Scheme        *runtime.Scheme
	Recorder      events.EventRecorder
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentscalingadapters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentscalingadapters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographpowerbudgets,verbs=get;list;watch

// Reconcile implements the reconciliation loop for DynamoGraphDeploymentScalingAdapter
func (r *DynamoGraphDeploymentScalingAdapterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// 1. Fetch the DynamoGraphDeploymentScalingAdapter
	adapter := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}
	if err := r.Get(ctx, req.NamespacedName, adapter); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Skip reconciliation if being deleted
	if !adapter.GetDeletionTimestamp().IsZero() {
		logger.V(1).Info("Adapter is being deleted, skipping reconciliation")
		return ctrl.Result{}, nil
	}

	// 2. Fetch the referenced DGD
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
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

	// 3. Find the target component in the DGD's components list.
	componentName := adapter.Spec.DGDRef.ComponentName
	component := dgd.GetComponentByName(componentName)
	if component == nil {
		logger.Info("Component referenced by adapter not found in DGD; waiting for adapter or DGD update",
			"component", componentName,
			"dgd", dgd.Name,
			"availableComponents", getComponentNames(dgd.Spec.Components))
		return ctrl.Result{}, nil
	}
	transactional := dgd.Annotations[nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation] ==
		nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence &&
		dynamo.IsWorkerComponent(string(component.ComponentType))
	if component.ScalingAdapter == nil && !transactional {
		logger.V(1).Info("Component no longer uses a scaling adapter; skipping replica propagation",
			"component", componentName,
			"dgd", dgd.Name)
		return ctrl.Result{}, nil
	}

	// Get current replicas from DGD (default to 1 if not set)
	currentReplicas := int32(1)
	if component.Replicas != nil {
		currentReplicas = *component.Replicas
	}

	// 4. Static adapters remain the DGD source of truth; transactional adapters are request-only.
	if !transactional && currentReplicas != adapter.Spec.Replicas {
		// Update the component's replicas in DGD.
		component.Replicas = &adapter.Spec.Replicas

		if err := r.Update(ctx, dgd); err != nil {
			logger.Error(err, "Failed to update DGD")
			r.Recorder.Eventf(adapter, dgd, corev1.EventTypeWarning, "UpdateFailed", "Update",
				"Failed to update DGD %s: %v", dgd.Name, err)
			return ctrl.Result{}, err
		}

		logger.Info("Scaled component",
			"dgd", dgd.Name,
			"component", componentName,
			"from", currentReplicas,
			"to", adapter.Spec.Replicas)

		r.Recorder.Eventf(adapter, dgd, corev1.EventTypeNormal, "Scaled", "Scale",
			"Scaled component %s from %d to %d replicas", componentName, currentReplicas, adapter.Spec.Replicas)

		// Record scaling event
		now := metav1.Now()
		adapter.Status.LastScaleTime = &now
		currentReplicas = adapter.Spec.Replicas
	}

	// 5. The Scale subresource status reports observed capacity. Static mode
	// retains its historical mirrored-target behavior; transactional mode keeps
	// requested, durably committed, and actually observed replicas distinct.
	actualReplicas := currentReplicas
	if transactional {
		actualReplicas = dgd.Status.Components[componentName].Replicas
		dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
		if err := r.Get(ctx, dgdKey, dgpb); err != nil {
			return ctrl.Result{}, err
		}
		committedReplicas, found := dgpb.Status.CommittedReplicaTargets[componentName]
		if !found {
			return ctrl.Result{}, fmt.Errorf(
				"power budget %s/%s has no committed target for component %q",
				dgpb.Namespace,
				dgpb.Name,
				componentName,
			)
		}
		if adapter.Spec.Replicas != committedReplicas &&
			(adapter.Status.PendingReason == "" ||
				adapter.Status.RequestedReplicas != adapter.Spec.Replicas ||
				adapter.Status.CommittedReplicas != committedReplicas) {
			// The DGD reconciler owns the complete-vector decision and publishes
			// its exact bounded reason. A reason for an older request must not be
			// rebound to this new tuple before that decision completes.
			if adapter.Status.PendingReason == "" {
				return ctrl.Result{}, nil
			}
			adapter.Status.ActualReplicas = actualReplicas
			adapter.Status.Replicas = actualReplicas
			adapter.Status.Selector = r.buildPodSelector(dgd.Name, componentName)
			if err := r.Status().Update(ctx, adapter); err != nil {
				logger.Error(err, "Failed to update adapter actual status")
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil
		}
		preserveInitializingZeroReason :=
			dgpb.Status.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing &&
				adapter.Spec.Replicas == 0 && committedReplicas == 0 &&
				adapter.Status.RequestedReplicas == 0 && adapter.Status.CommittedReplicas == 0 &&
				(adapter.Status.PendingReason == nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline ||
					adapter.Status.PendingReason == nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBelowMinimum)
		adapter.Status.RequestedReplicas = adapter.Spec.Replicas
		adapter.Status.CommittedReplicas = committedReplicas
		adapter.Status.ActualReplicas = actualReplicas
		if adapter.Spec.Replicas == committedReplicas && !preserveInitializingZeroReason {
			adapter.Status.PendingReason = ""
		}
	} else {
		adapter.Status.RequestedReplicas = actualReplicas
		adapter.Status.CommittedReplicas = actualReplicas
		adapter.Status.ActualReplicas = actualReplicas
		adapter.Status.PendingReason = ""
	}
	adapter.Status.Replicas = actualReplicas
	adapter.Status.Selector = r.buildPodSelector(dgd.Name, componentName)

	if err := r.Status().Update(ctx, adapter); err != nil {
		logger.Error(err, "Failed to update adapter status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// buildPodSelector constructs a label selector for the pods managed by this component.
func (r *DynamoGraphDeploymentScalingAdapterReconciler) buildPodSelector(dgdName, componentName string) string {
	// Pods are labeled with:
	// - nvidia.com/dynamo-graph-deployment-name = dgd.Name
	// - nvidia.com/dynamo-component = componentName
	return labels.SelectorFromSet(labels.Set{
		consts.KubeLabelDynamoGraphDeploymentName: dgdName,
		consts.KubeLabelDynamoComponent:           componentName,
	}).String()
}

// SetupWithManager sets up the controller with the Manager
func (r *DynamoGraphDeploymentScalingAdapterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{}, builder.WithPredicates(
			predicate.GenerationChangedPredicate{},
		)).
		Named(consts.ResourceTypeDynamoGraphDeploymentScalingAdapter).
		// Watch DGDs to sync both target and observed component replicas.
		Watches(
			&nvidiacomv1beta1.DynamoGraphDeployment{},
			handler.EnqueueRequestsFromMapFunc(r.findAdaptersForDGD),
			builder.WithPredicates(predicate.Funcs{
				CreateFunc: func(ce event.CreateEvent) bool { return false },
				DeleteFunc: func(de event.DeleteEvent) bool { return true },
				UpdateFunc: func(ue event.UpdateEvent) bool {
					oldDGD, okOld := ue.ObjectOld.(*nvidiacomv1beta1.DynamoGraphDeployment)
					newDGD, okNew := ue.ObjectNew.(*nvidiacomv1beta1.DynamoGraphDeployment)
					if !okOld || !okNew {
						return false
					}
					return !componentsEqual(oldDGD.Spec.Components, newDGD.Spec.Components) ||
						!reflect.DeepEqual(oldDGD.Status.Components, newDGD.Status.Components)
				},
				GenericFunc: func(ge event.GenericEvent) bool { return false },
			}),
		).
		Watches(
			&nvidiacomv1beta1.DynamoGraphPowerBudget{},
			handler.EnqueueRequestsFromMapFunc(r.findAdaptersForPowerBudget),
		).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(observability.NewObservedReconciler(r, consts.ResourceTypeDynamoGraphDeploymentScalingAdapter))
}

// findAdaptersForPowerBudget maps a DGPB status change to adapters owned by the
// same namespaced DGD.
func (r *DynamoGraphDeploymentScalingAdapterReconciler) findAdaptersForPowerBudget(
	ctx context.Context,
	obj client.Object,
) []reconcile.Request {
	dgpb, ok := obj.(*nvidiacomv1beta1.DynamoGraphPowerBudget)
	if !ok {
		return nil
	}
	return r.findAdaptersForDGD(ctx, &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: dgpb.Name, Namespace: dgpb.Namespace},
	})
}

// findAdaptersForDGD maps DGD changes to adapter reconcile requests.
func (r *DynamoGraphDeploymentScalingAdapterReconciler) findAdaptersForDGD(ctx context.Context, obj client.Object) []reconcile.Request {
	dgd, ok := obj.(*nvidiacomv1beta1.DynamoGraphDeployment)
	if !ok {
		return nil
	}

	adapterList := &nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapterList{}
	if err := r.List(ctx, adapterList,
		client.InNamespace(dgd.Namespace),
	); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list adapters for DGD", "dgd", dgd.Name)
		return nil
	}

	requests := make([]reconcile.Request, 0, len(adapterList.Items))
	for i := range adapterList.Items {
		if adapterList.Items[i].Spec.DGDRef.Name != dgd.Name {
			continue
		}
		requests = append(requests, reconcile.Request{
			NamespacedName: types.NamespacedName{
				Name:      adapterList.Items[i].Name,
				Namespace: adapterList.Items[i].Namespace,
			},
		})
	}

	return requests
}
