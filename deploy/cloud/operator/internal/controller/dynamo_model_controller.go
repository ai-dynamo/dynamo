/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/modelendpoint"
)

const (
	// Condition types
	ConditionTypeEndpointsReady = "EndpointsReady"
	ConditionTypeServicesFound  = "ServicesFound"

	// Condition reasons
	ReasonEndpointsDiscovered = "EndpointsDiscovered"
	ReasonNoReadyEndpoints    = "NoReadyEndpoints"
	ReasonNoEndpoints         = "NoEndpoints"
	ReasonServicesFound       = "ServicesFound"
	ReasonNoServicesFound     = "NoServicesFound"

	// Field index names
	dynamoModelBaseModelIndex = ".spec.baseModelName"
)

// DynamoModelReconciler reconciles a DynamoModel object
type DynamoModelReconciler struct {
	client.Client
	Recorder       record.EventRecorder
	EndpointClient *modelendpoint.Client
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomodels,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomodels/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomodels/finalizers,verbs=update
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch
// +kubebuilder:rbac:groups=discovery.k8s.io,resources=endpointslices,verbs=get;list;watch

// Reconcile handles the reconciliation loop for DynamoModel resources
func (r *DynamoModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logs := log.FromContext(ctx)

	// Fetch the DynamoModel
	model := &v1alpha1.DynamoModel{}
	if err := r.Get(ctx, req.NamespacedName, model); err != nil {
		if k8serrors.IsNotFound(err) {
			logs.Info("DynamoModel resource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logs.Error(err, "Failed to get DynamoModel")
		return ctrl.Result{}, err
	}

	logs = logs.WithValues("dynamoModel", model.Name, "namespace", model.Namespace, "baseModelName", model.Spec.BaseModelName)
	logs.Info("Reconciling DynamoModel")

	// Handle finalizer using common handler
	finalized, err := commoncontroller.HandleFinalizer(ctx, model, r.Client, r)
	if err != nil {
		return ctrl.Result{}, err
	}
	if finalized {
		// Object was being deleted and finalizer has been called
		return ctrl.Result{}, nil
	}

	// Get endpoint candidates (common logic)
	candidates, serviceNames, err := r.getEndpointCandidates(ctx, model)
	if err != nil {
		// Error already logged and status updated in helper
		return ctrl.Result{RequeueAfter: 30 * time.Second}, err
	}

	if len(candidates) == 0 {
		msg := fmt.Sprintf("No endpoint slices found for base model %s", model.Spec.BaseModelName)
		logs.Info(msg)
		r.Recorder.Event(model, corev1.EventTypeWarning, "NoEndpointsFound", msg)
		r.updateCondition(model, ConditionTypeServicesFound, metav1.ConditionFalse, ReasonNoServicesFound, msg)
		r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionFalse, ReasonNoEndpoints, msg)
		model.Status.Endpoints = nil
		model.Status.TotalEndpoints = 0
		model.Status.ReadyEndpoints = 0
		if err := r.Status().Update(ctx, model); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
	}

	// Initialize endpoint client if needed
	if r.EndpointClient == nil {
		r.EndpointClient = modelendpoint.NewClient()
	}

	// Load LoRA on all endpoints in parallel with bounded concurrency
	allEndpoints, probeErr := r.EndpointClient.LoadLoRA(ctx, candidates, model)
	hasFailures := probeErr != nil || countReadyEndpoints(allEndpoints) < len(allEndpoints)

	if probeErr != nil {
		logs.Error(probeErr, "Some endpoints failed during probing")
		r.Recorder.Event(model, corev1.EventTypeWarning, "PartialEndpointFailure",
			fmt.Sprintf("Some endpoints failed to load LoRA: %v", probeErr))
	}

	// Update service found condition based on whether we found any services
	if len(serviceNames) > 0 {
		r.updateCondition(model, ConditionTypeServicesFound, metav1.ConditionTrue, ReasonServicesFound,
			fmt.Sprintf("Found %d service(s)", len(serviceNames)))
	} else {
		r.updateCondition(model, ConditionTypeServicesFound, metav1.ConditionFalse, ReasonNoServicesFound,
			"No services associated with endpoint slices")
	}

	// Update status
	model.Status.Endpoints = allEndpoints
	model.Status.TotalEndpoints = len(allEndpoints)
	model.Status.ReadyEndpoints = countReadyEndpoints(allEndpoints)

	// Update conditions
	if model.Status.ReadyEndpoints > 0 {
		r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionTrue, ReasonEndpointsDiscovered,
			fmt.Sprintf("Found %d ready endpoint(s) out of %d total", model.Status.ReadyEndpoints, model.Status.TotalEndpoints))
		r.Recorder.Eventf(model, corev1.EventTypeNormal, "EndpointsReady",
			"Discovered %d ready endpoints for base model %s", model.Status.ReadyEndpoints, model.Spec.BaseModelName)
	} else if model.Status.TotalEndpoints > 0 {
		r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionFalse, ReasonNoReadyEndpoints,
			fmt.Sprintf("Found %d endpoint(s) but none are ready", model.Status.TotalEndpoints))
		r.Recorder.Event(model, corev1.EventTypeWarning, "NoReadyEndpoints", "Endpoints exist but none are ready")
	} else {
		r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionFalse, ReasonNoEndpoints, "No endpoints found")
	}

	if err := r.Status().Update(ctx, model); err != nil {
		logs.Error(err, "Failed to update DynamoModel status")
		return ctrl.Result{}, err
	}

	logs.Info("Successfully reconciled DynamoModel",
		"totalEndpoints", model.Status.TotalEndpoints,
		"readyEndpoints", model.Status.ReadyEndpoints)

	// Requeue if there were probe failures to retry loading LoRAs
	if hasFailures {
		logs.Info("Requeuing due to endpoint probe failures",
			"ready", model.Status.ReadyEndpoints,
			"total", model.Status.TotalEndpoints)
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
	}

	return ctrl.Result{}, nil
}

// countReadyEndpoints counts how many endpoints are ready
func countReadyEndpoints(endpoints []v1alpha1.EndpointInfo) int {
	count := 0
	for _, ep := range endpoints {
		if ep.Ready {
			count++
		}
	}
	return count
}

// updateCondition updates or adds a condition to the model's status
func (r *DynamoModelReconciler) updateCondition(model *v1alpha1.DynamoModel, condType string, status metav1.ConditionStatus, reason, message string) {
	condition := metav1.Condition{
		Type:               condType,
		Status:             status,
		ObservedGeneration: model.Generation,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
	meta.SetStatusCondition(&model.Status.Conditions, condition)
}

// SetupWithManager sets up the controller with the Manager
func (r *DynamoModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// Register field indexer for DynamoModels by base model name
	// This allows efficient O(1) queries: "get all DynamoModels for base-model X"
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&v1alpha1.DynamoModel{},
		dynamoModelBaseModelIndex,
		func(obj client.Object) []string {
			model := obj.(*v1alpha1.DynamoModel)
			return []string{model.Spec.BaseModelName}
		},
	); err != nil {
		return err
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.DynamoModel{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		// Watch EndpointSlices - reconcile when endpoints change (Service changes trigger EndpointSlice updates)
		Watches(
			&discoveryv1.EndpointSlice{},
			handler.EnqueueRequestsFromMapFunc(r.findModelsForEndpointSlice),
			builder.WithPredicates(predicate.Funcs{
				GenericFunc: func(e event.GenericEvent) bool { return false },
			}),
		).
		Complete(r)
}

// findModelsForEndpointSlice maps an EndpointSlice to DynamoModels
func (r *DynamoModelReconciler) findModelsForEndpointSlice(ctx context.Context, obj client.Object) []reconcile.Request {
	slice := obj.(*discoveryv1.EndpointSlice)
	logs := log.FromContext(ctx).WithValues("endpointSlice", slice.Name, "namespace", slice.Namespace)

	// Get the service name from the EndpointSlice label
	// Since service name = base model name, we can directly use it to find models
	serviceName, ok := slice.Labels[discoveryv1.LabelServiceName]
	if !ok {
		return nil
	}

	// Find all DynamoModels for this base model using field indexer
	requests, err := modelendpoint.FindModelsForBaseModel(ctx, r.Client, slice.Namespace, serviceName, dynamoModelBaseModelIndex)
	if err != nil {
		return nil
	}

	if len(requests) > 0 {
		logs.V(1).Info("EndpointSlice change triggered DynamoModel reconciliation", "modelCount", len(requests), "baseModel", serviceName)
	}

	return requests
}

// FinalizeResource implements the Finalizer interface
// Performs cleanup when a DynamoModel is being deleted
func (r *DynamoModelReconciler) FinalizeResource(ctx context.Context, model *v1alpha1.DynamoModel) error {
	logs := log.FromContext(ctx)

	logs.Info("Finalizing DynamoModel", "modelType", model.Spec.ModelType)

	// Only perform cleanup for LoRA models
	if model.Spec.ModelType == "lora" {
		// Get endpoint candidates (reusing common logic)
		candidates, _, err := r.getEndpointCandidates(ctx, model)
		if err != nil {
			logs.Info("Failed to get endpoints during deletion, continuing with resource deletion",
				"error", err.Error())
			r.Recorder.Event(model, corev1.EventTypeWarning, "CleanupFailed", err.Error())
			// Continue with deletion even if we can't get endpoints
		} else if len(candidates) > 0 {
			logs.Info("Unloading LoRA from endpoints", "endpointCount", len(candidates))

			// Initialize endpoint client if needed
			if r.EndpointClient == nil {
				r.EndpointClient = modelendpoint.NewClient()
			}

			// Unload LoRA from all endpoints in parallel
			if err := r.EndpointClient.UnloadLoRA(ctx, candidates, model.Spec.ModelName); err != nil {
				// Log as Info since we're continuing with deletion anyway (expected behavior)
				// Detailed failure information is already logged by the prober
				logs.Info("Some endpoints failed to unload LoRA, continuing with deletion",
					"error", err.Error())
				r.Recorder.Event(model, corev1.EventTypeWarning, "LoRAUnloadFailed",
					fmt.Sprintf("Failed to unload LoRA from some endpoints: %v", err))
				// Continue with deletion even if unload fails
			} else {
				logs.Info("Successfully unloaded LoRA from all endpoints")
				r.Recorder.Event(model, corev1.EventTypeNormal, "LoRAUnloaded",
					fmt.Sprintf("Unloaded LoRA from %d endpoint(s)", len(candidates)))
			}
		} else {
			logs.Info("No endpoints found for cleanup")
		}
	} else {
		logs.Info("Skipping cleanup for non-LoRA model")
	}

	logs.Info("Finalization completed successfully")
	return nil
}

// getEndpointCandidates fetches EndpointSlices and extracts endpoint candidates
// Returns candidates, service names, and error
func (r *DynamoModelReconciler) getEndpointCandidates(
	ctx context.Context,
	model *v1alpha1.DynamoModel,
) ([]modelendpoint.Candidate, map[string]bool, error) {
	logs := log.FromContext(ctx)

	// Hash the base model name for label-based discovery
	modelHash := dynamo.HashModelName(model.Spec.BaseModelName)

	// Query EndpointSlices directly by base model hash label
	// This label propagates from the Service to its EndpointSlices
	endpointSlices := &discoveryv1.EndpointSliceList{}
	if err := r.List(ctx, endpointSlices,
		client.InNamespace(model.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoBaseModelHash: modelHash},
	); err != nil {
		logs.Error(err, "Failed to list endpoint slices for model")
		r.Recorder.Event(model, corev1.EventTypeWarning, "EndpointDiscoveryFailed", err.Error())
		return nil, nil, err
	}

	if len(endpointSlices.Items) == 0 {
		return nil, nil, nil
	}

	logs.Info("Found endpoint slices for model", "count", len(endpointSlices.Items))

	// Extract pod-ready endpoint candidates from all EndpointSlices
	candidates, serviceNames := modelendpoint.ExtractCandidates(endpointSlices, int32(consts.DynamoSystemPort))

	return candidates, serviceNames, nil
}
