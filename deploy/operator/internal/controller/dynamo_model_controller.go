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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/validation"
)

const (
	// Condition types
	ConditionTypeEndpointsReady = "EndpointsReady"
	ConditionTypeServicesFound  = "ServicesFound"

	// Condition reasons
	ReasonAllEndpointsReady   = "AllEndpointsReady"
	ReasonEndpointsDiscovered = "EndpointsDiscovered"
	ReasonNotReady            = "NotReady"
	ReasonNoEndpoints         = "NoEndpoints"
	ReasonServicesFound       = "ServicesFound"
	ReasonNoServicesFound     = "NoServicesFound"

	// Field index names
	dynamoModelBaseModelHashIndex = ".spec.baseModelNameHash"

	// Requeue duration for retries when endpoints are not ready
	requeueAfterDuration = 30 * time.Second
)

// DynamoModelReconciler reconciles a DynamoModel object
type DynamoModelReconciler struct {
	client.Client
	Recorder       record.EventRecorder
	EndpointClient *modelendpoint.Client
	Config         commoncontroller.Config
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

	// Validate the DynamoModel spec (defense in depth - only when webhooks are disabled)
	if !r.Config.WebhooksEnabled {
		validator := webhookvalidation.NewDynamoModelValidator(model)
		if _, err := validator.Validate(); err != nil {
			logs.Error(err, "DynamoModel validation failed, refusing to reconcile")

			// Set validation error condition
			meta.SetStatusCondition(&model.Status.Conditions, metav1.Condition{
				Type:               "Valid",
				Status:             metav1.ConditionFalse,
				ObservedGeneration: model.Generation,
				Reason:             "ValidationFailed",
				Message:            fmt.Sprintf("Validation failed: %v", err),
			})

			// Update status and don't requeue (user must fix the spec)
			if statusErr := r.Status().Update(ctx, model); statusErr != nil {
				logs.Error(statusErr, "Failed to update DynamoModel status with validation error")
				return ctrl.Result{}, statusErr
			}

			// Record event for visibility
			r.Recorder.Event(model, corev1.EventTypeWarning, "ValidationFailed", err.Error())

			// Don't requeue - user must fix the spec
			logs.Info("DynamoModel is invalid, not reconciling until spec is fixed")
			return ctrl.Result{}, nil
		}

		// Set Valid condition to True
		meta.SetStatusCondition(&model.Status.Conditions, metav1.Condition{
			Type:               "Valid",
			Status:             metav1.ConditionTrue,
			ObservedGeneration: model.Generation,
			Reason:             "ValidationPassed",
			Message:            "DynamoModel spec is valid",
		})
	}

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
		// Let controller-runtime handle retry with exponential backoff
		return ctrl.Result{}, err
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
		// Don't requeue - we're watching EndpointSlices, so we'll be notified when they appear
		return ctrl.Result{}, nil
	}

	// Load LoRA on all endpoints in parallel with bounded concurrency
	allEndpoints, probeErr := r.EndpointClient.LoadLoRA(ctx, candidates, model)

	// Determine if we need to requeue based on model type
	// For LoRA models: requeue if there were probe errors OR if not all endpoints are ready
	// For base models: only requeue if there were probe errors (Ready is expected to be false)
	hasFailures := probeErr != nil
	if model.IsLoRA() {
		hasFailures = hasFailures || countReadyEndpoints(allEndpoints) < len(allEndpoints)
	}

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

	// Update conditions based on model type
	if model.IsLoRA() {
		// For LoRA models, check readiness - condition is True only when ALL endpoints are ready
		if model.Status.ReadyEndpoints == model.Status.TotalEndpoints && model.Status.TotalEndpoints > 0 {
			r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionTrue, ReasonAllEndpointsReady,
				fmt.Sprintf("All %d endpoint(s) are ready", model.Status.TotalEndpoints))
			r.Recorder.Eventf(model, corev1.EventTypeNormal, "EndpointsReady",
				"All %d endpoints ready for base model %s", model.Status.TotalEndpoints, model.Spec.BaseModelName)
		} else if model.Status.TotalEndpoints > 0 {
			r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionFalse, ReasonNotReady,
				fmt.Sprintf("Found %d ready endpoint(s) out of %d total", model.Status.ReadyEndpoints, model.Status.TotalEndpoints))
			r.Recorder.Eventf(model, corev1.EventTypeWarning, "NotReady",
				"Only %d of %d endpoints ready for base model %s", model.Status.ReadyEndpoints, model.Status.TotalEndpoints, model.Spec.BaseModelName)
		} else {
			r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionFalse, ReasonNoEndpoints, "No endpoints found")
		}
	} else {
		// For base models, just check that endpoints exist (readiness doesn't apply)
		if model.Status.TotalEndpoints > 0 {
			r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionTrue, ReasonEndpointsDiscovered,
				fmt.Sprintf("Found %d endpoint(s) for base model", model.Status.TotalEndpoints))
			r.Recorder.Eventf(model, corev1.EventTypeNormal, "EndpointsDiscovered",
				"Discovered %d endpoints for base model %s", model.Status.TotalEndpoints, model.Spec.BaseModelName)
		} else {
			r.updateCondition(model, ConditionTypeEndpointsReady, metav1.ConditionFalse, ReasonNoEndpoints, "No endpoints found")
		}
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
		return ctrl.Result{RequeueAfter: requeueAfterDuration}, nil
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
	// Register field indexer for DynamoModels by hash of base model name
	// This allows efficient O(1) queries: "get all DynamoModels for EndpointSlice with hash X"
	// The hash matches the label on EndpointSlices: nvidia.com/dynamo-base-model-hash
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&v1alpha1.DynamoModel{},
		dynamoModelBaseModelHashIndex,
		func(obj client.Object) []string {
			model := obj.(*v1alpha1.DynamoModel)
			// Hash the base model name using the same function used for EndpointSlice labels
			hash := dynamo.HashModelName(model.Spec.BaseModelName)
			return []string{hash}
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
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.Config)). // set the event filter to ignore resources handled by other controllers in namespace-restricted mode
		Complete(r)
}

// findModelsForEndpointSlice maps an EndpointSlice to DynamoModels
func (r *DynamoModelReconciler) findModelsForEndpointSlice(ctx context.Context, obj client.Object) []reconcile.Request {
	slice := obj.(*discoveryv1.EndpointSlice)
	logs := log.FromContext(ctx).WithValues("endpointSlice", slice.Name, "namespace", slice.Namespace)

	// Get the base model hash from the EndpointSlice label
	// This hash is set when the Service is created and matches our index
	baseModelHash, ok := slice.Labels[consts.KubeLabelDynamoBaseModelHash]
	if !ok {
		return nil
	}

	// Find all DynamoModels with this base model hash using field indexer
	// The indexer hashes each model's BaseModelName and we query by that hash
	requests, err := modelendpoint.FindModelsForBaseModel(ctx, r.Client, slice.Namespace, baseModelHash, dynamoModelBaseModelHashIndex)
	if err != nil {
		return nil
	}

	if len(requests) > 0 {
		logs.V(1).Info("EndpointSlice change triggered DynamoModel reconciliation",
			"modelCount", len(requests),
			"baseModelHash", baseModelHash)
	}

	return requests
}

// FinalizeResource implements the Finalizer interface
// Performs cleanup when a DynamoModel is being deleted
func (r *DynamoModelReconciler) FinalizeResource(ctx context.Context, model *v1alpha1.DynamoModel) error {
	logs := log.FromContext(ctx)

	logs.Info("Finalizing DynamoModel", "modelType", model.Spec.ModelType)

	// Only perform cleanup for LoRA models
	if model.IsLoRA() {
		// Get endpoint candidates (reusing common logic)
		candidates, _, err := r.getEndpointCandidates(ctx, model)
		if err != nil {
			logs.Info("Failed to get endpoints during deletion, continuing with resource deletion",
				"error", err.Error())
			r.Recorder.Event(model, corev1.EventTypeWarning, "CleanupFailed", err.Error())
			// Continue with deletion even if we can't get endpoints
		} else if len(candidates) > 0 {
			logs.Info("Unloading LoRA from endpoints", "endpointCount", len(candidates))

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
