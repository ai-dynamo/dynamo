/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"
	"net/http"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	controlleroptions "sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	ctrl "sigs.k8s.io/controller-runtime"
)

// SetupOptions selects the runtime provider registered with the Engine Group controller.
type SetupOptions struct {
	RuntimeProvider  RuntimeProvider
	WorkloadProvider WorkloadProvider
	GroveEnabled     bool
}

// Setup registers the Engine Group subsystem with the operator manager.
func Setup(mgr ctrl.Manager, opts SetupOptions) error {
	// Resolve the production provider unless the caller supplies another implementation.
	provider := opts.RuntimeProvider
	workload := opts.WorkloadProvider
	if workload == nil && opts.GroveEnabled {
		workload = &GroveWorkloadProvider{Client: mgr.GetClient()}
	}
	if provider == nil {
		provider = unavailableEngineGroupRuntimeProvider{}
		if workload != nil {
			provider = &runtimeProvider{Client: mgr.GetClient(), HTTP: &http.Client{}, Workload: workload}
		}
	}

	// Register reconciliation and watches together at the subsystem composition root.
	if err := (&Reconciler{
		Client:           mgr.GetClient(),
		RuntimeProvider:  provider,
		WorkloadProvider: workload,
	}).SetupWithManager(mgr, opts.GroveEnabled); err != nil {
		return fmt.Errorf("unable to create DynamoGraphDeploymentEngineGroup controller: %w", err)
	}
	return nil
}

// SetupWithManager registers the Engine Group controller and its status/finalizer writes.
func (r *Reconciler) SetupWithManager(mgr ctrl.Manager, groveEnabled bool) error {
	// Native clique deletion and recreation must wake the same logical world, even without Pod updates.
	controller := ctrl.NewControllerManagedBy(mgr).
		WithOptions(controlleroptions.Options{MaxConcurrentReconciles: engineGroupConcurrentReconciles}).
		For(&api.DynamoGraphDeploymentEngineGroup{}).
		Owns(&corev1.ConfigMap{}).
		Watches(&api.DynamoGraphDeployment{}, handler.EnqueueRequestsFromMapFunc(r.engineGroupsForParent), builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(r.engineGroupsForCapacity)).
		Named("dynamographdeploymentenginegroup")
	if groveEnabled {
		workload := &GroveWorkloadProvider{Client: r.Client}
		controller = controller.
			Watches(&grove.PodClique{}, handler.EnqueueRequestsFromMapFunc(r.engineGroupsForCapacity)).
			Watches(&grove.PodCliqueSet{}, handler.EnqueueRequestsFromMapFunc(workload.engineGroupsForParent)).
			Watches(&grove.PodCliqueScalingGroup{}, handler.EnqueueRequestsFromMapFunc(workload.engineGroupsForParent))
	}
	return controller.Complete(r)
}
