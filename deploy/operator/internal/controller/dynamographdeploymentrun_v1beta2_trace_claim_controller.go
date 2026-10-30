/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"slices"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// DynamoGraphDeploymentRunTraceClaimReconciler releases the finalizers by which Runs protect their
// trace claims when the Run reconciler cannot: a Run has no finalizer of its own, so it can be
// deleted, by its Request or with its namespace, before it released its claim. The claim keeps a
// Run's finalizer while that Run exists and is not being deleted, or while a Job the Run controls
// exists. The Run reconciler releases the finalizer of a completed Run itself.
type DynamoGraphDeploymentRunTraceClaimReconciler struct {
	client.Client
	APIReader     client.Reader
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commoncontroller.RuntimeConfig
}

func (r *DynamoGraphDeploymentRunTraceClaimReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	claim := &corev1.PersistentVolumeClaim{}
	if err := r.Get(ctx, req.NamespacedName, claim); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Drop the finalizer of every Run that can no longer execute against the claim.
	original := claim.DeepCopy()
	for _, finalizer := range original.Finalizers {
		uid, ok := strings.CutPrefix(finalizer, dgdrRunTraceClaimFinalizerPrefix)
		if !ok {
			continue
		}
		held, err := r.runHoldsClaim(ctx, claim, types.UID(uid))
		if err != nil {
			return ctrl.Result{}, err
		}
		if !held {
			controllerutil.RemoveFinalizer(claim, finalizer)
		}
	}
	if len(claim.Finalizers) == len(original.Finalizers) {
		return ctrl.Result{}, nil
	}
	if err := r.Patch(ctx, claim, client.MergeFromWithOptions(original, client.MergeFromWithOptimisticLock{})); err != nil {
		return ctrl.Result{}, fmt.Errorf("release trace PersistentVolumeClaim %q: %w", claim.Name, err)
	}
	return ctrl.Result{}, nil
}

// runHoldsClaim reports whether the Run with uid may still execute against claim: the Run exists
// and is not being deleted, or a Job it controls exists. A Run never appears after its finalizer, so
// the cache answers for the Run. The Job cache can miss a Job the Run created just before it was
// deleted, and releasing the claim cannot be undone, so the API server confirms that no such Job is
// left before the claim is released. claim must not be nil.
func (r *DynamoGraphDeploymentRunTraceClaimReconciler) runHoldsClaim(
	ctx context.Context,
	claim *corev1.PersistentVolumeClaim,
	uid types.UID,
) (bool, error) {
	runs := &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}
	if err := r.List(ctx, runs, client.InNamespace(claim.Namespace),
		client.MatchingFields{dgdrRunTraceClaimIndex: claim.Name}); err != nil {
		return false, fmt.Errorf("list the Runs that read trace PersistentVolumeClaim %q: %w", claim.Name, err)
	}
	for i := range runs.Items {
		if runs.Items[i].UID == uid && runs.Items[i].DeletionTimestamp == nil {
			return true, nil
		}
	}

	// A Job of the Run, as the cache and then the API server show it.
	jobs := &batchv1.JobList{}
	if err := r.List(ctx, jobs, client.InNamespace(claim.Namespace),
		client.MatchingFields{dgdrRunJobControllerUIDIndex: string(uid)}); err != nil {
		return false, fmt.Errorf("list the Jobs of Run %q: %w", uid, err)
	}
	if len(jobs.Items) > 0 {
		return true, nil
	}
	live := &batchv1.JobList{}
	if err := r.APIReader.List(ctx, live, client.InNamespace(claim.Namespace)); err != nil {
		return false, fmt.Errorf("list the Jobs of Run %q from the API: %w", uid, err)
	}
	for i := range live.Items {
		if slices.Contains(dgdrRunJobControllerUIDs(&live.Items[i]), string(uid)) {
			return true, nil
		}
	}
	return false, nil
}

// dgdrRunProtectsTraceClaim reports whether obj carries the finalizer of a Run.
func dgdrRunProtectsTraceClaim(obj client.Object) bool {
	return slices.ContainsFunc(obj.GetFinalizers(), func(finalizer string) bool {
		return strings.HasPrefix(finalizer, dgdrRunTraceClaimFinalizerPrefix)
	})
}

// dgdrRunTraceClaimOfRun maps a Run event to the trace claim the Run reads, if any.
func dgdrRunTraceClaimOfRun(_ context.Context, obj client.Object) []reconcile.Request {
	run, ok := obj.(*nvidiacomv1beta2.DynamoGraphDeploymentRun)
	if !ok {
		return nil
	}
	source := dgdrRunTraceClaim(run)
	if source == nil {
		return nil
	}
	return []reconcile.Request{{Namespace: run.Namespace, Name: source.ClaimName}}
}

// dgdrRunTraceClaimOfJob maps an event of a Job a Run controls to the trace claim the Job mounts, if
// any.
func dgdrRunTraceClaimOfJob(_ context.Context, obj client.Object) []reconcile.Request {
	job, ok := obj.(*batchv1.Job)
	if !ok || len(dgdrRunJobControllerUIDs(job)) == 0 {
		return nil
	}
	trace, found := dgdrRunVolume(job.Spec.Template.Spec.Volumes, dgdrRunTraceVolume)
	if !found || trace.PersistentVolumeClaim == nil {
		return nil
	}
	return []reconcile.Request{{Namespace: job.Namespace, Name: trace.PersistentVolumeClaim.ClaimName}}
}

// dgdrRunDeleteEvents passes only deletions: a Run or Job releases a claim only by going away, and
// claim events cover every other state, including the initial list after a restart.
var dgdrRunDeleteEvents = predicate.Funcs{
	CreateFunc:  func(event.CreateEvent) bool { return false },
	UpdateFunc:  func(event.UpdateEvent) bool { return false },
	DeleteFunc:  func(event.DeleteEvent) bool { return true },
	GenericFunc: func(event.GenericEvent) bool { return false },
}

// SetupWithManager registers the reconciler. It relies on the Run and Job indexes that the Run
// reconciler registers, so it is set up after that reconciler.
func (r *DynamoGraphDeploymentRunTraceClaimReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.APIReader == nil {
		r.APIReader = mgr.GetAPIReader()
	}
	return ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeploymentrun-v1beta2-trace-claim").
		For(&corev1.PersistentVolumeClaim{}, builder.WithPredicates(predicate.NewPredicateFuncs(dgdrRunProtectsTraceClaim))).
		Watches(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}, handler.EnqueueRequestsFromMapFunc(dgdrRunTraceClaimOfRun),
			builder.WithPredicates(dgdrRunDeleteEvents)).
		Watches(&batchv1.Job{}, handler.EnqueueRequestsFromMapFunc(dgdrRunTraceClaimOfJob),
			builder.WithPredicates(dgdrRunDeleteEvents)).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}
