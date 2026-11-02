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
//
// For a deleted Run, no Job does not mean no pod: garbage collection deletes the Run's Job in the
// background, before the Job's pod has stopped, so the finalizer can be released while that pod
// still has the trace file mounted. Kubernetes PersistentVolumeClaim protection, not this
// finalizer, keeps the claim until the pod is gone.
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

	// The finalizers of Runs that the cache shows gone and without a Job.
	stale, err := r.staleTraceClaimFinalizers(ctx, claim)
	if err != nil || len(stale) == 0 {
		return ctrl.Result{}, err
	}

	// The Job cache can miss a Job a Run created just before it was deleted, and releasing the claim
	// cannot be undone, so one uncached list of the namespace's Jobs confirms every stale finalizer.
	live := &batchv1.JobList{}
	if err := r.APIReader.List(ctx, live, client.InNamespace(claim.Namespace)); err != nil {
		return ctrl.Result{}, fmt.Errorf("list the Jobs of the Runs that read trace PersistentVolumeClaim %q from the API: %w",
			claim.Name, err)
	}
	for i := range live.Items {
		for _, uid := range dgdrRunJobControllerUIDs(&live.Items[i]) {
			delete(stale, dgdrRunTraceClaimFinalizer(types.UID(uid)))
		}
	}

	// Drop the finalizer of every Run that can no longer execute against the claim.
	original := claim.DeepCopy()
	for finalizer := range stale {
		controllerutil.RemoveFinalizer(claim, finalizer)
	}
	if len(claim.Finalizers) == len(original.Finalizers) {
		return ctrl.Result{}, nil
	}
	if err := r.Patch(ctx, claim, client.MergeFromWithOptions(original, client.MergeFromWithOptimisticLock{})); err != nil {
		return ctrl.Result{}, fmt.Errorf("release trace PersistentVolumeClaim %q: %w", claim.Name, err)
	}
	return ctrl.Result{}, nil
}

// staleTraceClaimFinalizers returns the Run finalizers on claim that the cache no longer justifies:
// the Run is gone or being deleted, and the cache shows no Job it controls. A Run never appears after
// its finalizer, so the cache answers for the Run; only the Jobs need the API server's confirmation,
// which the caller makes once for all of them. claim must not be nil.
func (r *DynamoGraphDeploymentRunTraceClaimReconciler) staleTraceClaimFinalizers(
	ctx context.Context,
	claim *corev1.PersistentVolumeClaim,
) (map[string]struct{}, error) {
	runs := &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}
	if err := r.List(ctx, runs, client.InNamespace(claim.Namespace),
		client.MatchingFields{dgdrRunTraceClaimIndex: claim.Name}); err != nil {
		return nil, fmt.Errorf("list the Runs that read trace PersistentVolumeClaim %q: %w", claim.Name, err)
	}
	active := make(map[string]struct{}, len(runs.Items))
	for i := range runs.Items {
		if runs.Items[i].DeletionTimestamp == nil {
			active[dgdrRunTraceClaimFinalizer(runs.Items[i].UID)] = struct{}{}
		}
	}

	stale := map[string]struct{}{}
	for _, finalizer := range claim.Finalizers {
		uid, ok := strings.CutPrefix(finalizer, dgdrRunTraceClaimFinalizerPrefix)
		if _, held := active[finalizer]; !ok || held {
			continue
		}

		// A Job of the Run as the cache shows it.
		jobs := &batchv1.JobList{}
		if err := r.List(ctx, jobs, client.InNamespace(claim.Namespace),
			client.MatchingFields{dgdrRunJobControllerUIDIndex: uid}); err != nil {
			return nil, fmt.Errorf("list the Jobs of Run %q: %w", uid, err)
		}
		if len(jobs.Items) == 0 {
			stale[finalizer] = struct{}{}
		}
	}
	return stale, nil
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
