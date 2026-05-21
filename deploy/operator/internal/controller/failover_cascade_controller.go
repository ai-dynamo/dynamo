/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// Grove labels that together uniquely identify an "engine group" — the set of
// pods (one per rank in multi-node, or a single pod in single-node) that share
// the same pod index within a PCSG replica. When any one of them terminates,
// the whole group must be torn down so Grove can recreate it as a healthy unit.
const (
	groveLabelPCSG             = "grove.io/podcliquescalinggroup"
	groveLabelPCSGReplicaIndex = "grove.io/podcliquescalinggroup-replica-index"
	groveLabelPodIndex         = "grove.io/podclique-pod-index"
)

// FailoverCascadeReconciler watches GMS failover pods (restartPolicy: Never)
// and cascade-deletes all pods in the same engine group when any member
// reaches a terminal phase (Failed or Succeeded). This ensures broken
// distributed inference groups are restarted cleanly by Grove.
//
// Background: GMS (GPU Memory Service) pods run with restartPolicy: Never so
// that Kubernetes does not attempt to restart them in-place — a partial
// restart would leave the distributed inference group in an inconsistent
// state. Instead, this controller detects the terminal pod and deletes the
// entire group.  Grove then sees the missing pods and recreates the whole
// group from scratch.
//
// An engine group is identified by three Grove labels:
//   - grove.io/podcliquescalinggroup              (PCSG name)
//   - grove.io/podcliquescalinggroup-replica-index (PCSG replica — which copy of the group)
//   - grove.io/podclique-pod-index                (pod index within the clique)
//
// Only pods carrying the dynamo failover engine-group-member label are
// considered; see failoverCascadePredicate().
type FailoverCascadeReconciler struct {
	client.Client
	Recorder record.EventRecorder
}

// NewFailoverCascadeReconciler creates a new reconciler.
func NewFailoverCascadeReconciler(c client.Client, recorder record.EventRecorder) *FailoverCascadeReconciler {
	return &FailoverCascadeReconciler{
		Client:   c,
		Recorder: recorder,
	}
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;delete;deletecollection

// Reconcile is called whenever a failover-eligible pod transitions to a
// terminal phase (see failoverCascadePredicate).
//
// DeleteAllOf is idempotent, so concurrent reconciles for multiple pods in the
// same engine group are harmless — the first deletes the group and subsequent
// calls are no-ops.
func (r *FailoverCascadeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	if !isTerminalPhase(pod.Status.Phase) {
		return ctrl.Result{}, nil
	}

	// Between predicate evaluation and reconcile execution, another reconcile
	// may have already cascade-deleted this pod. The pod still exists in the
	// API server but is marked for deletion — skip it.
	if pod.DeletionTimestamp != nil {
		return ctrl.Result{}, nil
	}

	// Defensive re-check of the engine-group-member label: the predicate
	// already filters on it at the informer layer, but labels can be removed
	// between predicate evaluation and reconcile. We never want to cascade-
	// delete a pod that has been explicitly unlabeled (e.g. an operator
	// manually quarantining a pod).
	if pod.Labels[commonconsts.KubeLabelDynamoFailoverEngineGroupMember] != commonconsts.KubeLabelValueTrue {
		return ctrl.Result{}, nil
	}

	pcsg := pod.Labels[groveLabelPCSG]
	pcsgReplica := pod.Labels[groveLabelPCSGReplicaIndex]
	podIndex := pod.Labels[groveLabelPodIndex]
	if pcsg == "" || pcsgReplica == "" || podIndex == "" {
		logger.Info("failover pod missing Grove labels, skipping cascade",
			"pod", pod.Name,
			groveLabelPCSG, pcsg,
			groveLabelPCSGReplicaIndex, pcsgReplica,
			groveLabelPodIndex, podIndex,
		)
		return ctrl.Result{}, nil
	}

	groupLabels := client.MatchingLabels{
		commonconsts.KubeLabelDynamoFailoverEngineGroupMember: commonconsts.KubeLabelValueTrue,
		groveLabelPCSG:             pcsg,
		groveLabelPCSGReplicaIndex: pcsgReplica,
		groveLabelPodIndex:         podIndex,
	}

	// Force delete (grace=0) intentionally: the distributed inference group is
	// already broken when we get here, so giving the surviving engines a SIGTERM
	// window only delays Grove's recreation of the cohort and risks leaving
	// half-torn-down NCCL/CUDA IPC state and stale UDS sockets on the shared
	// hostPath. We deliberately skip preStop hooks and the graceful shutdown
	// window; do NOT soften this to a positive grace period.
	if err := r.DeleteAllOf(ctx, &corev1.Pod{}, client.InNamespace(pod.Namespace), groupLabels, client.GracePeriodSeconds(0)); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to cascade-delete engine group: %w", err)
	}

	logger.Info("cascade-deleted engine group",
		"trigger", pod.Name,
		"pcsg", pcsg,
		"pcsgReplica", pcsgReplica,
		"podIndex", podIndex,
	)
	r.Recorder.Eventf(&pod, corev1.EventTypeWarning, "FailoverCascade",
		"Pod %s terminated (phase=%s); cascade-deleted engine group (pcsg=%s, replica=%s, index=%s)",
		pod.Name, pod.Status.Phase, pcsg, pcsgReplica, podIndex,
	)

	return ctrl.Result{}, nil
}

// SetupWithManager registers a controller that watches all Pods (not just
// owned ones) and uses failoverCascadePredicate to filter down to only the
// failover-eligible phase transitions.  EnqueueRequestForObject means the
// reconcile key is the pod itself (namespace/name), not a parent resource.
func (r *FailoverCascadeReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("gms-failover-cascade").
		Watches(&corev1.Pod{}, &handler.EnqueueRequestForObject{},
			builder.WithPredicates(failoverCascadePredicate()),
		).
		Complete(r)
}

func isTerminalPhase(phase corev1.PodPhase) bool {
	return phase == corev1.PodFailed || phase == corev1.PodSucceeded
}

// failoverCascadePredicate keeps the reconcile queue minimal by filtering
// events at the informer level, before they ever reach Reconcile().
//
// It accepts only pods carrying the dynamo failover engine-group-member label
// and only when they reach a terminal phase:
//
//   - CreateFunc: handles the edge case where the informer's initial list-watch
//     delivers a pod that is already Failed/Succeeded (e.g. the informer cache
//     started after the pod transitioned, so no Update event was observed).
//     Without this, such pods would be silently ignored and their engine group
//     would never be cascade-deleted.
//
//   - UpdateFunc: the primary path — fires when a Running/Pending pod
//     transitions to Failed/Succeeded.  Pods that already have a
//     deletionTimestamp are filtered out to avoid acting on pods that are
//     being terminated by an ongoing cascade or DGD deletion.
//
//   - DeleteFunc / GenericFunc: always suppressed — pod deletions are the
//     *result* of our cascade, not triggers for one.
func failoverCascadePredicate() predicate.Predicate {
	hasLabel := func(labels map[string]string) bool {
		return labels[commonconsts.KubeLabelDynamoFailoverEngineGroupMember] == commonconsts.KubeLabelValueTrue
	}

	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			if !hasLabel(e.Object.GetLabels()) {
				return false
			}
			pod, ok := e.Object.(*corev1.Pod)
			if !ok {
				return false
			}
			return isTerminalPhase(pod.Status.Phase)
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return false
		},
		GenericFunc: func(e event.GenericEvent) bool {
			return false
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			if !hasLabel(e.ObjectNew.GetLabels()) {
				return false
			}
			// Ignore pods already being deleted — this avoids reacting to
			// our own cascade-delete (which sets deletionTimestamp before
			// the pod actually disappears from the cache).
			if e.ObjectNew.GetDeletionTimestamp() != nil {
				return false
			}
			newPod, ok := e.ObjectNew.(*corev1.Pod)
			if !ok {
				return false
			}
			oldPod, ok := e.ObjectOld.(*corev1.Pod)
			if !ok {
				return false
			}
			// Only trigger on actual phase transitions to avoid processing
			// the same pod twice (e.g. a metadata update on an already-Failed pod).
			return !isTerminalPhase(oldPod.Status.Phase) && isTerminalPhase(newPod.Status.Phase)
		},
	}
}
