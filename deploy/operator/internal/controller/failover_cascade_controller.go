/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
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

// failoverCascadeReconciler watches GMS failover pods (restartPolicy: Never)
// and cascade-deletes all pods in the same engine group when any member
// reaches a terminal phase (Failed or Succeeded). This ensures broken
// distributed inference groups are restarted cleanly by Grove.
//
// Only pods carrying the Dynamo failover engine-group-member label are
// considered; see failoverCascadePredicate.
type failoverCascadeReconciler struct {
	client.Client
	recorder events.EventRecorder
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;delete;deletecollection

// Reconcile is called whenever a failover-eligible pod transitions to a
// terminal phase (see failoverCascadePredicate).
//
// DeleteAllOf is idempotent, so concurrent reconciles for multiple pods in the
// same engine group are harmless — the first deletes the group and subsequent
// calls are no-ops.
func (r *failoverCascadeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Snapshot+GMS restore targets currently use IntraPod recovery. InterPod
	// Snapshot+failover is follow-on work, so fail closed rather than cascade-
	// deleting a cohort when the unsupported labels overlap.
	if pod.Labels[snapshotprotocol.RestoreTargetLabel] == commonconsts.KubeLabelValueTrue {
		return ctrl.Result{}, nil
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
	restoreTargetRequirement, err := labels.NewRequirement(
		snapshotprotocol.RestoreTargetLabel,
		selection.NotEquals,
		[]string{commonconsts.KubeLabelValueTrue},
	)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to build restore-target selector: %w", err)
	}
	groupSelector := labels.SelectorFromSet(labels.Set(groupLabels)).Add(*restoreTargetRequirement)

	// Force delete (grace=0) intentionally: the distributed inference group is
	// already broken when we get here, so giving the surviving engines a SIGTERM
	// window only delays Grove's recreation of the cohort and risks leaving
	// half-torn-down NCCL/CUDA IPC state and stale UDS sockets on the shared
	// hostPath. We deliberately skip preStop hooks and the graceful shutdown
	// window; do NOT soften this to a positive grace period.
	if err := r.DeleteAllOf(ctx, &corev1.Pod{}, client.InNamespace(pod.Namespace),
		client.MatchingLabelsSelector{Selector: groupSelector}, client.GracePeriodSeconds(0)); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to cascade-delete engine group: %w", err)
	}

	logger.Info("cascade-deleted engine group",
		"trigger", pod.Name,
		"pcsg", pcsg,
		"pcsgReplica", pcsgReplica,
		"podIndex", podIndex,
	)
	r.recorder.Eventf(&pod, nil, corev1.EventTypeWarning, "FailoverCascade", "Delete",
		"Pod %s terminated (phase=%s); cascade-deleted engine group (pcsg=%s, replica=%s, index=%s)",
		pod.Name, pod.Status.Phase, pcsg, pcsgReplica, podIndex,
	)

	return ctrl.Result{}, nil
}

func (r *failoverCascadeReconciler) setupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("gms-failover-cascade").
		For(&corev1.Pod{}, builder.WithPredicates(failoverCascadePredicate())).
		Complete(r)
}

func isTerminalPhase(phase corev1.PodPhase) bool {
	return phase == corev1.PodFailed || phase == corev1.PodSucceeded
}

// failoverCascadePredicate keeps the reconcile queue minimal by filtering
// events at the informer level, before they ever reach Reconcile.
//
// It accepts only pods carrying the Dynamo failover engine-group-member label
// and only when they reach a terminal phase:
//
//   - CreateFunc handles an informer observing an already-terminal pod.
//   - UpdateFunc handles a transition into a terminal phase.
//   - DeleteFunc and GenericFunc suppress events that cannot initiate failover.
func failoverCascadePredicate() predicate.Predicate {
	isEligible := func(labels map[string]string) bool {
		return labels[commonconsts.KubeLabelDynamoFailoverEngineGroupMember] == commonconsts.KubeLabelValueTrue &&
			labels[snapshotprotocol.RestoreTargetLabel] != commonconsts.KubeLabelValueTrue
	}

	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			if !isEligible(e.Object.GetLabels()) {
				return false
			}
			pod, ok := e.Object.(*corev1.Pod)
			if !ok {
				return false
			}
			return isTerminalPhase(pod.Status.Phase)
		},
		DeleteFunc: func(event.DeleteEvent) bool {
			return false
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			if !isEligible(e.ObjectNew.GetLabels()) || e.ObjectNew.GetDeletionTimestamp() != nil {
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
			return !isTerminalPhase(oldPod.Status.Phase) && isTerminalPhase(newPod.Status.Phase)
		},
	}
}
