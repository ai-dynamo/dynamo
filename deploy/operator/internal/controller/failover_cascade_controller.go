/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"sync"
	"time"

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

const (
	groveLabelPCSG             = "grove.io/podcliquescalinggroup"
	groveLabelPCSGReplicaIndex = "grove.io/podcliquescalinggroup-replica-index"
	groveLabelPodIndex         = "grove.io/podclique-pod-index"
)

// cascadeCooldown prevents tight delete-recreate-fail loops by suppressing
// repeated cascade-deletes for the same engine group within this window.
const cascadeCooldown = 60 * time.Second

// FailoverCascadeReconciler watches GMS failover pods (restartPolicy: Never)
// and cascade-deletes all pods in the same engine group when any member
// reaches a terminal phase (Failed or Succeeded). This ensures broken
// distributed inference groups are restarted cleanly by Grove.
//
// An engine group is identified by three Grove labels:
//   - grove.io/podcliquescalinggroup           (PCSG name)
//   - grove.io/podcliquescalinggroup-replica-index (PCSG replica)
//   - grove.io/podclique-pod-index              (pod index within PCLQ)
type FailoverCascadeReconciler struct {
	client.Client
	Recorder record.EventRecorder
	Now      func() time.Time

	cooldownMu sync.Mutex
	cooldowns  map[string]time.Time
}

func NewFailoverCascadeReconciler(c client.Client, recorder record.EventRecorder) *FailoverCascadeReconciler {
	return &FailoverCascadeReconciler{
		Client:   c,
		Recorder: recorder,
		Now:      time.Now,
	}
}

func (r *FailoverCascadeReconciler) inCooldown(key string) bool {
	r.cooldownMu.Lock()
	defer r.cooldownMu.Unlock()
	if last, ok := r.cooldowns[key]; ok {
		return r.Now().Sub(last) < cascadeCooldown
	}
	return false
}

func (r *FailoverCascadeReconciler) setCooldown(key string) {
	r.cooldownMu.Lock()
	defer r.cooldownMu.Unlock()
	if r.cooldowns == nil {
		r.cooldowns = make(map[string]time.Time)
	}
	r.cooldowns[key] = r.Now()
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;delete;deletecollection

func (r *FailoverCascadeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	if pod.Status.Phase != corev1.PodFailed && pod.Status.Phase != corev1.PodSucceeded {
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

	groupKey := pcsg + "/" + pcsgReplica + "/" + podIndex
	if r.inCooldown(groupKey) {
		logger.V(1).Info("engine group in cascade cooldown, skipping",
			"trigger", pod.Name,
			"pcsg", pcsg,
			"pcsgReplica", pcsgReplica,
			"podIndex", podIndex,
		)
		return ctrl.Result{}, nil
	}

	groupLabels := client.MatchingLabels{
		commonconsts.KubeLabelDynamoFailoverEngineGroupMember: commonconsts.KubeLabelValueTrue,
		groveLabelPCSG:             pcsg,
		groveLabelPCSGReplicaIndex: pcsgReplica,
		groveLabelPodIndex:         podIndex,
	}

	if err := r.DeleteAllOf(ctx, &corev1.Pod{}, client.InNamespace(pod.Namespace), groupLabels); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to cascade-delete engine group: %w", err)
	}

	r.setCooldown(groupKey)

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

// failoverCascadePredicate filters to pods with the failover group label
// and only triggers on updates where the phase transitions to a terminal state.
func failoverCascadePredicate() predicate.Predicate {
	hasLabel := func(labels map[string]string) bool {
		return labels[commonconsts.KubeLabelDynamoFailoverEngineGroupMember] == commonconsts.KubeLabelValueTrue
	}

	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return false
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
			return !isTerminalPhase(oldPod.Status.Phase) && isTerminalPhase(newPod.Status.Phase)
		},
	}
}
