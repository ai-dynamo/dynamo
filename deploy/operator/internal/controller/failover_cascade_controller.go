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

// Grove labels that together uniquely identify an "engine group" — the set of
// pods (primary + shadows) that serve a single model replica.  When any one of
// them terminates, the whole group must be torn down so Grove can recreate it
// as a healthy unit.
const (
	groveLabelPCSG             = "grove.io/podcliquescalinggroup"
	groveLabelPCSGReplicaIndex = "grove.io/podcliquescalinggroup-replica-index"
	groveLabelPodIndex         = "grove.io/podclique-pod-index"
)

// cascadeCooldown prevents tight delete-recreate-fail loops by suppressing
// repeated cascade-deletes for the same engine group within this window.
// Only applies to same-generation events; pods created after the last
// cascade (new generation from Grove) bypass it entirely.
const cascadeCooldown = 60 * time.Second

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

	// Now is an injectable clock used in cooldown logic; defaults to time.Now.
	// Tests override it to control time progression.
	Now func() time.Time

	// cooldowns tracks the last cascade-delete timestamp per engine group key
	// ("pcsg/replica/index"). Used by inCooldown() to suppress duplicate
	// deletes from redundant watch events while still allowing new-generation
	// pods (recreated by Grove) to trigger an immediate cascade.
	cooldownMu sync.Mutex
	cooldowns  map[string]time.Time
}

// NewFailoverCascadeReconciler creates a reconciler with a real clock.
// Use the struct directly with a custom Now func in tests.
func NewFailoverCascadeReconciler(c client.Client, recorder record.EventRecorder) *FailoverCascadeReconciler {
	return &FailoverCascadeReconciler{
		Client:   c,
		Recorder: recorder,
		Now:      time.Now,
	}
}

// inCooldown reports whether a cascade for this group should be suppressed.
//
// Three cases:
//  1. No previous cascade recorded → not in cooldown (first time).
//  2. Pod was created AFTER the last cascade → not in cooldown.  This means
//     Grove already recreated the group and a pod in the new generation has
//     failed.  We must react immediately; suppressing would leave the failed
//     pods stuck in a terminal state forever (restartPolicy: Never).
//  3. Pod was created BEFORE the last cascade (same generation) and the
//     cooldown window has not elapsed → in cooldown.  This is a duplicate
//     event from the same generation (e.g. multiple pods in the group
//     transitioning to Failed almost simultaneously) and we can safely skip.
func (r *FailoverCascadeReconciler) inCooldown(key string, podCreationTime time.Time) bool {
	r.cooldownMu.Lock()
	defer r.cooldownMu.Unlock()
	lastCascade, ok := r.cooldowns[key]
	if !ok {
		return false // case 1
	}
	if podCreationTime.After(lastCascade) {
		return false // case 2
	}
	return r.Now().Sub(lastCascade) < cascadeCooldown // case 3
}

// setCooldown records the given timestamp as the last cascade event for the
// engine group. The caller is expected to capture the time BEFORE issuing the
// DeleteAllOf so that any replacement pods created by Grove during the
// deletion have a CreationTimestamp strictly after this value.
func (r *FailoverCascadeReconciler) setCooldown(key string, t time.Time) {
	r.cooldownMu.Lock()
	defer r.cooldownMu.Unlock()
	if r.cooldowns == nil {
		r.cooldowns = make(map[string]time.Time)
	}
	r.cooldowns[key] = t
}

// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;delete;deletecollection

// Reconcile is called whenever a failover-eligible pod transitions to a
// terminal phase (see failoverCascadePredicate).
//
// The flow:
//  1. Fetch the pod; bail if already deleted (race with the cascade itself).
//  2. Double-check it is truly terminal — the predicate should guarantee
//     this, but a belt-and-suspenders check costs nothing.
//  3. Extract the Grove labels that identify the engine group.
//  4. Check the cooldown to avoid tight delete loops from duplicate events.
//  5. Issue a DeleteAllOf with GracePeriodSeconds(0) to immediately kill
//     every pod in the group (primary + shadows).  GracePeriodSeconds(0)
//     bypasses terminationGracePeriodSeconds — critical for fast failover
//     because we want Grove to recreate the group ASAP rather than waiting
//     for slow graceful shutdowns.
//  6. Record the cascade in cooldowns and emit a Kubernetes event for
//     observability.
func (r *FailoverCascadeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Step 1: Fetch the triggering pod.
	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		if errors.IsNotFound(err) {
			// Pod was already cleaned up (e.g. by our own cascade or DGD deletion).
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Step 2: Only act on terminal pods.
	if pod.Status.Phase != corev1.PodFailed && pod.Status.Phase != corev1.PodSucceeded {
		return ctrl.Result{}, nil
	}

	// Step 3: Extract the three Grove labels that uniquely identify the engine
	// group.  All three are required; a missing label means the pod isn't
	// managed by Grove in the expected way and we shouldn't touch it.
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

	// Step 4: Build a unique key for this engine group and check the cooldown.
	// This prevents redundant cascades when multiple pods in the same group
	// fail almost simultaneously and each generates a watch event.
	groupKey := pcsg + "/" + pcsgReplica + "/" + podIndex
	if r.inCooldown(groupKey, pod.CreationTimestamp.Time) {
		logger.V(1).Info("engine group in cascade cooldown, skipping same-generation event",
			"trigger", pod.Name,
			"pcsg", pcsg,
			"pcsgReplica", pcsgReplica,
			"podIndex", podIndex,
		)
		return ctrl.Result{}, nil
	}

	// Step 5: Cascade-delete every pod in the engine group.
	// The label selector matches on both the dynamo failover label (our label)
	// and the three Grove labels (to scope to exactly this group).
	// GracePeriodSeconds(0) sends SIGKILL immediately — no graceful shutdown.
	//
	// IMPORTANT: capture the time BEFORE the delete so that any replacement
	// pods Grove creates while DeleteAllOf is in-flight will have a
	// CreationTimestamp strictly after cascadeTime. This ensures inCooldown()
	// correctly recognises them as new-generation and does not suppress them.
	groupLabels := client.MatchingLabels{
		commonconsts.KubeLabelDynamoFailoverEngineGroupMember: commonconsts.KubeLabelValueTrue,
		groveLabelPCSG:             pcsg,
		groveLabelPCSGReplicaIndex: pcsgReplica,
		groveLabelPodIndex:         podIndex,
	}

	cascadeTime := r.Now()
	if err := r.DeleteAllOf(ctx, &corev1.Pod{}, client.InNamespace(pod.Namespace), groupLabels, client.GracePeriodSeconds(0)); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to cascade-delete engine group: %w", err)
	}

	// Step 6: Record the cascade for cooldown tracking and emit a Kubernetes
	// event so operators can see what happened via kubectl describe / kubectl get events.
	r.setCooldown(groupKey, cascadeTime)

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
