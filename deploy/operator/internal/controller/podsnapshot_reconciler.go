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
	"errors"
	"fmt"
	"math/rand"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const (
	// podSnapshotFinalizer is set on the PodSnapshot so its bound PodSnapshotContent is
	// deleted before the PodSnapshot is removed.
	podSnapshotFinalizer = "nvidia.com/podsnapshotcontent-cleanup"

	// podSnapshotContentFieldManager is the Server-Side Apply field owner for SnapshotContents.
	podSnapshotContentFieldManager = "dynamo-podsnapshot-controller"

	// snapshotPodResolveBackoffBase is the minimum requeue delay while waiting for the
	// source pod to be scheduled; jitter is added on top to avoid a synchronized hot loop.
	snapshotPodResolveBackoffBase = 2 * time.Second

	// snapshotContentDeleteRequeue is the delay between cascade-delete progress checks.
	snapshotContentDeleteRequeue = time.Second

	// maxResourceNameLength is the Kubernetes object name limit (RFC 1123 subdomain).
	maxResourceNameLength = 253
)

// errPodSnapshotPodUnscheduled signals that the source pod is not yet scheduled and the
// reconcile should retry with backoff rather than fail.
var errPodSnapshotPodUnscheduled = errors.New("source pod is not yet scheduled to a node")

// PodSnapshotReconciler reconciles a PodSnapshot: it creates the bound, cluster-scoped
// PodSnapshotContent work order for the node agent, mirrors the agent's terminal status
// back to the PodSnapshot, and cascades deletion to the PodSnapshotContent.
type PodSnapshotReconciler struct {
	client.Client
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshots,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshots/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshots/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshotcontents,verbs=create;get;list;watch;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshotcontents/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch

// Reconcile drives a PodSnapshot through binding, status mirroring, and cascade deletion.
func (sr *PodSnapshotReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	snap := &nvidiacomv1alpha1.PodSnapshot{}
	if err := sr.Get(ctx, req.NamespacedName, snap); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !snap.GetDeletionTimestamp().IsZero() {
		return sr.handleDelete(ctx, snap)
	}

	if !controllerutil.ContainsFinalizer(snap, podSnapshotFinalizer) {
		controllerutil.AddFinalizer(snap, podSnapshotFinalizer)
		if err := sr.Update(ctx, snap); err != nil {
			return ctrl.Result{}, fmt.Errorf("add snapshot finalizer: %w", err)
		}
		return ctrl.Result{}, nil
	}

	pod, err := sr.getSourcePod(ctx, snap)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(1).Info("Source pod not found, backing off", "snapshot", snap.Name)
			return ctrl.Result{RequeueAfter: jitteredBackoff(snapshotPodResolveBackoffBase)}, nil
		}
		return ctrl.Result{}, err
	}
	if err := validateSourcePod(pod); err != nil {
		logger.V(1).Info("Source pod not ready, backing off", "snapshot", snap.Name, "reason", err.Error())
		return ctrl.Result{RequeueAfter: jitteredBackoff(snapshotPodResolveBackoffBase)}, nil
	}

	// The content is named from the PodSnapshot UID, not the checkpoint ID: the ID is a
	// restore-time concern that lives on the pod's labels, and the content does not need it.
	// UID is immutable and already populated here (the finalizer branch above round-tripped an
	// Update), so the name is stable and deterministic for the PodSnapshot's lifetime.
	contentName := podSnapshotContentName(snap)

	content, err := sr.ensurePodSnapshotContent(ctx, snap, contentName, pod)
	if err != nil {
		return ctrl.Result{}, err
	}
	// A freshly-created content always matches; only a pre-existing content whose
	// source pod was rescheduled to another node mismatches (spec is immutable).
	if content.Spec.Source.NodeName != pod.Spec.NodeName {
		return sr.failPodSnapshot(ctx, snap, "PodRescheduled",
			fmt.Errorf("source pod moved from node %q to %q; CRIU checkpoint cannot survive migration",
				content.Spec.Source.NodeName, pod.Spec.NodeName))
	}

	return sr.propagateStatus(ctx, snap, content)
}

// getSourcePod loads the source pod referenced by the PodSnapshot.
func (sr *PodSnapshotReconciler) getSourcePod(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot) (*corev1.Pod, error) {
	pod := &corev1.Pod{}
	key := client.ObjectKey{Namespace: snap.Namespace, Name: snap.Spec.Source.PodRef.Name}
	if err := sr.Get(ctx, key, pod); err != nil {
		return nil, err
	}
	return pod, nil
}

// validateSourcePod requires the pod to be scheduled to a node.
func validateSourcePod(pod *corev1.Pod) error {
	if pod.Spec.NodeName == "" {
		return errPodSnapshotPodUnscheduled
	}
	return nil
}

// ensurePodSnapshotContent returns the existing PodSnapshotContent or, when absent, creates the
// trigger via a single Server-Side Apply carrying the source ref and the node mirror label.
// The returned object is the source of truth for the reschedule guard.
func (sr *PodSnapshotReconciler) ensurePodSnapshotContent(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, contentName string, pod *corev1.Pod) (*nvidiacomv1alpha1.PodSnapshotContent, error) {
	existing := &nvidiacomv1alpha1.PodSnapshotContent{}
	if err := sr.Get(ctx, client.ObjectKey{Name: contentName}, existing); err == nil {
		return existing, nil
	} else if !apierrors.IsNotFound(err) {
		return nil, err
	}

	content := sr.buildPodSnapshotContent(snap, contentName, pod)
	if err := sr.Patch(ctx, content, client.Apply,
		client.FieldOwner(podSnapshotContentFieldManager), client.ForceOwnership); err != nil {
		sr.Recorder.Event(snap, corev1.EventTypeWarning, "SnapshotContentCreateFailed", err.Error())
		return nil, fmt.Errorf("apply PodSnapshotContent %q: %w", contentName, err)
	}
	return content, nil
}

// buildPodSnapshotContent constructs the desired cluster-scoped PodSnapshotContent for a PodSnapshot.
func (sr *PodSnapshotReconciler) buildPodSnapshotContent(snap *nvidiacomv1alpha1.PodSnapshot, contentName string, pod *corev1.Pod) *nvidiacomv1alpha1.PodSnapshotContent {
	return &nvidiacomv1alpha1.PodSnapshotContent{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "PodSnapshotContent",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: contentName,
			Labels: map[string]string{
				snapshotprotocol.SnapshotNodeLabel: pod.Spec.NodeName,
			},
		},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{
				Namespace: snap.Namespace,
				Name:      snap.Name,
				UID:       snap.UID,
			},
			Source: nvidiacomv1alpha1.PodSnapshotContentSource{
				PodRef:   nvidiacomv1alpha1.PodReference{Name: pod.Name, UID: pod.UID},
				NodeName: pod.Spec.NodeName,
			},
		},
	}
}

// propagateStatus records the binding and mirrors the PodSnapshotContent's terminal status to
// the PodSnapshot, defaulting to a Pending condition until the agent writes a result. It
// receives the content resolved earlier in the reconcile, so it never re-Gets it.
func (sr *PodSnapshotReconciler) propagateStatus(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, content *nvidiacomv1alpha1.PodSnapshotContent) (ctrl.Result, error) {
	changed := false
	if ptr.Deref(snap.Status.BoundPodSnapshotContentName, "") != content.Name {
		snap.Status.BoundPodSnapshotContentName = ptr.To(content.Name)
		changed = true
	}

	switch {
	case nvidiacomv1alpha1.IsPodSnapshotContentSucceeded(content):
		cond := meta.FindStatusCondition(content.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady)
		changed = sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionReady, metav1.ConditionTrue, cond.Reason, cond.Message) || changed
	case nvidiacomv1alpha1.IsPodSnapshotContentFailed(content):
		cond := meta.FindStatusCondition(content.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
		changed = sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionFailed, metav1.ConditionTrue, cond.Reason, cond.Message) || changed
	default:
		changed = sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionReady, metav1.ConditionFalse, "Pending", "Waiting for node agent to capture the checkpoint") || changed
	}

	if !changed {
		return ctrl.Result{}, nil
	}
	if err := sr.Status().Update(ctx, snap); err != nil {
		return ctrl.Result{}, fmt.Errorf("update snapshot status: %w", err)
	}
	return ctrl.Result{}, nil
}

// setCondition sets a status condition and reports whether it changed.
func (sr *PodSnapshotReconciler) setCondition(snap *nvidiacomv1alpha1.PodSnapshot, condType string, status metav1.ConditionStatus, reason, message string) bool {
	return meta.SetStatusCondition(&snap.Status.Conditions, metav1.Condition{
		Type:    condType,
		Status:  status,
		Reason:  reason,
		Message: message,
	})
}

// failPodSnapshot marks the PodSnapshot Failed terminally and records an event.
func (sr *PodSnapshotReconciler) failPodSnapshot(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, reason string, cause error) (ctrl.Result, error) {
	sr.Recorder.Event(snap, corev1.EventTypeWarning, reason, cause.Error())
	sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionFailed, metav1.ConditionTrue, reason, cause.Error())
	if err := sr.Status().Update(ctx, snap); err != nil {
		return ctrl.Result{}, fmt.Errorf("mark snapshot failed: %w", err)
	}
	return ctrl.Result{}, nil
}

// handleDelete cascades deletion to the bound PodSnapshotContent and blocks (requeues) until
// it is gone before dropping the PodSnapshot finalizer. The PodSnapshotContent carries no
// finalizer of its own, so the Delete takes effect immediately.
func (sr *PodSnapshotReconciler) handleDelete(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot) (ctrl.Result, error) {
	if !controllerutil.ContainsFinalizer(snap, podSnapshotFinalizer) {
		return ctrl.Result{}, nil
	}

	// status.BoundPodSnapshotContentName is the authoritative record of the content we created.
	// If it is unset, nothing was bound, so drop the finalizer. (Accepted orphan risk: a content
	// created via SSA but whose status write did not land before the process crashed AND the
	// PodSnapshot was deleted during that downtime would leak; deemed acceptable, not guarded.)
	contentName := ptr.Deref(snap.Status.BoundPodSnapshotContentName, "")
	if contentName == "" {
		controllerutil.RemoveFinalizer(snap, podSnapshotFinalizer)
		if err := sr.Update(ctx, snap); err != nil {
			return ctrl.Result{}, fmt.Errorf("remove snapshot finalizer: %w", err)
		}
		return ctrl.Result{}, nil
	}

	content := &nvidiacomv1alpha1.PodSnapshotContent{ObjectMeta: metav1.ObjectMeta{Name: contentName}}
	if err := sr.Delete(ctx, content); err != nil && !apierrors.IsNotFound(err) {
		return ctrl.Result{}, fmt.Errorf("delete PodSnapshotContent %q: %w", contentName, err)
	}

	// Block until the content is confirmed gone before releasing the PodSnapshot.
	if err := sr.Get(ctx, client.ObjectKey{Name: contentName}, &nvidiacomv1alpha1.PodSnapshotContent{}); err == nil {
		return ctrl.Result{RequeueAfter: snapshotContentDeleteRequeue}, nil
	} else if !apierrors.IsNotFound(err) {
		return ctrl.Result{}, fmt.Errorf("confirm PodSnapshotContent %q deleted: %w", contentName, err)
	}

	controllerutil.RemoveFinalizer(snap, podSnapshotFinalizer)
	if err := sr.Update(ctx, snap); err != nil {
		return ctrl.Result{}, fmt.Errorf("remove snapshot finalizer: %w", err)
	}
	return ctrl.Result{}, nil
}

// SetupWithManager wires the controller: it owns Snapshots and watches SnapshotContents,
// mapping a PodSnapshotContent back to its bound PodSnapshot via spec.snapshotRef.
func (sr *PodSnapshotReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.PodSnapshot{}).
		Watches(
			&nvidiacomv1alpha1.PodSnapshotContent{},
			handler.EnqueueRequestsFromMapFunc(podSnapshotContentToPodSnapshot),
		).
		Complete(sr)
}

// podSnapshotContentToPodSnapshot maps a PodSnapshotContent (including a delete-event tombstone) back
// to its bound PodSnapshot. It MUST unwrap cache.DeletedFinalStateUnknown so that the final
// PodSnapshotContent delete still re-enqueues the PodSnapshot and the cascade can complete.
func podSnapshotContentToPodSnapshot(ctx context.Context, obj client.Object) []reconcile.Request {
	ref, err := podSnapshotRefFromContentObj(obj)
	if err != nil {
		log.FromContext(ctx).Error(err, "Failed to map PodSnapshotContent to PodSnapshot")
		return nil
	}
	if ref.Name == "" {
		return nil
	}
	return []reconcile.Request{{NamespacedName: types.NamespacedName{Namespace: ref.Namespace, Name: ref.Name}}}
}

// podSnapshotRefFromContentObj extracts the bound PodSnapshot reference from a PodSnapshotContent,
// unwrapping a cache.DeletedFinalStateUnknown tombstone first so the final delete event
// still re-enqueues the PodSnapshot and the cascade can complete (F-2.2). It errors when the
// object is not a PodSnapshotContent (a malformed watch event, not a control-flow skip).
func podSnapshotRefFromContentObj(obj any) (nvidiacomv1alpha1.PodSnapshotReference, error) {
	if tombstone, isTombstone := obj.(cache.DeletedFinalStateUnknown); isTombstone {
		obj = tombstone.Obj
	}
	content, ok := obj.(*nvidiacomv1alpha1.PodSnapshotContent)
	if !ok {
		return nvidiacomv1alpha1.PodSnapshotReference{}, fmt.Errorf("expected *PodSnapshotContent, got %T", obj)
	}
	return content.Spec.PodSnapshotRef, nil
}

// podSnapshotContentName composes the deterministic cluster-scoped PodSnapshotContent name from
// the PodSnapshot UID, following the Kubernetes convention for naming a cluster-scoped object
// bound to a namespaced one (a dynamically provisioned PV is pvc-<PVC.UID>; an external-snapshotter
// content is snapcontent-<VolumeSnapshot.UID>). The UID-derived name is collision-proof cluster-wide
// and stable for the PodSnapshot's lifetime, so re-reconcile after a partial create Gets the same
// content rather than creating a duplicate.
func podSnapshotContentName(snap *nvidiacomv1alpha1.PodSnapshot) string {
	return "podsnapshotcontent-" + string(snap.UID)
}

// jitteredBackoff adds up to 50% jitter to a base delay to avoid synchronized requeues.
func jitteredBackoff(base time.Duration) time.Duration {
	return base + time.Duration(rand.Int63n(int64(base/2)+1))
}
