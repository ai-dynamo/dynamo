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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const (
	// podSnapshotFinalizer is set on the PodSnapshot so its bound PodSnapshotContent is
	// deleted before the PodSnapshot is removed.
	podSnapshotFinalizer = "nvidia.com/podsnapshotcontent-cleanup"

	// snapshotContentDeleteRequeue is the delay between cascade-delete progress checks.
	snapshotContentDeleteRequeue = time.Second
)

// errPodSnapshotPodUnscheduled signals that the source pod is not yet scheduled to a node; the
// reconcile returns it so controller-runtime requeues with backoff rather than failing the snapshot.
var errPodSnapshotPodUnscheduled = errors.New("source pod is not yet scheduled to a node")

// errPodSnapshotStalePodRef signals the live pod is a same-named recreation (UID mismatch) — a
// terminal mismatch, not a retryable condition.
var errPodSnapshotStalePodRef = errors.New("source pod UID does not match the pinned PodSnapshot source")

// errContentConflict marks an existing PodSnapshotContent that does not belong to this PodSnapshot.
var errContentConflict = errors.New("existing PodSnapshotContent belongs to another PodSnapshot")

// PodSnapshotReconciler reconciles a PodSnapshot: it creates the bound, cluster-scoped
// PodSnapshotContent work order for the node agent, mirrors the agent's terminal status
// back to the PodSnapshot, and cascades deletion to the PodSnapshotContent.
type PodSnapshotReconciler struct {
	client.Client
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig
	Recorder      record.EventRecorder
}

// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshots,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshots/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshots/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshotcontents,verbs=create;get;list;watch;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshotcontents/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch

// Reconcile drives a PodSnapshot through binding, status mirroring, and cascade deletion. It is a thin
// orchestrator: each branch delegates to a helper that owns that path's detail.
func (sr *PodSnapshotReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	snap := &nvidiacomv1alpha1.PodSnapshot{}
	if err := sr.Get(ctx, req.NamespacedName, snap); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !snap.GetDeletionTimestamp().IsZero() {
		return sr.handleDelete(ctx, snap)
	}
	if added, err := sr.ensureFinalizer(ctx, snap); err != nil || added {
		return ctrl.Result{}, err
	}
	if isPodSnapshotTerminal(snap) {
		return ctrl.Result{}, nil
	}

	// Once a PodSnapshotContent is bound it drives the snapshot; the live source pod is consulted
	// only on the path that has not yet created one.
	if boundName := ptr.Deref(snap.Status.BoundPodSnapshotContentName, ""); boundName != "" {
		return sr.mirrorBoundContent(ctx, snap, boundName)
	}
	return sr.captureFromSourcePod(ctx, snap)
}

// isPodSnapshotTerminal reports whether the snapshot has reached a terminal, sticky state. Only Failed
// is terminal: a failed snapshot never recovers, so we stop re-evaluating it. Ready is intentionally
// NOT terminal — it mirrors the content's status and may still change, so a Ready snapshot keeps
// reconciling (and re-mirroring) on later content-watch events.
func isPodSnapshotTerminal(snap *nvidiacomv1alpha1.PodSnapshot) bool {
	return meta.IsStatusConditionTrue(snap.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
}

// ensureFinalizer adds the cleanup finalizer when absent, reporting added=true when it patched (the
// caller returns early; the resulting watch event re-enqueues — so a fake-client test must reconcile
// twice when it does not pre-seed the finalizer).
func (sr *PodSnapshotReconciler) ensureFinalizer(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot) (bool, error) {
	if controllerutil.ContainsFinalizer(snap, podSnapshotFinalizer) {
		return false, nil
	}
	controllerutil.AddFinalizer(snap, podSnapshotFinalizer)
	if err := sr.Update(ctx, snap); err != nil {
		return false, fmt.Errorf("add snapshot finalizer: %w", err)
	}
	return true, nil
}

// mirrorBoundContent loads the bound PodSnapshotContent and mirrors its status onto the PodSnapshot.
// It does not resolve the source pod (the binding already exists).
func (sr *PodSnapshotReconciler) mirrorBoundContent(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, boundName string) (ctrl.Result, error) {
	content := &nvidiacomv1alpha1.PodSnapshotContent{}
	if err := sr.Get(ctx, client.ObjectKey{Name: boundName}, content); err != nil {
		return ctrl.Result{}, fmt.Errorf("get bound PodSnapshotContent %q: %w", boundName, err)
	}
	// Defense-in-depth: the content's backref to this snapshot. Its spec is immutable, so this is a
	// no-op after the first pass; it never triggers an extra API read (the Get above is needed anyway).
	if err := verifyContentBacklink(snap, content); err != nil {
		return sr.failPodSnapshot(ctx, snap, "ContentConflict", err)
	}
	return sr.propagateStatus(ctx, snap, content)
}

// captureFromSourcePod is the no-content path: it resolves and validates the live source pod, then
// creates the PodSnapshotContent and records the binding.
func (sr *PodSnapshotReconciler) captureFromSourcePod(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot) (ctrl.Result, error) {
	pod, err := sr.getSourcePod(ctx, snap)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return sr.failPodSnapshot(ctx, snap, "SourcePodNotFound",
				fmt.Errorf("source pod %q no longer exists", snap.Spec.Source.PodRef.Name))
		}
		return ctrl.Result{}, err
	}
	if err = validateSourcePod(snap, pod); err != nil {
		if errors.Is(err, errPodSnapshotPodUnscheduled) {
			return ctrl.Result{}, err // controller-runtime backs off and requeues on a returned error
		}
		if errors.Is(err, errPodSnapshotStalePodRef) {
			return sr.failPodSnapshot(ctx, snap, "StalePodReference", err)
		}
		return ctrl.Result{}, fmt.Errorf("validate source pod: %w", err)
	}

	content, err := sr.ensurePodSnapshotContent(ctx, snap, podSnapshotContentName(snap), pod)
	if err != nil {
		if errors.Is(err, errContentConflict) {
			return sr.failPodSnapshot(ctx, snap, "ContentConflict", err)
		}
		return ctrl.Result{}, err
	}
	return sr.bindContent(ctx, snap, content.Name)
}

// verifyContentBacklink errors when a content's backref does not point at this PodSnapshot
// (namespace/name, and uid when recorded). It is pod-free: the content↔pod relationship is the
// PodSnapshotContent's own concern, not the PodSnapshot reconciler's.
func verifyContentBacklink(snap *nvidiacomv1alpha1.PodSnapshot, content *nvidiacomv1alpha1.PodSnapshotContent) error {
	if ref := content.Spec.PodSnapshotRef; ref.Namespace != snap.Namespace || ref.Name != snap.Name ||
		(ref.UID != "" && ref.UID != snap.UID) {
		return fmt.Errorf("PodSnapshotContent %q is bound to %s/%s (uid %q), not %s/%s (uid %q)",
			content.Name, ref.Namespace, ref.Name, ref.UID, snap.Namespace, snap.Name, snap.UID)
	}
	return nil
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

// validateSourcePod requires the pod to be scheduled and, when the PodSnapshot pins a source pod UID,
// to match it (rejecting a same-named recreation). It returns one of two sentinels:
// errPodSnapshotPodUnscheduled (retryable) or errPodSnapshotStalePodRef (terminal).
func validateSourcePod(snap *nvidiacomv1alpha1.PodSnapshot, pod *corev1.Pod) error {
	if pod.Spec.NodeName == "" {
		return errPodSnapshotPodUnscheduled
	}
	if wantUID := snap.Spec.Source.PodRef.UID; wantUID != "" && pod.UID != wantUID {
		return fmt.Errorf("%w: live pod %q UID %q, want %q", errPodSnapshotStalePodRef, pod.Name, pod.UID, wantUID)
	}
	return nil
}

// ensurePodSnapshotContent creates the PodSnapshotContent trigger. The bound-content check already ran,
// so the content almost never exists here — assume it does not and Create it. The only way it can
// already exist is a prior reconcile that created it but crashed before the status write, surfaced as
// AlreadyExists; only then do we re-fetch, confirm it is ours (the spec is immutable), and adopt it.
func (sr *PodSnapshotReconciler) ensurePodSnapshotContent(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, contentName string, pod *corev1.Pod) (*nvidiacomv1alpha1.PodSnapshotContent, error) {
	content := sr.buildPodSnapshotContent(snap, contentName, pod)
	if err := sr.Create(ctx, content); err != nil {
		if apierrors.IsAlreadyExists(err) {
			existing := &nvidiacomv1alpha1.PodSnapshotContent{}
			// A NotFound here is a benign delete/recreate race; the returned error just requeues and the
			// next Create succeeds.
			if err := sr.Get(ctx, client.ObjectKey{Name: contentName}, existing); err != nil {
				return nil, fmt.Errorf("get existing PodSnapshotContent %q: %w", contentName, err)
			}
			if err := verifyContentBacklink(snap, existing); err != nil {
				// %v on the inner: only errContentConflict needs to be unwrappable by the caller.
				return nil, fmt.Errorf("%w: %v", errContentConflict, err)
			}
			return existing, nil
		}
		sr.Recorder.Event(snap, corev1.EventTypeWarning, "SnapshotContentCreateFailed", err.Error())
		return nil, fmt.Errorf("create PodSnapshotContent %q: %w", contentName, err)
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

// bindContent records the one-time binding of the created/adopted PodSnapshotContent. Mirroring the
// content's status is the bound path's job (mirrorBoundContent), driven by the content watch.
func (sr *PodSnapshotReconciler) bindContent(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, contentName string) (ctrl.Result, error) {
	snap.Status.BoundPodSnapshotContentName = ptr.To(contentName)
	if err := sr.Status().Update(ctx, snap); err != nil {
		return ctrl.Result{}, fmt.Errorf("record bound PodSnapshotContent %q: %w", contentName, err)
	}
	return ctrl.Result{}, nil
}

// propagateStatus mirrors the bound PodSnapshotContent's status onto the PodSnapshot, defaulting to a
// Pending condition until the agent writes a result. It receives the content resolved by the bound
// path, so it never re-Gets it, and writes status only when a condition actually changed.
func (sr *PodSnapshotReconciler) propagateStatus(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, content *nvidiacomv1alpha1.PodSnapshotContent) (ctrl.Result, error) {
	var changed bool
	switch {
	case nvidiacomv1alpha1.IsPodSnapshotContentSucceeded(content):
		cond := meta.FindStatusCondition(content.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady)
		changed = sr.markReady(snap, cond.Reason, cond.Message)
	case nvidiacomv1alpha1.IsPodSnapshotContentFailed(content):
		cond := meta.FindStatusCondition(content.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
		changed = sr.markFailed(snap, cond.Reason, cond.Message)
	default:
		changed = sr.markPending(snap, "Pending", "Waiting for node agent to capture the checkpoint")
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

// Ready and Failed are kept as a coherent, mutually-exclusive pair: at most one is True. Both
// False is the in-progress (Pending) state. markReady/markFailed set both ends so the snapshot can
// never report Ready=True and Failed=True simultaneously.

// markReady records terminal success: Ready=True, Failed=False.
func (sr *PodSnapshotReconciler) markReady(snap *nvidiacomv1alpha1.PodSnapshot, reason, message string) bool {
	changed := sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionReady, metav1.ConditionTrue, reason, message)
	return sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionFailed, metav1.ConditionFalse, reason, message) || changed
}

// markFailed records terminal failure: Failed=True, Ready=False. Failed is sticky (the reconcile
// short-circuits on a terminal snapshot, so it never transitions back to Ready).
func (sr *PodSnapshotReconciler) markFailed(snap *nvidiacomv1alpha1.PodSnapshot, reason, message string) bool {
	changed := sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionFailed, metav1.ConditionTrue, reason, message)
	return sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionReady, metav1.ConditionFalse, reason, message) || changed
}

// markPending records the in-progress state: Ready=False, Failed left absent (both-False). It never
// needs to clear Failed because Failed is terminal — isPodSnapshotTerminal short-circuits the reconcile
// before reaching here once Failed is set, so this is only hit while Failed is absent or False.
func (sr *PodSnapshotReconciler) markPending(snap *nvidiacomv1alpha1.PodSnapshot, reason, message string) bool {
	return sr.setCondition(snap, nvidiacomv1alpha1.PodSnapshotConditionReady, metav1.ConditionFalse, reason, message)
}

// failPodSnapshot marks the PodSnapshot Failed terminally (Failed=True, Ready=False) and records
// an event.
func (sr *PodSnapshotReconciler) failPodSnapshot(ctx context.Context, snap *nvidiacomv1alpha1.PodSnapshot, reason string, cause error) (ctrl.Result, error) {
	sr.Recorder.Event(snap, corev1.EventTypeWarning, reason, cause.Error())
	sr.markFailed(snap, reason, cause.Error())
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

	// status.BoundPodSnapshotContentName is the authoritative record of the content we created;
	// cascade-delete keys off it (we do NOT reconstruct the name from the PodSnapshot UID, because
	// a future bring-your-own-content mode will let the content name diverge from podsnapshotcontent-
	// <uid>). If it is unset, nothing was bound, so drop the finalizer.
	//
	// Accepted orphan risk (deemed acceptable, not guarded here): a content created via Create whose
	// status write did not land before the process crashed AND the PodSnapshot was deleted during
	// that downtime would leak. A future GC/cleanup policy will reclaim unbound content.
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
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(sr.Config, sr.RuntimeConfig)).
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
