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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func snapshotReconcilerScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(s)
	_ = corev1.AddToScheme(s)
	return s
}

func makeSnapshotReconciler(s *runtime.Scheme, objs ...client.Object) *PodSnapshotReconciler {
	return makeSnapshotReconcilerWithInterceptor(s, interceptor.Funcs{}, objs...)
}

// makeSnapshotReconcilerWithInterceptor builds a reconciler whose fake client routes calls through
// interceptor.Funcs, letting tests inject API errors or count calls on specific code paths.
func makeSnapshotReconcilerWithInterceptor(s *runtime.Scheme, funcs interceptor.Funcs, objs ...client.Object) *PodSnapshotReconciler {
	return &PodSnapshotReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).
			WithStatusSubresource(&nvidiacomv1alpha1.PodSnapshot{}, &nvidiacomv1alpha1.PodSnapshotContent{}).
			WithInterceptorFuncs(funcs).Build(),
		Recorder: record.NewFakeRecorder(10),
	}
}

func makeSnapshotForReconcile() *nvidiacomv1alpha1.PodSnapshot {
	return &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "podsnapshot-abc123",
			Namespace:  "inference",
			UID:        types.UID("snap-uid"),
			Finalizers: []string{podSnapshotFinalizer},
		},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0"}},
		},
	}
}

// scheduledPod builds a scheduled source pod named "worker-0" on node "node-a". The checkpoint ID
// lives on the pod label (the reconciler reads it from there); pass "" to omit the label and exercise
// the missing-id path.
func scheduledPod(checkpointID string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference", UID: types.UID("pod-uid-9")},
		Spec:       corev1.PodSpec{NodeName: "node-a"},
	}
	if checkpointID != "" {
		pod.Labels = map[string]string{snapshotprotocol.CheckpointIDLabel: checkpointID}
	}
	return pod
}

func reconcileSnapshot(t *testing.T, r *PodSnapshotReconciler, name string) ctrl.Result {
	t.Helper()
	res, err := r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "inference", Name: name}})
	require.NoError(t, err)
	return res
}

func TestSnapshotReconciler_PodUnscheduledBacksOff(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference"}}
	r := makeSnapshotReconciler(s, snap, pod)

	// Unscheduled pod: return the sentinel error so controller-runtime backs off and requeues; no
	// content is created and the snapshot is not failed.
	_, err := r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "inference", Name: snap.Name}})
	require.Error(t, err)
	assert.ErrorIs(t, err, errPodSnapshotPodUnscheduled)

	var contents nvidiacomv1alpha1.PodSnapshotContentList
	require.NoError(t, r.List(context.Background(), &contents))
	assert.Empty(t, contents.Items)
}

func TestSnapshotReconciler_BuildsWorkOrderAndBinds(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	r := makeSnapshotReconciler(s, snap, scheduledPod("abc123"))

	// Creation path records the binding and returns without a requeue; conditions are mirrored on the
	// next (bound-path) reconcile that the content watch drives.
	res := reconcileSnapshot(t, r, snap.Name)
	assert.Zero(t, res.RequeueAfter)

	content := &nvidiacomv1alpha1.PodSnapshotContent{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Name: "podsnapshotcontent-snap-uid"}, content))
	assert.Equal(t, "worker-0", content.Spec.Source.PodRef.Name)
	assert.Equal(t, types.UID("pod-uid-9"), content.Spec.Source.PodRef.UID)
	assert.Equal(t, "node-a", content.Spec.Source.NodeName)
	assert.Equal(t, "node-a", content.Labels[snapshotprotocol.SnapshotNodeLabel])
	assert.NotContains(t, content.Labels, snapshotprotocol.CheckpointIDLabel)
	assert.NotContains(t, content.Annotations, snapshotprotocol.CheckpointArtifactVersionAnnotation)
	assert.Empty(t, content.Finalizers)
	assert.Equal(t, "inference", content.Spec.PodSnapshotRef.Namespace)
	assert.Equal(t, snap.Name, content.Spec.PodSnapshotRef.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	require.NotNil(t, updated.Status.BoundPodSnapshotContentName)
	assert.Equal(t, "podsnapshotcontent-snap-uid", *updated.Status.BoundPodSnapshotContentName)
	// Bound but not yet mirrored: no conditions written on the create pass (Ready not True, Failed
	// absent — don't deref a nil condition); conditions appear on the next bound-path reconcile.
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	assert.Nil(t, meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}

func TestSnapshotReconciler_StalePodReferenceFails(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// The PodSnapshot pins a source pod UID that does not match the live pod (pod-uid-9):
	// a same-named recreation must not be captured as the wrong workload.
	snap.Spec.Source.PodRef.UID = types.UID("old-pod-uid")
	r := makeSnapshotReconciler(s, snap, scheduledPod("abc123"))

	reconcileSnapshot(t, r, snap.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	cond := meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, "StalePodReference", cond.Reason)

	var contents nvidiacomv1alpha1.PodSnapshotContentList
	require.NoError(t, r.List(context.Background(), &contents))
	assert.Empty(t, contents.Items)
}

func TestSnapshotReconciler_MissingSourcePodFails(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// No source pod registered: a PodSnapshot is tied to its live pod, so it must fail terminally
	// rather than loop on NotFound.
	r := makeSnapshotReconciler(s, snap)

	res := reconcileSnapshot(t, r, snap.Name)
	assert.Zero(t, res.RequeueAfter)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	failed := meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, failed)
	assert.Equal(t, metav1.ConditionTrue, failed.Status)
	assert.Equal(t, "SourcePodNotFound", failed.Reason)
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
}

func TestSnapshotReconciler_ReadySnapReMirrorsViaBoundContent(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// Ready is NOT terminal: a bound, already-Ready snapshot keeps reconciling. With its content still
	// Ready it stays Ready via the bound path — no source pod resolved, no flip to Failed.
	meta.SetStatusCondition(&snap.Status.Conditions, metav1.Condition{
		Type: nvidiacomv1alpha1.PodSnapshotConditionReady, Status: metav1.ConditionTrue, Reason: "Captured", Message: "done"})
	snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name, UID: "snap-uid"},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: "pod-uid-9"}, NodeName: "node-a"},
		},
		Status: nvidiacomv1alpha1.PodSnapshotContentStatus{
			Conditions: []metav1.Condition{{Type: nvidiacomv1alpha1.PodSnapshotConditionReady, Status: metav1.ConditionTrue, Reason: "Agent", Message: "done"}},
		},
	}
	r := makeSnapshotReconciler(s, snap, content) // no source pod

	reconcileSnapshot(t, r, snap.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.True(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}

func TestSnapshotReconciler_ReadySnapRevertsToPendingWhenContentPending(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// Ready is not terminal and may change: a bound Ready snapshot whose content has reverted to no
	// conditions (Pending) re-mirrors back to Ready=False, Failed absent.
	meta.SetStatusCondition(&snap.Status.Conditions, metav1.Condition{
		Type: nvidiacomv1alpha1.PodSnapshotConditionReady, Status: metav1.ConditionTrue, Reason: "Captured", Message: "done"})
	snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name, UID: "snap-uid"},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: "pod-uid-9"}, NodeName: "node-a"},
		},
	}
	r := makeSnapshotReconciler(s, snap, content)

	reconcileSnapshot(t, r, snap.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}

func TestSnapshotReconciler_AlreadyFailedShortCircuits(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	meta.SetStatusCondition(&snap.Status.Conditions, metav1.Condition{
		Type: nvidiacomv1alpha1.PodSnapshotConditionFailed, Status: metav1.ConditionTrue, Reason: "SourcePodNotFound", Message: "gone"})
	// Failed is terminal & sticky: even with a (now) live pod present, it never becomes Ready.
	r := makeSnapshotReconciler(s, snap, scheduledPod("abc123"))

	reconcileSnapshot(t, r, snap.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.True(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
}

func TestSnapshotReconciler_BoundContentMissingRequeuesError(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// Content already created (recorded in status) but absent from the client — most likely a stale
	// cache read right after creation. The reconcile returns an error (requeue with backoff); it
	// does NOT fail the snapshot, which stays Pending.
	snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
	r := makeSnapshotReconciler(s, snap) // no content, no pod

	_, err := r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "inference", Name: snap.Name}})
	require.Error(t, err)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
}

func TestSnapshotReconciler_BoundContentPendingNoRequeue(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
	// Bound content exists but the agent hasn't written a result yet (no conditions): the dominant
	// live steady-state. Mirror Pending, no requeue, no pod resolved.
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name, UID: "snap-uid"},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: "pod-uid-9"}, NodeName: "node-a"},
		},
	}
	r := makeSnapshotReconciler(s, snap, content) // no source pod

	res := reconcileSnapshot(t, r, snap.Name)
	assert.Zero(t, res.RequeueAfter)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	assert.Nil(t, meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}

func TestSnapshotReconciler_ContentConflictFails(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// An existing content at this snapshot's name but bound to a different PodSnapshot UID must not
	// be adopted (CSI claimRef model).
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name, UID: "other-snap-uid"},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: "pod-uid-9"}, NodeName: "node-a"},
		},
	}
	r := makeSnapshotReconciler(s, snap, content, scheduledPod("abc123"))

	reconcileSnapshot(t, r, snap.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	failed := meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, failed)
	assert.Equal(t, "ContentConflict", failed.Reason)
}

func TestSnapshotReconciler_AdoptsExistingContentAndMirrors(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// Crash-recovery adopt: the content was created on a prior reconcile but the binding was never
	// recorded (BoundPodSnapshotContentName unset). The first pass adopts (Create→AlreadyExists→Get→
	// backlink OK) and records the binding; the second pass (bound path) mirrors the agent's status.
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name, UID: "snap-uid"},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: "pod-uid-9"}, NodeName: "node-a"},
		},
		Status: nvidiacomv1alpha1.PodSnapshotContentStatus{
			Conditions: []metav1.Condition{{Type: nvidiacomv1alpha1.PodSnapshotConditionReady, Status: metav1.ConditionTrue, Reason: "Agent", Message: "done"}},
		},
	}
	r := makeSnapshotReconciler(s, snap, content, scheduledPod("abc123"))

	// First pass: adopt and bind, no mirroring yet (no ContentConflict).
	reconcileSnapshot(t, r, snap.Name)
	bound := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, bound))
	require.NotNil(t, bound.Status.BoundPodSnapshotContentName)
	assert.Equal(t, "podsnapshotcontent-snap-uid", *bound.Status.BoundPodSnapshotContentName)
	assert.Nil(t, meta.FindStatusCondition(bound.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))

	// Second pass (bound path): mirror the content's Ready status.
	reconcileSnapshot(t, r, snap.Name)
	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.True(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	// markReady keeps the pair coherent (Failed=False), so Failed is present-but-False, never True.
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}

func TestSnapshotReconciler_MirrorsReadyAndFailed(t *testing.T) {
	for _, tc := range []struct {
		name      string
		condType  string
		wantReady metav1.ConditionStatus
	}{
		{name: "ready", condType: nvidiacomv1alpha1.PodSnapshotConditionReady},
		{name: "failed", condType: nvidiacomv1alpha1.PodSnapshotConditionFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := snapshotReconcilerScheme()
			snap := makeSnapshotForReconcile()
			// Bound path: the binding is already recorded, so mirroring happens from the content
			// with NO source pod resolved.
			snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
			content := &nvidiacomv1alpha1.PodSnapshotContent{
				ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
				Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
					PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name, UID: "snap-uid"},
					Source: nvidiacomv1alpha1.PodSnapshotContentSource{
						PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", UID: "pod-uid-9"}, NodeName: "node-a",
					},
				},
				Status: nvidiacomv1alpha1.PodSnapshotContentStatus{
					Conditions: []metav1.Condition{{Type: tc.condType, Status: metav1.ConditionTrue, Reason: "Agent", Message: "done"}},
				},
			}
			r := makeSnapshotReconciler(s, snap, content) // no source pod needed on the bound path

			reconcileSnapshot(t, r, snap.Name)

			updated := &nvidiacomv1alpha1.PodSnapshot{}
			require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
			cond := meta.FindStatusCondition(updated.Status.Conditions, tc.condType)
			require.NotNil(t, cond)
			assert.Equal(t, metav1.ConditionTrue, cond.Status)
			// Mutual exclusion is written, not merely absent: markReady/markFailed set the opposite
			// condition to present-and-False (a dropped paired setCondition would leave it absent).
			opposite := nvidiacomv1alpha1.PodSnapshotConditionFailed
			if tc.condType == nvidiacomv1alpha1.PodSnapshotConditionFailed {
				opposite = nvidiacomv1alpha1.PodSnapshotConditionReady
			}
			oppCond := meta.FindStatusCondition(updated.Status.Conditions, opposite)
			require.NotNil(t, oppCond)
			assert.Equal(t, metav1.ConditionFalse, oppCond.Status)
		})
	}
}

func TestSnapshotReconciler_ProceedsWithoutCheckpointIDLabel(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// The source pod carries no checkpoint-id label: the content is named from the PodSnapshot
	// UID, not the ID, so reconcile proceeds and binds rather than failing.
	r := makeSnapshotReconciler(s, snap, scheduledPod(""))

	reconcileSnapshot(t, r, snap.Name)

	content := &nvidiacomv1alpha1.PodSnapshotContent{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Name: "podsnapshotcontent-snap-uid"}, content))

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	require.NotNil(t, updated.Status.BoundPodSnapshotContentName)
	assert.Equal(t, "podsnapshotcontent-snap-uid", *updated.Status.BoundPodSnapshotContentName)
	assert.Nil(t, meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}

func TestSnapshotReconciler_DeleteWithNilBoundDropsFinalizer(t *testing.T) {
	s := snapshotReconcilerScheme()
	now := metav1.Now()
	snap := makeSnapshotForReconcile()
	snap.DeletionTimestamp = &now
	// status.BoundPodSnapshotContentName is unset → nothing was bound → finalizer is dropped.
	r := makeSnapshotReconciler(s, snap)

	reconcileSnapshot(t, r, snap.Name)

	gone := &nvidiacomv1alpha1.PodSnapshot{}
	err := r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, gone)
	if err == nil {
		assert.False(t, controllerutil.ContainsFinalizer(gone, podSnapshotFinalizer))
	} else {
		assert.True(t, apierrors.IsNotFound(err))
	}
}

func TestSnapshotReconciler_CascadeDelete(t *testing.T) {
	s := snapshotReconcilerScheme()
	now := metav1.Now()
	snap := makeSnapshotForReconcile()
	snap.DeletionTimestamp = &now
	snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0"}, NodeName: "node-a"},
		},
	}
	r := makeSnapshotReconciler(s, snap, content)

	// The content carries no finalizer, so it is deleted immediately; one pass deletes
	// the content and, once confirmed gone, drops the PodSnapshot finalizer.
	reconcileSnapshot(t, r, snap.Name)
	err := r.Get(context.Background(), types.NamespacedName{Name: "podsnapshotcontent-snap-uid"}, &nvidiacomv1alpha1.PodSnapshotContent{})
	assert.True(t, apierrors.IsNotFound(err))

	gone := &nvidiacomv1alpha1.PodSnapshot{}
	err = r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, gone)
	if err == nil {
		assert.False(t, controllerutil.ContainsFinalizer(gone, podSnapshotFinalizer))
	} else {
		assert.True(t, apierrors.IsNotFound(err))
	}
}

func TestSnapshotContentToSnapshot_UnwrapsTombstone(t *testing.T) {
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-abc123"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: "podsnapshot-abc123"},
		},
	}

	direct := podSnapshotContentToPodSnapshot(context.Background(), content)
	require.Len(t, direct, 1)
	assert.Equal(t, "podsnapshot-abc123", direct[0].Name)

	tombstone := cache.DeletedFinalStateUnknown{Key: "podsnapshotcontent-abc123", Obj: content}
	ref, err := podSnapshotRefFromContentObj(tombstone)
	require.NoError(t, err)
	assert.Equal(t, "podsnapshot-abc123", ref.Name)
	assert.Equal(t, "inference", ref.Namespace)

	// A non-PodSnapshotContent object is a malformed watch event, surfaced as an error.
	_, err = podSnapshotRefFromContentObj(&corev1.Pod{})
	require.Error(t, err)
	assert.Empty(t, podSnapshotContentToPodSnapshot(context.Background(), &corev1.Pod{}))

	// A content with an empty backref name (malformed/partial) maps to no requests.
	emptyRef := &nvidiacomv1alpha1.PodSnapshotContent{ObjectMeta: metav1.ObjectMeta{Name: "orphan"}}
	assert.Empty(t, podSnapshotContentToPodSnapshot(context.Background(), emptyRef))
}

func TestSnapshotReconciler_EnsureFinalizerAddsThenProceeds(t *testing.T) {
	s := snapshotReconcilerScheme()
	// A fresh snapshot with no finalizer: the first reconcile only adds the finalizer and short-circuits
	// (the watch re-enqueues), so no content is created yet; the second reconcile does the real work.
	snap := &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshot-abc123", Namespace: "inference", UID: types.UID("snap-uid")},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0"}},
		},
	}
	r := makeSnapshotReconciler(s, snap, scheduledPod("abc123"))

	res := reconcileSnapshot(t, r, snap.Name)
	assert.Zero(t, res.RequeueAfter)

	withFinalizer := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, withFinalizer))
	assert.True(t, controllerutil.ContainsFinalizer(withFinalizer, podSnapshotFinalizer))
	assert.Nil(t, meta.FindStatusCondition(withFinalizer.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
	var contents nvidiacomv1alpha1.PodSnapshotContentList
	require.NoError(t, r.List(context.Background(), &contents))
	assert.Empty(t, contents.Items)

	// Second pass: finalizer present, so reconcile proceeds to create the content and bind.
	reconcileSnapshot(t, r, snap.Name)
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Name: "podsnapshotcontent-snap-uid"}, &nvidiacomv1alpha1.PodSnapshotContent{}))
	bound := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, bound))
	require.NotNil(t, bound.Status.BoundPodSnapshotContentName)
	assert.Equal(t, "podsnapshotcontent-snap-uid", *bound.Status.BoundPodSnapshotContentName)
}

func TestSnapshotReconciler_DeleteWithNoFinalizerIsNoop(t *testing.T) {
	s := snapshotReconcilerScheme()
	now := metav1.Now()
	// handleDelete must return immediately when the finalizer is already gone and never touch any
	// content. The fake client refuses to persist a deleting object without a finalizer, so the snap is
	// passed in-memory straight to handleDelete; the client holds only the content.
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: "podsnapshot-abc123", UID: "snap-uid"},
		},
	}
	r := makeSnapshotReconciler(s, content)
	snap := &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshot-abc123", Namespace: "inference", UID: types.UID("snap-uid"), DeletionTimestamp: &now},
	}

	res, err := r.handleDelete(context.Background(), snap)
	require.NoError(t, err)
	assert.Zero(t, res.RequeueAfter)

	assert.NoError(t, r.Get(context.Background(), types.NamespacedName{Name: "podsnapshotcontent-snap-uid"}, &nvidiacomv1alpha1.PodSnapshotContent{}))
}

func TestSnapshotReconciler_BoundContentConflictFails(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
	// The bound content's namespace/name match this snapshot but its UID does not — the UID arm of
	// verifyContentBacklink must fail it on the bound path (not the namespace arm).
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: "podsnapshot-abc123", UID: "other-snap-uid"},
		},
	}
	r := makeSnapshotReconciler(s, snap, content) // no source pod: the bound path must not need it

	reconcileSnapshot(t, r, snap.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	failed := meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, failed)
	assert.Equal(t, "ContentConflict", failed.Reason)
}

func TestSnapshotReconciler_PropagateStatusIsIdempotent(t *testing.T) {
	boundContent := func(reason string) *nvidiacomv1alpha1.PodSnapshotContent {
		return &nvidiacomv1alpha1.PodSnapshotContent{
			ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
			Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
				PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: "podsnapshot-abc123", UID: "snap-uid"},
			},
			Status: nvidiacomv1alpha1.PodSnapshotContentStatus{
				Conditions: []metav1.Condition{{Type: nvidiacomv1alpha1.PodSnapshotConditionReady, Status: metav1.ConditionTrue, Reason: reason, Message: "done"}},
			},
		}
	}

	t.Run("no-op when already mirrored", func(t *testing.T) {
		s := snapshotReconcilerScheme()
		snap := makeSnapshotForReconcile()
		snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
		// Snapshot already carries the exact conditions markReady would write, so propagateStatus must
		// detect no change and skip the status write entirely.
		meta.SetStatusCondition(&snap.Status.Conditions, metav1.Condition{Type: nvidiacomv1alpha1.PodSnapshotConditionReady, Status: metav1.ConditionTrue, Reason: "Agent", Message: "done"})
		meta.SetStatusCondition(&snap.Status.Conditions, metav1.Condition{Type: nvidiacomv1alpha1.PodSnapshotConditionFailed, Status: metav1.ConditionFalse, Reason: "Agent", Message: "done"})

		count := 0
		funcs := interceptor.Funcs{SubResourceUpdate: func(ctx context.Context, c client.Client, sub string, obj client.Object, opts ...client.SubResourceUpdateOption) error {
			if sub == "status" {
				count++
			}
			return c.SubResource(sub).Update(ctx, obj, opts...)
		}}
		r := makeSnapshotReconcilerWithInterceptor(s, funcs, snap, boundContent("Agent"))

		reconcileSnapshot(t, r, snap.Name)
		assert.Zero(t, count, "no status write expected when conditions are unchanged")
	})

	t.Run("writes when a condition changed", func(t *testing.T) {
		s := snapshotReconcilerScheme()
		snap := makeSnapshotForReconcile()
		snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
		meta.SetStatusCondition(&snap.Status.Conditions, metav1.Condition{Type: nvidiacomv1alpha1.PodSnapshotConditionReady, Status: metav1.ConditionTrue, Reason: "Old", Message: "done"})

		count := 0
		funcs := interceptor.Funcs{SubResourceUpdate: func(ctx context.Context, c client.Client, sub string, obj client.Object, opts ...client.SubResourceUpdateOption) error {
			if sub == "status" {
				count++
			}
			return c.SubResource(sub).Update(ctx, obj, opts...)
		}}
		r := makeSnapshotReconcilerWithInterceptor(s, funcs, snap, boundContent("New"))

		reconcileSnapshot(t, r, snap.Name)
		assert.Equal(t, 1, count)
		updated := &nvidiacomv1alpha1.PodSnapshot{}
		require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
		assert.Equal(t, "New", meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady).Reason)
	})
}

func TestSnapshotReconciler_SourcePodGetErrorRequeues(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile() // finalizer pre-seeded so reconcile reaches getSourcePod
	// A non-NotFound error resolving the source pod must requeue (returned error), never fail the
	// snapshot terminally. The interceptor errors only for the pod Get, not the snapshot Get.
	funcs := interceptor.Funcs{Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		if _, ok := obj.(*corev1.Pod); ok {
			return errors.New("transient API error")
		}
		return c.Get(ctx, key, obj, opts...)
	}}
	r := makeSnapshotReconcilerWithInterceptor(s, funcs, snap, scheduledPod("abc123"))

	_, err := r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "inference", Name: snap.Name}})
	require.Error(t, err)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.Nil(t, meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
}

func TestSnapshotReconciler_ContentCreateErrorEmitsEventAndRequeues(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// A non-AlreadyExists Create error must requeue (returned error), record a warning event, and not
	// fail the snapshot terminally.
	funcs := interceptor.Funcs{Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
		if _, ok := obj.(*nvidiacomv1alpha1.PodSnapshotContent); ok {
			return errors.New("apiserver rejected create")
		}
		return c.Create(ctx, obj, opts...)
	}}
	r := makeSnapshotReconcilerWithInterceptor(s, funcs, snap, scheduledPod("abc123"))

	_, err := r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "inference", Name: snap.Name}})
	require.Error(t, err)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	assert.Nil(t, meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))

	recorder := r.Recorder.(*record.FakeRecorder)
	select {
	case ev := <-recorder.Events:
		assert.Contains(t, ev, "SnapshotContentCreateFailed")
	default:
		t.Fatal("expected a SnapshotContentCreateFailed warning event")
	}
}

func TestSnapshotReconciler_CascadeDeleteRequeuesUntilContentGone(t *testing.T) {
	s := snapshotReconcilerScheme()
	now := metav1.Now()
	snap := makeSnapshotForReconcile()
	snap.DeletionTimestamp = &now
	snap.Status.BoundPodSnapshotContentName = ptr.To("podsnapshotcontent-snap-uid")
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name},
		},
	}
	// No-op the FIRST Delete only, so the content survives one pass and the requeue-while-present arm
	// fires; subsequent Deletes pass through. handleDelete calls Delete exactly once per pass — adjust
	// the threshold if that ever changes.
	deleteCalls := 0
	funcs := interceptor.Funcs{Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
		if _, ok := obj.(*nvidiacomv1alpha1.PodSnapshotContent); ok {
			deleteCalls++
			if deleteCalls == 1 {
				return nil // swallow the first delete: content remains present
			}
		}
		return c.Delete(ctx, obj, opts...)
	}}
	r := makeSnapshotReconcilerWithInterceptor(s, funcs, snap, content)

	// Pass 1: content still present → requeue, finalizer retained.
	res, err := r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "inference", Name: snap.Name}})
	require.NoError(t, err)
	assert.Equal(t, 1, deleteCalls)
	assert.Equal(t, snapshotContentDeleteRequeue, res.RequeueAfter)
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Name: "podsnapshotcontent-snap-uid"}, &nvidiacomv1alpha1.PodSnapshotContent{}))
	stillDeleting := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, stillDeleting))
	assert.True(t, controllerutil.ContainsFinalizer(stillDeleting, podSnapshotFinalizer))

	// Pass 2: real delete removes the content → finalizer dropped (snapshot gone).
	reconcileSnapshot(t, r, snap.Name)
	assert.True(t, apierrors.IsNotFound(r.Get(context.Background(), types.NamespacedName{Name: "podsnapshotcontent-snap-uid"}, &nvidiacomv1alpha1.PodSnapshotContent{})))
	gone := &nvidiacomv1alpha1.PodSnapshot{}
	err = r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, gone)
	if err == nil {
		assert.False(t, controllerutil.ContainsFinalizer(gone, podSnapshotFinalizer))
	} else {
		assert.True(t, apierrors.IsNotFound(err))
	}
}

// filterExcludedNamespaces implements commonController.ExcludedNamespacesInterface over a fixed list.
type filterExcludedNamespaces []string

// Contains reports whether namespace is in the stubbed exclusion list.
func (f filterExcludedNamespaces) Contains(namespace string) bool {
	return slices.Contains(f, namespace)
}

// contentWithRefNamespace builds a cluster-scoped PodSnapshotContent bound to a PodSnapshot in ns.
func contentWithRefNamespace(ns string) *nvidiacomv1alpha1.PodSnapshotContent {
	return &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-x"},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: ns, Name: "snap-x"},
		},
	}
}

func TestPodSnapshotContentEventFilter(t *testing.T) {
	tests := []struct {
		name       string
		restricted string
		excluded   []string
		obj        client.Object
		want       bool
	}{
		{name: "restricted mode admits matching snapshot ref namespace", restricted: "prod", obj: contentWithRefNamespace("prod"), want: true},
		{name: "restricted mode drops mismatched snapshot ref namespace", restricted: "prod", obj: contentWithRefNamespace("other"), want: false},
		{name: "restricted mode drops empty snapshot ref namespace", restricted: "prod", obj: contentWithRefNamespace(""), want: false},
		{name: "cluster-wide drops excluded snapshot ref namespace", excluded: []string{"banned"}, obj: contentWithRefNamespace("banned"), want: false},
		{name: "cluster-wide drops ephemeral snapshot ref namespace", obj: contentWithRefNamespace("ci-ephemeral-1"), want: false},
		{name: "cluster-wide admits normal snapshot ref namespace", obj: contentWithRefNamespace("prod"), want: true},
		{name: "non-content object is dropped", restricted: "prod", obj: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "prod", Name: "p"}}, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &configv1alpha1.OperatorConfiguration{}
			config.Namespace.Restricted = tt.restricted
			runtimeConfig := &commonController.RuntimeConfig{}
			if tt.excluded != nil {
				runtimeConfig.ExcludedNamespaces = filterExcludedNamespaces(tt.excluded)
			}
			pred := podSnapshotContentEventFilter(config, runtimeConfig)
			assert.Equal(t, tt.want, pred.Create(event.CreateEvent{Object: tt.obj}))
		})
	}

	t.Run("delete event admits matching snapshot ref namespace", func(t *testing.T) {
		config := &configv1alpha1.OperatorConfiguration{}
		config.Namespace.Restricted = "prod"
		pred := podSnapshotContentEventFilter(config, &commonController.RuntimeConfig{})
		assert.True(t, pred.Delete(event.DeleteEvent{Object: contentWithRefNamespace("prod")}))
	})
}
