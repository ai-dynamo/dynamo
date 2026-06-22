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
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func snapshotReconcilerScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(s)
	_ = corev1.AddToScheme(s)
	return s
}

func makeSnapshotReconciler(s *runtime.Scheme, objs ...client.Object) *PodSnapshotReconciler {
	return &PodSnapshotReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).
			WithStatusSubresource(&nvidiacomv1alpha1.PodSnapshot{}, &nvidiacomv1alpha1.PodSnapshotContent{}).Build(),
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

// scheduledPod builds a scheduled source pod named "worker-0". The checkpoint ID lives on the pod
// label (the reconciler reads it from there); pass "" to omit the label and exercise the
// missing-id path.
func scheduledPod(node, checkpointID string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: "inference", UID: types.UID("pod-uid-9")},
		Spec:       corev1.PodSpec{NodeName: node},
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

	res := reconcileSnapshot(t, r, snap.Name)
	assert.Positive(t, res.RequeueAfter)

	var contents nvidiacomv1alpha1.PodSnapshotContentList
	require.NoError(t, r.List(context.Background(), &contents))
	assert.Empty(t, contents.Items)
}

func TestSnapshotReconciler_BuildsWorkOrderAndBinds(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	r := makeSnapshotReconciler(s, snap, scheduledPod("node-a", "abc123"))

	reconcileSnapshot(t, r, snap.Name)

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
}

func TestSnapshotReconciler_StalePodReferenceFails(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// The PodSnapshot pins a source pod UID that does not match the live pod (pod-uid-9):
	// a same-named recreation must not be captured as the wrong workload.
	snap.Spec.Source.PodRef.UID = types.UID("old-pod-uid")
	r := makeSnapshotReconciler(s, snap, scheduledPod("node-a", "abc123"))

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
			content := &nvidiacomv1alpha1.PodSnapshotContent{
				ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid", Finalizers: []string{podSnapshotFinalizer}},
				Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
					PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name},
					Source: nvidiacomv1alpha1.PodSnapshotContentSource{
						PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0"}, NodeName: "node-a",
					},
				},
				Status: nvidiacomv1alpha1.PodSnapshotContentStatus{
					Conditions: []metav1.Condition{{Type: tc.condType, Status: metav1.ConditionTrue, Reason: "Agent", Message: "done"}},
				},
			}
			r := makeSnapshotReconciler(s, snap, content, scheduledPod("node-a", "abc123"))

			reconcileSnapshot(t, r, snap.Name)

			updated := &nvidiacomv1alpha1.PodSnapshot{}
			require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
			cond := meta.FindStatusCondition(updated.Status.Conditions, tc.condType)
			require.NotNil(t, cond)
			assert.Equal(t, metav1.ConditionTrue, cond.Status)
		})
	}
}

func TestSnapshotReconciler_RescheduleFailsSnapshot(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	content := &nvidiacomv1alpha1.PodSnapshotContent{
		ObjectMeta: metav1.ObjectMeta{Name: "podsnapshotcontent-snap-uid", Finalizers: []string{podSnapshotFinalizer}},
		Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
			PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{Namespace: "inference", Name: snap.Name},
			Source:         nvidiacomv1alpha1.PodSnapshotContentSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0"}, NodeName: "node-a"},
		},
	}
	// Pod now runs on a different node than the bound content.
	r := makeSnapshotReconciler(s, snap, content, scheduledPod("node-b", "abc123"))

	reconcileSnapshot(t, r, snap.Name)

	updated := &nvidiacomv1alpha1.PodSnapshot{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Namespace: "inference", Name: snap.Name}, updated))
	cond := meta.FindStatusCondition(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed)
	require.NotNil(t, cond)
	assert.Equal(t, metav1.ConditionTrue, cond.Status)
	assert.Equal(t, "PodRescheduled", cond.Reason)
}

func TestSnapshotReconciler_ProceedsWithoutCheckpointIDLabel(t *testing.T) {
	s := snapshotReconcilerScheme()
	snap := makeSnapshotForReconcile()
	// The source pod carries no checkpoint-id label: the content is named from the PodSnapshot
	// UID, not the ID, so reconcile proceeds and binds rather than failing.
	r := makeSnapshotReconciler(s, snap, scheduledPod("node-a", ""))

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
}
