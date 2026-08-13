/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpointjob"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

const (
	cascadeTestNamespace = "test-ns"
	cascadeTestPCSG      = "my-pcsg"
)

func newFailoverPod(name string, phase corev1.PodPhase, replicaIdx, podIdx string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cascadeTestNamespace,
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoFailoverEngineGroupMember: commonconsts.KubeLabelValueTrue,
				groveLabelPCSG:             cascadeTestPCSG,
				groveLabelPCSGReplicaIndex: replicaIdx,
				groveLabelPodIndex:         podIdx,
			},
		},
		Status: corev1.PodStatus{Phase: phase},
	}
}

func newCascadeReconciler(objs ...client.Object) (*failoverCascadeReconciler, client.Client) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	cb := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&corev1.Pod{})
	for _, o := range objs {
		cb = cb.WithObjects(o)
	}
	c := cb.Build()

	return &failoverCascadeReconciler{
		Client:   c,
		recorder: events.NewFakeRecorder(16),
	}, c
}

func TestFailoverCascade_FailedPodDeletesEntireGroup(t *testing.T) {

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "0", "0")
	sibling1 := newFailoverPod("gms-0-0", corev1.PodRunning, "0", "0")
	sibling2 := newFailoverPod("wkr-1-0", corev1.PodRunning, "0", "0")

	r, c := newCascadeReconciler(failedPod, sibling1, sibling2)

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Empty(t, remaining.Items, "all pods in the engine group should be deleted")
}

func TestFailoverCascade_SucceededPodDeletesEntireGroup(t *testing.T) {

	succeededPod := newFailoverPod("ldr-0", corev1.PodSucceeded, "0", "0")
	sibling := newFailoverPod("gms-0-0", corev1.PodRunning, "0", "0")

	r, c := newCascadeReconciler(succeededPod, sibling)

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Empty(t, remaining.Items, "succeeded pod should also trigger cascade")
}

func TestFailoverCascade_DifferentGroupUnaffected(t *testing.T) {
	t.Log("Build a failed trigger, exact sibling, and Pods differing in each selector dimension")
	failedPod := newFailoverPod("trigger", corev1.PodFailed, "0", "0")
	sibling := newFailoverPod("sibling", corev1.PodRunning, "0", "0")
	differentPCSG := newFailoverPod("different-pcsg", corev1.PodRunning, "0", "0")
	differentPCSG.Labels[groveLabelPCSG] = "other-pcsg"
	differentReplica := newFailoverPod("different-replica", corev1.PodRunning, "1", "0")
	differentPodIndex := newFailoverPod("different-pod-index", corev1.PodRunning, "0", "1")
	differentMember := newFailoverPod("different-member", corev1.PodRunning, "0", "0")
	differentMember.Labels[commonconsts.KubeLabelDynamoFailoverEngineGroupMember] = "false"
	restoreTarget := newFailoverPod("restore-target", corev1.PodRunning, "0", "0")
	restoreTarget.Labels[snapshotprotocol.RestoreTargetLabel] = commonconsts.KubeLabelValueTrue
	differentNamespace := newFailoverPod("different-namespace", corev1.PodRunning, "0", "0")
	differentNamespace.Namespace = "other-ns"

	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))

	var deleteOptions client.DeleteAllOfOptions
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&corev1.Pod{}).
		WithObjects(
			failedPod,
			sibling,
			differentPCSG,
			differentReplica,
			differentPodIndex,
			differentMember,
			restoreTarget,
			differentNamespace,
		).
		WithInterceptorFuncs(interceptor.Funcs{
			DeleteAllOf: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteAllOfOption) error {
				deleteOptions.ApplyOptions(opts)
				return c.DeleteAllOf(ctx, obj, opts...)
			},
		}).
		Build()
	r := &failoverCascadeReconciler{Client: c, recorder: events.NewFakeRecorder(16)}

	t.Log("Reconcile the failed trigger and capture the destructive delete options")
	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKeyFromObject(failedPod),
	})
	require.NoError(t, err)

	t.Log("Verify the delete is namespace-scoped, excludes restore targets, and uses zero grace")
	assert.Equal(t, cascadeTestNamespace, deleteOptions.Namespace)
	assert.True(t, deleteOptions.LabelSelector.Matches(labels.Set(failedPod.Labels)))
	assert.True(t, deleteOptions.LabelSelector.Matches(labels.Set(sibling.Labels)))
	assert.False(t, deleteOptions.LabelSelector.Matches(labels.Set(restoreTarget.Labels)))
	require.NotNil(t, deleteOptions.GracePeriodSeconds)
	assert.Zero(t, *deleteOptions.GracePeriodSeconds)

	t.Log("Verify only the exact engine group was deleted")
	require.True(t, apierrors.IsNotFound(c.Get(context.Background(), client.ObjectKeyFromObject(failedPod), &corev1.Pod{})))
	require.True(t, apierrors.IsNotFound(c.Get(context.Background(), client.ObjectKeyFromObject(sibling), &corev1.Pod{})))
	for _, pod := range []*corev1.Pod{
		differentPCSG,
		differentReplica,
		differentPodIndex,
		differentMember,
		differentNamespace,
	} {
		require.NoError(t, c.Get(context.Background(), client.ObjectKeyFromObject(pod), &corev1.Pod{}))
	}
	require.NoError(t, c.Get(context.Background(), client.ObjectKeyFromObject(restoreTarget), &corev1.Pod{}))
}

func TestFailoverCascade_MultipleFailedPodsAllDeleted(t *testing.T) {

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "0", "0")
	alsoFailed := newFailoverPod("wkr-1-0", corev1.PodFailed, "0", "0")
	running := newFailoverPod("gms-0-0", corev1.PodRunning, "0", "0")

	r, c := newCascadeReconciler(failedPod, alsoFailed, running)

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Empty(t, remaining.Items, "all pods in the engine group should be deleted")
}

func TestFailoverCascade_PodWithoutLabelIgnored(t *testing.T) {

	unlabeled := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "random-pod",
			Namespace: cascadeTestNamespace,
		},
		Status: corev1.PodStatus{Phase: corev1.PodFailed},
	}

	r, _ := newCascadeReconciler(unlabeled)

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "random-pod", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestFailoverCascade_NonFailedPodIsNoop(t *testing.T) {

	runningPod := newFailoverPod("ldr-0", corev1.PodRunning, "0", "0")
	sibling := newFailoverPod("gms-0-0", corev1.PodRunning, "0", "0")

	r, c := newCascadeReconciler(runningPod, sibling)

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Len(t, remaining.Items, 2, "running pod should not trigger cascade")
}

func TestFailoverCascade_NotFoundPodIsNoop(t *testing.T) {
	r, _ := newCascadeReconciler()

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "gone", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestFailoverCascade_MissingGroveLabelsIsNoop(t *testing.T) {

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "partial-labels",
			Namespace: cascadeTestNamespace,
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoFailoverEngineGroupMember: commonconsts.KubeLabelValueTrue,
				groveLabelPCSG: cascadeTestPCSG,
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodFailed},
	}

	r, _ := newCascadeReconciler(pod)

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "partial-labels", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestFailoverCascade_DifferentPCSGReplicaUnaffected(t *testing.T) {

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "0", "0")
	differentReplica := newFailoverPod("ldr-r1-0", corev1.PodRunning, "1", "0")

	r, c := newCascadeReconciler(failedPod, differentReplica)

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Len(t, remaining.Items, 1, "only the different PCSG replica pod should remain")
	assert.Equal(t, "ldr-r1-0", remaining.Items[0].Name)
}

func TestFailoverCascade_DeletingPodIsSkipped(t *testing.T) {

	now := metav1.Now()

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "0", "0")
	failedPod.DeletionTimestamp = &now
	failedPod.DeletionGracePeriodSeconds = ptr.To(int64(0))
	failedPod.Finalizers = []string{"test-finalizer"}
	sibling := newFailoverPod("gms-0-0", corev1.PodRunning, "0", "0")

	r, c := newCascadeReconciler(failedPod, sibling)

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Len(t, remaining.Items, 2, "already-deleting pod should not trigger a cascade")
}

func TestFailoverCascade_ConcurrentReconcileIsIdempotent(t *testing.T) {

	pod1 := newFailoverPod("ldr-0", corev1.PodFailed, "0", "0")
	pod2 := newFailoverPod("wkr-1-0", corev1.PodFailed, "0", "0")

	r, c := newCascadeReconciler(pod1, pod2)

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)

	// Second reconcile for the other pod — it's already gone (NotFound).
	_, err = r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "wkr-1-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Empty(t, remaining.Items)
}

func TestFailoverCascade_SnapshotRestoreTargetCannotDeleteSibling(t *testing.T) {
	t.Log("Build a terminal Snapshot restore target with every failover label and a matching sibling")
	pod := gmsPodReplacementTestPod("snapshot-uid", 0)
	pod.Status.Phase = corev1.PodFailed
	pod.Labels[commonconsts.KubeLabelDynamoFailoverEngineGroupMember] = commonconsts.KubeLabelValueTrue
	pod.Labels[groveLabelPCSG] = cascadeTestPCSG
	pod.Labels[groveLabelPCSGReplicaIndex] = "0"
	pod.Labels[groveLabelPodIndex] = "0"
	sibling := newFailoverPod("sibling", corev1.PodRunning, "0", "0")
	sibling.Namespace = pod.Namespace

	t.Log("Verify the failover predicate rejects the unsupported Snapshot overlap")
	pred := failoverCascadePredicate()
	assert.False(t, pred.Create(event.CreateEvent{Object: pod}))
	oldPod := pod.DeepCopy()
	oldPod.Status.Phase = corev1.PodRunning
	assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: pod}))

	t.Log("Verify native GMS replacement does not admit the Pod before its sidecar restarts")
	assert.False(t, gmsPodReplacementPredicate().Create(event.CreateEvent{Object: pod}))

	t.Log("Reconcile defensively and verify neither the trigger nor sibling is cascade-deleted")
	r, c := newCascadeReconciler(pod, sibling)
	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKeyFromObject(pod),
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
	require.NoError(t, c.Get(context.Background(), client.ObjectKeyFromObject(pod), &corev1.Pod{}))
	require.NoError(t, c.Get(context.Background(), client.ObjectKeyFromObject(sibling), &corev1.Pod{}))

	t.Log("Verify native GMS replacement independently admits the same Pod after its sidecar restarts")
	pod.Status.InitContainerStatuses[0].RestartCount = 1
	assert.Equal(t, commonconsts.KubeLabelValueTrue, pod.Labels[snapshotprotocol.RestoreTargetLabel])
	assert.True(t, gmsPodReplacementPredicate().Create(event.CreateEvent{Object: pod}))
}
