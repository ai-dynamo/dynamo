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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const cascadeTestNamespace = "test-ns"

func newFailoverPod(name string, phase corev1.PodPhase, pcsg, replicaIdx, podIdx string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cascadeTestNamespace,
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoFailoverEngineGroupMember: commonconsts.KubeLabelValueTrue,
				groveLabelPCSG:             pcsg,
				groveLabelPCSGReplicaIndex: replicaIdx,
				groveLabelPodIndex:         podIdx,
			},
		},
		Status: corev1.PodStatus{Phase: phase},
	}
}

func newCascadeReconciler(objs ...client.Object) (*FailoverCascadeReconciler, client.Client) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	cb := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&corev1.Pod{})
	for _, o := range objs {
		cb = cb.WithObjects(o)
	}
	c := cb.Build()

	return NewFailoverCascadeReconciler(c, record.NewFakeRecorder(16)), c
}

func TestFailoverCascade_FailedPodDeletesEntireGroup(t *testing.T) {

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "my-pcsg", "0", "0")
	sibling1 := newFailoverPod("gms-0-0", corev1.PodRunning, "my-pcsg", "0", "0")
	sibling2 := newFailoverPod("wkr-1-0", corev1.PodRunning, "my-pcsg", "0", "0")

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

	succeededPod := newFailoverPod("ldr-0", corev1.PodSucceeded, "my-pcsg", "0", "0")
	sibling := newFailoverPod("gms-0-0", corev1.PodRunning, "my-pcsg", "0", "0")

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

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "my-pcsg", "0", "0")
	differentGroup := newFailoverPod("ldr-1", corev1.PodRunning, "my-pcsg", "0", "1")

	r, c := newCascadeReconciler(failedPod, differentGroup)

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "ldr-0", Namespace: cascadeTestNamespace},
	})
	require.NoError(t, err)

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(cascadeTestNamespace)))
	assert.Len(t, remaining.Items, 1, "only the different engine group pod should remain")
	assert.Equal(t, "ldr-1", remaining.Items[0].Name)
}

func TestFailoverCascade_MultipleFailedPodsAllDeleted(t *testing.T) {

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "my-pcsg", "0", "0")
	alsoFailed := newFailoverPod("wkr-1-0", corev1.PodFailed, "my-pcsg", "0", "0")
	running := newFailoverPod("gms-0-0", corev1.PodRunning, "my-pcsg", "0", "0")

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

	runningPod := newFailoverPod("ldr-0", corev1.PodRunning, "my-pcsg", "0", "0")
	sibling := newFailoverPod("gms-0-0", corev1.PodRunning, "my-pcsg", "0", "0")

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
				groveLabelPCSG:                            "my-pcsg",
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

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "my-pcsg", "0", "0")
	differentReplica := newFailoverPod("ldr-r1-0", corev1.PodRunning, "my-pcsg", "1", "0")

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

	failedPod := newFailoverPod("ldr-0", corev1.PodFailed, "my-pcsg", "0", "0")
	failedPod.DeletionTimestamp = &now
	failedPod.DeletionGracePeriodSeconds = ptr.To(int64(0))
	failedPod.Finalizers = []string{"test-finalizer"}
	sibling := newFailoverPod("gms-0-0", corev1.PodRunning, "my-pcsg", "0", "0")

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

	pod1 := newFailoverPod("ldr-0", corev1.PodFailed, "my-pcsg", "0", "0")
	pod2 := newFailoverPod("wkr-1-0", corev1.PodFailed, "my-pcsg", "0", "0")

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
