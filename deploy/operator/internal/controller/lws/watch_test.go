/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"sort"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

func TestPodObservationPredicate(t *testing.T) {
	f := newFixture()
	pred := PodObservationPredicate()

	t.Run("create admits first current nonzero exit at restartCount0", func(t *testing.T) {
		t.Log("Deliver a create of a failed LWS Pod")
		assert.True(t, pred.Create(event.CreateEvent{Object: f.leaderPod}))
	})

	t.Run("create admits healthy set-name-hinted Pod as late dependency", func(t *testing.T) {
		pod := f.leaderPod.DeepCopy()
		healthyMain(pod, f.readyLTT)
		t.Log("Deliver a create of a healthy leader that can unblock a failed follower")
		assert.True(t, pred.Create(event.CreateEvent{Object: pod}))
	})

	t.Run("create ignores unlabeled Pods", func(t *testing.T) {
		pod := f.leaderPod.DeepCopy()
		delete(pod.Labels, leaderworkersetv1.SetNameLabelKey)
		assert.False(t, pred.Create(event.CreateEvent{Object: pod}))
	})

	t.Run("update admits FinishedAt Ready owner label UID and deletion changes", func(t *testing.T) {
		cases := []struct {
			name   string
			mutate func(*corev1.Pod)
			want   bool
		}{
			{
				name: "FinishedAt change",
				mutate: func(pod *corev1.Pod) {
					pod.Status.ContainerStatuses[0].State.Terminated.FinishedAt = metav1.NewTime(f.now)
				},
				want: true,
			},
			{
				name: "Ready recovery",
				mutate: func(pod *corev1.Pod) {
					healthyMain(pod, f.readyLTT)
				},
				want: true,
			},
			{
				name: "owner change",
				mutate: func(pod *corev1.Pod) {
					pod.OwnerReferences[0].UID = "other"
				},
				want: true,
			},
			{
				name: "revision label change",
				mutate: func(pod *corev1.Pod) {
					pod.Labels[leaderworkersetv1.RevisionKey] = "other"
				},
				want: true,
			},
			{
				name: "UID change",
				mutate: func(pod *corev1.Pod) {
					pod.UID = "replaced"
				},
				want: true,
			},
			{
				name: "deletion",
				mutate: func(pod *corev1.Pod) {
					pod.DeletionTimestamp = &f.finished
					pod.Finalizers = []string{"test.nvidia.com/hold"}
				},
				want: true,
			},
			{
				name: "unrelated resourceVersion only",
				mutate: func(pod *corev1.Pod) {
					pod.ResourceVersion = "999"
				},
			},
		}
		for _, tt := range cases {
			t.Run(tt.name, func(t *testing.T) {
				oldPod := f.leaderPod.DeepCopy()
				newPod := f.leaderPod.DeepCopy()
				tt.mutate(newPod)
				assert.Equal(t, tt.want, pred.Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}))
			})
		}
	})

	t.Run("delete admits labeled Pods", func(t *testing.T) {
		assert.True(t, pred.Delete(event.DeleteEvent{Object: f.leaderPod}))
	})
}

func TestStatefulSetObservationPredicate(t *testing.T) {
	f := newFixture()
	pred := StatefulSetObservationPredicate()

	t.Run("create and delete admit LWS-owned or labeled STS", func(t *testing.T) {
		assert.True(t, pred.Create(event.CreateEvent{Object: f.leaderSTS}))
		assert.True(t, pred.Delete(event.DeleteEvent{Object: f.workerSTS}))
	})

	t.Run("update admits template owner identity and excludes unrelated RV", func(t *testing.T) {
		oldSTS := f.leaderSTS.DeepCopy()
		newSTS := f.leaderSTS.DeepCopy()
		newSTS.ResourceVersion = "999"
		assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: oldSTS, ObjectNew: newSTS}))

		newSTS = f.leaderSTS.DeepCopy()
		newSTS.Spec.Template.Labels[leaderworkersetv1.RevisionKey] = "other"
		assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: oldSTS, ObjectNew: newSTS}))
	})

	t.Run("unrelated StatefulSet is excluded", func(t *testing.T) {
		sts := &appsv1.StatefulSet{ObjectMeta: metav1.ObjectMeta{Name: "unrelated", Namespace: testNamespace}}
		assert.False(t, pred.Create(event.CreateEvent{Object: sts}))
	})
}

func TestEnqueueDCDForPodAndStatefulSet(t *testing.T) {
	f := newFixture()
	reader := f.reader(f.dcd, f.lws, f.leaderSTS, f.leaderPod)

	t.Run("Pod maps through set-name hint to verified DCD-owned LWS", func(t *testing.T) {
		t.Log("Deliver create and verify the DCD is enqueued")
		assert.Equal(t, []string{"decode"}, enqueueNames(t, EnqueueDCDForPod(reader), nil, f.leaderPod))
	})

	t.Run("Pod update maps old and new sides", func(t *testing.T) {
		oldPod := f.leaderPod.DeepCopy()
		oldPod.Labels[leaderworkersetv1.SetNameLabelKey] = "other"
		other := f.lws.DeepCopy()
		other.Name = "other"
		other.UID = "other-lws"
		other.OwnerReferences = []metav1.OwnerReference{{
			APIVersion: "nvidia.com/v1beta1",
			Kind:       "DynamoComponentDeployment",
			Name:       "other-dcd",
			UID:        "other-dcd",
			Controller: ptr.To(true),
		}}
		mapped := f.reader(f.dcd, f.lws, other)

		t.Log("Deliver an update that moves the Pod between LWS names")
		assert.Equal(t, []string{"decode", "other-dcd"}, enqueueNames(t, EnqueueDCDForPod(mapped), oldPod, f.leaderPod))
	})

	t.Run("StatefulSet create maps a previously missing dependency", func(t *testing.T) {
		t.Log("Deliver a labeled STS create")
		assert.Equal(t, []string{"decode"}, enqueueNames(t, EnqueueDCDForStatefulSet(reader), nil, f.leaderSTS))
	})

	t.Run("unrelated Pod is excluded", func(t *testing.T) {
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
			Name: "deploy-pod", Namespace: testNamespace,
			Labels: map[string]string{commonconsts.KubeLabelDynamoSelector: "decode"},
		}}
		assert.Empty(t, enqueueNames(t, EnqueueDCDForPod(reader), nil, pod))
	})

	t.Run("healthy leader create still routes through set-name hint", func(t *testing.T) {
		pod := f.leaderPod.DeepCopy()
		healthyMain(pod, f.readyLTT)
		assert.Equal(t, []string{"decode"}, enqueueNames(t, EnqueueDCDForPod(reader), nil, pod))
	})
}

func enqueueNames(t *testing.T, eventHandler handler.EventHandler, oldObj, newObj client.Object) []string {
	t.Helper()
	queue := workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[reconcile.Request]())
	defer queue.ShutDown()

	if oldObj == nil {
		eventHandler.Create(t.Context(), event.CreateEvent{Object: newObj}, queue)
	} else {
		eventHandler.Update(t.Context(), event.UpdateEvent{ObjectOld: oldObj, ObjectNew: newObj}, queue)
	}

	var names []string
	for queue.Len() > 0 {
		req, _ := queue.Get()
		names = append(names, req.Name)
		queue.Done(req)
	}
	sort.Strings(names)
	return names
}
