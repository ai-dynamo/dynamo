/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pods

import (
	"sort"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestOwnerNames(t *testing.T) {
	tests := []struct {
		name      string
		ownerName OwnerName
		labels    map[string]string
		want      string
		owned     bool
	}{
		{name: "DCD by selector label", ownerName: DCDName, labels: map[string]string{commonconsts.KubeLabelDynamoSelector: "graph-frontend"}, want: "graph-frontend", owned: true},
		{name: "no selector label", ownerName: DCDName},
		{name: "DGD by graph and component labels", ownerName: DGDName, labels: componentLabels("graph", "frontend"), want: "graph", owned: true},
		{name: "graph label without a component", ownerName: DGDName, labels: map[string]string{commonconsts.KubeLabelDynamoGraphDeploymentName: "graph"}, want: "graph"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name, owned := tt.ownerName(testPod("inference", "pod", tt.labels, 0))
			assert.Equal(t, tt.owned, owned)
			if owned {
				assert.Equal(t, tt.want, name)
			}
		})
	}
}

func TestEnqueueOwnersEnqueuesPreviousOwnerOnUpdate(t *testing.T) {
	t.Log("Relabel a crash-looping Pod from one graph to another")
	oldPod := testPod("inference", "worker-a", componentLabels("old-graph", "worker"), 3)
	newPod := testPod("inference", "worker-a", componentLabels("new-graph", "worker"), 3)
	queue := workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[reconcile.Request]())
	defer queue.ShutDown()

	t.Log("Deliver the update to the handler")
	EnqueueOwners(DGDName).Update(t.Context(), event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}, queue)

	t.Log("Verify both graphs are reconciled")
	var names []string
	for queue.Len() > 0 {
		request, _ := queue.Get()
		assert.Equal(t, "inference", request.Namespace)
		names = append(names, request.Name)
		queue.Done(request)
	}
	sort.Strings(names)
	assert.Equal(t, []string{"new-graph", "old-graph"}, names)
}

func TestEnqueueOwnersSkipsUnownedPods(t *testing.T) {
	queue := workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[reconcile.Request]())
	defer queue.ShutDown()

	EnqueueOwners(DGDName).Create(t.Context(), event.CreateEvent{Object: testPod("inference", "unlabeled", nil, 3)}, queue)

	assert.Zero(t, queue.Len())
}

func TestCrashLoopPredicate(t *testing.T) {
	owned := componentLabels("graph", "worker")
	recovered := testPod("inference", "worker-a", owned, 2)
	recovered.Status.ContainerStatuses[0].State.Waiting = nil

	t.Run("create", func(t *testing.T) {
		for _, tt := range []struct {
			name string
			pod  testPodSpec
			want bool
		}{
			{name: "crash-looping owned Pod", pod: testPodSpec{labels: owned, restarts: 3}, want: true},
			{name: "healthy owned Pod", pod: testPodSpec{labels: owned}},
			{name: "crash-looping unowned Pod", pod: testPodSpec{restarts: 3}},
		} {
			t.Run(tt.name, func(t *testing.T) {
				pod := testPod("inference", "worker-a", tt.pod.labels, tt.pod.restarts)
				assert.Equal(t, tt.want, CrashLoopPredicate(DGDName).Create(event.CreateEvent{Object: pod}))
			})
		}
	})

	t.Run("delete", func(t *testing.T) {
		assert.True(t, CrashLoopPredicate(DGDName).Delete(event.DeleteEvent{Object: testPod("inference", "worker-a", owned, 0)}))
		assert.False(t, CrashLoopPredicate(DGDName).Delete(event.DeleteEvent{Object: testPod("inference", "worker-a", nil, 3)}))
	})

	t.Run("update", func(t *testing.T) {
		for _, tt := range []struct {
			name   string
			oldPod testPodSpec
			newPod testPodSpec
			want   bool
		}{
			{name: "diagnosis changes", oldPod: testPodSpec{labels: owned, restarts: 1}, newPod: testPodSpec{labels: owned, restarts: 2}, want: true},
			{name: "diagnosis unchanged", oldPod: testPodSpec{labels: owned, restarts: 2}, newPod: testPodSpec{labels: owned, restarts: 2}},
			{name: "healthy Pod moves to another graph", oldPod: testPodSpec{labels: owned}, newPod: testPodSpec{labels: componentLabels("other", "worker")}, want: true},
			{name: "Pod leaves its graph", oldPod: testPodSpec{labels: owned, restarts: 2}, newPod: testPodSpec{restarts: 2}, want: true},
			{name: "unowned Pod crash-loops", oldPod: testPodSpec{restarts: 1}, newPod: testPodSpec{restarts: 2}},
		} {
			t.Run(tt.name, func(t *testing.T) {
				oldPod := testPod("inference", "worker-a", tt.oldPod.labels, tt.oldPod.restarts)
				newPod := testPod("inference", "worker-a", tt.newPod.labels, tt.newPod.restarts)
				assert.Equal(t, tt.want, CrashLoopPredicate(DGDName).Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}))
			})
		}
		assert.True(t, CrashLoopPredicate(DGDName).Update(event.UpdateEvent{
			ObjectOld: testPod("inference", "worker-a", owned, 2),
			ObjectNew: recovered,
		}), "recovery delegates to CrashLoopChanged")
	})
}

// testPodSpec selects the labels and restart count of a testPod.
type testPodSpec struct {
	labels   map[string]string
	restarts int32
}
