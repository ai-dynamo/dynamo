/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pods

import (
	"sort"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
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
		{
			name:      "DCD by selector label",
			ownerName: DCDName,
			labels:    map[string]string{commonconsts.KubeLabelDynamoSelector: "graph-frontend"},
			want:      "graph-frontend",
			owned:     true,
		},
		{
			name:      "no selector label",
			ownerName: DCDName,
		},
		{
			name:      "Grove Pod with the selector label",
			ownerName: DCDName,
			labels: map[string]string{
				commonconsts.KubeLabelDynamoSelector: "graph-frontend",
				grovecommon.LabelPodClique:           "graph-0-frontend",
			},
		},
		{
			name:      "DGD by graph and component labels",
			ownerName: DGDName,
			labels:    componentLabels("graph", "frontend"),
			want:      "graph",
			owned:     true,
		},
		{
			name:      "graph label without a component",
			ownerName: DGDName,
			labels:    map[string]string{commonconsts.KubeLabelDynamoGraphDeploymentName: "graph"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Resolve the owner of a Pod with the case's labels")
			name, owned := tt.ownerName(testPod("inference", "pod", tt.labels, 0))

			t.Log("Verify the owner and whether the Pod has one")
			assert.Equal(t, tt.owned, owned)
			if owned {
				assert.Equal(t, tt.want, name)
			}
		})
	}
}

func TestEnqueueOwners(t *testing.T) {
	tests := []struct {
		name   string
		oldPod *corev1.Pod // nil delivers a create of newPod
		newPod *corev1.Pod
		want   []string
	}{
		{
			name:   "update moving a Pod between graphs",
			oldPod: testPod("inference", "worker-a", componentLabels("old-graph", "worker"), 3),
			newPod: testPod("inference", "worker-a", componentLabels("new-graph", "worker"), 3),
			want:   []string{"new-graph", "old-graph"},
		},
		{
			name:   "create of an owned Pod",
			newPod: testPod("inference", "worker-a", componentLabels("graph", "worker"), 3),
			want:   []string{"graph"},
		},
		{
			name:   "create of an unowned Pod",
			newPod: testPod("inference", "unlabeled", nil, 3),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Deliver the event to the handler")
			queue := workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[reconcile.Request]())
			defer queue.ShutDown()
			if tt.oldPod == nil {
				EnqueueOwners(DGDName).Create(t.Context(), event.CreateEvent{Object: tt.newPod}, queue)
			} else {
				EnqueueOwners(DGDName).Update(t.Context(), event.UpdateEvent{ObjectOld: tt.oldPod, ObjectNew: tt.newPod}, queue)
			}

			t.Log("Verify the enqueued graphs in the Pod's namespace")
			var names []string
			for queue.Len() > 0 {
				request, _ := queue.Get()
				assert.Equal(t, "inference", request.Namespace)
				names = append(names, request.Name)
				queue.Done(request)
			}
			sort.Strings(names)
			assert.Equal(t, tt.want, names)
		})
	}
}

func TestCrashLoopPredicate(t *testing.T) {
	owned := componentLabels("graph", "worker")
	recovered := testPod("inference", "worker-a", owned, 2)
	recovered.Status.ContainerStatuses[0].State.Waiting = nil

	t.Run("create and delete", func(t *testing.T) {
		for _, tt := range []struct {
			name     string
			labels   map[string]string
			restarts int32
			want     bool
		}{
			{name: "crash-looping owned Pod", labels: owned, restarts: 3, want: true},
			{name: "healthy owned Pod", labels: owned},
			{name: "crash-looping unowned Pod", restarts: 3},
		} {
			t.Run(tt.name, func(t *testing.T) {
				t.Log("Deliver a create and a delete of the Pod")
				pod := testPod("inference", "worker-a", tt.labels, tt.restarts)
				created := CrashLoopPredicate(DGDName).Create(event.CreateEvent{Object: pod})
				deleted := CrashLoopPredicate(DGDName).Delete(event.DeleteEvent{Object: pod})

				t.Log("Verify only an owned crash-looping Pod can change a diagnosis")
				assert.Equal(t, tt.want, created)
				assert.Equal(t, tt.want, deleted)
			})
		}
	})

	t.Run("update", func(t *testing.T) {
		for _, tt := range []struct {
			name                   string
			oldLabels, newLabels   map[string]string
			oldRestart, newRestart int32
			want                   bool
		}{
			{name: "diagnosis changes", oldLabels: owned, newLabels: owned, oldRestart: 1, newRestart: 2, want: true},
			{name: "diagnosis unchanged", oldLabels: owned, newLabels: owned, oldRestart: 2, newRestart: 2},
			{name: "Pod moves to another graph", oldLabels: owned, newLabels: componentLabels("other", "worker"), want: true},
			{name: "Pod leaves its graph", oldLabels: owned, oldRestart: 2, newRestart: 2, want: true},
			{name: "unowned Pod relabeled", oldLabels: map[string]string{commonconsts.KubeLabelDynamoGraphDeploymentName: "graph"}, newLabels: map[string]string{commonconsts.KubeLabelDynamoGraphDeploymentName: "other"}, oldRestart: 2, newRestart: 2},
			{name: "unowned Pod crash-loops", oldRestart: 1, newRestart: 2},
		} {
			t.Run(tt.name, func(t *testing.T) {
				t.Log("Deliver an update between the two Pod states")
				oldPod := testPod("inference", "worker-a", tt.oldLabels, tt.oldRestart)
				newPod := testPod("inference", "worker-a", tt.newLabels, tt.newRestart)
				got := CrashLoopPredicate(DGDName).Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod})

				t.Log("Verify whether an owner needs reconciling")
				assert.Equal(t, tt.want, got)
			})
		}
	})

	t.Run("recovery delegates to the diagnosis comparison", func(t *testing.T) {
		t.Log("Deliver an update from a crash loop to a running container")
		got := CrashLoopPredicate(DGDName).Update(event.UpdateEvent{ObjectOld: testPod("inference", "worker-a", owned, 2), ObjectNew: recovered})

		t.Log("Verify the owner is reconciled")
		assert.True(t, got)
	})
}
