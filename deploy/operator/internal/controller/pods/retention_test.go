/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pods

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	toolscache "k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestInformerCacheRetainsControllerPodsAndTracksMembership(t *testing.T) {
	t.Log("Create workload, failover and profiling Pods alongside unrelated Pods")
	ctx := t.Context()
	cfg, writer := startAPIServer(t)
	controller := true
	for _, item := range []struct {
		name   string
		labels map[string]string
		job    string
	}{
		{name: "component", labels: componentLabels("graph", "worker")},
		{name: "standalone", labels: map[string]string{consts.KubeLabelDynamoSelector: "standalone"}},
		{name: "gms", labels: map[string]string{consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoNamespace: "runtime"}},
		{name: "failover", labels: map[string]string{consts.KubeLabelDynamoFailoverEngineGroupMember: consts.KubeLabelValueTrue}},
		{name: "legacy-profiler", job: "profile-request"},
		{name: "run-profiler", job: "request-aisimulate"},
		{name: "unrelated"},
		{name: "unrelated-job", job: "batch-job"},
	} {
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: item.name, Namespace: "default", Labels: item.labels}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example.invalid/worker:test"}}}}
		if item.job != "" {
			pod.OwnerReferences = []metav1.OwnerReference{{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job", Name: item.job, UID: types.UID(item.job), Controller: &controller}}
		}
		require.NoError(t, writer.Create(ctx, pod))
	}

	t.Log("Synchronize the production cache and preserve indexed workload queries")
	podCache := startCache(t, cfg)
	var pods corev1.PodList
	require.NoError(t, podCache.List(ctx, &pods))
	require.Len(t, pods.Items, 6)
	require.NoError(t, podCache.List(ctx, &pods, client.InNamespace("default"), client.MatchingFields{ComponentIndex: "graph/worker"}))
	require.Len(t, pods.Items, 1)
	require.Equal(t, "component", pods.Items[0].Name)
	indexed, err := ComponentCrashLoops(ctx, podCache, "default", "graph", []string{"worker"})
	require.NoError(t, err)
	require.Empty(t, indexed)
	key := client.ObjectKey{Namespace: "default", Name: "unrelated"}
	var unrelated corev1.Pod
	require.True(t, apierrors.IsNotFound(podCache.Get(ctx, key, &unrelated)))

	t.Log("Observe an unrelated Pod entering the controller interest set")
	informer, err := podCache.GetInformer(ctx, &corev1.Pod{})
	require.NoError(t, err)
	deletedGraph := make(chan string, 8)
	_, err = informer.AddEventHandler(toolscache.ResourceEventHandlerFuncs{DeleteFunc: func(obj any) {
		if pod, ok := obj.(*corev1.Pod); ok && pod.Name == "unrelated" {
			deletedGraph <- pod.Labels[consts.KubeLabelDynamoGraphDeploymentName]
		}
	}})
	require.NoError(t, err)
	require.NoError(t, writer.Get(ctx, key, &unrelated))
	before := unrelated.DeepCopy()
	unrelated.Labels = componentLabels("new-graph", "worker")
	require.NoError(t, writer.Patch(ctx, &unrelated, client.MergeFrom(before)))
	require.Eventually(t, func() bool {
		var pod corev1.Pod
		return podCache.Get(ctx, key, &pod) == nil && pod.Labels[consts.KubeLabelDynamoGraphDeploymentName] == "new-graph"
	}, 10*time.Second, 10*time.Millisecond)

	t.Log("Remove its labels and deliver deletion using the previous controller identity")
	before = unrelated.DeepCopy()
	unrelated.Labels = nil
	require.NoError(t, writer.Patch(ctx, &unrelated, client.MergeFrom(before)))
	require.Eventually(t, func() bool { return apierrors.IsNotFound(podCache.Get(ctx, key, &corev1.Pod{})) }, 10*time.Second, 10*time.Millisecond)
	require.NoError(t, podCache.List(ctx, &pods, client.InNamespace("default"), client.MatchingFields{ComponentIndex: "new-graph/worker"}))
	require.Empty(t, pods.Items, "leaving Pods must be removed from the workload index")
	select {
	case graph := <-deletedGraph:
		require.Equal(t, "new-graph", graph)
	case <-time.After(10 * time.Second):
		t.Fatal("membership-removal event was not delivered")
	}

	t.Log("Process unrelated watch traffic without retaining it or stalling later events")
	require.NoError(t, createPod(ctx, writer, "more-noise", nil, 0))
	var component corev1.Pod
	componentKey := client.ObjectKey{Namespace: "default", Name: "component"}
	require.NoError(t, writer.Get(ctx, componentKey, &component))
	before = component.DeepCopy()
	component.Annotations = map[string]string{"test.example.com/observed": "after-noise"}
	require.NoError(t, writer.Patch(ctx, &component, client.MergeFrom(before)))
	require.Eventually(t, func() bool {
		var pod corev1.Pod
		return podCache.Get(ctx, componentKey, &pod) == nil && pod.Annotations["test.example.com/observed"] == "after-noise"
	}, 10*time.Second, 10*time.Millisecond)
	require.NoError(t, podCache.List(ctx, &pods))
	require.Len(t, pods.Items, 6)

	t.Log("Delete an admitted Pod and rebuild from a fresh filtered list")
	require.NoError(t, writer.Delete(ctx, &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "failover", Namespace: "default"}}, client.GracePeriodSeconds(0)))
	require.Eventually(t, func() bool {
		return apierrors.IsNotFound(podCache.Get(ctx, client.ObjectKey{Namespace: "default", Name: "failover"}, &corev1.Pod{}))
	}, 10*time.Second, 10*time.Millisecond)
	rebuilt := startCache(t, cfg)
	require.NoError(t, rebuilt.List(ctx, &pods))
	require.Len(t, pods.Items, 5)
}
