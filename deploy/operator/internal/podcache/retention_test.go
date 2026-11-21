/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package podcache

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	toolscache "k8s.io/client-go/tools/cache"
	"k8s.io/utils/ptr"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestPodRetentionWatchMembership(t *testing.T) {
	t.Log("Watch an unrelated Pod enter, leave, and reenter scope before the indexer catches up")
	source := watch.NewRaceFreeFake()
	lw := &retainedPodListWatch{source: &toolscache.ListWatch{
		WatchFuncWithContext: func(context.Context, metav1.ListOptions) (watch.Interface, error) { return source, nil },
	}}
	stream, err := lw.WatchWithContext(context.Background(), metav1.ListOptions{})
	require.NoError(t, err)
	defer stream.Stop()
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "worker", ResourceVersion: "1"}}
	source.Add(pod.DeepCopy())
	event := nextRetentionEvent(t, stream)
	assert.Equal(t, watch.Bookmark, event.Type)
	assert.Equal(t, "1", event.Object.(*corev1.Pod).ResourceVersion)
	assert.Empty(t, event.Object.(*corev1.Pod).Annotations)

	t.Log("Admit a label transition as an Add and retain only its projected identity")
	pod.Labels = map[string]string{consts.KubeLabelDynamoSelector: "worker"}
	pod.Spec.ServiceAccountName = "discard"
	pod.ResourceVersion = "2"
	source.Modify(pod.DeepCopy())
	assert.Equal(t, watch.Added, nextRetentionEvent(t, stream).Type)
	require.Eventually(t, func() bool {
		lw.mu.Lock()
		defer lw.mu.Unlock()
		return len(lw.admitted) == 1
	}, time.Second, time.Millisecond)
	lw.mu.Lock()
	assert.Len(t, lw.admitted, 1)
	assert.Empty(t, lw.admitted["ns/worker"].Spec.ServiceAccountName)
	lw.mu.Unlock()

	t.Log("Leaving scope deletes the last admitted identity using the new resource version")
	pod.Labels = nil
	pod.ResourceVersion = "3"
	source.Modify(pod.DeepCopy())
	event = nextRetentionEvent(t, stream)
	assert.Equal(t, watch.Deleted, event.Type)
	deleted := event.Object.(*corev1.Pod)
	assert.Equal(t, "worker", deleted.Labels[consts.KubeLabelDynamoSelector])
	assert.Equal(t, "3", deleted.ResourceVersion)

	t.Log("A later unrelated deletion does not create a tombstone")
	source.Delete(pod.DeepCopy())
	assert.Equal(t, watch.Bookmark, nextRetentionEvent(t, stream).Type)
	lw.mu.Lock()
	assert.Empty(t, lw.admitted)
	lw.mu.Unlock()
}

func TestPodRetentionListPagesAndConsistency(t *testing.T) {
	t.Log("An empty filtered first page must preserve pagination and collection resource version")
	lw := &retainedPodListWatch{source: &toolscache.ListWatch{
		ListWithContextFunc: func(_ context.Context, options metav1.ListOptions) (runtime.Object, error) {
			if options.Continue == "" {
				return &corev1.PodList{ListMeta: metav1.ListMeta{ResourceVersion: "8", Continue: "next"}, Items: []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "unrelated"}}}}, nil
			}
			return &corev1.PodList{ListMeta: metav1.ListMeta{ResourceVersion: "8"}, Items: []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "worker", Labels: map[string]string{consts.KubeLabelDynamoSelector: "worker"}}}}}, nil
		},
	}}
	obj, err := lw.List(metav1.ListOptions{})
	require.NoError(t, err)
	assert.Empty(t, obj.(*corev1.PodList).Items)
	assert.Equal(t, "next", obj.(*corev1.PodList).Continue)
	assert.Equal(t, "8", obj.(*corev1.PodList).ResourceVersion)
	assert.Empty(t, lw.admitted)

	t.Log("Publish membership only after all pages, replacing obsolete entries")
	lw.admitted = map[string]*corev1.Pod{"/stale": {ObjectMeta: metav1.ObjectMeta{Name: "stale"}}}
	_, err = lw.List(metav1.ListOptions{Continue: "next"})
	require.NoError(t, err)
	assert.Contains(t, lw.admitted, "/worker")
	assert.NotContains(t, lw.admitted, "/stale")

	t.Log("A watch-list consistency check cannot overwrite active membership")
	lw.active = 1
	lw.admitted = map[string]*corev1.Pod{"/active": {ObjectMeta: metav1.ObjectMeta{Name: "active"}}}
	_, err = lw.List(metav1.ListOptions{})
	require.NoError(t, err)
	_, err = lw.List(metav1.ListOptions{Continue: "next"})
	require.NoError(t, err)
	assert.Contains(t, lw.admitted, "/active")
	assert.NotContains(t, lw.admitted, "/worker")
}

func TestPodRetentionWatchListAndCancellation(t *testing.T) {
	t.Log("Each initial-event stream starts a fresh membership snapshot")
	source := watch.NewRaceFreeFake()
	lw := &retainedPodListWatch{
		source:   &toolscache.ListWatch{WatchFuncWithContext: func(context.Context, metav1.ListOptions) (watch.Interface, error) { return source, nil }},
		admitted: map[string]*corev1.Pod{"/stale": {ObjectMeta: metav1.ObjectMeta{Name: "stale"}}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := lw.WatchWithContext(ctx, metav1.ListOptions{SendInitialEvents: ptr.To(true)})
	require.NoError(t, err)
	defer stream.Stop()
	initialEnd := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "10", Annotations: map[string]string{metav1.InitialEventsAnnotationKey: "true"}}}
	source.Add(&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "unrelated", ResourceVersion: "2"}})
	source.Action(watch.Bookmark, initialEnd)
	event := nextRetentionEvent(t, stream)
	assert.Equal(t, watch.Bookmark, event.Type)
	assert.Same(t, initialEnd, event.Object)
	lw.mu.Lock()
	assert.Empty(t, lw.admitted)
	lw.mu.Unlock()

	t.Log("Cancellation closes a watch even while the downstream consumer is idle")
	source.Add(&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "unrelated", ResourceVersion: "11"}})
	cancel()
	select {
	case <-stream.(*retainedPodWatch).done:
	case <-time.After(2 * time.Second):
		t.Fatal("watch did not stop after cancellation")
	}
}

func nextRetentionEvent(t *testing.T, stream watch.Interface) watch.Event {
	t.Helper()
	select {
	case event, open := <-stream.ResultChan():
		require.True(t, open)
		return event
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for watch event")
		return watch.Event{}
	}
}

func TestPodRetentionInterest(t *testing.T) {
	t.Log("Retain the union of existing Pod contracts without requiring newly stamped labels")
	cases := []struct {
		name   string
		labels map[string]string
		owner  *metav1.OwnerReference
		want   bool
	}{
		{name: "unrelated"},
		{name: "DGD diagnostic", labels: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "graph"}, want: true},
		{name: "standalone DCD", labels: map[string]string{consts.KubeLabelDynamoSelector: "worker"}, want: true},
		{name: "GMS standalone", labels: map[string]string{consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoNamespace: "graph"}, want: true},
		{name: "component alone", labels: map[string]string{consts.KubeLabelDynamoComponent: "worker"}},
		{name: "failover", labels: map[string]string{consts.KubeLabelDynamoFailoverEngineGroupMember: consts.KubeLabelValueTrue}, want: true},
		{name: "empty marker", labels: map[string]string{consts.KubeLabelDynamoSelector: ""}},
		{name: "profiler job", owner: &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "profile-graph", Controller: ptr.To(true)}, want: true},
		{name: "run job", owner: &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "run-aisimulate", Controller: ptr.To(true)}, want: true},
		{name: "unrelated job", owner: &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "other", Controller: ptr.To(true)}},
		{name: "noncontroller job", owner: &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "profile-graph"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: tc.labels}}
			if tc.owner != nil {
				pod.OwnerReferences = []metav1.OwnerReference{*tc.owner}
			}
			assert.Equal(t, tc.want, interestedPod(pod))
		})
	}
}

func TestPodRetentionInformerStoreAndIndex(t *testing.T) {
	t.Log("Populate an informer with one indexed Dynamo Pod and many unrelated Pods")
	items := make([]corev1.Pod, 100)
	for i := range items {
		items[i].Name = "unrelated-" + strconv.Itoa(i)
	}
	items = append(items, corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker", Labels: map[string]string{consts.KubeLabelDynamoSelector: "worker"}}})
	source := watch.NewRaceFreeFake()
	lw := &retainedPodListWatch{
		unsupportedWatchList: true,
		source: &toolscache.ListWatch{
			ListWithContextFunc: func(context.Context, metav1.ListOptions) (runtime.Object, error) {
				return &corev1.PodList{ListMeta: metav1.ListMeta{ResourceVersion: "1"}, Items: items}, nil
			},
			WatchFuncWithContext: func(context.Context, metav1.ListOptions) (watch.Interface, error) { return source, nil },
		},
	}
	informer := toolscache.NewSharedIndexInformer(lw, &corev1.Pod{}, 0, toolscache.Indexers{
		"selector": func(obj any) ([]string, error) {
			return []string{obj.(*corev1.Pod).Labels[consts.KubeLabelDynamoSelector]}, nil
		},
	})
	require.NoError(t, informer.SetTransform(Transform))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	done := make(chan struct{})
	go func() { defer close(done); informer.RunWithContext(ctx) }()
	t.Cleanup(func() { cancel(); <-done })
	require.True(t, toolscache.WaitForCacheSync(ctx.Done(), informer.HasSynced))
	assert.Len(t, informer.GetStore().List(), 1)
	indexed, err := informer.GetIndexer().ByIndex("selector", "worker")
	require.NoError(t, err)
	assert.Len(t, indexed, 1)

	t.Log("Admit a relabeled Pod and remove it from both store and index when it leaves scope")
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new-worker", ResourceVersion: "2", Labels: map[string]string{consts.KubeLabelDynamoSelector: "worker"}}}
	source.Modify(pod.DeepCopy())
	require.Eventually(t, func() bool { return len(informer.GetStore().List()) == 2 }, 2*time.Second, time.Millisecond)
	pod.Labels = nil
	pod.ResourceVersion = "3"
	source.Modify(pod.DeepCopy())
	require.Eventually(t, func() bool { return len(informer.GetStore().List()) == 1 }, 2*time.Second, time.Millisecond)
	indexed, err = informer.GetIndexer().ByIndex("selector", "worker")
	require.NoError(t, err)
	assert.Len(t, indexed, 1)
}

func TestPodRetentionConsistencyListWhileConsumerPaused(t *testing.T) {
	t.Log("Pause watch consumption after initial sync while the API sends another event")
	source := watch.NewRaceFreeFake()
	lw := &retainedPodListWatch{source: &toolscache.ListWatch{
		ListWithContextFunc: func(context.Context, metav1.ListOptions) (runtime.Object, error) {
			return &corev1.PodList{}, nil
		},
		WatchFuncWithContext: func(context.Context, metav1.ListOptions) (watch.Interface, error) { return source, nil },
	}}
	stream, err := lw.WatchWithContext(context.Background(), metav1.ListOptions{SendInitialEvents: ptr.To(true)})
	require.NoError(t, err)
	defer stream.Stop()
	source.Action(watch.Bookmark, &corev1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "10", Annotations: map[string]string{metav1.InitialEventsAnnotationKey: "true"}}})
	require.Equal(t, watch.Bookmark, nextRetentionEvent(t, stream).Type)
	source.Add(&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker", Labels: map[string]string{consts.KubeLabelDynamoSelector: "worker"}}})
	require.Eventually(t, func() bool { return len(source.ResultChan()) == 0 }, time.Second, time.Millisecond)

	t.Log("Consistency LIST must complete without waiting for watch consumption")
	listed := make(chan error, 1)
	go func() {
		_, listErr := lw.ListWithContext(context.Background(), metav1.ListOptions{})
		listed <- listErr
	}()
	select {
	case err := <-listed:
		require.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("consistency LIST blocked behind watch delivery")
	}
	assert.Equal(t, watch.Added, nextRetentionEvent(t, stream).Type)
}

func TestPodRetentionDiscardsIncompleteInitialStream(t *testing.T) {
	t.Log("An interrupted initial watch-list must discard its partial membership")
	source := watch.NewRaceFreeFake()
	lw := &retainedPodListWatch{source: &toolscache.ListWatch{
		WatchFuncWithContext: func(context.Context, metav1.ListOptions) (watch.Interface, error) { return source, nil },
	}}
	stream, err := lw.WatchWithContext(context.Background(), metav1.ListOptions{SendInitialEvents: ptr.To(true)})
	require.NoError(t, err)
	defer stream.Stop()
	source.Add(&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker", Labels: map[string]string{consts.KubeLabelDynamoSelector: "worker"}}})
	assert.Equal(t, watch.Added, nextRetentionEvent(t, stream).Type)
	stream.Stop()
	assert.Empty(t, lw.admitted)
}
