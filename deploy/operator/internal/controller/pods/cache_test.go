/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pods

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
	poddiagnostics "github.com/ai-dynamo/dynamo/deploy/operator/internal/pods"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

// startAPIServer starts an envtest API server for tb, skipping without
// KUBEBUILDER_ASSETS, and returns a client that writes to it directly.
func startAPIServer(tb testing.TB) (*rest.Config, client.Client) {
	tb.Helper()
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		tb.Skip("KUBEBUILDER_ASSETS is not set")
	}
	env := &envtest.Environment{}
	cfg, err := env.Start()
	require.NoError(tb, err)
	tb.Cleanup(func() { require.NoError(tb, env.Stop()) })
	writer, err := client.New(cfg, client.Options{Scheme: scheme.Scheme})
	require.NoError(tb, err)
	return cfg, writer
}

// startCache starts the informer cache the operator's manager uses for Pods,
// with this package's indexes registered, and waits for its initial list.
func startCache(tb testing.TB, cfg *rest.Config) cache.Cache {
	tb.Helper()
	options := cache.Options{Scheme: scheme.Scheme}
	require.NoError(tb, podcache.Configure(&options))
	podCache, err := cache.New(cfg, options)
	require.NoError(tb, err)
	ctx, cancel := context.WithCancel(context.Background())
	tb.Cleanup(cancel)
	require.NoError(tb, podCache.IndexField(ctx, &corev1.Pod{}, ComponentIndex, ComponentIndexValues))
	require.NoError(tb, podCache.IndexField(ctx, &corev1.Pod{}, SelectorIndex, SelectorIndexValues))
	go func() { _ = podCache.Start(ctx) }()
	require.True(tb, podCache.WaitForCacheSync(ctx))
	return podCache
}

// createPod creates a Pod in the default namespace with labels and, when restarts
// is positive, reports its main container as crash-looping through the status subresource.
func createPod(ctx context.Context, writer client.Client, name string, labels map[string]string, restarts int32) error {
	pod := &corev1.Pod{
		Name:      name,
		Namespace: "default",
		Labels:    labels,
		Spec:      corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "busybox:1.36"}}},
	}
	if err := writer.Create(ctx, pod); err != nil {
		return err
	}
	if restarts == 0 {
		return nil
	}
	pod.Status = testPod("default", name, labels, restarts).Status
	return writer.Status().Update(ctx, pod)
}

func TestInformerCacheDiagnosesExistingCrashLoopAfterRestart(t *testing.T) {
	ctx := t.Context()
	cfg, writer := startAPIServer(t)

	t.Log("Create a crash-looping Pod before any cache exists to deliver an event for it")
	require.NoError(t, createPod(ctx, writer, "graph-frontend-a", componentLabels("graph", "frontend"), 3))
	want := map[string]poddiagnostics.CrashLoop{
		"frontend": {Pod: "graph-frontend-a", Container: "main", RestartCount: 3, ExitCode: 1, Reason: "Error"},
	}

	t.Log("Diagnose the component from a freshly synchronized cache")
	got, err := ComponentCrashLoops(ctx, startCache(t, cfg), "default", "graph", []string{"frontend"})
	require.NoError(t, err)
	assert.Equal(t, want, got)

	t.Log("Rebuild the cache as after a controller restart and diagnose again")
	got, err = ComponentCrashLoops(ctx, startCache(t, cfg), "default", "graph", []string{"frontend"})
	require.NoError(t, err)
	assert.Equal(t, want, got)
}

func TestInformerCacheConvergesAfterCoalescedUpdates(t *testing.T) {
	ctx := t.Context()
	cfg, writer := startAPIServer(t)
	podCache := startCache(t, cfg)
	labels := componentLabels("graph", "worker")
	require.NoError(t, createPod(ctx, writer, "graph-worker-a", labels, 0))

	t.Log("Write a crash loop, deeper restarts and recovery back to back")
	pod := &corev1.Pod{}
	require.NoError(t, writer.Get(ctx, client.ObjectKey{Namespace: "default", Name: "graph-worker-a"}, pod))
	for _, restarts := range []int32{2, 3, 4} {
		pod.Status = testPod("default", pod.Name, labels, restarts).Status
		require.NoError(t, writer.Status().Update(ctx, pod))
	}
	pod.Status.ContainerStatuses[0].State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	require.NoError(t, writer.Status().Update(ctx, pod))

	t.Log("Verify the cached diagnosis converges to the recovered state")
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		got, err := ComponentCrashLoops(ctx, podCache, "default", "graph", []string{"worker"})
		require.NoError(c, err)
		assert.Empty(c, got)
	}, 10*time.Second, 50*time.Millisecond)

	t.Log("Crash again and verify the cached diagnosis reports the latest restart count")
	pod.Status = testPod("default", pod.Name, labels, 5).Status
	require.NoError(t, writer.Status().Update(ctx, pod))
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		got, err := ComponentCrashLoops(ctx, podCache, "default", "graph", []string{"worker"})
		require.NoError(c, err)
		assert.Equal(c, int32(5), got["worker"].RestartCount)
	}, 10*time.Second, 50*time.Millisecond)
}

// BenchmarkCachedCrashLoopLookup measures one component's crash-loop lookup
// through the informer cache as unrelated Pods in the same namespace grow. The
// label-selector variant is the unindexed baseline.
func BenchmarkCachedCrashLoopLookup(b *testing.B) {
	ctx := context.Background()
	cfg, writer := startAPIServer(b)

	b.Log("Create the affected component: four Pods selected by one Deployment, one crash-looping")
	affected := map[string]string{
		commonconsts.KubeLabelDynamoGraphDeploymentName: "graph",
		commonconsts.KubeLabelDynamoComponent:           "frontend",
		commonconsts.KubeLabelDynamoSelector:            "graph-frontend",
	}
	for i := range 4 {
		require.NoError(b, createPod(ctx, writer, fmt.Sprintf("graph-frontend-%d", i), affected, int32(3*(i/3))))
	}
	deployment := &appsv1.Deployment{
		Name:      "graph-frontend",
		Namespace: "default",
		Spec: appsv1.DeploymentSpec{Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{commonconsts.KubeLabelDynamoSelector: "graph-frontend"},
		}},
	}

	created := 0
	for _, unrelated := range []int{1000, 5000, 10000} {
		b.Logf("Grow to %d unrelated Pods spread over %d components of other graphs", unrelated, unrelated/10)
		createUnrelatedPods(b, ctx, writer, created, unrelated)
		created = unrelated
		podCache := startCache(b, cfg)

		b.Run(fmt.Sprintf("component-index/unrelated=%d", unrelated), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				crashLoops, err := ComponentCrashLoops(ctx, podCache, "default", "graph", []string{"frontend"})
				if err != nil || len(crashLoops) != 1 {
					b.Fatalf("ComponentCrashLoops = %v, %v", crashLoops, err)
				}
			}
		})
		b.Run(fmt.Sprintf("deployment-index/unrelated=%d", unrelated), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, found, err := FindDeploymentCrashLoop(ctx, podCache, deployment); err != nil || !found {
					b.Fatalf("FindDeploymentCrashLoop = %v, %v", found, err)
				}
			}
		})
		b.Run(fmt.Sprintf("label-selector-scan/unrelated=%d", unrelated), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				podList := &corev1.PodList{}
				if err := podCache.List(ctx, podList, client.InNamespace("default"), client.MatchingLabels{
					commonconsts.KubeLabelDynamoSelector: "graph-frontend",
				}); err != nil || len(podList.Items) != 4 {
					b.Fatalf("List = %d Pods, %v", len(podList.Items), err)
				}
			}
		})
	}
}

// createUnrelatedPods creates Pods from index from up to to, ten per component
// of other graphs, all labeled like the affected component's Pods.
func createUnrelatedPods(tb testing.TB, ctx context.Context, writer client.Client, from, to int) {
	tb.Helper()
	var wg sync.WaitGroup
	errs := make(chan error, to-from)
	work := make(chan int)
	for range 32 {
		wg.Go(func() {
			for i := range work {
				graph := fmt.Sprintf("other-%d", i/100)
				component := fmt.Sprintf("component-%d", i/10)
				labels := map[string]string{
					commonconsts.KubeLabelDynamoGraphDeploymentName: graph,
					commonconsts.KubeLabelDynamoComponent:           component,
					commonconsts.KubeLabelDynamoSelector:            graph + "-" + component,
				}
				if err := createPod(ctx, writer, fmt.Sprintf("unrelated-%d", i), labels, 0); err != nil {
					errs <- err
				}
			}
		})
	}
	for i := from; i < to; i++ {
		work <- i
	}
	close(work)
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(tb, err)
	}
}
