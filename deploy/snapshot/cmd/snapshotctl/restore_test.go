package main

import (
	"context"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func TestWaitForRestoreWatchesUntilCompleted(t *testing.T) {
	t.Parallel()

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "restore-pod",
			Namespace: "test-ns",
			Annotations: map[string]string{
				snapshotprotocol.RestoreStatusAnnotation: "running",
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
	clientset := fake.NewSimpleClientset(pod)
	podWatch := watch.NewFake()
	watchStarted := make(chan struct{})
	clientset.PrependWatchReactor("pods", func(action k8stesting.Action) (bool, watch.Interface, error) {
		close(watchStarted)
		return true, podWatch, nil
	})
	t.Cleanup(podWatch.Stop)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resultCh := make(chan struct {
		status string
		err    error
	}, 1)
	go func() {
		status, err := waitForRestore(ctx, clientset, "test-ns", "restore-pod")
		resultCh <- struct {
			status string
			err    error
		}{status: status, err: err}
	}()

	<-watchStarted

	completedPod := pod.DeepCopy()
	completedPod.Annotations[snapshotprotocol.RestoreStatusAnnotation] = snapshotprotocol.RestoreStatusCompleted
	podWatch.Modify(completedPod)

	select {
	case result := <-resultCh:
		if result.err != nil {
			t.Fatalf("waitForRestore returned error: %v", result.err)
		}
		if result.status != snapshotprotocol.RestoreStatusCompleted {
			t.Fatalf("waitForRestore status = %q, want %q", result.status, snapshotprotocol.RestoreStatusCompleted)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waitForRestore did not return after watch update")
	}
}
