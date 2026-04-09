package main

import (
	"context"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func TestWaitForCheckpointWatchesUntilCompleted(t *testing.T) {
	t.Parallel()

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "checkpoint-job",
			Namespace: "test-ns",
			Annotations: map[string]string{
				snapshotprotocol.CheckpointStatusAnnotation: "running",
			},
		},
	}
	clientset := fake.NewSimpleClientset(job)
	jobWatch := watch.NewFake()
	watchStarted := make(chan struct{})
	clientset.PrependWatchReactor("jobs", func(action k8stesting.Action) (bool, watch.Interface, error) {
		close(watchStarted)
		return true, jobWatch, nil
	})
	t.Cleanup(jobWatch.Stop)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resultCh := make(chan struct {
		status string
		err    error
	}, 1)
	go func() {
		status, err := waitForCheckpoint(ctx, clientset, "test-ns", "checkpoint-job")
		resultCh <- struct {
			status string
			err    error
		}{status: status, err: err}
	}()

	<-watchStarted

	completedJob := job.DeepCopy()
	completedJob.Annotations[snapshotprotocol.CheckpointStatusAnnotation] = snapshotprotocol.CheckpointStatusCompleted
	jobWatch.Modify(completedJob)

	select {
	case result := <-resultCh:
		if result.err != nil {
			t.Fatalf("waitForCheckpoint returned error: %v", result.err)
		}
		if result.status != snapshotprotocol.CheckpointStatusCompleted {
			t.Fatalf("waitForCheckpoint status = %q, want %q", result.status, snapshotprotocol.CheckpointStatusCompleted)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waitForCheckpoint did not return after watch update")
	}
}
