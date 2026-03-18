package controller

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/types"
)

const testNodeName = "test-node"
const testContainerID = "test-container"

// makeTestController creates a NodeController with a fake k8s client and nil executors.
// The fake clientset is empty so any goroutine launched by runCheckpoint/runRestore
// will fail on the first annotatePod call and exit cleanly.
func makeTestController(t *testing.T, objs ...runtime.Object) *NodeController {
	t.Helper()
	return &NodeController{
		config: &types.AgentConfig{
			NodeName: testNodeName,
		},
		clientset: fake.NewClientset(objs...),
		log:       testr.New(t),
		holderID:  "test-holder",
		inFlight:  make(map[string]struct{}),
		stopCh:    make(chan struct{}),
	}
}

func makeLease(namespace, name, holder string, renewTime time.Time) *coordinationv1.Lease {
	leaseDurationSeconds := int32(checkpointLeaseDuration.Seconds())
	renewMicroTime := metav1.NewMicroTime(renewTime)
	return &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       &holder,
			LeaseDurationSeconds: &leaseDurationSeconds,
			AcquireTime:          &renewMicroTime,
			RenewTime:            &renewMicroTime,
		},
	}
}

func makePod(name, namespace, nodeName string, phase corev1.PodPhase, ready bool, labels, annotations map[string]string) *corev1.Pod {
	var conditions []corev1.PodCondition
	if ready {
		conditions = append(conditions, corev1.PodCondition{
			Type:   corev1.PodReady,
			Status: corev1.ConditionTrue,
		})
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{
				{Name: "main"},
			},
		},
		Status: corev1.PodStatus{
			Phase:      phase,
			Conditions: conditions,
		},
	}
}

func TestReconcileCheckpointPod(t *testing.T) {
	tests := []struct {
		name       string
		nodeName   string
		phase      corev1.PodPhase
		ready      bool
		hash       string
		annotation string
		lease      *coordinationv1.Lease
		preSeed    bool // pre-populate inFlight to test deduplication
		want       bool // true = pod passes filtering and triggers checkpoint
	}{
		{
			name:     "happy path",
			nodeName: testNodeName,
			phase:    corev1.PodRunning,
			ready:    true,
			hash:     "abc123",
			want:     true,
		},
		{
			name:     "wrong node",
			nodeName: "other-node",
			phase:    corev1.PodRunning,
			ready:    true,
			hash:     "abc123",
			want:     false,
		},
		{
			name:     "not running",
			nodeName: testNodeName,
			phase:    corev1.PodPending,
			ready:    false,
			hash:     "abc123",
			want:     false,
		},
		{
			name:     "running but not ready",
			nodeName: testNodeName,
			phase:    corev1.PodRunning,
			ready:    false,
			hash:     "abc123",
			want:     false,
		},
		{
			name:     "missing hash label",
			nodeName: testNodeName,
			phase:    corev1.PodRunning,
			ready:    true,
			hash:     "",
			want:     false,
		},
		{
			name:       "already completed",
			nodeName:   testNodeName,
			phase:      corev1.PodRunning,
			ready:      true,
			hash:       "abc123",
			annotation: "completed",
			want:       false,
		},
		{
			name:       "already failed",
			nodeName:   testNodeName,
			phase:      corev1.PodRunning,
			ready:      true,
			hash:       "abc123",
			annotation: "failed",
			want:       false,
		},
		{
			name:     "active lease held elsewhere",
			nodeName: testNodeName,
			phase:    corev1.PodRunning,
			ready:    true,
			hash:     "abc123",
			lease:    makeLease("default", "checkpoint-job", "other-holder", time.Now()),
			want:     false,
		},
		{
			name:     "expired lease can be reclaimed",
			nodeName: testNodeName,
			phase:    corev1.PodRunning,
			ready:    true,
			hash:     "abc123",
			lease:    makeLease("default", "checkpoint-job", "other-holder", time.Now().Add(-checkpointLeaseDuration-time.Second)),
			want:     true,
		},
		{
			name:     "duplicate in-flight",
			nodeName: testNodeName,
			phase:    corev1.PodRunning,
			ready:    true,
			hash:     "abc123",
			preSeed:  true,
			want:     false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			labels := map[string]string{
				kubeLabelIsCheckpointSource:    "true",
				"batch.kubernetes.io/job-name": "checkpoint-job",
			}
			if tc.hash != "" {
				labels[kubeLabelCheckpointHash] = tc.hash
			}

			job := &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "checkpoint-job",
					Namespace: "default",
				},
			}
			if tc.annotation != "" {
				job.Annotations = map[string]string{
					kubeAnnotationCheckpointStatus: tc.annotation,
				}
			}

			var annotations map[string]string
			if tc.hash != "" {
				annotations = map[string]string{
					kubeAnnotationCheckpointLocation:    "/checkpoints/" + tc.hash,
					kubeAnnotationCheckpointStorageType: "pvc",
				}
			}
			pod := makePod("test-pod", "default", tc.nodeName, tc.phase, tc.ready, labels, annotations)
			objs := []runtime.Object{job}
			if tc.lease != nil {
				objs = append(objs, tc.lease)
			}

			w := makeTestController(t, objs...)
			ctx := context.Background()

			if tc.preSeed {
				w.inFlight["default/test-pod"] = struct{}{}
			}

			w.reconcileCheckpointPod(ctx, pod)

			// tryAcquire adds to inFlight synchronously before launching the goroutine.
			// For filtered pods, inFlight stays at its original size.
			triggered := len(w.inFlight) > 0 && !tc.preSeed
			if tc.preSeed {
				// Duplicate: inFlight was 1 before and should remain exactly 1
				triggered = false
			}

			if triggered != tc.want {
				t.Errorf("triggered = %v, want %v (inFlight=%d, preSeed=%v)", triggered, tc.want, len(w.inFlight), tc.preSeed)
			}

			// Let the background goroutine (if any) finish before the test ends
			if tc.want {
				time.Sleep(50 * time.Millisecond)
			}
		})
	}
}

func TestReconcileRestorePod(t *testing.T) {
	tests := []struct {
		name                  string
		nodeName              string
		phase                 corev1.PodPhase
		ready                 bool
		hash                  string
		annotationStatus      string
		annotationContainerID string
		createDir             bool // whether to create the checkpoint dir on disk
		preSeed               bool
		want                  bool
	}{
		{
			name:      "happy path",
			nodeName:  testNodeName,
			phase:     corev1.PodRunning,
			ready:     false,
			hash:      "abc123",
			createDir: true,
			want:      true,
		},
		{
			name:      "wrong node",
			nodeName:  "other-node",
			phase:     corev1.PodRunning,
			ready:     false,
			hash:      "abc123",
			createDir: true,
			want:      false,
		},
		{
			name:      "not running",
			nodeName:  testNodeName,
			phase:     corev1.PodPending,
			ready:     false,
			hash:      "abc123",
			createDir: true,
			want:      false,
		},
		{
			name:      "already ready",
			nodeName:  testNodeName,
			phase:     corev1.PodRunning,
			ready:     true,
			hash:      "abc123",
			createDir: true,
			want:      false,
		},
		{
			name:     "missing hash",
			nodeName: testNodeName,
			phase:    corev1.PodRunning,
			ready:    false,
			hash:     "",
			want:     false,
		},
		{
			name:      "invalid hash with path traversal",
			nodeName:  testNodeName,
			phase:     corev1.PodRunning,
			ready:     false,
			hash:      "../bad",
			createDir: true,
			want:      false,
		},
		{
			name:                  "already completed for same container",
			nodeName:              testNodeName,
			phase:                 corev1.PodRunning,
			ready:                 false,
			hash:                  "abc123",
			annotationStatus:      "completed",
			annotationContainerID: testContainerID,
			createDir:             true,
			want:                  false,
		},
		{
			name:                  "already in progress for same container",
			nodeName:              testNodeName,
			phase:                 corev1.PodRunning,
			ready:                 false,
			hash:                  "abc123",
			annotationStatus:      "in_progress",
			annotationContainerID: testContainerID,
			createDir:             true,
			want:                  false,
		},
		{
			name:                  "completed for previous container retries",
			nodeName:              testNodeName,
			phase:                 corev1.PodRunning,
			ready:                 false,
			hash:                  "abc123",
			annotationStatus:      "completed",
			annotationContainerID: "old-container",
			createDir:             true,
			want:                  true,
		},
		{
			name:                  "in progress for previous container retries",
			nodeName:              testNodeName,
			phase:                 corev1.PodRunning,
			ready:                 false,
			hash:                  "abc123",
			annotationStatus:      "in_progress",
			annotationContainerID: "old-container",
			createDir:             true,
			want:                  true,
		},
		{
			name:      "checkpoint not on disk",
			nodeName:  testNodeName,
			phase:     corev1.PodRunning,
			ready:     false,
			hash:      "abc123",
			createDir: false,
			want:      false,
		},
		{
			name:      "duplicate in-flight",
			nodeName:  testNodeName,
			phase:     corev1.PodRunning,
			ready:     false,
			hash:      "abc123",
			createDir: true,
			preSeed:   true,
			want:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			labels := map[string]string{
				kubeLabelIsRestoreTarget: "true",
			}
			if tc.hash != "" {
				labels[kubeLabelCheckpointHash] = tc.hash
			}

			w := makeTestController(t)
			checkpointDir := t.TempDir()

			var annotations map[string]string
			if tc.annotationStatus != "" {
				annotations = map[string]string{
					kubeAnnotationRestoreStatus:      tc.annotationStatus,
					kubeAnnotationRestoreContainerID: tc.annotationContainerID,
				}
			}
			if tc.hash != "" {
				if annotations == nil {
					annotations = make(map[string]string)
				}
				annotations[kubeAnnotationCheckpointLocation] = filepath.Join(checkpointDir, tc.hash)
				annotations[kubeAnnotationCheckpointStorageType] = "pvc"
			}

			pod := makePod("test-pod", "default", tc.nodeName, tc.phase, tc.ready, labels, annotations)
			pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
				Name:        "main",
				Ready:       tc.ready,
				ContainerID: "containerd://" + testContainerID,
			}}

			if tc.createDir && tc.hash != "" {
				dir := filepath.Join(checkpointDir, tc.hash)
				if err := os.MkdirAll(dir, 0o755); err != nil {
					t.Fatalf("failed to create checkpoint dir: %v", err)
				}
			}

			ctx := context.Background()

			if tc.preSeed {
				w.inFlight["default/test-pod/"+testContainerID] = struct{}{}
			}

			w.reconcileRestorePod(ctx, pod)

			triggered := len(w.inFlight) > 0 && !tc.preSeed
			if tc.preSeed {
				triggered = false
			}

			if triggered != tc.want {
				t.Errorf("triggered = %v, want %v (inFlight=%d, preSeed=%v)", triggered, tc.want, len(w.inFlight), tc.preSeed)
			}

			// Let the background goroutine (if any) finish before the test ends
			if tc.want {
				time.Sleep(50 * time.Millisecond)
			}
		})
	}
}

func TestRunCheckpointKeepsLeaseAndInFlightOnTerminalStatusPatchFailure(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			Labels: map[string]string{
				"batch.kubernetes.io/job-name": "checkpoint-job",
			},
		},
	}
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "checkpoint-job",
			Namespace: "default",
		},
	}
	lease := makeLease("default", "checkpoint-job", "test-holder", time.Now())

	clientset := fake.NewClientset(pod.DeepCopy(), job, lease)
	patchCalls := 0
	clientset.PrependReactor("patch", "jobs", func(clientgotesting.Action) (bool, runtime.Object, error) {
		patchCalls++
		return true, nil, errors.New("terminal patch failed")
	})

	w := &NodeController{
		config: &types.AgentConfig{
			NodeName: testNodeName,
		},
		clientset: clientset,
		log:       testr.New(t),
		holderID:  "test-holder",
		inFlight: map[string]struct{}{
			"default/test-pod": {},
		},
		stopCh: make(chan struct{}),
	}

	err := w.runCheckpoint(context.Background(), pod, job, "abc123", filepath.Join(t.TempDir(), "abc123"), "pvc", "default/test-pod")
	if err == nil {
		t.Fatal("expected terminal checkpoint status update to fail")
	}
	if _, ok := w.inFlight["default/test-pod"]; !ok {
		t.Fatal("checkpoint terminal status failure should keep pod in-flight")
	}
	if patchCalls != 1 {
		t.Fatalf("patchCalls = %d, want %d", patchCalls, 1)
	}

	remainingLease, err := clientset.CoordinationV1().Leases("default").Get(context.Background(), "checkpoint-job", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("expected checkpoint lease to remain after terminal status patch failure: %v", err)
	}
	if remainingLease.Spec.HolderIdentity == nil || *remainingLease.Spec.HolderIdentity != "test-holder" {
		t.Fatalf("unexpected remaining lease holder: %#v", remainingLease.Spec.HolderIdentity)
	}
}
