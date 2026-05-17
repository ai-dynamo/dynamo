package controller

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const testNodeName = "test-node"
const testContainerID = "test-container"

// fakeRuntime is a minimal Runtime implementation for controller reconciliation
// tests.
type fakeRuntime struct {
	containerIDByPod string
}

var _ snapshotruntime.Runtime = (*fakeRuntime)(nil)

func (r *fakeRuntime) ResolveContainer(ctx context.Context, id string) (int, *specs.Spec, error) {
	return 0, nil, errors.New("not implemented")
}
func (r *fakeRuntime) ResolveContainerIDByPod(ctx context.Context, pod, ns, ctr string) (string, error) {
	if r.containerIDByPod != "" {
		return r.containerIDByPod, nil
	}
	return "", errors.New("not implemented")
}
func (r *fakeRuntime) ResolveContainerByPod(ctx context.Context, pod, ns, ctr string) (int, *specs.Spec, error) {
	return 0, nil, errors.New("not implemented")
}
func (r *fakeRuntime) Close() error { return nil }

// makeTestController creates a NodeController with a fake k8s client and nil executors.
// The fake clientset is empty so any goroutine launched by runCheckpoint/runRestore
// will fail on the first annotatePod call and exit cleanly.
func makeTestController(t *testing.T, objs ...runtime.Object) *NodeController {
	t.Helper()
	return &NodeController{
		config: &types.AgentConfig{
			NodeName: testNodeName,
			Storage: types.StorageSpec{
				Type:     snapshotprotocol.StorageTypePVC,
				BasePath: t.TempDir(),
			},
		},
		clientset: fake.NewClientset(objs...),
		runtime:   &fakeRuntime{},
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
	// The snapshot contract requires the target-containers annotation on
	// every checkpoint/restore pod; stamp it here so individual cases do
	// not have to repeat themselves.
	merged := map[string]string{
		snapshotprotocol.TargetContainersAnnotation: "main",
	}
	for k, v := range annotations {
		merged[k] = v
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Labels:      labels,
			Annotations: merged,
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
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "main", Ready: ready, ContainerID: "containerd://" + testContainerID},
			},
		},
	}
}

func TestCheckpointLocationsFromPod(t *testing.T) {
	pod := makePod(
		"test-pod",
		"default",
		testNodeName,
		corev1.PodRunning,
		true,
		nil,
		map[string]string{
			snapshotprotocol.CheckpointArtifactVersionAnnotation: "2",
		},
	)

	t.Run("agent mount uses the agent-visible path", func(t *testing.T) {
		w := makeTestController(t)
		w.config.Storage.BasePath = "/checkpoints"

		locations, err := w.checkpointLocationsFromPod(pod, "abc123", 0)
		if err != nil {
			t.Fatalf("checkpointLocationsFromPod() error = %v", err)
		}

		expected := "/checkpoints/abc123/versions/2"
		if locations.HostPath != expected {
			t.Fatalf("HostPath = %q, want %q", locations.HostPath, expected)
		}
		if locations.ContainerPath != expected {
			t.Fatalf("ContainerPath = %q, want %q", locations.ContainerPath, expected)
		}
	})

	t.Run("pod mount uses the target container root from host proc", func(t *testing.T) {
		w := makeTestController(t)
		w.config.Storage.BasePath = "/checkpoints"
		w.config.Storage.AccessMode = types.StorageAccessModePodMount

		locations, err := w.checkpointLocationsFromPod(pod, "abc123", 1234)
		if err != nil {
			t.Fatalf("checkpointLocationsFromPod() error = %v", err)
		}

		expectedContainerPath := "/checkpoints/abc123/versions/2"
		expectedHostPath := filepath.Join(snapshotruntime.HostProcPath, "1234", "root", "checkpoints/abc123/versions/2")
		if locations.HostPath != expectedHostPath {
			t.Fatalf("HostPath = %q, want %q", locations.HostPath, expectedHostPath)
		}
		if locations.ContainerPath != expectedContainerPath {
			t.Fatalf("ContainerPath = %q, want %q", locations.ContainerPath, expectedContainerPath)
		}
	})

	t.Run("pod storage annotation overrides agent base path", func(t *testing.T) {
		annotatedPod := pod.DeepCopy()
		annotatedPod.Annotations[snapshotprotocol.CheckpointStorageBasePathAnnotation] = "/pod-checkpoints/"

		w := makeTestController(t)
		w.config.Storage.BasePath = "/agent-checkpoints"

		locations, err := w.checkpointLocationsFromPod(annotatedPod, "abc123", 0)
		if err != nil {
			t.Fatalf("checkpointLocationsFromPod() error = %v", err)
		}

		expected := "/pod-checkpoints/abc123/versions/2"
		if locations.HostPath != expected {
			t.Fatalf("HostPath = %q, want %q", locations.HostPath, expected)
		}
		if locations.ContainerPath != expected {
			t.Fatalf("ContainerPath = %q, want %q", locations.ContainerPath, expected)
		}
	})

	t.Run("blank pod storage annotation falls back to agent base path", func(t *testing.T) {
		annotatedPod := pod.DeepCopy()
		annotatedPod.Annotations[snapshotprotocol.CheckpointStorageBasePathAnnotation] = "   "

		w := makeTestController(t)
		w.config.Storage.BasePath = "/agent-checkpoints"

		locations, err := w.checkpointLocationsFromPod(annotatedPod, "abc123", 0)
		if err != nil {
			t.Fatalf("checkpointLocationsFromPod() error = %v", err)
		}

		expected := "/agent-checkpoints/abc123/versions/2"
		if locations.HostPath != expected {
			t.Fatalf("HostPath = %q, want %q", locations.HostPath, expected)
		}
		if locations.ContainerPath != expected {
			t.Fatalf("ContainerPath = %q, want %q", locations.ContainerPath, expected)
		}
	})

	t.Run("missing base path returns an error", func(t *testing.T) {
		w := makeTestController(t)
		w.config.Storage.BasePath = ""

		if _, err := w.checkpointLocationsFromPod(pod, "abc123", 0); err == nil {
			t.Fatal("expected error for missing base path")
		}
	})

	t.Run("non-clean base path returns an error", func(t *testing.T) {
		annotatedPod := pod.DeepCopy()
		annotatedPod.Annotations[snapshotprotocol.CheckpointStorageBasePathAnnotation] = "/checkpoints/../escape"

		w := makeTestController(t)
		w.config.Storage.BasePath = "/agent-checkpoints"
		w.config.Storage.AccessMode = types.StorageAccessModePodMount

		_, err := w.checkpointLocationsFromPod(annotatedPod, "abc123", 1234)
		if err == nil {
			t.Fatal("expected error for non-clean checkpoint location")
		}
		if !strings.Contains(err.Error(), "absolute, clean") {
			t.Fatalf("expected clean-path validation error, got: %v", err)
		}
	})

	t.Run("pod mount requires a host PID", func(t *testing.T) {
		w := makeTestController(t)
		w.config.Storage.BasePath = "/checkpoints"
		w.config.Storage.AccessMode = types.StorageAccessModePodMount

		if _, err := w.checkpointLocationsFromPod(pod, "abc123", 0); err == nil {
			t.Fatal("expected error for missing host PID")
		}
	})
}

func TestRestoreCheckpointReady(t *testing.T) {
	w := makeTestController(t)
	log := testr.New(t)

	t.Run("existing directory is ready", func(t *testing.T) {
		dir := t.TempDir()
		ready, err := w.restoreCheckpointReady(log, "default/test-pod", "abc123", dir)
		if err != nil {
			t.Fatalf("restoreCheckpointReady() error = %v", err)
		}
		if !ready {
			t.Fatal("expected checkpoint directory to be ready")
		}
	})

	t.Run("missing directory is not ready", func(t *testing.T) {
		ready, err := w.restoreCheckpointReady(log, "default/test-pod", "abc123", filepath.Join(t.TempDir(), "missing"))
		if err != nil {
			t.Fatalf("restoreCheckpointReady() error = %v", err)
		}
		if ready {
			t.Fatal("expected missing checkpoint directory to be not ready")
		}
	})

	t.Run("file is rejected", func(t *testing.T) {
		filePath := filepath.Join(t.TempDir(), "checkpoint")
		if err := os.WriteFile(filePath, []byte("not a directory"), 0o600); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}

		_, err := w.restoreCheckpointReady(log, "default/test-pod", "abc123", filePath)
		if err == nil {
			t.Fatal("expected file checkpoint location to be rejected")
		}
		if !strings.Contains(err.Error(), "not a directory") {
			t.Fatalf("expected not-a-directory error, got: %v", err)
		}
	})
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
				snapshotprotocol.CheckpointSourceLabel: "true",
				"batch.kubernetes.io/job-name":         "checkpoint-job",
			}
			if tc.hash != "" {
				labels[snapshotprotocol.CheckpointIDLabel] = tc.hash
			}

			job := &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "checkpoint-job",
					Namespace: "default",
				},
			}
			if tc.annotation != "" {
				job.Annotations = map[string]string{
					snapshotprotocol.CheckpointStatusAnnotation: tc.annotation,
				}
			}

			pod := makePod("test-pod", "default", tc.nodeName, tc.phase, tc.ready, labels, nil)
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
			name:      "ready placeholder still restores",
			nodeName:  testNodeName,
			phase:     corev1.PodRunning,
			ready:     true,
			hash:      "abc123",
			createDir: true,
			want:      true,
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
			name:                  "in progress for same container retries after restart",
			nodeName:              testNodeName,
			phase:                 corev1.PodRunning,
			ready:                 false,
			hash:                  "abc123",
			annotationStatus:      "in_progress",
			annotationContainerID: testContainerID,
			createDir:             true,
			want:                  true,
		},
		{
			name:                  "already failed for same container",
			nodeName:              testNodeName,
			phase:                 corev1.PodRunning,
			ready:                 false,
			hash:                  "abc123",
			annotationStatus:      "failed",
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
			name:                  "failed for previous container retries",
			nodeName:              testNodeName,
			phase:                 corev1.PodRunning,
			ready:                 false,
			hash:                  "abc123",
			annotationStatus:      "failed",
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
			// Restore pods are identified by snapshot-agent as
			// (CheckpointIDLabel present, CheckpointSourceLabel absent),
			// so the restore informer's label selector does the filtering.
			// The hash-missing case deliberately omits the label to exercise
			// the early-return branch in reconcileRestorePod.
			labels := map[string]string{}
			if tc.hash != "" {
				labels[snapshotprotocol.CheckpointIDLabel] = tc.hash
			}

			w := makeTestController(t)
			var annotations map[string]string
			if tc.annotationStatus != "" {
				annotations = map[string]string{
					snapshotprotocol.RestoreStatusAnnotationPrefix + "main":      tc.annotationStatus,
					snapshotprotocol.RestoreContainerIDAnnotationPrefix + "main": tc.annotationContainerID,
				}
			}

			pod := makePod("test-pod", "default", tc.nodeName, tc.phase, tc.ready, labels, annotations)
			pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
				Name:        "main",
				Ready:       tc.ready,
				ContainerID: "containerd://" + testContainerID,
			}}

			if tc.createDir && tc.hash != "" {
				dir := filepath.Join(w.config.Storage.BasePath, tc.hash, "versions", snapshotprotocol.DefaultCheckpointArtifactVersion)
				if err := os.MkdirAll(dir, 0o755); err != nil {
					t.Fatalf("failed to create checkpoint dir: %v", err)
				}
			}

			ctx := context.Background()

			if tc.preSeed {
				w.inFlight["default/test-pod/main/"+testContainerID] = struct{}{}
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

func TestReconcileRestorePodRejectsTargetNameThatCannotFitStatusAnnotation(t *testing.T) {
	checkpointID := "abc123"
	containerName := "restore-target-with-long-name-123456"
	w := makeTestController(t)
	dir := filepath.Join(w.config.Storage.BasePath, checkpointID, "versions", snapshotprotocol.DefaultCheckpointArtifactVersion)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("failed to create checkpoint dir: %v", err)
	}

	pod := makePod(
		"test-pod",
		"default",
		testNodeName,
		corev1.PodRunning,
		false,
		map[string]string{snapshotprotocol.CheckpointIDLabel: checkpointID},
		map[string]string{snapshotprotocol.TargetContainersAnnotation: containerName},
	)
	pod.Spec.Containers[0].Name = containerName
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name:        containerName,
		ContainerID: "containerd://" + testContainerID,
	}}

	w.reconcileRestorePod(context.Background(), pod)
	if len(w.inFlight) != 0 {
		t.Fatalf("expected restore not to start for overlong annotation key, got inFlight=%v", w.inFlight)
	}
}

func TestReconcileRestorePodResolvesContainerBeforePodStatus(t *testing.T) {
	labels := map[string]string{
		snapshotprotocol.CheckpointIDLabel: "abc123",
	}
	w := makeTestController(t)
	w.runtime = &fakeRuntime{containerIDByPod: testContainerID}
	clientset := w.clientset.(*fake.Clientset)

	pod := makePod("test-pod", "default", testNodeName, corev1.PodRunning, false, labels, nil)
	pod.Status.ContainerStatuses = nil
	dir := filepath.Join(w.config.Storage.BasePath, "abc123", "versions", snapshotprotocol.DefaultCheckpointArtifactVersion)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("failed to create checkpoint dir: %v", err)
	}

	w.reconcileRestorePod(context.Background(), pod)

	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		for _, action := range clientset.Actions() {
			create, ok := action.(clientgotesting.CreateAction)
			if !ok || create.GetResource().Resource != "events" {
				continue
			}
			event, ok := create.GetObject().(*corev1.Event)
			if ok && event.Reason == "RestoreRequested" {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("expected RestoreRequested event after node-runtime container resolution; actions=%#v", clientset.Actions())
}

func TestPollForContainerIDSkipsWhenRestoreAttemptAlreadyHeld(t *testing.T) {
	checkpointID := "abc123"
	labels := map[string]string{
		snapshotprotocol.CheckpointIDLabel: checkpointID,
	}
	stalePod := makePod("test-pod", "default", testNodeName, corev1.PodRunning, false, labels, nil)
	stalePod.Status.ContainerStatuses = nil

	w := makeTestController(t)
	w.runtime = &fakeRuntime{containerIDByPod: testContainerID}
	clientset := w.clientset.(*fake.Clientset)
	dir := filepath.Join(w.config.Storage.BasePath, checkpointID, "versions", snapshotprotocol.DefaultCheckpointArtifactVersion)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("failed to create checkpoint dir: %v", err)
	}

	resolveKey := "default/test-pod/main/resolve"
	restoreAttemptKey := "default/test-pod/main/" + testContainerID
	w.inFlight[resolveKey] = struct{}{}
	w.inFlight[restoreAttemptKey] = struct{}{}
	w.pollForContainerID(context.Background(), stalePod, "main", checkpointID, "default/test-pod", resolveKey)

	if _, held := w.inFlight[resolveKey]; held {
		t.Fatal("expected resolver key to be released")
	}
	if _, held := w.inFlight[restoreAttemptKey]; !held {
		t.Fatal("expected existing restore attempt key to remain held")
	}
	for _, action := range clientset.Actions() {
		create, ok := action.(clientgotesting.CreateAction)
		if !ok || create.GetResource().Resource != "events" {
			continue
		}
		event, ok := create.GetObject().(*corev1.Event)
		if ok && event.Reason == "RestoreRequested" {
			t.Fatalf("stale resolver should not start restore while attempt key is held; actions=%#v", clientset.Actions())
		}
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
			Storage: types.StorageSpec{
				Type:     snapshotprotocol.StorageTypePVC,
				BasePath: t.TempDir(),
			},
		},
		clientset: clientset,
		runtime:   &fakeRuntime{},
		log:       testr.New(t),
		holderID:  "test-holder",
		inFlight: map[string]struct{}{
			"default/test-pod": {},
		},
		stopCh: make(chan struct{}),
	}

	err := w.runCheckpoint(context.Background(), pod, job, "abc123", "main", "default/test-pod", time.Now())
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
