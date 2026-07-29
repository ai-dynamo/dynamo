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
	containerIDByPod     string
	resolvedContainerIDs []string
	// resolveContainerPID, when set, is returned by ResolveContainer with no error so the
	// capture path can advance past container resolution.
	resolveContainerPID int
}

var _ snapshotruntime.Runtime = (*fakeRuntime)(nil)

func (r *fakeRuntime) ResolveContainer(ctx context.Context, id string) (int, *specs.Spec, error) {
	r.resolvedContainerIDs = append(r.resolvedContainerIDs, id)
	if r.resolveContainerPID > 0 {
		return r.resolveContainerPID, nil, nil
	}
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
// The fake clientset is empty so any goroutine launched by the restore path will fail on
// the first annotatePod call and exit cleanly.
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

func sawEventReason(clientset *fake.Clientset, reason string) bool {
	for _, action := range clientset.Actions() {
		create, ok := action.(clientgotesting.CreateAction)
		if !ok || create.GetResource().Resource != "events" {
			continue
		}
		event, ok := create.GetObject().(*corev1.Event)
		if ok && event.Reason == reason {
			return true
		}
	}
	return false
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
			name:      "pending pod with status container id still restores",
			nodeName:  testNodeName,
			phase:     corev1.PodPending,
			ready:     false,
			hash:      "abc123",
			createDir: true,
			want:      true,
		},
		{
			name:      "succeeded pod does not restore",
			nodeName:  testNodeName,
			phase:     corev1.PodSucceeded,
			ready:     false,
			hash:      "abc123",
			createDir: true,
			want:      false,
		},
		{
			name:      "failed pod does not restore",
			nodeName:  testNodeName,
			phase:     corev1.PodFailed,
			ready:     false,
			hash:      "abc123",
			createDir: true,
			want:      false,
		},
		{
			name:      "unknown pod does not restore",
			nodeName:  testNodeName,
			phase:     corev1.PodUnknown,
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

			triggered := sawEventReason(w.clientset.(*fake.Clientset), "RestoreRequested")

			if triggered != tc.want {
				t.Errorf("triggered = %v, want %v (inFlight=%d, preSeed=%v, actions=%#v)", triggered, tc.want, len(w.inFlight), tc.preSeed, w.clientset.(*fake.Clientset).Actions())
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

	pod := makePod("test-pod", "default", testNodeName, corev1.PodRunning, false, labels, nil)
	pod.Status.ContainerStatuses = nil
	w := makeTestController(t, pod)
	w.runtime = &fakeRuntime{containerIDByPod: testContainerID}
	clientset := w.clientset.(*fake.Clientset)
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

func TestReconcileRestorePodPollsRuntimeBeforePodRunning(t *testing.T) {
	labels := map[string]string{
		snapshotprotocol.CheckpointIDLabel: "abc123",
	}

	pod := makePod("test-pod", "default", testNodeName, corev1.PodPending, false, labels, nil)
	pod.Status.ContainerStatuses = nil
	w := makeTestController(t, pod)
	w.runtime = &fakeRuntime{containerIDByPod: testContainerID}
	clientset := w.clientset.(*fake.Clientset)
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
	t.Fatalf("expected RestoreRequested event from runtime polling before PodRunning; actions=%#v", clientset.Actions())
}

func TestPollForContainerIDSkipsTerminalLivePod(t *testing.T) {
	checkpointID := "abc123"
	labels := map[string]string{
		snapshotprotocol.CheckpointIDLabel: checkpointID,
	}
	stalePod := makePod("test-pod", "default", testNodeName, corev1.PodPending, false, labels, nil)
	stalePod.Status.ContainerStatuses = nil
	livePod := stalePod.DeepCopy()
	livePod.Status.Phase = corev1.PodSucceeded

	w := makeTestController(t, livePod)
	w.runtime = &fakeRuntime{containerIDByPod: testContainerID}
	clientset := w.clientset.(*fake.Clientset)
	dir := filepath.Join(w.config.Storage.BasePath, checkpointID, "versions", snapshotprotocol.DefaultCheckpointArtifactVersion)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("failed to create checkpoint dir: %v", err)
	}

	resolveKey := "default/test-pod/main/resolve"
	w.inFlight[resolveKey] = struct{}{}
	w.pollForContainerID(context.Background(), stalePod, "main", checkpointID, "default/test-pod", resolveKey)

	if _, held := w.inFlight[resolveKey]; held {
		t.Fatal("expected resolver key to be released")
	}
	for _, action := range clientset.Actions() {
		create, ok := action.(clientgotesting.CreateAction)
		if !ok || create.GetResource().Resource != "events" {
			continue
		}
		event, ok := create.GetObject().(*corev1.Event)
		if ok && event.Reason == "RestoreRequested" {
			t.Fatalf("stale resolver should not start restore for terminal live pod; actions=%#v", clientset.Actions())
		}
	}
}

func TestPollForContainerIDSkipsWhenRestoreAttemptAlreadyHeld(t *testing.T) {
	checkpointID := "abc123"
	labels := map[string]string{
		snapshotprotocol.CheckpointIDLabel: checkpointID,
	}
	stalePod := makePod("test-pod", "default", testNodeName, corev1.PodRunning, false, labels, nil)
	stalePod.Status.ContainerStatuses = nil

	w := makeTestController(t, stalePod)
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

