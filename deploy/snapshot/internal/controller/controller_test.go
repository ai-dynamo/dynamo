package controller

import (
	"context"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsmount"
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
	resolveContainerPID   int
	rejectCanceledContext bool
}

var _ snapshotruntime.Runtime = (*fakeRuntime)(nil)

func (r *fakeRuntime) ResolveContainer(ctx context.Context, id string) (int, *specs.Spec, error) {
	r.resolvedContainerIDs = append(r.resolvedContainerIDs, id)
	if r.rejectCanceledContext && ctx.Err() != nil {
		return 0, nil, ctx.Err()
	}
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

// noopInjector is a no-op Mounter used in tests that do not exercise
// the injection path. It prevents a nil-pointer panic if runRestore is ever
// reached by a test that was previously relying on Phase 1 failing first.
type noopInjector struct{}

func (noopInjector) Mount(_ context.Context, _ int, _, _ string) (nsmount.MountPoint, error) {
	return noopMountPoint{}, nil
}

type noopMountPoint struct{}

func (noopMountPoint) Unmount(_ context.Context, _ bool) error { return nil }
func (noopMountPoint) Path(name string) (string, error)        { return "", nil }
func (noopMountPoint) NsFd() *os.File                          { return nil }

// errorInjector always fails the bundle mount phase.
type errorInjector struct{ err error }

func (e errorInjector) Mount(_ context.Context, _ int, _, _ string) (nsmount.MountPoint, error) {
	return nil, e.err
}

var _ executor.Mounter = noopInjector{}
var _ executor.Mounter = errorInjector{}

// makeTestController creates a NodeController with a fake k8s client and nil executors.
// The fake clientset is empty so any goroutine launched by the restore path will fail on
// the first annotatePod call and exit cleanly.
func makeTestController(t *testing.T, objs ...runtime.Object) *NodeController {
	t.Helper()
	return &NodeController{
		config: &types.AgentConfig{
			NodeName: testNodeName,
			Storage: types.StorageSpec{
				Type:     "pvc",
				BasePath: t.TempDir(),
			},
		},
		clientset: fake.NewClientset(objs...),
		runtime:   &fakeRuntime{},
		injector:  executor.Mounters{Bundle: noopInjector{}, Artifact: noopInjector{}},
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

func TestArtifactPathForPod(t *testing.T) {
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

	t.Run("the agent base path and the pod version compose the artifact path", func(t *testing.T) {
		w := makeTestController(t)
		w.config.Storage.BasePath = "/checkpoints"

		got, err := w.artifactPathForPod(pod, "abc123")
		require.NoError(t, err)
		assert.Equal(t, "/checkpoints/abc123/versions/2", got)
	})

	t.Run("a missing version annotation falls back to the protocol default", func(t *testing.T) {
		unversioned := pod.DeepCopy()
		delete(unversioned.Annotations, snapshotprotocol.CheckpointArtifactVersionAnnotation)

		w := makeTestController(t)
		w.config.Storage.BasePath = "/checkpoints"

		got, err := w.artifactPathForPod(unversioned, "abc123")
		require.NoError(t, err)
		assert.Equal(t, "/checkpoints/abc123/versions/"+snapshotprotocol.DefaultCheckpointArtifactVersion, got)
	})

	t.Run("a pod cannot redirect the artifact path", func(t *testing.T) {
		tests := []struct {
			name         string
			basePath     string
			checkpointID string
			version      string
		}{
			{name: "missing base path", basePath: "", checkpointID: "abc123", version: "2"},
			{name: "relative base path", basePath: "checkpoints", checkpointID: "abc123", version: "2"},
			{name: "unclean base path", basePath: "/checkpoints/../escape", checkpointID: "abc123", version: "2"},
			{name: "traversal in the checkpoint ID", basePath: "/checkpoints", checkpointID: "../escape", version: "2"},
			{name: "separator in the checkpoint ID", basePath: "/checkpoints", checkpointID: "a/b", version: "2"},
			{name: "traversal in the version", basePath: "/checkpoints", checkpointID: "abc123", version: ".."},
			{name: "separator in the version", basePath: "/checkpoints", checkpointID: "abc123", version: "2/3"},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				hostile := pod.DeepCopy()
				hostile.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation] = tc.version

				w := makeTestController(t)
				w.config.Storage.BasePath = tc.basePath

				_, err := w.artifactPathForPod(hostile, tc.checkpointID)
				require.Error(t, err)
			})
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

func TestRunRestoreEmitsRestoreFailedEventOnInjectError(t *testing.T) {
	checkpointID := "test-checkpoint"
	pod := makePod("test-pod", "default", testNodeName, corev1.PodRunning, true,
		map[string]string{snapshotprotocol.CheckpointIDLabel: checkpointID}, nil)

	injectErr := errors.New("injector unavailable")
	w := makeTestController(t, pod)

	// The agent resolves the artifact under its own base path, so the fixture
	// has to sit where the agent will look rather than anywhere the pod names.
	checkpointDir := filepath.Join(
		w.config.Storage.BasePath, checkpointID, "versions",
		snapshotprotocol.DefaultCheckpointArtifactVersion,
	)
	require.NoError(t, os.MkdirAll(checkpointDir, 0o755))
	require.NoError(t, types.WriteManifest(checkpointDir, &types.CheckpointManifest{CheckpointID: checkpointID}))

	// math.MaxInt32 is above any real kernel pid_max (≤4194304) so SendSignalToPID
	// returns ESRCH instead of killing the test process.
	w.runtime = &fakeRuntime{resolveContainerPID: math.MaxInt32}
	w.injector = executor.Mounters{Bundle: errorInjector{err: injectErr}, Artifact: errorInjector{err: injectErr}}

	_ = w.runRestore(
		context.Background(), pod, "main", "ctr-abc", checkpointID,
		"default/test-pod/main/ctr-abc",
		time.Time{},
	)

	if !sawEventReason(w.clientset.(*fake.Clientset), "RestoreFailed") {
		t.Fatal("expected RestoreFailed event when injector returns an error")
	}
}

func startRestoreTarget(t *testing.T) (int, chan error) {
	t.Helper()
	cmd := exec.Command("sleep", "60")
	require.NoError(t, cmd.Start())
	waitCh := make(chan error, 1)
	go func() { waitCh <- cmd.Wait() }()
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		select {
		case <-waitCh:
		case <-time.After(time.Second):
		}
	})
	return cmd.Process.Pid, waitCh
}

func requireTargetKilled(t *testing.T, waitCh chan error) {
	t.Helper()
	select {
	case err := <-waitCh:
		var exitErr *exec.ExitError
		require.ErrorAs(t, err, &exitErr)
		status, ok := exitErr.Sys().(syscall.WaitStatus)
		require.True(t, ok)
		assert.True(t, status.Signaled())
		assert.Equal(t, syscall.SIGKILL, status.Signal())
		waitCh <- err // let test cleanup observe that Wait already completed
	case <-time.After(5 * time.Second):
		t.Fatal("restore failure did not terminate the target")
	}
}

func TestRunRestore_AllFailureStagesKillTheTarget(t *testing.T) {
	tests := []struct {
		name          string
		rejectPatches bool
		prepare       func(t *testing.T, w *NodeController, pod *corev1.Pod, checkpointID string)
	}{
		{
			name: "invalid artifact coordinates",
			prepare: func(_ *testing.T, _ *NodeController, pod *corev1.Pod, _ string) {
				pod.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation] = "../escape"
			},
		},
		{
			name: "artifact path is not a directory",
			prepare: func(t *testing.T, w *NodeController, _ *corev1.Pod, checkpointID string) {
				path := filepath.Join(w.config.Storage.BasePath, checkpointID, "versions", snapshotprotocol.DefaultCheckpointArtifactVersion)
				require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
				require.NoError(t, os.WriteFile(path, []byte("not a directory"), 0o600))
			},
		},
		{
			name: "mount helper fails after artifact validation",
			prepare: func(t *testing.T, w *NodeController, _ *corev1.Pod, checkpointID string) {
				path := filepath.Join(w.config.Storage.BasePath, checkpointID, "versions", snapshotprotocol.DefaultCheckpointArtifactVersion)
				require.NoError(t, os.MkdirAll(path, 0o755))
				require.NoError(t, types.WriteManifest(path, &types.CheckpointManifest{CheckpointID: checkpointID}))
				failing := errorInjector{err: errors.New("mount helper failed")}
				w.injector = executor.Mounters{Bundle: failing, Artifact: failing}
			},
		},
		{
			name:          "failure status cannot be persisted",
			rejectPatches: true,
			prepare: func(_ *testing.T, _ *NodeController, pod *corev1.Pod, _ string) {
				pod.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation] = "../escape"
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			checkpointID := "test-checkpoint"
			pod := makePod("test-pod", "default", testNodeName, corev1.PodRunning, true,
				map[string]string{snapshotprotocol.CheckpointIDLabel: checkpointID}, nil)
			w := makeTestController(t, pod)
			pid, waitCh := startRestoreTarget(t)
			w.runtime = &fakeRuntime{resolveContainerPID: pid}
			tc.prepare(t, w, pod, checkpointID)
			if tc.rejectPatches {
				w.clientset.(*fake.Clientset).PrependReactor("patch", "pods", func(clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("status patch rejected")
				})
			}

			_ = w.runRestore(
				context.Background(), pod, "main", "ctr-abc", checkpointID,
				"default/test-pod/main/ctr-abc", time.Time{},
			)

			requireTargetKilled(t, waitCh)
			if !tc.rejectPatches {
				got, err := w.clientset.CoreV1().Pods(pod.Namespace).Get(context.Background(), pod.Name, metav1.GetOptions{})
				require.NoError(t, err)
				keys, err := snapshotprotocol.RestoreStatusAnnotationKeysFor("main")
				require.NoError(t, err)
				assert.Equal(t, snapshotprotocol.RestoreStatusFailed, got.Annotations[keys.Status])
			}
			assert.True(t, sawEventReason(w.clientset.(*fake.Clientset), "RestoreFailed"))
		})
	}
}

func TestRunRestore_CanceledParentStillMarksFailedAndKills(t *testing.T) {
	checkpointID := "test-checkpoint"
	pod := makePod("test-pod", "default", testNodeName, corev1.PodRunning, true,
		map[string]string{snapshotprotocol.CheckpointIDLabel: checkpointID},
		map[string]string{snapshotprotocol.CheckpointArtifactVersionAnnotation: "../escape"},
	)
	w := makeTestController(t, pod)
	pid, waitCh := startRestoreTarget(t)
	w.runtime = &fakeRuntime{resolveContainerPID: pid, rejectCanceledContext: true}

	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	err := w.runRestore(
		canceledCtx, pod, "main", "ctr-abc", checkpointID,
		"default/test-pod/main/ctr-abc", time.Time{},
	)

	require.ErrorContains(t, err, "artifact version")
	requireTargetKilled(t, waitCh)
	got, getErr := w.clientset.CoreV1().Pods(pod.Namespace).Get(context.Background(), pod.Name, metav1.GetOptions{})
	require.NoError(t, getErr)
	keys, keyErr := snapshotprotocol.RestoreStatusAnnotationKeysFor("main")
	require.NoError(t, keyErr)
	assert.Equal(t, snapshotprotocol.RestoreStatusFailed, got.Annotations[keys.Status])
}
