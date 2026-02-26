package watcher

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

const testNodeName = "test-node"

// makeTestWatcher creates a Watcher with a fake k8s client and nil orchestrators.
// The fake clientset is empty so any goroutine launched by doCheckpoint/doRestore
// will fail on the first annotatePod call and exit cleanly.
func makeTestWatcher(t *testing.T) *Watcher {
	t.Helper()
	return &Watcher{
		config: &types.AgentConfig{
			NodeName: testNodeName,
			BasePath: t.TempDir(),
		},
		clientset: fake.NewSimpleClientset(),
		log:       testr.New(t),
		inFlight:  make(map[string]struct{}),
		stopCh:    make(chan struct{}),
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

func TestHandleCheckpointPodEvent(t *testing.T) {
	tests := []struct {
		name       string
		nodeName   string
		phase      corev1.PodPhase
		ready      bool
		hash       string
		annotation string
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
			name:       "already in progress",
			nodeName:   testNodeName,
			phase:      corev1.PodRunning,
			ready:      true,
			hash:       "abc123",
			annotation: "in_progress",
			want:       false,
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
				kubeLabelIsCheckpointSource: "true",
			}
			if tc.hash != "" {
				labels[kubeLabelCheckpointHash] = tc.hash
			}

			var annotations map[string]string
			if tc.annotation != "" {
				annotations = map[string]string{
					kubeAnnotationCheckpointStatus: tc.annotation,
				}
			}

			pod := makePod("test-pod", "default", tc.nodeName, tc.phase, tc.ready, labels, annotations)
			w := makeTestWatcher(t)
			ctx := context.Background()

			if tc.preSeed {
				w.inFlight["default/test-pod"] = struct{}{}
			}

			w.handleCheckpointPodEvent(ctx, pod)

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

func TestHandleRestorePodEvent(t *testing.T) {
	tests := []struct {
		name       string
		nodeName   string
		phase      corev1.PodPhase
		ready      bool
		hash       string
		annotation string
		createDir  bool // whether to create the checkpoint dir on disk
		preSeed    bool
		want       bool
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
			name:       "already completed",
			nodeName:   testNodeName,
			phase:      corev1.PodRunning,
			ready:      false,
			hash:       "abc123",
			annotation: "completed",
			createDir:  true,
			want:       false,
		},
		{
			name:       "already in progress",
			nodeName:   testNodeName,
			phase:      corev1.PodRunning,
			ready:      false,
			hash:       "abc123",
			annotation: "in_progress",
			createDir:  true,
			want:       false,
		},
		{
			name:       "already failed",
			nodeName:   testNodeName,
			phase:      corev1.PodRunning,
			ready:      false,
			hash:       "abc123",
			annotation: "failed",
			createDir:  true,
			want:       false,
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

			var annotations map[string]string
			if tc.annotation != "" {
				annotations = map[string]string{
					kubeAnnotationRestoreStatus: tc.annotation,
				}
			}

			pod := makePod("test-pod", "default", tc.nodeName, tc.phase, tc.ready, labels, annotations)
			w := makeTestWatcher(t)

			if tc.createDir && tc.hash != "" {
				dir := filepath.Join(w.config.BasePath, tc.hash)
				if err := os.MkdirAll(dir, 0o755); err != nil {
					t.Fatalf("failed to create checkpoint dir: %v", err)
				}
			}

			ctx := context.Background()

			if tc.preSeed {
				w.inFlight["default/test-pod"] = struct{}{}
			}

			w.handleRestorePodEvent(ctx, pod)

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
