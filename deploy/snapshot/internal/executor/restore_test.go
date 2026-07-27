package executor

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/go-logr/logr/testr"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

type restoreFakeRuntime struct {
	resolvedID      string
	resolveByPodHit bool
}

func (r *restoreFakeRuntime) ResolveContainer(ctx context.Context, id string) (int, *specs.Spec, error) {
	r.resolvedID = id
	return 123, &specs.Spec{}, nil
}

func (r *restoreFakeRuntime) ResolveContainerIDByPod(ctx context.Context, pod, ns, ctr string) (string, error) {
	return "", errors.New("pod lookup should not be used")
}

func (r *restoreFakeRuntime) ResolveContainerByPod(ctx context.Context, pod, ns, ctr string) (int, *specs.Spec, error) {
	r.resolveByPodHit = true
	return 0, nil, errors.New("pod lookup should not be used")
}

func (r *restoreFakeRuntime) Close() error { return nil }

func TestExecNSRestoreRejectsRelativeContainerCheckpointLocation(t *testing.T) {
	_, err := execNSRestore(
		context.Background(),
		testr.New(t),
		RestoreRequest{
			ContainerCheckpointLocation: "relative/checkpoint",
			NSRestorePath:               "/usr/local/bin/nsrestore",
		},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/host/checkpoints/abc123",
			PlaceholderPID: 1,
		},
	)
	if err == nil {
		t.Fatal("expected relative container checkpoint location to be rejected")
	}
	if !strings.Contains(err.Error(), "absolute") {
		t.Fatalf("expected absolute-path validation error, got: %v", err)
	}
}

func TestInspectRestoreUsesContainerIDWhenProvided(t *testing.T) {
	checkpointDir := t.TempDir()
	manifest := types.NewCheckpointManifest(
		"checkpoint-123",
		types.CRIUDumpManifest{},
		types.NewSourcePodManifest("source-id", 456, "node-1", "source-pod", "default", "10.0.0.11", nil),
		types.OverlayManifest{},
	)
	if err := types.WriteManifest(checkpointDir, manifest); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}

	rt := &restoreFakeRuntime{}
	_, err := inspectRestore(
		context.Background(),
		rt,
		testr.New(t),
		RestoreRequest{
			CheckpointID:       "checkpoint-123",
			CheckpointLocation: checkpointDir,
			ContainerID:        "placeholder-id",
			PodName:            "virtual-pod-name",
			PodNamespace:       "default",
			ContainerName:      "main",
		},
	)
	if err != nil {
		t.Fatalf("inspectRestore: %v", err)
	}
	if rt.resolvedID != "placeholder-id" {
		t.Fatalf("ResolveContainer called with %q, want placeholder-id", rt.resolvedID)
	}
	if rt.resolveByPodHit {
		t.Fatal("ResolveContainerByPod should not be used when ContainerID is provided")
	}
}

func TestReadTargetGPUUUIDOrderWaitsForCapablePlaceholder(t *testing.T) {
	calls := 0
	got, err := readTargetGPUUUIDOrder(
		context.Background(),
		123,
		true,
		func(pid int) ([]byte, error) {
			if pid != 123 {
				t.Fatalf("PID = %d, want 123", pid)
			}
			calls++
			if calls == 1 {
				return nil, os.ErrNotExist
			}
			return []byte("GPU-aaaaaaaa-1111-2222-3333-444444444444\n"), nil
		},
	)
	if err != nil {
		t.Fatalf("readTargetGPUUUIDOrder: %v", err)
	}
	if calls != 2 {
		t.Fatalf("read calls = %d, want 2", calls)
	}
	if string(got) != "GPU-aaaaaaaa-1111-2222-3333-444444444444\n" {
		t.Fatalf("unexpected GPU order %q", got)
	}
}

func TestReadTargetGPUUUIDOrderKeepsLegacyFallbackImmediate(t *testing.T) {
	calls := 0
	_, err := readTargetGPUUUIDOrder(
		context.Background(),
		123,
		false,
		func(int) ([]byte, error) {
			calls++
			return nil, os.ErrNotExist
		},
	)
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("error = %v, want os.ErrNotExist", err)
	}
	if calls != 1 {
		t.Fatalf("read calls = %d, want 1", calls)
	}
}

func TestHasProcessEnv(t *testing.T) {
	spec := &specs.Spec{
		Process: &specs.Process{
			Env: []string{"OTHER=value", "DYN_SNAPSHOT_GPU_ORDER_HANDSHAKE=1"},
		},
	}
	if !hasProcessEnv(spec, "DYN_SNAPSHOT_GPU_ORDER_HANDSHAKE", "1") {
		t.Fatal("expected capability env to be found")
	}
	if hasProcessEnv(spec, "DYN_SNAPSHOT_GPU_ORDER_HANDSHAKE", "0") {
		t.Fatal("unexpected capability value match")
	}
}
