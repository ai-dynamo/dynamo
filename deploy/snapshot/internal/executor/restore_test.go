package executor

import (
	"context"
	"errors"
	"os"
	"path/filepath"
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

func TestExecNSRestoreRejectsInvalidCUDATransferSettings(t *testing.T) {
	_, err := execNSRestore(
		context.Background(),
		testr.New(t),
		RestoreRequest{
			ContainerCheckpointLocation: "/checkpoints/example",
			NSRestorePath:               "/usr/local/bin/nsrestore",
			CUDATransfer: types.CUDATransferSettings{
				BufferCount: 9,
				ChunkBytes:  types.DefaultCUDATransferChunkBytes,
			},
		},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/host/checkpoints/example",
			PlaceholderPID: 1,
		},
	)
	if err == nil || !strings.Contains(err.Error(), "invalid CUDA transfer settings") {
		t.Fatalf("execNSRestore() error = %v, want transfer validation error", err)
	}
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

func TestRestoreInNamespaceRejectsUnknownCUDAStorageModeBeforeRootfs(t *testing.T) {
	checkpointDir := t.TempDir()
	manifest := types.NewCheckpointManifest(
		"checkpoint-unknown-storage",
		types.CRIUDumpManifest{
			ExtMnt: map[string]string{"/": "/"},
		},
		types.NewSourcePodManifest("source-id", 456, "node-1", "source-pod", "default", "", nil),
		types.OverlayManifest{},
	)
	manifest.CUDA = types.CUDAManifest{
		PIDs:           []int{456},
		SourceGPUUUIDs: []string{"GPU-aaa"},
		StorageMode:    "object-store",
	}
	if err := types.WriteManifest(checkpointDir, manifest); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}

	// If validation moves after rootfs application, this invalid archive makes
	// that regression fail with a rootfs error instead of doing privileged work.
	if err := os.WriteFile(filepath.Join(checkpointDir, "rootfs-diff.tar"), []byte("not a tar archive"), 0o600); err != nil {
		t.Fatalf("WriteFile(rootfs-diff.tar): %v", err)
	}

	_, err := RestoreInNamespace(
		context.Background(),
		RestoreOptions{CheckpointPath: checkpointDir},
		testr.New(t),
	)
	if err == nil {
		t.Fatal("RestoreInNamespace() error = nil, want unsupported CUDA storage mode")
	}
	if !strings.Contains(err.Error(), "invalid CUDA artifact metadata") ||
		!strings.Contains(err.Error(), "unsupported CUDA artifact storage mode") {
		t.Fatalf("RestoreInNamespace() error = %v, want early CUDA metadata validation", err)
	}
}
