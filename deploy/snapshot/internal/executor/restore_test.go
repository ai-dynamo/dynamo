package executor

import (
	"context"
	"errors"
	"slices"
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

func TestNSRestoreArgsIncludeRootFSWorkers(t *testing.T) {
	args, err := nsRestoreArgs(
		RestoreRequest{
			NSRestorePath: "/usr/local/bin/nsrestore",
			RootFSWorkers: 23,
			TargetPodIP:   "10.0.0.1",
		},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/checkpoints/checkpoint",
			PlaceholderPID: 42,
		},
	)
	if err != nil {
		t.Fatalf("nsRestoreArgs: %v", err)
	}
	index := slices.Index(args, "--rootfs-workers")
	if index < 0 || index+1 >= len(args) || args[index+1] != "23" {
		t.Fatalf("--rootfs-workers not propagated in %v", args)
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
