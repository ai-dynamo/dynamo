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

func TestExecNSRestorePassesExplicitInetRemap(t *testing.T) {
	binDir := t.TempDir()
	argsPath := filepath.Join(t.TempDir(), "args")
	nsenterPath := filepath.Join(binDir, "nsenter")
	script := `#!/bin/sh
printf '%s\n' "$@" > "$NSENTER_ARGS_FILE"
printf '{"restoredPID":321}\n'
`
	if err := os.WriteFile(nsenterPath, []byte(script), 0o755); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("NSENTER_ARGS_FILE", argsPath)

	nsrestorePath := filepath.Join(t.TempDir(), "nsrestore")
	if err := os.WriteFile(nsrestorePath, []byte("new nsrestore"), 0o755); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	explicit := "10.0.0.11 10.1.4.21\n10.0.0.12 10.1.4.22"
	result, err := execNSRestore(
		context.Background(),
		testr.New(t),
		RestoreRequest{
			NSRestorePath: nsrestorePath,
			TargetPodIP:   "10.1.4.21",
			InetRemap:     explicit,
		},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/host/checkpoints/abc123",
			PlaceholderPID: 123,
		},
	)
	if err != nil {
		t.Fatalf("execNSRestore: %v", err)
	}
	if result.RestoredPID != 321 {
		t.Fatalf("RestoredPID = %d, want 321", result.RestoredPID)
	}
	args, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !strings.Contains(string(args), "--inet-remap\n"+explicit+"\n") {
		t.Fatalf("nsrestore args do not contain explicit remap: %q", args)
	}
	if !strings.Contains(string(args), "--\n/proc/self/fd/3\n") {
		t.Fatalf("nsrestore did not use the inherited agent binary: %q", args)
	}
}

func TestExecNSRestoreBlankInetRemapUsesPlaceholderBinary(t *testing.T) {
	binDir := t.TempDir()
	argsPath := filepath.Join(t.TempDir(), "args")
	nsenterPath := filepath.Join(binDir, "nsenter")
	script := `#!/bin/sh
printf '%s\n' "$@" > "$NSENTER_ARGS_FILE"
printf '{"restoredPID":321}\n'
`
	if err := os.WriteFile(nsenterPath, []byte(script), 0o755); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("NSENTER_ARGS_FILE", argsPath)

	_, err := execNSRestore(
		context.Background(),
		testr.New(t),
		RestoreRequest{
			NSRestorePath: "/usr/local/bin/nsrestore",
			TargetPodIP:   "10.1.4.21",
			InetRemap:     " \n\t",
		},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/host/checkpoints/abc123",
			PlaceholderPID: 123,
		},
	)
	if err != nil {
		t.Fatalf("execNSRestore: %v", err)
	}
	args, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !strings.Contains(string(args), "--\n/usr/local/bin/nsrestore\n") {
		t.Fatalf("nsrestore did not use the placeholder binary: %q", args)
	}
	if strings.Contains(string(args), "--inet-remap") {
		t.Fatalf("blank explicit remap was forwarded: %q", args)
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
