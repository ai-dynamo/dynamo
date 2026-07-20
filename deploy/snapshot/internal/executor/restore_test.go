package executor

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/go-logr/logr/testr"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

type restoreFakeRuntime struct {
	resolvedID      string
	resolveByPodHit bool
}

func TestCanonicalizeCheckpointPathSurvivesNestedExtraFiles(t *testing.T) {
	if os.Getenv("DYNAMO_SNAPSHOT_CHECKPOINT_PATH_HELPER") == "1" {
		stable, err := os.ReadFile(os.Getenv("DYNAMO_SNAPSHOT_STABLE_PATH"))
		if err != nil {
			fmt.Fprintf(os.Stderr, "read stable checkpoint path: %v", err)
			os.Exit(1)
		}
		self, err := os.ReadFile(os.Getenv("DYNAMO_SNAPSHOT_SELF_PATH"))
		if err != nil {
			fmt.Fprintf(os.Stderr, "read child checkpoint path: %v", err)
			os.Exit(1)
		}
		fmt.Printf("%s|%s", stable, self)
		os.Exit(0)
	}

	checkpointDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(checkpointDir, "marker"), []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	checkpoint, err := os.Open(checkpointDir)
	if err != nil {
		t.Fatal(err)
	}
	defer checkpoint.Close()

	selfPath := fmt.Sprintf("/proc/self/fd/%d", checkpoint.Fd())
	stablePath := canonicalizeCheckpointPath(selfPath)
	wantStablePath := fmt.Sprintf("/proc/%d/fd/%d", os.Getpid(), checkpoint.Fd())
	if stablePath != wantStablePath {
		t.Fatalf("canonicalizeCheckpointPath(%q) = %q, want %q", selfPath, stablePath, wantStablePath)
	}
	if _, err := os.Stat(fmt.Sprintf("/proc/%d", os.Getpid())); err != nil {
		t.Fatalf("current PID is not visible in the active proc mount: %v", err)
	}

	childDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(childDir, "marker"), []byte("nested-child"), 0o600); err != nil {
		t.Fatal(err)
	}
	childCheckpoint, err := os.Open(childDir)
	if err != nil {
		t.Fatal(err)
	}
	defer childCheckpoint.Close()

	extraFileCount := max(3, int(checkpoint.Fd())-2)
	extraFiles := make([]*os.File, extraFileCount)
	for i := range extraFiles {
		extraFiles[i] = childCheckpoint
	}
	command := exec.Command(os.Args[0], "-test.run=^TestCanonicalizeCheckpointPathSurvivesNestedExtraFiles$")
	command.Env = append(
		os.Environ(),
		"DYNAMO_SNAPSHOT_CHECKPOINT_PATH_HELPER=1",
		"DYNAMO_SNAPSHOT_STABLE_PATH="+filepath.Join(stablePath, "marker"),
		"DYNAMO_SNAPSHOT_SELF_PATH="+filepath.Join(selfPath, "marker"),
	)
	command.ExtraFiles = extraFiles
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("nested child: %v\n%s", err, output)
	}
	if string(output) != "checkpoint|nested-child" {
		t.Fatalf("nested child paths resolved to %q, want %q", output, "checkpoint|nested-child")
	}
}

func TestNSenterTargetSyntaxPreservesInheritedFD(t *testing.T) {
	args := nsenterTargetArgs(os.Getpid())
	for _, arg := range args {
		if arg == "-r" || strings.HasPrefix(arg, "--root") ||
			strings.HasPrefix(arg, "--wd") {
			t.Fatalf("nsenter unexpectedly changes caller root/cwd: %q", arg)
		}
	}

	if os.Getenv("DYNAMO_SNAPSHOT_PRIVILEGED_TEST") != "1" || os.Geteuid() != 0 {
		t.Skip("privileged nsenter test requires root and DYNAMO_SNAPSHOT_PRIVILEGED_TEST=1")
	}
	file, err := os.CreateTemp(t.TempDir(), "nsenter-fd")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if _, err := file.WriteString("inherited"); err != nil {
		t.Fatal(err)
	}
	if _, err := file.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	rootfs, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer rootfs.Close()
	workspace, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer workspace.Close()
	args = append(args, "--", "cat", "/proc/self/fd/5")
	command := exec.Command("nsenter", args...)
	command.ExtraFiles = []*os.File{rootfs, workspace, file}
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("nsenter: %v\n%s", err, output)
	}
	if string(output) != "inherited" {
		t.Fatalf("inherited FD output = %q", output)
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

func TestExecNSRestoreRequiresInheritedFiles(t *testing.T) {
	rootfs, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer rootfs.Close()
	workspace, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer workspace.Close()
	_, err = execNSRestore(
		context.Background(),
		testr.New(t),
		RestoreRequest{NSRestorePath: "/usr/local/bin/nsrestore"},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/host/checkpoints/abc123",
			PlaceholderPID: 1,
		},
		rootfs,
		workspace,
		nil,
	)
	if err == nil {
		t.Fatal("expected missing inherited checkpoint directory to be rejected")
	}
	if !strings.Contains(err.Error(), "checkpoint") {
		t.Fatalf("expected inherited-file validation error, got: %v", err)
	}
}

func TestStderrTail(t *testing.T) {
	stderr := append([]byte("€"), bytes.Repeat([]byte("x"), nsRestoreStderrTailLimit-1)...)
	got := stderrTail(stderr)

	if len(got) > nsRestoreStderrTailLimit {
		t.Fatalf("stderrTail returned %d bytes, limit is %d", len(got), nsRestoreStderrTailLimit)
	}
	if !utf8.Valid(got) {
		t.Fatalf("stderrTail split a UTF-8 rune: %q", got)
	}
	if want := bytes.Repeat([]byte("x"), nsRestoreStderrTailLimit-1); !bytes.Equal(got, want) {
		t.Fatal("stderrTail did not preserve the final complete bytes")
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
	manifest.RootFSSHA256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
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
