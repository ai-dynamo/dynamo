// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux && integration

// Package nsintegration exercises the real ns-bind-mount helper against a real
// mount namespace. It is the only layer that can demonstrate the kernel-enforced
// properties the design relies on: the unit tests stop at the process boundary
// and can only show that the right argv was produced.
//
// Run as root on Linux:
//
//	go test -tags=integration ./internal/nsmount/nsintegration/
package nsintegration

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsmount"
)

// Destinations are compile-time constants on both sides of the exec boundary:
// #define in the helper, const in Go. Taking them from Go here means every
// mount assertion below doubles as a drift check — if the two ever disagree the
// mount lands where the helper says and these tests look where the agent says,
// so they fail instead of the agent silently tracking the wrong path.
const (
	artifactDst = nsmount.CheckpointDst
	bundleDst   = nsmount.SnapshotBinDst
	bundleSrc   = nsmount.SnapshotBinSrc
)

// helperPath is the compiled ns-bind-mount binary, set by TestMain.
var helperPath string

func TestMain(m *testing.M) {
	// Not being root is an environment fact, so skipping is honest. A missing
	// compiler is a broken setup: skipping there would let this suite report
	// green forever on an image without gcc, which is the exact failure mode
	// this layer exists to prevent.
	if os.Geteuid() != 0 {
		fmt.Fprintln(os.Stderr, "nsintegration: skipping, must run as root")
		os.Exit(0)
	}
	compiler, err := exec.LookPath("gcc")
	if err != nil {
		fmt.Fprintln(os.Stderr, "nsintegration: gcc is required to build ns-bind-mount; this is a broken environment, not a skip")
		os.Exit(1)
	}
	for _, tool := range []string{"unshare", "nsenter"} {
		if _, err := exec.LookPath(tool); err != nil {
			fmt.Fprintf(os.Stderr, "nsintegration: %s is required to drive a foreign mount namespace\n", tool)
			os.Exit(1)
		}
	}

	dir, err := os.MkdirTemp("", "ns-bind-mount-build")
	if err != nil {
		fmt.Fprintln(os.Stderr, "nsintegration: mkdtemp:", err)
		os.Exit(1)
	}
	helperPath = filepath.Join(dir, "ns-bind-mount")
	build := exec.Command(compiler, "-Wall", "-Wextra", "-O2", "-o", helperPath, "../../../cmd/ns-bind-mount/main.c")
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "nsintegration: compiling ns-bind-mount failed: %v\n%s\n", err, out)
		os.Exit(1)
	}

	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// target is a live process owning a private mount namespace. The test process
// cannot enter that namespace itself — Go's multithreaded runtime cannot call
// setns(CLONE_NEWNS), which is the whole reason the C helper exists — so all
// assertions are made either by reading the target's mountinfo from outside or
// by running a probe through nsenter.
type target struct {
	t   *testing.T
	pid int
}

func newTarget(t *testing.T) *target {
	t.Helper()

	// Private propagation keeps the target's mounts from reaching the host, and
	// a fresh tmpfs on /tmp keeps each target's destinations independent of
	// every other test's leftovers.
	cmd := exec.Command("unshare", "--mount", "--propagation", "private",
		"sh", "-c", "mount -t tmpfs tmpfs /tmp && exec sleep 600")
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
	require.NoError(t, cmd.Start())

	tg := &target{t: t, pid: cmd.Process.Pid}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})

	// /proc/<pid>/ns/mnt exists from the moment the process does, and until
	// unshare(2) actually runs it still names the *caller's* namespace. Pinning
	// it that early would silently mount on the host, so wait for the inode to
	// differ from our own rather than merely for the path to appear.
	self := namespaceInode(t, "/proc/self/ns/mnt")
	require.Eventually(t, func() bool {
		ino, err := namespaceInodeErr(fmt.Sprintf("/proc/%d/ns/mnt", tg.pid))
		return err == nil && ino != self
	}, 5*time.Second, 5*time.Millisecond, "target never entered its own mount namespace")

	// The namespace exists before the tmpfs inside it does. Mounting during that
	// window puts the artifact on the underlying /tmp and the tmpfs then lands
	// on top and hides it — the mount reports success and the probe reads
	// nothing. Wait for the tmpfs itself, not just the namespace.
	require.Eventually(t, func() bool {
		return strings.Contains(tg.mountEntry("/tmp"), " tmpfs ")
	}, 5*time.Second, 5*time.Millisecond, "target never mounted its private /tmp")

	return tg
}

func namespaceInode(t *testing.T, path string) uint64 {
	t.Helper()
	ino, err := namespaceInodeErr(path)
	require.NoError(t, err)
	return ino
}

func namespaceInodeErr(path string) (uint64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, fmt.Errorf("unexpected stat type for %s", path)
	}
	return st.Ino, nil
}

// nsFD pins the target's mount namespace, mirroring what the agent does before
// invoking the helper.
func (tg *target) nsFD() *os.File {
	tg.t.Helper()
	fd, err := os.Open(fmt.Sprintf("/proc/%d/ns/mnt", tg.pid))
	require.NoError(tg.t, err)
	tg.t.Cleanup(func() { _ = fd.Close() })
	return fd
}

// run executes the helper with the given subcommand arguments, passing the
// namespace descriptor at fd 3 and any source descriptor at fd 4.
func (tg *target) run(extraFiles []*os.File, args ...string) (string, error) {
	tg.t.Helper()
	cmd := exec.Command(helperPath, args...)
	cmd.ExtraFiles = extraFiles
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// mount runs the helper's generic mount with the given attribute tokens.
func (tg *target) mount(src, dst string, attrs ...string) (string, error) {
	tg.t.Helper()
	args := append([]string{"mount-fd", "3", src, dst}, attrs...)
	return tg.run([]*os.File{tg.nsFD()}, args...)
}

// mountArtifact applies the artifact policy the agent uses: read-only, noexec.
func (tg *target) mountArtifact(srcDir string) (string, error) {
	tg.t.Helper()
	return tg.mount(srcDir, artifactDst, "ro", "nosuid", "nodev", "noexec")
}

// unmountArtifact applies the artifact's strict unmount.
func (tg *target) unmountArtifact(extra ...string) (string, error) {
	tg.t.Helper()
	args := append([]string{"umount-fd", "3", artifactDst, "strict"}, extra...)
	return tg.run([]*os.File{tg.nsFD()}, args...)
}

// probe runs a shell command inside the target's mount namespace and reports
// whether it succeeded. Enforcement claims (noexec, read-only) can only be
// observed this way: mountinfo says what the kernel was told, a probe says what
// the kernel does.
func (tg *target) probe(script string) (string, error) {
	tg.t.Helper()
	out, err := exec.Command("nsenter", "--mount", "--target", strconv.Itoa(tg.pid),
		"sh", "-c", script).CombinedOutput()
	return string(out), err
}

// mountEntry returns the mountinfo line for mountPoint inside the target, or ""
// if nothing is mounted there.
func (tg *target) mountEntry(mountPoint string) string {
	tg.t.Helper()
	f, err := os.Open(fmt.Sprintf("/proc/%d/mountinfo", tg.pid))
	require.NoError(tg.t, err)
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		// Field 5 (1-indexed) is the mount point.
		if len(fields) >= 5 && fields[4] == mountPoint {
			return scanner.Text()
		}
	}
	require.NoError(tg.t, scanner.Err())
	return ""
}

// artifactSource builds a directory shaped like a checkpoint artifact, with an
// executable in it so exec can be probed.
func artifactSource(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "manifest.yaml"), []byte("checkpointId: abc\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "payload"), []byte("#!/bin/sh\necho ran\n"), 0o755))
	return dir
}

func TestArtifactMount_IsReadOnlyAndNoexec(t *testing.T) {
	tg := newTarget(t)
	src := artifactSource(t)

	out, err := tg.mountArtifact(src)
	require.NoError(t, err, "artifact mount failed: %s", out)

	entry := tg.mountEntry(artifactDst)
	require.NotEmpty(t, entry, "artifact is not mounted in the target namespace")
	for _, opt := range []string{"ro", "noexec", "nosuid", "nodev"} {
		assert.Contains(t, entry, opt, "mountinfo is missing %s: %s", opt, entry)
	}

	// The checkpoint is readable, which is the point of mounting it at all.
	content, err := tg.probe("cat " + artifactDst + "/manifest.yaml")
	require.NoError(t, err, "checkpoint must be readable: %s", content)
	assert.Contains(t, content, "checkpointId")

	// Writes are refused by the kernel, not merely unrequested.
	out, err = tg.probe("echo tampered > " + artifactDst + "/manifest.yaml")
	assert.Error(t, err, "write through a read-only mount must fail, got: %s", out)

	// Execution is refused. A checkpoint is data the agent brought in from
	// shared storage; letting the container run it would turn the mount into a
	// code-injection channel.
	out, err = tg.probe(artifactDst + "/payload")
	assert.Error(t, err, "exec through a noexec mount must fail, got: %s", out)
}

func TestArtifactMount_IsInvisibleOutsideTheTarget(t *testing.T) {
	tg := newTarget(t)

	out, err := tg.mountArtifact(artifactSource(t))
	require.NoError(t, err, out)

	// The mount lives in the target's namespace only. If it showed up on the
	// host, every container on the node would inherit another pod's checkpoint.
	hostEntry, err := os.ReadFile("/proc/self/mountinfo")
	require.NoError(t, err)
	for _, line := range strings.Split(string(hostEntry), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 5 {
			assert.NotEqual(t, artifactDst, fields[4], "artifact mount leaked into the host namespace")
		}
	}
}

func TestArtifactMount_ExposesOnlyTheSelectedDirectory(t *testing.T) {
	tg := newTarget(t)
	parent := t.TempDir()
	selected := filepath.Join(parent, "selected-version")
	require.NoError(t, os.Mkdir(selected, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(selected, "selected"), []byte("visible"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(parent, "sibling-version"), []byte("secret"), 0o644))

	out, err := tg.mountArtifact(selected)
	require.NoError(t, err, out)

	out, err = tg.probe("cat " + artifactDst + "/selected")
	require.NoError(t, err, "selected artifact is not readable: %s", out)
	assert.Equal(t, "visible", strings.TrimSpace(out))

	// `..` is resolved in the target namespace. It reaches /tmp, not the
	// source directory's parent, so sibling versions and checkpoints remain
	// unreachable even when their names are known.
	out, err = tg.probe("cat " + artifactDst + "/../sibling-version")
	assert.Error(t, err, "source sibling escaped through the mount parent: %s", out)
}

func TestArtifactMount_CloneIsNotRecursive(t *testing.T) {
	tg := newTarget(t)
	src := artifactSource(t)

	// A submount under the source stands in for anything the shared checkpoint
	// filesystem might have nested beneath the artifact directory.
	sub := filepath.Join(src, "nested")
	require.NoError(t, os.Mkdir(sub, 0o755))
	require.NoError(t, unix.Mount("tmpfs", sub, "tmpfs", 0, ""))
	t.Cleanup(func() { _ = unix.Unmount(sub, unix.MNT_DETACH) })
	require.NoError(t, os.WriteFile(filepath.Join(sub, "secret"), []byte("hidden"), 0o644))

	out, err := tg.mountArtifact(src)
	require.NoError(t, err, out)

	// OPEN_TREE_CLONE without AT_RECURSIVE copies only the top mount, so the
	// submount's contents must not follow it into the container.
	out, err = tg.probe("cat " + artifactDst + "/nested/secret")
	assert.Error(t, err, "a non-recursive clone must not carry submounts, got: %s", out)
}

func TestResolveArtifact_RejectsNestedMounts(t *testing.T) {
	basePath := t.TempDir()
	artifactPath := filepath.Join(basePath, "checkpoint-123", "versions", "1")
	nested := filepath.Join(artifactPath, "nested")
	require.NoError(t, os.MkdirAll(nested, 0o755))
	require.NoError(t, unix.Mount("tmpfs", nested, "tmpfs", 0, ""))
	t.Cleanup(func() { _ = unix.Unmount(nested, unix.MNT_DETACH) })

	_, err := nsmount.ResolveArtifact(basePath, "checkpoint-123", "1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nested mount")
}

func TestArtifactMount_RejectsUnsafeDestinations(t *testing.T) {
	tests := []struct {
		name    string
		prepare string // shell run inside the target before mounting
		wantMsg string
	}{
		{
			// Content already at the destination would be shadowed by the mount
			// and reappear on unmount, so the helper refuses rather than hide it.
			name:    "non-empty destination",
			prepare: "mkdir -p " + artifactDst + " && touch " + artifactDst + "/squatter",
			wantMsg: "not empty",
		},
		{
			name:    "destination is a regular file",
			prepare: "touch " + artifactDst,
			wantMsg: "is not a directory",
		},
		{
			// A symlinked destination is how a compromised container would
			// redirect a privileged mount somewhere it chose. The single
			// openat2 that opens the destination rejects it, so the mount is
			// never attempted against the link target.
			name:    "symlinked destination",
			prepare: "mkdir -p /tmp/elsewhere && ln -s /tmp/elsewhere " + artifactDst,
			wantMsg: "is a symlink",
		},
		{
			name:    "destination already carries a mount",
			prepare: "mkdir -p " + artifactDst + " && mount -t tmpfs tmpfs " + artifactDst,
			wantMsg: "already a mountpoint",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tg := newTarget(t)
			out, err := tg.probe(tc.prepare)
			require.NoError(t, err, "preparing the destination failed: %s", out)

			out, err = tg.mountArtifact(artifactSource(t))
			require.Error(t, err, "mount should have been refused, got: %s", out)
			assert.Contains(t, out, tc.wantMsg)
		})
	}
}

func TestArtifactMount_DestinationOwnershipControlsRemoval(t *testing.T) {
	t.Run("existing empty directory is retained", func(t *testing.T) {
		tg := newTarget(t)
		out, err := tg.probe("mkdir -p " + artifactDst)
		require.NoError(t, err, out)

		out, err = tg.mountArtifact(artifactSource(t))
		require.NoError(t, err, out)
		assert.Contains(t, out, "created_dst=0")
		out, err = tg.unmountArtifact()
		require.NoError(t, err, out)
		out, err = tg.probe("test -d " + artifactDst)
		require.NoError(t, err, "pre-existing destination was removed: %s", out)
	})

	t.Run("helper-created directory is removed", func(t *testing.T) {
		tg := newTarget(t)
		out, err := tg.mountArtifact(artifactSource(t))
		require.NoError(t, err, out)
		assert.Contains(t, out, "created_dst=1")
		out, err = tg.unmountArtifact("created")
		require.NoError(t, err, out)
		out, err = tg.probe("test ! -e " + artifactDst)
		require.NoError(t, err, "helper-created destination remains: %s", out)
	})
}

func TestArtifactMount_WorksWithoutProcInTheTarget(t *testing.T) {
	tg := newTarget(t)
	// A minimal container need not mount /proc. The helper used to answer
	// "is the destination already a mountpoint?" by reading
	// /proc/self/mountinfo, which fails closed there and so refused every
	// mount into such a container. The destination descriptor answers the same
	// question with no /proc dependency at all.
	out, err := tg.probe("umount -l /proc 2>/dev/null; test ! -e /proc/self/mountinfo")
	require.NoError(t, err, "could not remove /proc from the target: %s", out)

	out, err = tg.mountArtifact(artifactSource(t))
	require.NoError(t, err, "mount must not depend on /proc inside the target: %s", out)
	assert.NotEmpty(t, tg.mountEntry(artifactDst))
}

func TestArtifactUnmount_RefusesASymlinkedDestination(t *testing.T) {
	tg := newTarget(t)
	out, err := tg.mountArtifact(artifactSource(t))
	require.NoError(t, err, out)

	// Replace the leaf with a symlink to a different mount. Without
	// UMOUNT_NOFOLLOW the helper would resolve the link and unmount whatever it
	// points at, which is the target's to choose.
	out, err = tg.probe("umount " + artifactDst +
		" && rmdir " + artifactDst +
		" && mkdir -p /tmp/decoy && mount -t tmpfs tmpfs /tmp/decoy" +
		" && ln -s /tmp/decoy " + artifactDst)
	require.NoError(t, err, "could not stage the swapped destination: %s", out)

	out, err = tg.unmountArtifact()
	require.Error(t, err, "unmount followed a symlinked destination: %s", out)
	assert.NotEmpty(t, tg.mountEntry("/tmp/decoy"),
		"the decoy mount was unmounted through the symlink")
}

func TestArtifactUnmount_IsStrictAndSurfacesBusy(t *testing.T) {
	tg := newTarget(t)

	out, err := tg.mountArtifact(artifactSource(t))
	require.NoError(t, err, out)

	// Hold a file open inside the target so the strict unmount cannot succeed.
	// The holder announces itself on the container root filesystem, which the
	// test process shares; racing the open would let the unmount succeed and
	// quietly turn this into a test of nothing.
	// Not under /tmp: the target replaced /tmp with its own tmpfs, so a marker
	// written there would be invisible to this process.
	readyDir, err := os.MkdirTemp("/run", "nsintegration")
	require.NoError(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(readyDir) })
	ready := filepath.Join(readyDir, "holder-ready")
	holder := exec.Command("nsenter", "--mount", "--target", strconv.Itoa(tg.pid),
		// The final exec matters: without it sh forks for sleep, and killing sh
		// would leave a child still holding the descriptor.
		"sh", "-c", "exec 3< "+artifactDst+"/manifest.yaml && touch "+ready+" && exec sleep 60")
	require.NoError(t, holder.Start())
	defer func() {
		_ = holder.Process.Kill()
		_, _ = holder.Process.Wait()
	}()
	require.Eventually(t, func() bool {
		_, err := os.Stat(ready)
		return err == nil
	}, 5*time.Second, 10*time.Millisecond, "holder never opened a file under the mount")
	// Strict umount2 (flags=0) reports EBUSY instead of silently deferring.
	// That failure is what triggers destroying the container, so it must not be
	// swallowed the way the bundle's lazy detach is.
	out, err = tg.unmountArtifact()
	require.Error(t, err, "strict unmount should have failed while a file is open: %s", out)
	assert.Contains(t, strings.ToLower(out), "busy")
	assert.NotEmpty(t, tg.mountEntry(artifactDst), "the mount must still be present after a failed strict unmount")

	// Once the holder is gone the same call succeeds and the destination is
	// reclaimed, proving the failure was the open file and not a broken argv.
	_ = holder.Process.Kill()
	_, _ = holder.Process.Wait()
	require.Eventually(t, func() bool {
		_, err := tg.unmountArtifact("created")
		return err == nil && tg.mountEntry(artifactDst) == ""
	}, 5*time.Second, 100*time.Millisecond, "strict unmount never succeeded after the holder exited")
}

func TestBundleMount_IsExecutableAndDetachesLazily(t *testing.T) {
	// The bundle source is compile-time in the helper, so it has to exist on the
	// host for this case. Use a unique probe and never overwrite the real
	// nsrestore binary when the suite runs inside the agent image.
	createdBundleDir := false
	if _, err := os.Stat(bundleSrc); os.IsNotExist(err) {
		require.NoError(t, os.Mkdir(bundleSrc, 0o755))
		createdBundleDir = true
	} else {
		require.NoError(t, err)
	}
	probeName := fmt.Sprintf(".nsintegration-probe-%d", os.Getpid())
	probePath := filepath.Join(bundleSrc, probeName)
	probe, err := os.OpenFile(probePath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o755)
	require.NoError(t, err)
	_, err = probe.WriteString("#!/bin/sh\necho bundle-probe\n")
	require.NoError(t, err)
	require.NoError(t, probe.Close())
	t.Cleanup(func() {
		_ = os.Remove(probePath)
		if createdBundleDir {
			_ = os.Remove(bundleSrc)
		}
	})

	tg := newTarget(t)
	out, err := tg.mount(bundleSrc, bundleDst, "ro", "nosuid", "nodev")
	require.NoError(t, err, "bundle mount failed: %s", out)

	entry := tg.mountEntry(bundleDst)
	require.NotEmpty(t, entry, "bundle is not mounted in the target namespace")
	for _, opt := range []string{"ro", "nosuid", "nodev"} {
		assert.Contains(t, entry, opt, "mountinfo is missing %s: %s", opt, entry)
	}
	// The bundle exists to be run: unlike the artifact it must not be noexec.
	assert.NotContains(t, entry, "noexec")

	out, err = tg.probe(bundleDst + "/" + probeName)
	require.NoError(t, err, "the agent bundle must be executable inside the target: %s", out)
	assert.Contains(t, out, "bundle-probe")

	// The bundle detaches lazily so a still-running nsrestore cannot fail an
	// otherwise successful restore.
	out, err = tg.run([]*os.File{tg.nsFD()}, "umount-fd", "3", bundleDst, "created")
	require.NoError(t, err, out)
	assert.Empty(t, tg.mountEntry(bundleDst))
}

func TestHelper_RejectsUnknownOperations(t *testing.T) {
	tg := newTarget(t)

	tests := []struct {
		name string
		args []string
	}{
		{name: "unknown subcommand", args: []string{"mount-anything-fd", "3"}},
		{name: "mount without a destination", args: []string{"mount-fd", "3", "/some/src"}},
		{name: "unknown mount attribute", args: []string{"mount-fd", "3", "/some/src", "/tmp/x", "rw"}},
		{name: "misspelled mount attribute", args: []string{"mount-fd", "3", "/some/src", "/tmp/x", "noexecc"}},
		{name: "non-numeric descriptor", args: []string{"mount-fd", "three", "/some/src", "/tmp/x"}},
		{name: "empty descriptor", args: []string{"mount-fd", "", "/some/src", "/tmp/x"}},
		{name: "unmount without a destination", args: []string{"umount-fd", "3"}},
		{name: "unknown unmount flag", args: []string{"umount-fd", "3", "/tmp/x", "keep"}},
		{name: "no arguments at all", args: []string{"mount-fd"}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := tg.run([]*os.File{tg.nsFD()}, tc.args...)
			assert.Error(t, err, "helper accepted %v: %s", tc.args, out)
		})
	}
}

// TestMount_AttributesAreInputsNotAssumptions mounts the same source twice with
// different attributes and shows the difference reaching the kernel. This is the
// case that could not exist while the helper derived attributes from a
// destination-specific subcommand: there was no way to ask for anything else.
// It is also what catches an attr_clr that is zero rather than the complement,
// since a clone inherits its source mount's flags.
func TestMount_AttributesAreInputsNotAssumptions(t *testing.T) {
	src := artifactSource(t)

	t.Run("noexec requested, execution refused", func(t *testing.T) {
		tg := newTarget(t)
		out, err := tg.mount(src, artifactDst, "ro", "nosuid", "nodev", "noexec")
		require.NoError(t, err, out)

		assert.Contains(t, tg.mountEntry(artifactDst), "noexec")
		out, err = tg.probe(artifactDst + "/payload")
		assert.Error(t, err, "noexec was requested but execution succeeded: %s", out)
		out, err = tg.probe("echo tampered > " + artifactDst + "/manifest.yaml")
		assert.Error(t, err, "ro was requested but the write succeeded: %s", out)
	})

	t.Run("noexec not requested, execution allowed", func(t *testing.T) {
		tg := newTarget(t)
		out, err := tg.mount(src, artifactDst, "ro", "nosuid", "nodev")
		require.NoError(t, err, out)

		assert.NotContains(t, tg.mountEntry(artifactDst), "noexec")
		out, err = tg.probe(artifactDst + "/payload")
		require.NoError(t, err, "noexec was not requested but execution failed: %s", out)
		assert.Contains(t, out, "ran")
		// Read-only still applies: the two attributes move independently.
		out, err = tg.probe("echo tampered > " + artifactDst + "/manifest.yaml")
		assert.Error(t, err, "ro was requested but the write succeeded: %s", out)
	})
}

// TestMount_ClearsInheritedAttributes pins attr_clr. A cloned tree inherits its
// source mount's attributes, so an attribute the caller did not ask for has to
// be cleared explicitly or it survives into the target. That only shows up when
// the source mount itself carries the attribute, which is why this test puts the
// source on a noexec filesystem rather than the ordinary container root.
func TestMount_ClearsInheritedAttributes(t *testing.T) {
	tg := newTarget(t)

	// A noexec source. Execution is impossible here by construction.
	src := t.TempDir()
	require.NoError(t, unix.Mount("tmpfs", src, "tmpfs", unix.MS_NOEXEC, ""))
	t.Cleanup(func() { _ = unix.Unmount(src, unix.MNT_DETACH) })
	require.NoError(t, os.WriteFile(filepath.Join(src, "payload"), []byte("#!/bin/sh\necho ran\n"), 0o755))

	// Mount it without noexec. The clone starts out noexec by inheritance, so
	// the helper must clear what was not requested.
	out, err := tg.mount(src, artifactDst, "ro", "nosuid", "nodev")
	require.NoError(t, err, out)

	assert.NotContains(t, tg.mountEntry(artifactDst), "noexec",
		"noexec was inherited from the source and not cleared")
	out, err = tg.probe(artifactDst + "/payload")
	require.NoError(t, err, "inherited noexec survived into the target: %s", out)
	assert.Contains(t, out, "ran")
}
