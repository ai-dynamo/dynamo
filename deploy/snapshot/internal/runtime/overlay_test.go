package runtime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestBuildExclusions(t *testing.T) {
	got := buildExclusions(
		types.OverlaySettings{Exclusions: []string{
			"/proc",
			"*/.cache/huggingface",
			"*/__pycache__",
			"*.pyc",
		}},
		[]string{"/mounted[ab]"},
		[]string{".wh.removed"},
	)
	want := []string{
		"proc",
		".cache/huggingface", "... .cache/huggingface",
		"__pycache__", "... __pycache__",
		"*.pyc", "... *.pyc",
		`\m\o\u\n\t\e\d\[\a\b\]`,
		`\.\w\h\.\r\e\m\o\v\e\d`,
	}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("buildExclusions() = %#v, want %#v", got, want)
	}
}

func TestCaptureRootfsDiffIsMandatory(t *testing.T) {
	upper := t.TempDir()
	checkpoint := t.TempDir()
	t.Setenv("PATH", t.TempDir())
	if _, err := CaptureRootfsDiff(
		context.Background(),
		upper,
		checkpoint,
		types.OverlaySettings{},
		nil,
	); err == nil {
		t.Fatal("expected missing mksquashfs to fail capture")
	}
	if _, err := os.Stat(filepath.Join(checkpoint, RootfsDiffFilename)); !os.IsNotExist(err) {
		t.Fatalf("failed capture left a rootfs artifact: %v", err)
	}
}

func TestRunMksquashfsExclusions(t *testing.T) {
	requireTool(t, "mksquashfs")
	requireTool(t, "unsquashfs")
	source := t.TempDir()
	for _, path := range []string{
		".cache/huggingface",
		"__pycache__",
		"nested/.cache/huggingface",
		"nested/__pycache__",
		"literal[ab]",
		"literala",
	} {
		if err := os.MkdirAll(filepath.Join(source, path), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{
		".cache/huggingface/root",
		"__pycache__/root",
		"nested/.cache/huggingface/child",
		"nested/__pycache__/child",
		"top.pyc",
		"nested/child.pyc",
		"nested/keep",
		"literal[ab]/excluded",
		"literala/decoy",
	} {
		if err := os.WriteFile(filepath.Join(source, path), nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	image := filepath.Join(t.TempDir(), RootfsDiffFilename)
	exclusions := buildExclusions(types.OverlaySettings{Exclusions: []string{
		"*/.cache/huggingface",
		"*/__pycache__",
		"*.pyc",
	}}, []string{"/literal[ab]"}, nil)
	if err := runMksquashfs(context.Background(), source, image, exclusions); err != nil {
		t.Fatal(err)
	}
	output, err := exec.Command("unsquashfs", "-ll", image).CombinedOutput()
	if err != nil {
		t.Fatalf("unsquashfs: %v\n%s", err, output)
	}
	listing := string(output)
	if strings.Contains(listing, ".cache/huggingface") ||
		strings.Contains(listing, "__pycache__") ||
		strings.Contains(listing, ".pyc") ||
		strings.Contains(listing, "literal[ab]") ||
		!strings.Contains(listing, "nested/keep") ||
		!strings.Contains(listing, "literala/decoy") {
		t.Fatalf("unexpected SquashFS listing:\n%s", listing)
	}
}

func TestRunMksquashfsCancellationRemovesOutput(t *testing.T) {
	binDir := t.TempDir()
	mksquashfs := filepath.Join(binDir, "mksquashfs")
	if err := os.WriteFile(
		mksquashfs,
		[]byte("#!/bin/sh\n: > \"$2\"\n: > \"$FAKE_MKSQUASHFS_STARTED\"\nexec sleep 60\n"),
		0o755,
	); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	started := filepath.Join(t.TempDir(), "started")
	t.Setenv("FAKE_MKSQUASHFS_STARTED", started)
	output := filepath.Join(t.TempDir(), RootfsDiffFilename)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := runMksquashfs(ctx, t.TempDir(), output, nil); err == nil {
		t.Fatal("expected cancellation to fail mksquashfs")
	}
	if _, err := os.Stat(started); err != nil {
		t.Fatalf("fake mksquashfs did not start: %v", err)
	}
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatalf("cancelled mksquashfs left output: %v", err)
	}
}

func TestScanOverlayUpperHandlesOpaqueXAndY(t *testing.T) {
	upper := t.TempDir()
	for _, path := range []string{"opaque-x", "opaque-y", "nested"} {
		if err := os.Mkdir(filepath.Join(upper, path), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := unix.Lsetxattr(
		filepath.Join(upper, "opaque-x"),
		"user.overlay.opaque",
		[]byte("x"),
		0,
	); err != nil {
		t.Skipf("overlay xattr not supported by test filesystem: %v", err)
	}
	if err := unix.Lsetxattr(
		filepath.Join(upper, "opaque-y"),
		"user.overlay.opaque",
		[]byte("y"),
		0,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(upper, "nested", ".wh.removed"), nil, 0o600); err != nil {
		t.Fatal(err)
	}

	deletions, excluded, err := scanOverlayUpper(upper)
	if err != nil {
		t.Fatal(err)
	}
	if len(deletions.OpaqueDirectory) != 1 ||
		deletions.OpaqueDirectory[0] != "opaque-y" {
		t.Fatalf("opaque directories = %#v, want [opaque-y]", deletions.OpaqueDirectory)
	}
	if len(deletions.Whiteouts) != 1 ||
		deletions.Whiteouts[0] != "nested/removed" {
		t.Fatalf("whiteouts = %#v, want [nested/removed]", deletions.Whiteouts)
	}
	if len(excluded) != 1 || excluded[0] != "nested/.wh.removed" {
		t.Fatalf("excluded overlay entries = %#v", excluded)
	}
}

func TestOpenAndValidateRootfsUsesManifestDigestAndNoFollow(t *testing.T) {
	requireTool(t, "mksquashfs")
	source := t.TempDir()
	if err := os.WriteFile(filepath.Join(source, "file"), []byte("rootfs"), 0o600); err != nil {
		t.Fatal(err)
	}
	imagePath := filepath.Join(t.TempDir(), RootfsDiffFilename)
	if err := runMksquashfs(context.Background(), source, imagePath, nil); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(content)
	digest := hex.EncodeToString(sum[:])
	image, got, err := openAndValidateRootfs(imagePath, digest)
	if err != nil {
		t.Fatal(err)
	}
	image.Close()
	if got != digest {
		t.Fatalf("digest = %s, want %s", got, digest)
	}
	if _, _, err := openAndValidateRootfs(imagePath, strings.Repeat("1", 64)); err == nil {
		t.Fatal("expected digest mismatch")
	}

	link := filepath.Join(t.TempDir(), RootfsDiffFilename)
	if err := os.Symlink(imagePath, link); err != nil {
		t.Fatal(err)
	}
	if _, _, err := openAndValidateRootfs(link, digest); err == nil {
		t.Fatal("expected final-component symlink to be rejected")
	}
}

func TestReadDeletedFilesRejectsTraversal(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), deletedFilesFilename)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if _, err := file.WriteString(`{"whiteouts":["../escape"]}`); err != nil {
		t.Fatal(err)
	}
	if _, err := readDeletedFilesFile(file); err == nil {
		t.Fatal("expected traversal to be rejected")
	}
}

func TestValidateRelativeArtifactPathRejectsParent(t *testing.T) {
	if err := validateRelativeArtifactPath(".."); err == nil {
		t.Fatal("expected bare parent path to be rejected")
	}
}

func TestDeletedFilesSidecarRoundTrip(t *testing.T) {
	checkpoint := t.TempDir()
	if err := writeDeletedFiles(
		checkpoint,
		&deletedFiles{Whiteouts: []string{"removed"}},
	); err != nil {
		t.Fatal(err)
	}
	file, err := OpenDeletedFiles(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	files, err := readDeletedFilesFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if len(files.Whiteouts) != 1 || files.Whiteouts[0] != "removed" {
		t.Fatalf("whiteouts = %#v, want [removed]", files.Whiteouts)
	}
}

func TestDeletedFilesSidecarNonregularBoundaries(t *testing.T) {
	checkpoint := t.TempDir()
	path := filepath.Join(checkpoint, deletedFilesFilename)
	if err := unix.Mkfifo(path, 0o600); err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	go func() {
		file, err := OpenDeletedFiles(checkpoint)
		if file != nil {
			file.Close()
		}
		result <- err
	}()
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("expected FIFO sidecar to be rejected")
		}
	case <-time.After(time.Second):
		t.Fatal("opening FIFO sidecar blocked")
	}

	fd, err := unix.Open(path, unix.O_RDWR|unix.O_NONBLOCK|unix.O_CLOEXEC, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Close(fd)
	if _, err := readDeletedFilesFD(fd); err == nil {
		t.Fatal("expected inherited FIFO FD to be rejected")
	}
}

func TestOpenDeletedFilesRejectsOversizedSparseFile(t *testing.T) {
	checkpoint := t.TempDir()
	path := filepath.Join(checkpoint, deletedFilesFilename)
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(maxDeletedFilesSidecarSize + 1); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if file, err := OpenDeletedFiles(checkpoint); err == nil {
		file.Close()
		t.Fatal("expected oversized deletion sidecar to be rejected")
	}
}

func requireTool(t *testing.T, name string) {
	t.Helper()
	if _, err := exec.LookPath(name); err != nil {
		t.Skipf("%s is not installed", name)
	}
}
