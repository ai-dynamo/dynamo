package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/testr"
	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestBuildExclusions(t *testing.T) {
	tests := []struct {
		name     string
		settings types.OverlaySettings
		want     map[string]bool
	}{
		{
			name: "normalizes rooted paths",
			settings: types.OverlaySettings{
				Exclusions: []string{"/proc", "/sys", "/root/.cache", "/tmp"},
			},
			want: map[string]bool{
				"./proc":        true,
				"./sys":         true,
				"./root/.cache": true,
				"./tmp":         true,
			},
		},
		{
			name: "strips leading dot and slash before prepending ./",
			settings: types.OverlaySettings{
				Exclusions: []string{"./proc", "/sys", "tmp"},
			},
			want: map[string]bool{
				"./proc": true,
				"./sys":  true,
				"./tmp":  true,
			},
		},
		{
			name: "glob patterns starting with * are untouched",
			settings: types.OverlaySettings{
				Exclusions: []string{
					"*/.cache/huggingface",
					"*.pyc",
					"*/__pycache__",
				},
			},
			want: map[string]bool{
				"*/.cache/huggingface": true,
				"*.pyc":                true,
				"*/__pycache__":        true,
			},
		},
		{
			name:     "empty settings produces empty slice",
			settings: types.OverlaySettings{},
			want:     map[string]bool{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := buildExclusions(test.settings)
			gotSet := make(map[string]bool, len(got))
			for _, value := range got {
				gotSet[value] = true
			}
			for expected := range test.want {
				if !gotSet[expected] {
					t.Errorf(
						"expected %q in exclusions, got %v",
						expected,
						got,
					)
				}
			}
			if len(got) != len(test.want) {
				t.Errorf(
					"len(exclusions) = %d, want %d; got %v",
					len(got),
					len(test.want),
					got,
				)
			}
		})
	}
}

func TestRootfsDirectoryHardlinksToExistingRegularFile(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	first := filepath.Join(source, "first")
	if err := os.WriteFile(first, []byte("new"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(first, filepath.Join(source, "second")); err != nil {
		t.Fatal(err)
	}
	if _, err := CaptureRootfsDiff(
		source,
		checkpoint,
		types.OverlaySettings{},
		nil,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(target, "first"),
		[]byte("existing"),
		0600,
	); err != nil {
		t.Fatal(err)
	}
	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err != nil {
		t.Fatal(err)
	}

	var firstStat, secondStat unix.Stat_t
	if err := unix.Stat(filepath.Join(target, "first"), &firstStat); err != nil {
		t.Fatal(err)
	}
	if err := unix.Stat(filepath.Join(target, "second"), &secondStat); err != nil {
		t.Fatal(err)
	}
	if firstStat.Ino != secondStat.Ino {
		t.Fatalf(
			"hardlinks have different inodes: %d and %d",
			firstStat.Ino,
			secondStat.Ino,
		)
	}
	data, err := os.ReadFile(filepath.Join(target, "second"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "existing" {
		t.Fatalf("hardlink content = %q", data)
	}
}

func TestFindWhiteoutFiles(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T, dir string)
		want  []string
	}{
		{
			name: "top-level whiteout",
			setup: func(t *testing.T, dir string) {
				t.Helper()
				if err := os.WriteFile(
					filepath.Join(dir, ".wh.somefile"),
					nil,
					0644,
				); err != nil {
					t.Fatalf("write whiteout: %v", err)
				}
			},
			want: []string{"somefile"},
		},
		{
			name: "nested whiteout returns relative path",
			setup: func(t *testing.T, dir string) {
				t.Helper()
				sub := filepath.Join(dir, "subdir")
				if err := os.MkdirAll(sub, 0755); err != nil {
					t.Fatalf("mkdir subdir: %v", err)
				}
				if err := os.WriteFile(
					filepath.Join(sub, ".wh.nested"),
					nil,
					0644,
				); err != nil {
					t.Fatalf("write nested whiteout: %v", err)
				}
			},
			want: []string{"subdir/nested"},
		},
		{
			name: "no whiteouts returns empty",
			setup: func(t *testing.T, dir string) {
				t.Helper()
				if err := os.WriteFile(
					filepath.Join(dir, "regular"),
					nil,
					0644,
				); err != nil {
					t.Fatalf("write regular file: %v", err)
				}
			},
		},
		{
			name:  "empty dir returns empty",
			setup: func(*testing.T, string) {},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			test.setup(t, dir)
			got, err := findWhiteoutFiles(dir)
			if err != nil {
				t.Fatalf("findWhiteoutFiles: %v", err)
			}
			if len(got) != len(test.want) {
				t.Fatalf("got %v, want %v", got, test.want)
			}
			for i := range test.want {
				if got[i] != test.want[i] {
					t.Errorf(
						"got[%d] = %q, want %q",
						i,
						got[i],
						test.want[i],
					)
				}
			}
		})
	}
}

func TestCaptureDeletedFiles(t *testing.T) {
	t.Run("dir with whiteouts writes JSON and returns true", func(t *testing.T) {
		upperDir := t.TempDir()
		checkpointDir := t.TempDir()
		if err := os.WriteFile(
			filepath.Join(upperDir, ".wh.removed"),
			nil,
			0644,
		); err != nil {
			t.Fatalf("write whiteout: %v", err)
		}

		found, err := CaptureDeletedFiles(upperDir, checkpointDir)
		if err != nil {
			t.Fatalf("CaptureDeletedFiles: %v", err)
		}
		if !found {
			t.Fatal("expected found=true")
		}

		data, err := os.ReadFile(
			filepath.Join(checkpointDir, deletedFilesFilename),
		)
		if err != nil {
			t.Fatalf("read deleted-files.json: %v", err)
		}
		var files []string
		if err := json.Unmarshal(data, &files); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(files) != 1 || files[0] != "removed" {
			t.Errorf("got %v, want [removed]", files)
		}
	})

	t.Run("dir with no whiteouts returns false and no file", func(t *testing.T) {
		upperDir := t.TempDir()
		checkpointDir := t.TempDir()
		if err := os.WriteFile(
			filepath.Join(upperDir, "normalfile"),
			nil,
			0644,
		); err != nil {
			t.Fatalf("write regular file: %v", err)
		}

		found, err := CaptureDeletedFiles(upperDir, checkpointDir)
		if err != nil {
			t.Fatalf("CaptureDeletedFiles: %v", err)
		}
		if found {
			t.Fatal("expected found=false")
		}
		if _, err := os.Stat(
			filepath.Join(checkpointDir, deletedFilesFilename),
		); !os.IsNotExist(err) {
			t.Error("deleted-files.json should not exist")
		}
	})

	t.Run("empty upperDir returns false", func(t *testing.T) {
		found, err := CaptureDeletedFiles("", t.TempDir())
		if err != nil {
			t.Fatalf("CaptureDeletedFiles: %v", err)
		}
		if found {
			t.Fatal("expected found=false for empty upperDir")
		}
	})
}

func TestApplyDeletedFiles(t *testing.T) {
	log := testr.New(t)

	t.Run("deletes listed files from target", func(t *testing.T) {
		checkpointDir := t.TempDir()
		targetRoot := t.TempDir()
		if err := os.WriteFile(
			filepath.Join(targetRoot, "old-cache"),
			[]byte("data"),
			0644,
		); err != nil {
			t.Fatalf("write target file: %v", err)
		}

		data, err := json.Marshal([]string{"old-cache"})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(
			filepath.Join(checkpointDir, deletedFilesFilename),
			data,
			0644,
		); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
		if _, err := os.Stat(
			filepath.Join(targetRoot, "old-cache"),
		); !os.IsNotExist(err) {
			t.Error("old-cache should have been deleted")
		}
	})

	t.Run("missing deleted-files.json is a no-op", func(t *testing.T) {
		if err := ApplyDeletedFiles(
			t.TempDir(),
			t.TempDir(),
			log,
		); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})

	t.Run("path traversal entry is skipped", func(t *testing.T) {
		checkpointDir := t.TempDir()
		targetRoot := t.TempDir()
		outsideDir := t.TempDir()
		secretFile := filepath.Join(outsideDir, "passwd")
		if err := os.WriteFile(secretFile, []byte("secret"), 0644); err != nil {
			t.Fatalf("write secret file: %v", err)
		}

		rel, err := filepath.Rel(targetRoot, secretFile)
		if err != nil {
			t.Fatalf("build relative path: %v", err)
		}
		data, err := json.Marshal([]string{rel})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(
			filepath.Join(checkpointDir, deletedFilesFilename),
			data,
			0644,
		); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
		if _, err := os.Stat(secretFile); err != nil {
			t.Error("path traversal deleted the outside file")
		}
	})

	t.Run("already-missing file causes no error", func(t *testing.T) {
		checkpointDir := t.TempDir()
		data, err := json.Marshal([]string{"nonexistent"})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(
			filepath.Join(checkpointDir, deletedFilesFilename),
			data,
			0644,
		); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(
			checkpointDir,
			t.TempDir(),
			log,
		); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})

	t.Run("empty entry is skipped", func(t *testing.T) {
		checkpointDir := t.TempDir()
		data, err := json.Marshal([]string{""})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(
			filepath.Join(checkpointDir, deletedFilesFilename),
			data,
			0644,
		); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(
			checkpointDir,
			t.TempDir(),
			log,
		); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})
}

func TestRootfsWorkerCount(t *testing.T) {
	tests := []struct {
		value   string
		want    int
		wantErr bool
	}{
		{value: "1", want: 1},
		{value: "32", want: 32},
		{value: "256", want: 256},
		{value: "0", wantErr: true},
		{value: "-1", wantErr: true},
		{value: "257", wantErr: true},
		{value: "invalid", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.value, func(t *testing.T) {
			got, err := rootfsWorkerCount(test.value)
			if test.wantErr {
				if err == nil {
					t.Fatalf("rootfsWorkerCount(%q) succeeded", test.value)
				}
				return
			}
			if err != nil {
				t.Fatalf("rootfsWorkerCount(%q): %v", test.value, err)
			}
			if got != test.want {
				t.Fatalf("rootfsWorkerCount(%q) = %d, want %d", test.value, got, test.want)
			}
		})
	}

	if got, err := rootfsWorkerCount(""); err != nil ||
		got != defaultRootfsWorkers {
		t.Fatalf("default rootfsWorkerCount() = %d, %v", got, err)
	}
}

func TestCaptureAndApplyRootfsDirectory(t *testing.T) {
	t.Setenv(RootfsWorkersEnv, "4")
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()

	if err := os.MkdirAll(filepath.Join(source, "cache", "nested"), 0750); err != nil {
		t.Fatal(err)
	}
	const fileCount = 128
	var expectedBytes int64
	for i := range fileCount {
		data := []byte(fmt.Sprintf("parallel-file-%03d", i))
		expectedBytes += int64(len(data))
		path := filepath.Join(source, "cache", "nested", fmt.Sprintf("%03d", i))
		if err := os.WriteFile(path, data, 0640); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink("../nested/000", filepath.Join(source, "cache", "link")); err != nil {
		t.Fatal(err)
	}

	artifact, err := CaptureRootfsDiff(
		source,
		checkpoint,
		types.OverlaySettings{},
		nil,
	)
	if err != nil {
		t.Fatalf("CaptureRootfsDiff: %v", err)
	}
	if artifact != filepath.Join(checkpoint, rootfsDirectoryFilename) {
		t.Fatalf("artifact = %q", artifact)
	}
	metadata, present, err := rootfsDirectoryArtifact(checkpoint)
	if err != nil {
		t.Fatalf("rootfsDirectoryArtifact: %v", err)
	}
	if !present {
		t.Fatal("directory artifact is not present")
	}
	if metadata.Entries != fileCount+4 {
		t.Fatalf("metadata entries = %d, want %d", metadata.Entries, fileCount+4)
	}
	if metadata.Bytes != expectedBytes {
		t.Fatalf("metadata bytes = %d, want %d", metadata.Bytes, expectedBytes)
	}

	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err != nil {
		t.Fatalf("ApplyRootfsDiff: %v", err)
	}
	for i := range fileCount {
		path := filepath.Join(target, "cache", "nested", fmt.Sprintf("%03d", i))
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		want := fmt.Sprintf("parallel-file-%03d", i)
		if string(data) != want {
			t.Fatalf("%s = %q, want %q", path, data, want)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0640 {
			t.Fatalf("%s mode = %#o, want 0640", path, info.Mode().Perm())
		}
	}
	link, err := os.Readlink(filepath.Join(target, "cache", "link"))
	if err != nil {
		t.Fatal(err)
	}
	if link != "../nested/000" {
		t.Fatalf("symlink target = %q", link)
	}
}

func TestApplyRootfsDirectorySkipsExistingEntries(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	if err := os.Mkdir(filepath.Join(source, "cache"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "cache", "existing"), []byte("new"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "cache", "new"), []byte("copied"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := CaptureRootfsDiff(source, checkpoint, types.OverlaySettings{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(target, "cache"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "cache", "existing"), []byte("old"), 0600); err != nil {
		t.Fatal(err)
	}

	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(target, "cache", "existing"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "old" {
		t.Fatalf("existing file was overwritten with %q", data)
	}
	data, err = os.ReadFile(filepath.Join(target, "cache", "new"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "copied" {
		t.Fatalf("new file = %q", data)
	}
	info, err := os.Stat(filepath.Join(target, "cache"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0700 {
		t.Fatalf("existing directory mode = %#o, want 0700", info.Mode().Perm())
	}
}

func TestRootfsDirectoryPreservesHardlinksSparseDataAndXattrs(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	first := filepath.Join(source, "sparse")

	file, err := os.OpenFile(first, os.O_CREATE|os.O_WRONLY, 0641)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteAt([]byte("begin"), 0); err != nil {
		file.Close()
		t.Fatal(err)
	}
	const endOffset = int64(8 << 20)
	if _, err := file.WriteAt([]byte("end"), endOffset); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(first, filepath.Join(source, "hardlink")); err != nil {
		t.Fatal(err)
	}
	if err := unix.Setxattr(first, "user.dynamo-test", []byte("xattr-value"), 0); err != nil {
		if errors.Is(err, unix.ENOTSUP) || errors.Is(err, unix.EPERM) {
			t.Skipf("test filesystem does not permit user xattrs: %v", err)
		}
		t.Fatal(err)
	}

	if _, err := CaptureRootfsDiff(source, checkpoint, types.OverlaySettings{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err != nil {
		t.Fatal(err)
	}

	var firstStat, linkStat unix.Stat_t
	if err := unix.Stat(filepath.Join(target, "sparse"), &firstStat); err != nil {
		t.Fatal(err)
	}
	if err := unix.Stat(filepath.Join(target, "hardlink"), &linkStat); err != nil {
		t.Fatal(err)
	}
	if firstStat.Ino != linkStat.Ino {
		t.Fatalf("hardlinks have different inodes: %d and %d", firstStat.Ino, linkStat.Ino)
	}
	if firstStat.Blocks*512 >= firstStat.Size {
		t.Fatalf("restored file is not sparse: blocks=%d size=%d", firstStat.Blocks, firstStat.Size)
	}
	file, err = os.Open(filepath.Join(target, "sparse"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	got := make([]byte, 3)
	if _, err := file.ReadAt(got, endOffset); err != nil {
		t.Fatal(err)
	}
	if string(got) != "end" {
		t.Fatalf("sparse tail = %q", got)
	}
	value := make([]byte, 64)
	size, err := unix.Getxattr(
		filepath.Join(target, "sparse"),
		"user.dynamo-test",
		value,
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(value[:size]) != "xattr-value" {
		t.Fatalf("xattr = %q", value[:size])
	}
}

func TestCaptureRootfsDirectoryExclusions(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	for path, data := range map[string]string{
		"keep/file":                            "keep",
		"proc/hidden":                          "proc",
		"home/user/.cache/huggingface/model":   "model",
		"home/user/package/__pycache__/x":      "bytecode",
		"home/user/package/generated-file.pyc": "bytecode",
		"mnt/model/weights":                    "bind mount",
	} {
		fullPath := filepath.Join(source, path)
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(fullPath, []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}

	_, err := CaptureRootfsDiff(
		source,
		checkpoint,
		types.OverlaySettings{Exclusions: []string{
			"/proc",
			"*/.cache/huggingface",
			"*/__pycache__",
			"*.pyc",
		}},
		[]string{"/mnt/model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	artifact := filepath.Join(checkpoint, rootfsDirectoryFilename)
	if data, err := os.ReadFile(filepath.Join(artifact, "keep", "file")); err != nil ||
		string(data) != "keep" {
		t.Fatalf("retained file = %q, %v", data, err)
	}
	for _, path := range []string{
		"proc",
		"home/user/.cache/huggingface",
		"home/user/package/__pycache__",
		"home/user/package/generated-file.pyc",
		"mnt/model",
	} {
		if _, err := os.Lstat(filepath.Join(artifact, path)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("excluded path %s exists: %v", path, err)
		}
	}
}

func TestCaptureRootfsDirectoryOmitsWhiteouts(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	if err := os.WriteFile(
		filepath.Join(source, ".wh.removed"),
		nil,
		0600,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := CaptureRootfsDiff(
		source,
		checkpoint,
		types.OverlaySettings{},
		nil,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(
		filepath.Join(checkpoint, rootfsDirectoryFilename, ".wh.removed"),
	); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("whiteout was copied into directory artifact: %v", err)
	}
	found, err := CaptureDeletedFiles(source, checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("CaptureDeletedFiles did not record the whiteout")
	}
	data, err := os.ReadFile(
		filepath.Join(checkpoint, deletedFilesFilename),
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `["removed"]` {
		t.Fatalf("deleted files = %s", data)
	}
}

func TestRootfsDirectoryRequiresValidCompleteMarker(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T, checkpoint string)
	}{
		{
			name: "directory without marker",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(filepath.Join(checkpoint, rootfsDirectoryFilename), 0755); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "marker without directory",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				writeTestMetadata(t, checkpoint, rootfsDirectoryMetadata{
					Format:  rootfsDirectoryFormat,
					Version: rootfsDirectoryVersion,
					Entries: 1,
				})
			},
		},
		{
			name: "malformed marker",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(filepath.Join(checkpoint, rootfsDirectoryFilename), 0755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(
					filepath.Join(checkpoint, rootfsMetadataFilename),
					[]byte("{not-json"),
					0644,
				); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "unknown version",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(filepath.Join(checkpoint, rootfsDirectoryFilename), 0755); err != nil {
					t.Fatal(err)
				}
				writeTestMetadata(t, checkpoint, rootfsDirectoryMetadata{
					Format:  rootfsDirectoryFormat,
					Version: rootfsDirectoryVersion + 1,
					Entries: 1,
				})
			},
		},
		{
			name: "entry count mismatch",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(filepath.Join(checkpoint, rootfsDirectoryFilename), 0755); err != nil {
					t.Fatal(err)
				}
				writeTestMetadata(t, checkpoint, rootfsDirectoryMetadata{
					Format:  rootfsDirectoryFormat,
					Version: rootfsDirectoryVersion,
					Entries: 2,
				})
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			checkpoint := t.TempDir()
			test.setup(t, checkpoint)
			if err := ApplyRootfsDiff(
				checkpoint,
				t.TempDir(),
				testr.New(t),
			); err == nil {
				t.Fatal("ApplyRootfsDiff succeeded")
			}
		})
	}
}

func TestRootfsDirectoryRejectsUnsupportedTypes(t *testing.T) {
	t.Run("capture fifo", func(t *testing.T) {
		source := t.TempDir()
		if err := unix.Mkfifo(filepath.Join(source, "fifo"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := CaptureRootfsDiff(
			source,
			t.TempDir(),
			types.OverlaySettings{},
			nil,
		); err == nil || !strings.Contains(err.Error(), "unsupported") {
			t.Fatalf("CaptureRootfsDiff error = %v", err)
		}
	})

	t.Run("restore fifo", func(t *testing.T) {
		checkpoint := t.TempDir()
		artifact := filepath.Join(checkpoint, rootfsDirectoryFilename)
		if err := os.Mkdir(artifact, 0755); err != nil {
			t.Fatal(err)
		}
		if err := unix.Mkfifo(filepath.Join(artifact, "fifo"), 0600); err != nil {
			t.Fatal(err)
		}
		writeTestMetadata(t, checkpoint, rootfsDirectoryMetadata{
			Format:  rootfsDirectoryFormat,
			Version: rootfsDirectoryVersion,
			Entries: 2,
		})
		if err := ApplyRootfsDiff(
			checkpoint,
			t.TempDir(),
			testr.New(t),
		); err == nil || !strings.Contains(err.Error(), "unsupported") {
			t.Fatalf("ApplyRootfsDiff error = %v", err)
		}
	})
}

func TestCaptureRootfsDirectoryCancellationDoesNotPublish(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	checkpoint := t.TempDir()
	_, err := CaptureRootfsDiffContext(
		ctx,
		t.TempDir(),
		checkpoint,
		types.OverlaySettings{},
		nil,
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("CaptureRootfsDiffContext error = %v", err)
	}
	for _, name := range []string{
		rootfsDirectoryFilename,
		rootfsMetadataFilename,
	} {
		if _, err := os.Lstat(filepath.Join(checkpoint, name)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s was published: %v", name, err)
		}
	}
}

func TestApplyRootfsDirectoryPropagatesWorkerError(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	if err := os.Mkdir(filepath.Join(source, "parent"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "parent", "file"), []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := CaptureRootfsDiff(source, checkpoint, types.OverlaySettings{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "parent"), []byte("obstruction"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err == nil {
		t.Fatal("ApplyRootfsDiff succeeded")
	}
}

func TestApplyRootfsDirectoryPreventsDestinationSymlinkTraversal(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	outside := t.TempDir()
	if err := os.Mkdir(filepath.Join(source, "escape"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "escape", "pwned"), []byte("bad"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := CaptureRootfsDiff(source, checkpoint, types.OverlaySettings{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(target, "escape")); err != nil {
		t.Fatal(err)
	}

	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err == nil {
		t.Fatal("ApplyRootfsDiff followed destination symlink")
	}
	if _, err := os.Stat(filepath.Join(outside, "pwned")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("path escaped target root: %v", err)
	}
}

func TestRootfsDirectoryPreservesSymlinkWithoutFollowingIt(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	outside := t.TempDir()
	outsideFile := filepath.Join(outside, "secret")
	if err := os.WriteFile(outsideFile, []byte("secret"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outsideFile, filepath.Join(source, "outside")); err != nil {
		t.Fatal(err)
	}

	if _, err := CaptureRootfsDiff(source, checkpoint, types.OverlaySettings{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err != nil {
		t.Fatal(err)
	}
	link, err := os.Readlink(filepath.Join(target, "outside"))
	if err != nil {
		t.Fatal(err)
	}
	if link != outsideFile {
		t.Fatalf("symlink = %q, want %q", link, outsideFile)
	}
	data, err := os.ReadFile(outsideFile)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "secret" {
		t.Fatalf("outside file changed to %q", data)
	}
}

func TestValidateRelativePath(t *testing.T) {
	for _, path := range []string{"file", "dir/file", "."} {
		if err := validateRelativePath(path); err != nil {
			t.Errorf("validateRelativePath(%q): %v", path, err)
		}
	}
	for _, path := range []string{"", "/absolute", "../escape", "dir/../escape", "dir//file"} {
		if err := validateRelativePath(path); err == nil {
			t.Errorf("validateRelativePath(%q) succeeded", path)
		}
	}
}

func TestRootfsDirectoryPreservesTimestamps(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	path := filepath.Join(source, "timestamp")
	if err := os.WriteFile(path, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}
	want := time.Unix(123456789, 123456789)
	if err := os.Chtimes(path, want, want); err != nil {
		t.Fatal(err)
	}
	if _, err := CaptureRootfsDiff(source, checkpoint, types.OverlaySettings{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := ApplyRootfsDiff(checkpoint, target, testr.New(t)); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(filepath.Join(target, "timestamp"))
	if err != nil {
		t.Fatal(err)
	}
	if !info.ModTime().Equal(want) {
		t.Fatalf("mtime = %s, want %s", info.ModTime(), want)
	}
}

// BenchmarkRootfsDirectory reconstructs the retained 12.8k-entry inventory
// with sparse files. It measures metadata/concurrency, not physical payload I/O.
func BenchmarkRootfsDirectory(b *testing.B) {
	source := createRootfsBenchmarkFixture(b)

	b.Run("Capture", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			checkpoint, err := os.MkdirTemp("", "rootfs-bench-checkpoint-")
			if err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			_, err = CaptureRootfsDiff(
				source,
				checkpoint,
				types.OverlaySettings{},
				nil,
			)
			b.StopTimer()
			if err != nil {
				os.RemoveAll(checkpoint)
				b.Fatal(err)
			}
			if err := os.RemoveAll(checkpoint); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
		}
	})

	checkpoint, err := os.MkdirTemp("", "rootfs-bench-checkpoint-")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(checkpoint)
	if _, err := CaptureRootfsDiff(
		source,
		checkpoint,
		types.OverlaySettings{},
		nil,
	); err != nil {
		b.Fatalf("prepare restore artifact: %v", err)
	}

	b.Run("Restore", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			target, err := os.MkdirTemp("", "rootfs-bench-target-")
			if err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			err = ApplyRootfsDiff(checkpoint, target, logr.Discard())
			b.StopTimer()
			if err != nil {
				os.RemoveAll(target)
				b.Fatal(err)
			}
			if err := os.RemoveAll(target); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
		}
	})
}

func createRootfsBenchmarkFixture(b *testing.B) string {
	b.Helper()
	const (
		directoryCount = 2505
		regularCount   = 10422
		symlinkCount   = 39
		logicalBytes   = int64(875568633)
	)

	root := filepath.Join(b.TempDir(), "source")
	if err := os.Mkdir(root, 0755); err != nil {
		b.Fatal(err)
	}
	for i := 1; i < directoryCount; i++ {
		if err := os.Mkdir(
			filepath.Join(root, fmt.Sprintf("d%04d", i)),
			0755,
		); err != nil {
			b.Fatal(err)
		}
	}

	baseSize := logicalBytes / regularCount
	largerFiles := logicalBytes % regularCount
	for i := range regularCount {
		size := baseSize
		if int64(i) < largerFiles {
			size++
		}
		directory := i%(directoryCount-1) + 1
		path := filepath.Join(
			root,
			fmt.Sprintf("d%04d", directory),
			fmt.Sprintf("f%05d", i),
		)
		file, err := os.Create(path)
		if err != nil {
			b.Fatal(err)
		}
		if err := file.Truncate(size); err != nil {
			file.Close()
			b.Fatal(err)
		}
		if err := file.Close(); err != nil {
			b.Fatal(err)
		}
	}
	for i := 1; i <= symlinkCount; i++ {
		if err := os.Symlink(
			fmt.Sprintf("f%05d", i-1),
			filepath.Join(
				root,
				fmt.Sprintf("d%04d", i),
				fmt.Sprintf("l%02d", i),
			),
		); err != nil {
			b.Fatal(err)
		}
	}
	return root
}

func writeTestMetadata(
	t *testing.T,
	checkpoint string,
	metadata rootfsDirectoryMetadata,
) {
	t.Helper()
	if err := writeRootfsMetadata(checkpoint, metadata); err != nil {
		t.Fatal(err)
	}
}
