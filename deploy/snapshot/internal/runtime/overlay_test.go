package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestRootfsDirectoryRoundTrip(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	if err := os.Mkdir(filepath.Join(source, "bin"), 0750); err != nil {
		t.Fatal(err)
	}
	executable := filepath.Join(source, "bin", "run")
	if err := os.WriteFile(executable, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}
	wantTime := time.Unix(1_700_000_000, 123_000_000)
	if err := os.Chtimes(executable, wantTime, wantTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("bin/run", filepath.Join(source, "latest")); err != nil {
		t.Fatal(err)
	}
	if _, err := CaptureRootfsDiff(
		context.Background(),
		source,
		checkpoint,
		types.OverlaySettings{},
		nil,
		4,
		testr.New(t),
	); err != nil {
		t.Fatalf("CaptureRootfsDiff: %v", err)
	}
	if _, err := os.Stat(filepath.Join(checkpoint, "rootfs-diff.tar")); err == nil {
		t.Fatal("tar artifact must not be created")
	}
	deletedData, err := os.ReadFile(
		filepath.Join(checkpoint, deletedFilesFilename),
	)
	if err != nil {
		t.Fatalf("read empty deletion metadata: %v", err)
	}
	if string(deletedData) != "[]" {
		t.Fatalf("empty deletion metadata = %s, want []", deletedData)
	}
	if err := ApplyRootfsDiff(
		context.Background(),
		checkpoint,
		target,
		3,
		testr.New(t),
	); err != nil {
		t.Fatalf("ApplyRootfsDiff: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(target, "bin", "run"))
	if err != nil || string(data) != "#!/bin/sh\n" {
		t.Fatalf("restored file = %q, %v", data, err)
	}
	info, err := os.Stat(filepath.Join(target, "bin", "run"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0755 {
		t.Fatalf("restored mode = %#o, want 0755", info.Mode().Perm())
	}
	if !info.ModTime().Equal(wantTime) {
		t.Fatalf("restored mtime = %v, want %v", info.ModTime(), wantTime)
	}
	link, err := os.Readlink(filepath.Join(target, "latest"))
	if err != nil || link != "bin/run" {
		t.Fatalf("restored symlink = %q, %v", link, err)
	}
	sourceInfo, err := os.Lstat(executable)
	if err != nil {
		t.Fatal(err)
	}
	targetInfo, err := os.Lstat(filepath.Join(target, "bin", "run"))
	if err != nil {
		t.Fatal(err)
	}
	if os.Geteuid() == 0 {
		sourceStat := sourceInfo.Sys().(*syscall.Stat_t)
		targetStat := targetInfo.Sys().(*syscall.Stat_t)
		if sourceStat.Uid != targetStat.Uid || sourceStat.Gid != targetStat.Gid {
			t.Fatalf(
				"restored owner = %d:%d, want %d:%d",
				targetStat.Uid,
				targetStat.Gid,
				sourceStat.Uid,
				sourceStat.Gid,
			)
		}
	}
}

func TestCaptureDeletedFilesRejectsMalformedWhiteouts(t *testing.T) {
	for _, directory := range []string{"", "nested"} {
		for _, suffix := range []string{"", ".", ".."} {
			name := "top-level"
			if directory != "" {
				name = directory
			}
			t.Run(name+"/"+suffix, func(t *testing.T) {
				source := t.TempDir()
				checkpoint := t.TempDir()
				parent := filepath.Join(source, directory)
				if err := os.MkdirAll(parent, 0755); err != nil {
					t.Fatal(err)
				}
				writeTestFile(t, filepath.Join(parent, ".wh."+suffix), nil)
				_, err := CaptureRootfsDiff(
					context.Background(),
					source,
					checkpoint,
					types.OverlaySettings{},
					nil,
					1,
					testr.New(t),
				)
				if err == nil || !strings.Contains(err.Error(), "unsafe whiteout target") {
					t.Fatalf("CaptureRootfsDiff() error = %v", err)
				}
				if info, err := os.Stat(parent); err != nil || !info.IsDir() {
					t.Fatalf("whiteout removed its parent %q: %v", parent, err)
				}
				if _, err := os.Lstat(
					filepath.Join(checkpoint, deletedFilesFilename),
				); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("deletion metadata was published: %v", err)
				}
			})
		}
	}
}

func TestCopyRegularFilesCancellationAndError(t *testing.T) {
	source := t.TempDir()
	target := t.TempDir()
	sourceFD, err := openRoot(source, false)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Close(sourceFD)
	targetFD, err := openRoot(target, true)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Close(targetFD)
	entry := rootfsEntry{path: "missing", mode: unix.S_IFREG}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = copyRegularFiles(ctx, sourceFD, targetFD, []rootfsEntry{entry}, 2, false)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("copyRegularFiles cancellation error = %v", err)
	}

	err = copyRegularFiles(
		context.Background(),
		sourceFD,
		targetFD,
		[]rootfsEntry{entry},
		2,
		false,
	)
	if err == nil || !strings.Contains(err.Error(), "open source") {
		t.Fatalf("copyRegularFiles worker error = %v", err)
	}
}

func TestCopyRootfsCancellationLoops(t *testing.T) {
	t.Run("symlinks", func(t *testing.T) {
		source := t.TempDir()
		target := t.TempDir()
		if err := os.Symlink("target", filepath.Join(source, "a")); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink("target", filepath.Join(source, "b")); err != nil {
			t.Fatal(err)
		}
		entries, _, _, err := scanRootfs(
			context.Background(),
			source,
			types.OverlaySettings{},
			nil,
			false,
		)
		if err != nil {
			t.Fatal(err)
		}
		ctx := newCancelOnErrCallContext(4)
		err = copyRootfs(ctx, source, target, entries, 1, false)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("copyRootfs() error = %v, want context.Canceled", err)
		}
		if _, err := os.Lstat(filepath.Join(target, "a")); err != nil {
			t.Fatalf("first symlink was not copied: %v", err)
		}
		if _, err := os.Lstat(filepath.Join(target, "b")); !errors.Is(
			err,
			os.ErrNotExist,
		) {
			t.Fatalf("second symlink was published after cancellation: %v", err)
		}
	})

	t.Run("reverse directory metadata", func(t *testing.T) {
		source := t.TempDir()
		target := t.TempDir()
		directory := filepath.Join(source, "directory")
		if err := os.Mkdir(directory, 0500); err != nil {
			t.Fatal(err)
		}
		entries, _, _, err := scanRootfs(
			context.Background(),
			source,
			types.OverlaySettings{},
			nil,
			false,
		)
		if err != nil {
			t.Fatal(err)
		}
		ctx := newCancelOnErrCallContext(3)
		err = copyRootfs(ctx, source, target, entries, 1, false)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("copyRootfs() error = %v, want context.Canceled", err)
		}
		info, err := os.Stat(filepath.Join(target, "directory"))
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() == 0500 {
			t.Fatal("directory metadata was published after cancellation")
		}
	})
}

func TestApplyRootfsDiffSkipsExisting(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	if err := os.WriteFile(filepath.Join(source, "file"), []byte("new"), 0644); err != nil {
		t.Fatal(err)
	}
	captureRootfsForTest(t, source, checkpoint)
	if err := os.WriteFile(filepath.Join(target, "file"), []byte("old"), 0600); err != nil {
		t.Fatal(err)
	}

	if err := ApplyRootfsDiff(
		context.Background(),
		checkpoint,
		target,
		2,
		testr.New(t),
	); err != nil {
		t.Fatalf("ApplyRootfsDiff: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(target, "file"))
	if err != nil || string(data) != "old" {
		t.Fatalf("existing file = %q, %v", data, err)
	}
}

func TestRootfsArtifactValidation(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*testing.T, string)
		want  string
	}{
		{
			name: "missing",
			want: "missing rootfs directory artifact",
		},
		{
			name: "tar only",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				writeTestFile(t, filepath.Join(checkpoint, "rootfs-diff.tar"), nil)
			},
			want: "legacy tar checkpoints are unsupported",
		},
		{
			name: "incomplete",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(
					filepath.Join(checkpoint, rootfsDirectoryFilename),
					0755,
				); err != nil {
					t.Fatal(err)
				}
			},
			want: "incomplete rootfs directory artifact",
		},
		{
			name: "malformed",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(
					filepath.Join(checkpoint, rootfsDirectoryFilename),
					0755,
				); err != nil {
					t.Fatal(err)
				}
				writeTestFile(
					t,
					filepath.Join(checkpoint, rootfsMetadataFilename),
					[]byte("{"),
				)
			},
			want: "parse rootfs completion metadata",
		},
		{
			name: "unsupported version",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(
					filepath.Join(checkpoint, rootfsDirectoryFilename),
					0755,
				); err != nil {
					t.Fatal(err)
				}
				data, err := json.Marshal(rootfsMetadata{
					Format:  rootfsDirectoryFormat,
					Version: rootfsDirectoryVersion + 1,
				})
				if err != nil {
					t.Fatal(err)
				}
				writeTestFile(
					t,
					filepath.Join(checkpoint, rootfsMetadataFilename),
					data,
				)
			},
			want: "unsupported rootfs directory format",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			checkpoint := t.TempDir()
			if test.setup != nil {
				test.setup(t, checkpoint)
			}
			err := ApplyRootfsDiff(
				context.Background(),
				checkpoint,
				t.TempDir(),
				1,
				testr.New(t),
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ApplyRootfsDiff() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestCaptureRootfsExclusionsAndBindDestinations(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	for name := range map[string]string{
		"keep/file":                 "keep",
		"proc/hidden":               "excluded",
		"data/hidden":               "bind",
		"root/.cache/huggingface/x": "glob",
		"pkg/__pycache__/x":         "glob",
		"pkg/x.pyc":                 "glob",
	} {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(source, name)), 0755); err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, filepath.Join(source, name), []byte(name))
	}
	if _, err := CaptureRootfsDiff(
		context.Background(),
		source,
		checkpoint,
		types.OverlaySettings{Exclusions: []string{
			"/proc",
			"*/.cache/huggingface",
			"*/__pycache__",
			"*.pyc",
		}},
		[]string{"/data"},
		2,
		testr.New(t),
	); err != nil {
		t.Fatalf("CaptureRootfsDiff: %v", err)
	}
	if err := ApplyRootfsDiff(
		context.Background(),
		checkpoint,
		target,
		2,
		testr.New(t),
	); err != nil {
		t.Fatalf("ApplyRootfsDiff: %v", err)
	}
	if _, err := os.Stat(filepath.Join(target, "keep", "file")); err != nil {
		t.Fatal("included file was not restored")
	}
	for _, name := range []string{
		"proc",
		"data",
		"root/.cache/huggingface",
		"pkg/__pycache__",
		"pkg/x.pyc",
	} {
		if _, err := os.Lstat(filepath.Join(target, name)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("excluded path %s exists: %v", name, err)
		}
	}
}

func TestCaptureRootfsRejectsUnsupportedEntries(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*testing.T, string)
		want  string
	}{
		{
			name: "hardlink",
			setup: func(t *testing.T, root string) {
				t.Helper()
				file := filepath.Join(root, "file")
				writeTestFile(t, file, []byte("data"))
				if err := os.Link(file, filepath.Join(root, "link")); err != nil {
					t.Fatal(err)
				}
			},
			want: "unsupported hardlink",
		},
		{
			name: "sparse",
			setup: func(t *testing.T, root string) {
				t.Helper()
				file, err := os.Create(filepath.Join(root, "sparse"))
				if err != nil {
					t.Fatal(err)
				}
				if err := file.Truncate(1 << 20); err != nil {
					t.Fatal(err)
				}
				file.Close()
			},
			want: "unsupported sparse file",
		},
		{
			name: "fifo",
			setup: func(t *testing.T, root string) {
				t.Helper()
				if err := unix.Mkfifo(filepath.Join(root, "fifo"), 0600); err != nil {
					t.Fatal(err)
				}
			},
			want: "unsupported rootfs entry",
		},
		{
			name: "opaque marker",
			setup: func(t *testing.T, root string) {
				t.Helper()
				writeTestFile(t, filepath.Join(root, ".wh..wh..opq"), nil)
			},
			want: "unsupported opaque whiteout",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := t.TempDir()
			test.setup(t, source)
			_, err := CaptureRootfsDiff(
				context.Background(),
				source,
				t.TempDir(),
				types.OverlaySettings{},
				nil,
				2,
				testr.New(t),
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("CaptureRootfsDiff() error = %v, want %q", err, test.want)
			}
		})
	}

	t.Run("xattr", func(t *testing.T) {
		source := t.TempDir()
		file := filepath.Join(source, "file")
		writeTestFile(t, file, nil)
		if err := unix.Setxattr(file, "user.test", []byte("x"), 0); err != nil {
			if errors.Is(err, unix.ENOTSUP) || errors.Is(err, unix.EPERM) {
				t.Skipf("xattrs unavailable: %v", err)
			}
			t.Fatal(err)
		}
		_, err := CaptureRootfsDiff(
			context.Background(),
			source,
			t.TempDir(),
			types.OverlaySettings{},
			nil,
			1,
			testr.New(t),
		)
		if err == nil || !strings.Contains(err.Error(), "unsupported xattrs") {
			t.Fatalf("CaptureRootfsDiff() error = %v", err)
		}
	})
}

func TestCaptureRootfsCancellation(t *testing.T) {
	tests := []struct {
		name    string
		trigger func(string) context.Context
	}{
		{
			name: rootfsDirectoryFilename,
			trigger: func(checkpoint string) context.Context {
				return newCancelWhenPathExistsContext(
					filepath.Join(checkpoint, rootfsDirectoryFilename),
				)
			},
		},
		{
			name: deletedFilesFilename,
			trigger: func(checkpoint string) context.Context {
				return newCancelWhenPathExistsContext(
					filepath.Join(checkpoint, deletedFilesFilename),
				)
			},
		},
		{
			name: rootfsMetadataFilename,
			trigger: func(checkpoint string) context.Context {
				return newCancelWhenPathPrefixExistsContext(
					checkpoint,
					".rootfs-meta-",
				)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := t.TempDir()
			checkpoint := t.TempDir()
			writeTestFile(t, filepath.Join(source, "file"), []byte("data"))
			ctx := test.trigger(checkpoint)
			_, err := CaptureRootfsDiff(
				ctx,
				source,
				checkpoint,
				types.OverlaySettings{},
				nil,
				4,
				testr.New(t),
			)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf(
					"CaptureRootfsDiff() error = %v, want context.Canceled",
					err,
				)
			}
			for _, name := range []string{
				rootfsDirectoryFilename,
				deletedFilesFilename,
				rootfsMetadataFilename,
			} {
				if _, err := os.Lstat(filepath.Join(checkpoint, name)); !errors.Is(
					err,
					os.ErrNotExist,
				) {
					t.Fatalf("%s was published after cancellation: %v", name, err)
				}
			}
		})
	}
}

func TestApplyRootfsRejectsDestinationSymlink(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	target := t.TempDir()
	outside := t.TempDir()
	if err := os.Mkdir(filepath.Join(source, "parent"), 0755); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(source, "parent", "file"), []byte("data"))
	captureRootfsForTest(t, source, checkpoint)
	if err := os.Symlink(outside, filepath.Join(target, "parent")); err != nil {
		t.Fatal(err)
	}
	err := ApplyRootfsDiff(
		context.Background(),
		checkpoint,
		target,
		2,
		testr.New(t),
	)
	if err == nil {
		t.Fatal("ApplyRootfsDiff succeeded through destination symlink")
	}
	if _, err := os.Stat(filepath.Join(outside, "file")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("outside file exists: %v", err)
	}
}

func TestApplyDeletedFiles(t *testing.T) {
	t.Run("deletes without following symlink", func(t *testing.T) {
		checkpoint := t.TempDir()
		target := t.TempDir()
		outside := t.TempDir()
		writeTestFile(t, filepath.Join(outside, "secret"), []byte("secret"))
		if err := os.Symlink(outside, filepath.Join(target, "link")); err != nil {
			t.Fatal(err)
		}
		writeDeletedFiles(t, checkpoint, []string{"link/secret"})
		err := ApplyDeletedFiles(checkpoint, target, testr.New(t))
		if err == nil {
			t.Fatal("ApplyDeletedFiles followed an intermediate symlink")
		}
		if _, err := os.Stat(filepath.Join(outside, "secret")); err != nil {
			t.Fatalf("outside file was removed: %v", err)
		}
	})

	t.Run("validates all paths before deleting", func(t *testing.T) {
		checkpoint := t.TempDir()
		target := t.TempDir()
		writeTestFile(t, filepath.Join(target, "keep"), nil)
		writeDeletedFiles(t, checkpoint, []string{"keep", "../escape"})
		if err := ApplyDeletedFiles(checkpoint, target, testr.New(t)); err == nil {
			t.Fatal("ApplyDeletedFiles accepted path traversal")
		}
		if _, err := os.Stat(filepath.Join(target, "keep")); err != nil {
			t.Fatal("valid entry was deleted before invalid entry failed")
		}
	})

	t.Run("removes directory tree", func(t *testing.T) {
		checkpoint := t.TempDir()
		target := t.TempDir()
		if err := os.MkdirAll(filepath.Join(target, "old", "nested"), 0755); err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, filepath.Join(target, "old", "nested", "file"), nil)
		writeDeletedFiles(t, checkpoint, []string{"old"})
		if err := ApplyDeletedFiles(checkpoint, target, testr.New(t)); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
		if _, err := os.Stat(filepath.Join(target, "old")); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("deleted directory exists: %v", err)
		}
	})
}

func TestDeletedFilesMetadataValidation(t *testing.T) {
	t.Run("missing", func(t *testing.T) {
		checkpoint := t.TempDir()
		writeRootfsContract(t, checkpoint, 0)
		if err := os.Remove(
			filepath.Join(checkpoint, deletedFilesFilename),
		); err != nil {
			t.Fatal(err)
		}
		err := ApplyRootfsDiff(
			context.Background(),
			checkpoint,
			t.TempDir(),
			1,
			testr.New(t),
		)
		if err == nil || !strings.Contains(err.Error(), "open deleted files metadata") {
			t.Fatalf("ApplyRootfsDiff() error = %v", err)
		}
	})

	t.Run("count mismatch", func(t *testing.T) {
		checkpoint := t.TempDir()
		writeRootfsContract(t, checkpoint, 1)
		writeTestFile(
			t,
			filepath.Join(checkpoint, deletedFilesFilename),
			[]byte("[]"),
		)
		err := ApplyRootfsDiff(
			context.Background(),
			checkpoint,
			t.TempDir(),
			1,
			testr.New(t),
		)
		if err == nil || !strings.Contains(err.Error(), "count mismatch") {
			t.Fatalf("ApplyRootfsDiff() error = %v", err)
		}
	})

	t.Run("missing presence marker", func(t *testing.T) {
		checkpoint := t.TempDir()
		if err := os.Mkdir(
			filepath.Join(checkpoint, rootfsDirectoryFilename),
			0755,
		); err != nil {
			t.Fatal(err)
		}
		writeTestJSON(t, filepath.Join(checkpoint, rootfsMetadataFilename), rootfsMetadata{
			Format:  rootfsDirectoryFormat,
			Version: rootfsDirectoryVersion,
		})
		writeTestFile(
			t,
			filepath.Join(checkpoint, deletedFilesFilename),
			[]byte("[]"),
		)
		err := ApplyRootfsDiff(
			context.Background(),
			checkpoint,
			t.TempDir(),
			1,
			testr.New(t),
		)
		if err == nil || !strings.Contains(
			err.Error(),
			"invalid rootfs directory artifact counts",
		) {
			t.Fatalf("ApplyRootfsDiff() error = %v", err)
		}
	})
}

func TestReadDeletedFilesRejectsUnsafeFiles(t *testing.T) {
	tests := []struct {
		name  string
		setup func(*testing.T, string)
	}{
		{
			name: "symlink",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Symlink(
					"/dev/null",
					filepath.Join(checkpoint, deletedFilesFilename),
				); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "fifo",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := unix.Mkfifo(
					filepath.Join(checkpoint, deletedFilesFilename),
					0600,
				); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "directory",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				if err := os.Mkdir(
					filepath.Join(checkpoint, deletedFilesFilename),
					0700,
				); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "oversize",
			setup: func(t *testing.T, checkpoint string) {
				t.Helper()
				file, err := os.Create(
					filepath.Join(checkpoint, deletedFilesFilename),
				)
				if err != nil {
					t.Fatal(err)
				}
				if err := file.Truncate(maxDeletedFilesSize + 1); err != nil {
					file.Close()
					t.Fatal(err)
				}
				if err := file.Close(); err != nil {
					t.Fatal(err)
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			checkpoint := t.TempDir()
			test.setup(t, checkpoint)
			if _, err := readDeletedFiles(checkpoint, 0); err == nil {
				t.Fatal("readDeletedFiles() accepted unsafe deletion metadata")
			}
		})
	}
}

func TestDeletedFilesLimits(t *testing.T) {
	t.Run("path length", func(t *testing.T) {
		data, err := json.Marshal([]string{
			strings.Repeat("a", maxRootfsPathLength+1),
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := parseDeletedFiles(data); err == nil ||
			!strings.Contains(err.Error(), "unsafe rootfs path") {
			t.Fatalf("parseDeletedFiles() error = %v", err)
		}
	})

	t.Run("deletion count", func(t *testing.T) {
		var data strings.Builder
		data.Grow(4*maxDeletedFiles + 3)
		data.WriteByte('[')
		for i := 0; i <= maxDeletedFiles; i++ {
			if i > 0 {
				data.WriteByte(',')
			}
			data.WriteString(`"a"`)
		}
		data.WriteByte(']')
		if _, err := parseDeletedFiles([]byte(data.String())); err == nil ||
			!strings.Contains(err.Error(), "more than") {
			t.Fatalf("parseDeletedFiles() error = %v", err)
		}
	})

	t.Run("rootfs entry count", func(t *testing.T) {
		checkpoint := t.TempDir()
		if err := os.Mkdir(
			filepath.Join(checkpoint, rootfsDirectoryFilename),
			0755,
		); err != nil {
			t.Fatal(err)
		}
		writeTestJSON(t, filepath.Join(checkpoint, rootfsMetadataFilename), rootfsMetadata{
			Format:              rootfsDirectoryFormat,
			Version:             rootfsDirectoryVersion,
			Entries:             maxRootfsEntries + 1,
			DeletedFilesPresent: true,
		})
		if _, err := readRootfsMetadata(checkpoint); err == nil ||
			!strings.Contains(err.Error(), "invalid rootfs directory artifact counts") {
			t.Fatalf("readRootfsMetadata() error = %v", err)
		}
	})
}

func TestCaptureDeletedFilesDeterministicAndExcluded(t *testing.T) {
	source := t.TempDir()
	checkpoint := t.TempDir()
	if err := os.MkdirAll(filepath.Join(source, "nested"), 0755); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(source, ".wh.z"), nil)
	writeTestFile(t, filepath.Join(source, "nested", ".wh.a"), nil)
	writeTestFile(t, filepath.Join(source, "nested", ".wh.skip"), nil)
	_, err := CaptureRootfsDiff(
		context.Background(),
		source,
		checkpoint,
		types.OverlaySettings{Exclusions: []string{"*/.wh.skip"}},
		nil,
		2,
		testr.New(t),
	)
	if err != nil {
		t.Fatalf("CaptureRootfsDiff: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(checkpoint, deletedFilesFilename))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `["nested/a","z"]` {
		t.Fatalf("deleted-files.json = %s", data)
	}
}

func TestCaptureDeletedFilesRejectsNativeWhiteout(t *testing.T) {
	source := t.TempDir()
	if err := unix.Mkfifo(filepath.Join(source, ".wh.native"), 0600); err != nil {
		t.Fatal(err)
	}
	_, err := CaptureRootfsDiff(
		context.Background(),
		source,
		t.TempDir(),
		types.OverlaySettings{},
		nil,
		1,
		testr.New(t),
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported native whiteout") {
		t.Fatalf("CaptureRootfsDiff() error = %v", err)
	}
}

func TestOpenBeneathNoXdev(t *testing.T) {
	rootFD, err := openRoot("/", true)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Close(rootFD)
	_, err = openBeneath(rootFD, "proc/self", unix.O_PATH)
	if err == nil {
		t.Fatal("openBeneath crossed into /proc mount")
	}
	if !errors.Is(err, unix.EXDEV) {
		t.Fatalf("openBeneath() error = %v, want EXDEV", err)
	}
}

func TestScanRootfsRejectsMountBoundary(t *testing.T) {
	var rootStat, procStat unix.Statx_t
	if err := statxPath("/", &rootStat); err != nil {
		t.Fatal(err)
	}
	if err := statxPath("/proc", &procStat); err != nil {
		t.Skipf("/proc unavailable: %v", err)
	}
	if rootStat.Mnt_id == procStat.Mnt_id {
		t.Skip("/proc is not a distinct mount")
	}
	rootEntries, err := os.ReadDir("/")
	if err != nil {
		t.Fatal(err)
	}
	exclusions := make([]string, 0, len(rootEntries)-1)
	for _, entry := range rootEntries {
		if entry.Name() != "proc" {
			exclusions = append(exclusions, "/"+entry.Name())
		}
	}
	_, _, _, err = scanRootfs(
		context.Background(),
		"/",
		types.OverlaySettings{},
		exclusions,
		false,
	)
	if err == nil || !strings.Contains(err.Error(), "mount boundary") {
		t.Fatalf("scanRootfs() error = %v, want mount boundary", err)
	}
}

func captureRootfsForTest(t *testing.T, source, checkpoint string) {
	t.Helper()
	if _, err := CaptureRootfsDiff(
		context.Background(),
		source,
		checkpoint,
		types.OverlaySettings{},
		nil,
		2,
		testr.New(t),
	); err != nil {
		t.Fatalf("CaptureRootfsDiff: %v", err)
	}
}

func writeTestFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func writeDeletedFiles(t *testing.T, checkpoint string, paths []string) {
	t.Helper()
	if _, err := os.Stat(
		filepath.Join(checkpoint, rootfsDirectoryFilename),
	); errors.Is(err, os.ErrNotExist) {
		if err := os.Mkdir(
			filepath.Join(checkpoint, rootfsDirectoryFilename),
			0755,
		); err != nil {
			t.Fatal(err)
		}
	}
	writeTestJSON(t, filepath.Join(checkpoint, deletedFilesFilename), paths)
	writeTestJSON(t, filepath.Join(checkpoint, rootfsMetadataFilename), rootfsMetadata{
		Format:              rootfsDirectoryFormat,
		Version:             rootfsDirectoryVersion,
		DeletedFilesPresent: true,
		Deletions:           int64(len(paths)),
	})
}

func writeRootfsContract(t *testing.T, checkpoint string, deletions int64) {
	t.Helper()
	if err := os.Mkdir(
		filepath.Join(checkpoint, rootfsDirectoryFilename),
		0755,
	); err != nil {
		t.Fatal(err)
	}
	writeTestJSON(t, filepath.Join(checkpoint, rootfsMetadataFilename), rootfsMetadata{
		Format:              rootfsDirectoryFormat,
		Version:             rootfsDirectoryVersion,
		DeletedFilesPresent: true,
		Deletions:           deletions,
	})
	writeTestFile(
		t,
		filepath.Join(checkpoint, deletedFilesFilename),
		[]byte("[]"),
	)
}

func writeTestJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, path, data)
}

type cancelWhenPathExistsContext struct {
	context.Context
	cancel context.CancelFunc
	once   sync.Once
	exists func() bool
}

func newCancelWhenPathExistsContext(trigger string) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	return &cancelWhenPathExistsContext{
		Context: ctx,
		cancel:  cancel,
		exists: func() bool {
			_, err := os.Lstat(trigger)
			return err == nil
		},
	}
}

func newCancelWhenPathPrefixExistsContext(
	directory string,
	prefix string,
) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	return &cancelWhenPathExistsContext{
		Context: ctx,
		cancel:  cancel,
		exists: func() bool {
			entries, err := os.ReadDir(directory)
			if err != nil {
				return false
			}
			for _, entry := range entries {
				if strings.HasPrefix(entry.Name(), prefix) {
					return true
				}
			}
			return false
		},
	}
}

func (c *cancelWhenPathExistsContext) Err() error {
	if c.exists() {
		c.once.Do(c.cancel)
	}
	return c.Context.Err()
}

type cancelOnErrCallContext struct {
	context.Context
	cancel context.CancelFunc
	call   int32
	count  atomic.Int32
}

func newCancelOnErrCallContext(call int32) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	return &cancelOnErrCallContext{
		Context: ctx,
		cancel:  cancel,
		call:    call,
	}
}

func (c *cancelOnErrCallContext) Err() error {
	if c.count.Add(1) == c.call {
		c.cancel()
	}
	return c.Context.Err()
}
