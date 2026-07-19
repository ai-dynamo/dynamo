package runtime

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr/testr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestBuildExclusions(t *testing.T) {
	tests := []struct {
		name     string
		settings types.OverlaySettings
		want     map[string]bool // expected entries (true = must be present)
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
				Exclusions: []string{"*/.cache/huggingface", "*.pyc", "*/__pycache__"},
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

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := buildExclusions(tc.settings)
			gotSet := make(map[string]bool, len(got))
			for _, v := range got {
				gotSet[v] = true
			}
			for expected := range tc.want {
				if !gotSet[expected] {
					t.Errorf("expected %q in exclusions, got %v", expected, got)
				}
			}
			if len(got) != len(tc.want) {
				t.Errorf("len(exclusions) = %d, want %d; got %v", len(got), len(tc.want), got)
			}
		})
	}
}

func TestFindWhiteoutFiles(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T, dir string) // create files in temp dir
		want  []string
	}{
		{
			name: "top-level whiteout",
			setup: func(t *testing.T, dir string) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(dir, ".wh.somefile"), nil, 0644); err != nil {
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
				if err := os.WriteFile(filepath.Join(sub, ".wh.nested"), nil, 0644); err != nil {
					t.Fatalf("write nested whiteout: %v", err)
				}
			},
			want: []string{"subdir/nested"},
		},
		{
			name: "no whiteouts returns empty",
			setup: func(t *testing.T, dir string) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(dir, "regular"), nil, 0644); err != nil {
					t.Fatalf("write regular file: %v", err)
				}
			},
			want: nil,
		},
		{
			name:  "empty dir returns empty",
			setup: func(*testing.T, string) {},
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tc.setup(t, dir)
			got, err := findWhiteoutFiles(dir)
			if err != nil {
				t.Fatalf("findWhiteoutFiles: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("got[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestCaptureRootfsDiff(t *testing.T) {
	t.Run("writes valid archive atomically", func(t *testing.T) {
		upperDir := t.TempDir()
		checkpointDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(upperDir, "generated.txt"), []byte("runtime data"), 0644); err != nil {
			t.Fatalf("write upperdir file: %v", err)
		}

		got, err := CaptureRootfsDiff(upperDir, checkpointDir, types.OverlaySettings{}, nil)
		if err != nil {
			t.Fatalf("CaptureRootfsDiff: %v", err)
		}
		want := filepath.Join(checkpointDir, rootfsDiffFilename)
		if got != want {
			t.Fatalf("got path %q, want %q", got, want)
		}
		info, err := os.Stat(want)
		if err != nil {
			t.Fatalf("stat rootfs diff: %v", err)
		}
		if info.Size() == 0 {
			t.Fatal("rootfs diff should not be empty")
		}
		matches, err := filepath.Glob(filepath.Join(checkpointDir, rootfsDiffFilename+".*.tmp"))
		if err != nil {
			t.Fatalf("glob temp files: %v", err)
		}
		if len(matches) != 0 {
			t.Fatalf("temporary rootfs diff files were not cleaned up: %v", matches)
		}

		targetRoot := t.TempDir()
		if err := ApplyRootfsDiff(checkpointDir, targetRoot, testr.New(t)); err != nil {
			t.Fatalf("ApplyRootfsDiff: %v", err)
		}
		data, err := os.ReadFile(filepath.Join(targetRoot, "generated.txt"))
		if err != nil {
			t.Fatalf("read extracted file: %v", err)
		}
		if string(data) != "runtime data" {
			t.Fatalf("extracted data = %q, want %q", string(data), "runtime data")
		}
	})

	t.Run("failed capture does not publish partial archive", func(t *testing.T) {
		checkpointDir := t.TempDir()
		missingUpperDir := filepath.Join(t.TempDir(), "missing")

		if _, err := CaptureRootfsDiff(missingUpperDir, checkpointDir, types.OverlaySettings{}, nil); err == nil {
			t.Fatal("CaptureRootfsDiff should fail for missing upperdir")
		}
		if _, err := os.Stat(filepath.Join(checkpointDir, rootfsDiffFilename)); !os.IsNotExist(err) {
			t.Fatalf("rootfs diff should not be published after failure, stat error: %v", err)
		}
		matches, err := filepath.Glob(filepath.Join(checkpointDir, rootfsDiffFilename+".*.tmp"))
		if err != nil {
			t.Fatalf("glob temp files: %v", err)
		}
		if len(matches) != 0 {
			t.Fatalf("temporary rootfs diff files were not cleaned up: %v", matches)
		}
	})
}

func TestApplyRootfsDiff(t *testing.T) {
	t.Run("missing archive is no-op", func(t *testing.T) {
		if err := ApplyRootfsDiff(t.TempDir(), t.TempDir(), testr.New(t)); err != nil {
			t.Fatalf("ApplyRootfsDiff: %v", err)
		}
	})

	t.Run("empty archive is no-op", func(t *testing.T) {
		checkpointDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(checkpointDir, rootfsDiffFilename), nil, 0644); err != nil {
			t.Fatalf("write empty rootfs diff: %v", err)
		}
		if err := ApplyRootfsDiff(checkpointDir, t.TempDir(), testr.New(t)); err != nil {
			t.Fatalf("ApplyRootfsDiff: %v", err)
		}
	})

	t.Run("non-empty invalid archive fails", func(t *testing.T) {
		checkpointDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(checkpointDir, rootfsDiffFilename), []byte("not a tar archive"), 0644); err != nil {
			t.Fatalf("write invalid rootfs diff: %v", err)
		}
		if err := ApplyRootfsDiff(checkpointDir, t.TempDir(), testr.New(t)); err == nil {
			t.Fatal("ApplyRootfsDiff should fail for invalid non-empty archive")
		}
	})

	t.Run("non-ENOENT stat error is propagated", func(t *testing.T) {
		// Pass a regular file as checkpointPath so stat of
		// checkpointPath/rootfs-diff.tar returns ENOTDIR, not ENOENT.
		f, err := os.CreateTemp(t.TempDir(), "not-a-dir")
		if err != nil {
			t.Fatalf("create temp file: %v", err)
		}
		f.Close()
		if err := ApplyRootfsDiff(f.Name(), t.TempDir(), testr.New(t)); err == nil {
			t.Fatal("ApplyRootfsDiff should propagate non-ENOENT stat error")
		}
	})
}

func TestCaptureDeletedFiles(t *testing.T) {
	t.Run("dir with whiteouts writes JSON and returns true", func(t *testing.T) {
		upperDir := t.TempDir()
		checkpointDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(upperDir, ".wh.removed"), nil, 0644); err != nil {
			t.Fatalf("write whiteout: %v", err)
		}

		found, err := CaptureDeletedFiles(upperDir, checkpointDir)
		if err != nil {
			t.Fatalf("CaptureDeletedFiles: %v", err)
		}
		if !found {
			t.Fatal("expected found=true")
		}

		data, err := os.ReadFile(filepath.Join(checkpointDir, deletedFilesFilename))
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
		if err := os.WriteFile(filepath.Join(upperDir, "normalfile"), nil, 0644); err != nil {
			t.Fatalf("write regular file: %v", err)
		}

		found, err := CaptureDeletedFiles(upperDir, checkpointDir)
		if err != nil {
			t.Fatalf("CaptureDeletedFiles: %v", err)
		}
		if found {
			t.Fatal("expected found=false")
		}
		if _, err := os.Stat(filepath.Join(checkpointDir, deletedFilesFilename)); !os.IsNotExist(err) {
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

		// Create target file that should be deleted
		if err := os.WriteFile(filepath.Join(targetRoot, "old-cache"), []byte("data"), 0644); err != nil {
			t.Fatalf("write target file: %v", err)
		}

		// Write deleted-files.json
		data, err := json.Marshal([]string{"old-cache"})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}

		if _, err := os.Stat(filepath.Join(targetRoot, "old-cache")); !os.IsNotExist(err) {
			t.Error("old-cache should have been deleted")
		}
	})

	t.Run("missing deleted-files.json is a no-op", func(t *testing.T) {
		if err := ApplyDeletedFiles(t.TempDir(), t.TempDir(), log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})

	t.Run("path traversal entry is skipped", func(t *testing.T) {
		checkpointDir := t.TempDir()
		targetRoot := t.TempDir()

		// Create a file outside targetRoot that the traversal would try to delete
		outsideDir := t.TempDir()
		secretFile := filepath.Join(outsideDir, "passwd")
		if err := os.WriteFile(secretFile, []byte("secret"), 0644); err != nil {
			t.Fatalf("write secret file: %v", err)
		}

		// Construct a relative path that escapes targetRoot
		rel, err := filepath.Rel(targetRoot, secretFile)
		if err != nil {
			t.Fatalf("build relative path: %v", err)
		}
		data, err := json.Marshal([]string{rel})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}

		// The file outside targetRoot must still exist
		if _, err := os.Stat(secretFile); err != nil {
			t.Error("path traversal should have been blocked, but file was deleted")
		}
	})

	t.Run("already-missing file causes no error", func(t *testing.T) {
		checkpointDir := t.TempDir()
		targetRoot := t.TempDir()

		data, err := json.Marshal([]string{"nonexistent"})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})

	t.Run("empty entry is skipped", func(t *testing.T) {
		checkpointDir := t.TempDir()
		targetRoot := t.TempDir()

		data, err := json.Marshal([]string{""})
		if err != nil {
			t.Fatalf("marshal deleted files: %v", err)
		}
		if err := os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644); err != nil {
			t.Fatalf("write deleted-files.json: %v", err)
		}

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})
}
