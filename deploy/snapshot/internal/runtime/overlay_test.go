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
				Exclusions: []string{"*/.cache/huggingface", "*/.cache/vllm/torch_compile_cache", "*.pyc", "*/__pycache__"},
			},
			want: map[string]bool{
				"*/.cache/huggingface":              true,
				"*/.cache/vllm/torch_compile_cache": true,
				"*.pyc":                             true,
				"*/__pycache__":                     true,
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
