package common

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr/testr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

func TestBuildExclusions(t *testing.T) {
	tests := []struct {
		name     string
		settings types.OverlaySettings
		want     map[string]bool // expected entries (true = must be present)
	}{
		{
			name: "merges all lists and normalizes paths",
			settings: types.OverlaySettings{
				SystemDirs:           []string{"/proc", "/sys"},
				CacheDirs:            []string{"/root/.cache"},
				AdditionalExclusions: []string{"/tmp"},
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
				SystemDirs: []string{"./proc", "/sys", "tmp"},
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
				AdditionalExclusions: []string{"*.pyc", "*/__pycache__"},
			},
			want: map[string]bool{
				"*.pyc":         true,
				"*/__pycache__": true,
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
		setup func(dir string) // create files in temp dir
		want  []string
	}{
		{
			name: "top-level whiteout",
			setup: func(dir string) {
				os.WriteFile(filepath.Join(dir, ".wh.somefile"), nil, 0644)
			},
			want: []string{"somefile"},
		},
		{
			name: "nested whiteout returns relative path",
			setup: func(dir string) {
				sub := filepath.Join(dir, "subdir")
				os.MkdirAll(sub, 0755)
				os.WriteFile(filepath.Join(sub, ".wh.nested"), nil, 0644)
			},
			want: []string{"subdir/nested"},
		},
		{
			name:  "no whiteouts returns empty",
			setup: func(dir string) { os.WriteFile(filepath.Join(dir, "regular"), nil, 0644) },
			want:  nil,
		},
		{
			name:  "empty dir returns empty",
			setup: func(dir string) {},
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tc.setup(dir)
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
		os.WriteFile(filepath.Join(upperDir, ".wh.removed"), nil, 0644)

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
		os.WriteFile(filepath.Join(upperDir, "normalfile"), nil, 0644)

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
		os.WriteFile(filepath.Join(targetRoot, "old-cache"), []byte("data"), 0644)

		// Write deleted-files.json
		data, _ := json.Marshal([]string{"old-cache"})
		os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644)

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
		os.WriteFile(secretFile, []byte("secret"), 0644)

		// Construct a relative path that escapes targetRoot
		rel, _ := filepath.Rel(targetRoot, secretFile)
		data, _ := json.Marshal([]string{rel})
		os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644)

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

		data, _ := json.Marshal([]string{"nonexistent"})
		os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644)

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})

	t.Run("empty entry is skipped", func(t *testing.T) {
		checkpointDir := t.TempDir()
		targetRoot := t.TempDir()

		data, _ := json.Marshal([]string{""})
		os.WriteFile(filepath.Join(checkpointDir, deletedFilesFilename), data, 0644)

		if err := ApplyDeletedFiles(checkpointDir, targetRoot, log); err != nil {
			t.Fatalf("ApplyDeletedFiles: %v", err)
		}
	})
}
