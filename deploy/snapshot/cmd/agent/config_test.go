package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestLoadConfigRootFSWorkers(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	config := `
rootfs:
  workers: 23
storage:
  type: pvc
  basePath: /checkpoints
restore:
  nsRestorePath: /usr/local/bin/nsrestore
  restoreTimeoutSeconds: 60
`
	if err := os.WriteFile(path, []byte(config), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if cfg.RootFS.Workers != 23 {
		t.Fatalf("RootFS.Workers = %d, want 23", cfg.RootFS.Workers)
	}
}

func TestLoadConfigRootFSWorkersDefaultAndInvalid(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	config := `
storage:
  type: pvc
  basePath: /checkpoints
restore:
  nsRestorePath: /usr/local/bin/nsrestore
  restoreTimeoutSeconds: 60
`
	if err := os.WriteFile(path, []byte(config), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate default: %v", err)
	}
	if cfg.RootFS.Workers != types.DefaultRootFSWorkers {
		t.Fatalf(
			"RootFS.Workers = %d, want %d",
			cfg.RootFS.Workers,
			types.DefaultRootFSWorkers,
		)
	}

	cfg.RootFS.Workers = types.MaxRootFSWorkers + 1
	err = cfg.Validate()
	if err == nil || !strings.Contains(err.Error(), "rootfs.workers") {
		t.Fatalf("Validate invalid workers error = %v", err)
	}
}
