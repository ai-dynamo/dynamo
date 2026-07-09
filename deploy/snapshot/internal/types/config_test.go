package types

import (
	"fmt"
	"testing"
)

func validAgentConfig() *AgentConfig {
	return &AgentConfig{
		Storage: StorageSpec{
			Type:     "pvc",
			BasePath: "/checkpoints",
		},
		Restore: RestoreSpec{
			NSRestorePath:         "/usr/local/bin/nsrestore",
			RestoreTimeoutSeconds: 60,
		},
	}
}

func TestAgentConfigValidateRootFSWorkers(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		cfg := validAgentConfig()
		if err := cfg.Validate(); err != nil {
			t.Fatalf("Validate() error = %v", err)
		}
		if cfg.RootFS.Workers != DefaultRootFSWorkers {
			t.Fatalf(
				"RootFS.Workers = %d, want %d",
				cfg.RootFS.Workers,
				DefaultRootFSWorkers,
			)
		}
	})

	for _, workers := range []int{-1, MaxRootFSWorkers + 1} {
		t.Run(fmt.Sprintf("rejects_%d", workers), func(t *testing.T) {
			cfg := validAgentConfig()
			cfg.RootFS.Workers = workers
			if err := cfg.Validate(); err == nil {
				t.Fatalf("Validate() accepted %d workers", workers)
			}
		})
	}
}

func TestAgentConfigValidateRequiresAbsoluteStorageBasePath(t *testing.T) {
	cfg := validAgentConfig()
	cfg.Storage.BasePath = "checkpoints"

	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for relative storage base path")
	}
}

func TestAgentConfigValidateNormalizesStorageFields(t *testing.T) {
	cfg := validAgentConfig()
	cfg.Storage.BasePath = " /checkpoints "
	cfg.Storage.AccessMode = " podMount "

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if cfg.Storage.BasePath != "/checkpoints" {
		t.Fatalf("Storage.BasePath = %q, want %q", cfg.Storage.BasePath, "/checkpoints")
	}
	if cfg.Storage.AccessMode != StorageAccessModePodMount {
		t.Fatalf("Storage.AccessMode = %q, want %q", cfg.Storage.AccessMode, StorageAccessModePodMount)
	}
}

func TestAgentConfigValidateDefaultsStorageAccessMode(t *testing.T) {
	cfg := validAgentConfig()

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if cfg.Storage.AccessMode != StorageAccessModeAgentMount {
		t.Fatalf("Storage.AccessMode = %q, want %q", cfg.Storage.AccessMode, StorageAccessModeAgentMount)
	}
}
