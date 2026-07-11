package types

import "testing"

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

func TestAgentConfigValidateDefaultsCUDAStorageMode(t *testing.T) {
	cfg := validAgentConfig()

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if cfg.CUDACheckpoint.StorageMode != CUDAStorageModeLegacy {
		t.Fatalf("CUDA storage mode = %q, want %q", cfg.CUDACheckpoint.StorageMode, CUDAStorageModeLegacy)
	}
}

func TestAgentConfigValidateCUDAStorageMode(t *testing.T) {
	cfg := validAgentConfig()
	cfg.CUDACheckpoint.StorageMode = " POSIX "

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if cfg.CUDACheckpoint.StorageMode != CUDAStorageModePOSIX {
		t.Fatalf("CUDA storage mode = %q, want %q", cfg.CUDACheckpoint.StorageMode, CUDAStorageModePOSIX)
	}

	cfg.CUDACheckpoint.StorageMode = "object-store"
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected unsupported CUDA storage mode error")
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
