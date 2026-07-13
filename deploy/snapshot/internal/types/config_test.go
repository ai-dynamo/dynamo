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

func TestAgentConfigRejectsAgentStorageOverlappingCUDAHelperDirectory(t *testing.T) {
	for _, basePath := range []string{
		"/",
		"/run",
		"/run/cuda-checkpoint-helper",
		"/run/cuda-checkpoint-helper/checkpoints",
	} {
		cfg := validAgentConfig()
		cfg.Storage.BasePath = basePath
		if err := cfg.Validate(); err == nil {
			t.Errorf("Validate() accepted overlapping agent storage path %q", basePath)
		}
	}

	cfg := validAgentConfig()
	cfg.Storage.AccessMode = StorageAccessModePodMount
	cfg.Storage.BasePath = "/run/cuda-checkpoint-helper/checkpoints"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("pod-mounted storage does not collide in the agent pod: %v", err)
	}
}

func TestCUDATransferSettingsWithDefaults(t *testing.T) {
	got := (CUDATransferSettings{}).WithDefaults()
	if got.BufferCount != DefaultCUDATransferBufferCount || got.ChunkBytes != DefaultCUDATransferChunkBytes {
		t.Fatalf("WithDefaults() = %+v, want 1 slot and 64 MiB", got)
	}
}

func TestAgentConfigValidateCUDATransferSettings(t *testing.T) {
	cfg := validAgentConfig()
	bufferCount := 4
	chunkBytes := uint64(32 * 1024 * 1024)
	cfg.CUDACheckpoint.TransferBufferCount = &bufferCount
	cfg.CUDACheckpoint.TransferChunkBytes = &chunkBytes

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if got := cfg.CUDACheckpoint.TransferSettings(); got.BufferCount != bufferCount || got.ChunkBytes != chunkBytes {
		t.Fatalf("CUDA transfer settings = %+v, want count=%d chunk=%d", got, bufferCount, chunkBytes)
	}

	tooManyBuffers := 8
	tooLargeChunk := uint64(256 * 1024 * 1024)
	cfg.CUDACheckpoint.TransferBufferCount = &tooManyBuffers
	cfg.CUDACheckpoint.TransferChunkBytes = &tooLargeChunk
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected excessive per-device pinned memory to be rejected")
	}
}

func TestAgentConfigValidateDefaultsCUDATransferSettings(t *testing.T) {
	cfg := validAgentConfig()

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	settings := cfg.CUDACheckpoint.TransferSettings()
	if settings.BufferCount != DefaultCUDATransferBufferCount || settings.ChunkBytes != DefaultCUDATransferChunkBytes {
		t.Fatalf("CUDA transfer settings = %+v, want defaults", settings)
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
