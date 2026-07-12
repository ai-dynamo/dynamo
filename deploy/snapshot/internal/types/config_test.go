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
		cfg.CUDACheckpoint.StorageMode = CUDAStorageModePOSIX
		cfg.CUDACheckpoint.DaemonSocketPath = "/run/cuda-checkpoint-helper/helper.sock"
		if err := cfg.Validate(); err == nil {
			t.Errorf("Validate() accepted overlapping agent storage path %q", basePath)
		}
	}

	cfg := validAgentConfig()
	cfg.Storage.AccessMode = StorageAccessModePodMount
	cfg.Storage.BasePath = "/run/cuda-checkpoint-helper/checkpoints"
	cfg.CUDACheckpoint.StorageMode = CUDAStorageModePOSIX
	cfg.CUDACheckpoint.DaemonSocketPath = "/run/cuda-checkpoint-helper/helper.sock"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("pod-mounted storage does not collide in the agent pod: %v", err)
	}
}

func TestAgentConfigRejectsUnsafeCUDAHelperSocketPaths(t *testing.T) {
	for _, socket := range []string{
		"/helper.sock",
		"/host/proc/helper.sock",
		"/run/cuda-checkpoint-helper/../helper.sock",
		"/run/cuda-checkpoint-helper/" + "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.sock",
	} {
		cfg := validAgentConfig()
		cfg.CUDACheckpoint.StorageMode = CUDAStorageModePOSIX
		cfg.CUDACheckpoint.DaemonSocketPath = socket
		if err := cfg.Validate(); err == nil {
			t.Errorf("Validate() accepted unsafe daemon socket path %q", socket)
		}
	}
}

func TestAgentConfigValidateCUDAHelperDaemon(t *testing.T) {
	cfg := validAgentConfig()
	cfg.CUDACheckpoint.StorageMode = CUDAStorageModePOSIX
	cfg.CUDACheckpoint.DaemonSocketPath = "/run/cuda-checkpoint-helper/helper.sock"
	cfg.CUDACheckpoint.DaemonFallback = true
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	settings := cfg.CUDACheckpoint.TransferSettings()
	if settings.DaemonSocket != cfg.CUDACheckpoint.DaemonSocketPath || !settings.DaemonFallback {
		t.Fatalf("TransferSettings() = %+v, want daemon configuration", settings)
	}

	cfg.CUDACheckpoint.DaemonSocketPath = "relative.sock"
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected relative daemon socket path to be rejected")
	}
	cfg.CUDACheckpoint.DaemonSocketPath = "/run/cuda-checkpoint-helper/helper.sock"
	cfg.CUDACheckpoint.StorageMode = CUDAStorageModeLegacy
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected daemon socket with legacy storage mode to be rejected")
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

func TestAgentConfigValidateDefaultsCUDAStorageMode(t *testing.T) {
	cfg := validAgentConfig()

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if cfg.CUDACheckpoint.StorageMode != CUDAStorageModeLegacy {
		t.Fatalf("CUDA storage mode = %q, want %q", cfg.CUDACheckpoint.StorageMode, CUDAStorageModeLegacy)
	}
	settings := cfg.CUDACheckpoint.TransferSettings()
	if settings.BufferCount != DefaultCUDATransferBufferCount || settings.ChunkBytes != DefaultCUDATransferChunkBytes {
		t.Fatalf("CUDA transfer settings = %+v, want defaults", settings)
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
