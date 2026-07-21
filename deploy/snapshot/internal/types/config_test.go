package types

import "testing"

func uint32Ptr(value uint32) *uint32 {
	return &value
}

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

func TestCRIUSettingsValidateCompression(t *testing.T) {
	tests := []struct {
		name     string
		settings CRIUSettings
		wantMode string
		wantErr  bool
	}{
		{name: "empty defaults off", settings: CRIUSettings{}, wantMode: CRIUCompressionModeOff},
		{name: "per page defaults acceleration", settings: CRIUSettings{CompressionMode: " PER-PAGE "}, wantMode: CRIUCompressionModePerPage},
		{name: "region defaults size", settings: CRIUSettings{CompressionMode: CRIUCompressionModeRegion}, wantMode: CRIUCompressionModeRegion},
		{name: "automatic decompression", settings: CRIUSettings{DecompressThreads: uint32Ptr(0)}, wantMode: CRIUCompressionModeOff},
		{name: "invalid mode", settings: CRIUSettings{CompressionMode: "gzip"}, wantErr: true},
		{name: "invalid acceleration", settings: CRIUSettings{CompressionMode: CRIUCompressionModePerPage, CompressionAcceleration: MaxCRIUCompressionAcceleration + 1}, wantErr: true},
		{name: "unaligned region", settings: CRIUSettings{CompressionMode: CRIUCompressionModeRegion, CompressionRegionSize: 12345}, wantErr: true},
		{name: "oversized region", settings: CRIUSettings{CompressionMode: CRIUCompressionModeRegion, CompressionRegionSize: MaxCRIUCompressionRegionSize + criuPageSize}, wantErr: true},
		{name: "too many threads", settings: CRIUSettings{DecompressThreads: uint32Ptr(MaxCRIUDecompressThreads + 1)}, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.settings.Validate()
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected validation error")
				}
				return
			}
			if err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			if tc.settings.CompressionMode != tc.wantMode {
				t.Fatalf("CompressionMode = %q, want %q", tc.settings.CompressionMode, tc.wantMode)
			}
			if tc.wantMode != CRIUCompressionModeOff && tc.settings.CompressionAcceleration != DefaultCRIUCompressionAcceleration {
				t.Fatalf("CompressionAcceleration = %d, want %d", tc.settings.CompressionAcceleration, DefaultCRIUCompressionAcceleration)
			}
			if tc.wantMode == CRIUCompressionModeRegion && tc.settings.CompressionRegionSize != DefaultCRIUCompressionRegionSize {
				t.Fatalf("CompressionRegionSize = %d, want %d", tc.settings.CompressionRegionSize, DefaultCRIUCompressionRegionSize)
			}
		})
	}
}
