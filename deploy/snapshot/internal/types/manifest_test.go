package types

import (
	"os"
	"path/filepath"
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"google.golang.org/protobuf/proto"
)

func TestManifestRoundTrip(t *testing.T) {
	dir := t.TempDir()

	original := NewCheckpointManifest(
		"sha256:abc123",
		CRIUDumpManifest{
			CRIU: CRIUSettings{
				LogLevel: 4,
				ShellJob: true,
				LibDir:   "/usr/lib/criu",
			},
			ExtMnt:   map[string]string{"/etc/hostname": "/etc/hostname", "/proc/acpi": "/dev/null"},
			External: []string{"net[12345]:extNetNs"},
			SkipMnt:  []string{"/proc/kcore"},
		},
		NewSourcePodManifest("ctr-abc", 42, "node-1", "my-pod", "default", "10.0.0.11", []string{"pipe:[111]", "pipe:[222]", "pipe:[333]"}),
		OverlayManifest{
			Exclusions:     OverlaySettings{Exclusions: []string{"/proc", "/sys"}},
			UpperDir:       "/var/lib/containerd/upper",
			ExternalPaths:  []string{"/proc/acpi"},
			BindMountDests: []string{"/data"},
		},
	)
	original.CUDA = NewCUDAManifest([]int{42, 43}, []string{"GPU-aaa", "GPU-bbb"})
	original.CUDA.VMMInterpose = true
	original.CUDA.VMMGeneration = "00112233445566778899aabbccddeeff"
	original.CUDA.VMMStateDigest = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"

	if err := WriteManifest(dir, original); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}

	loaded, err := ReadManifest(dir)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}

	// Verify key fields survived the round-trip
	if loaded.CheckpointID != original.CheckpointID {
		t.Errorf("CheckpointID = %q, want %q", loaded.CheckpointID, original.CheckpointID)
	}
	if loaded.CRIUDump.CRIU.LogLevel != 4 {
		t.Errorf("CRIU.LogLevel = %d, want 4", loaded.CRIUDump.CRIU.LogLevel)
	}
	if loaded.CRIUDump.CRIU.ShellJob != true {
		t.Error("CRIU.ShellJob should be true")
	}
	if len(loaded.CRIUDump.ExtMnt) != 2 {
		t.Errorf("ExtMnt count = %d, want 2", len(loaded.CRIUDump.ExtMnt))
	}
	if loaded.CRIUDump.ExtMnt["/etc/hostname"] != "/etc/hostname" {
		t.Errorf("ExtMnt[/etc/hostname] = %q", loaded.CRIUDump.ExtMnt["/etc/hostname"])
	}
	if len(loaded.CRIUDump.External) != 1 || loaded.CRIUDump.External[0] != "net[12345]:extNetNs" {
		t.Errorf("External = %v", loaded.CRIUDump.External)
	}
	if len(loaded.CRIUDump.SkipMnt) != 1 || loaded.CRIUDump.SkipMnt[0] != "/proc/kcore" {
		t.Errorf("SkipMnt = %v", loaded.CRIUDump.SkipMnt)
	}
	if loaded.K8s.PodName != "my-pod" {
		t.Errorf("K8s.PodName = %q", loaded.K8s.PodName)
	}
	if loaded.K8s.PodIP != "10.0.0.11" {
		t.Errorf("K8s.PodIP = %q", loaded.K8s.PodIP)
	}
	if len(loaded.K8s.StdioFDs) != 3 {
		t.Errorf("StdioFDs count = %d, want 3", len(loaded.K8s.StdioFDs))
	}
	if loaded.Overlay.UpperDir != "/var/lib/containerd/upper" {
		t.Errorf("Overlay.UpperDir = %q", loaded.Overlay.UpperDir)
	}
	if len(loaded.Overlay.BindMountDests) != 1 || loaded.Overlay.BindMountDests[0] != "/data" {
		t.Errorf("Overlay.BindMountDests = %v", loaded.Overlay.BindMountDests)
	}
	if len(loaded.CUDA.PIDs) != 2 || loaded.CUDA.PIDs[0] != 42 {
		t.Errorf("CUDA.PIDs = %v", loaded.CUDA.PIDs)
	}
	if len(loaded.CUDA.SourceGPUUUIDs) != 2 || loaded.CUDA.SourceGPUUUIDs[0] != "GPU-aaa" {
		t.Errorf("CUDA.SourceGPUUUIDs = %v", loaded.CUDA.SourceGPUUUIDs)
	}
	if !loaded.CUDA.VMMInterpose ||
		loaded.CUDA.VMMGeneration != original.CUDA.VMMGeneration ||
		loaded.CUDA.VMMStateDigest != original.CUDA.VMMStateDigest {
		t.Errorf(
			"CUDA VMM metadata = %t/%q/%q",
			loaded.CUDA.VMMInterpose,
			loaded.CUDA.VMMGeneration,
			loaded.CUDA.VMMStateDigest,
		)
	}
}

func TestManifestRejectsInconsistentVMMMetadata(t *testing.T) {
	validGeneration := "00112233445566778899aabbccddeeff"
	validDigest := "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
	tests := []struct {
		name string
		cuda CUDAManifest
	}{
		{name: "disabled with generation", cuda: CUDAManifest{VMMGeneration: validGeneration}},
		{name: "disabled with digest", cuda: CUDAManifest{VMMStateDigest: validDigest}},
		{name: "enabled without PIDs", cuda: CUDAManifest{
			VMMInterpose: true, VMMGeneration: validGeneration, VMMStateDigest: validDigest,
		}},
		{name: "enabled with invalid generation", cuda: CUDAManifest{
			PIDs: []int{1}, VMMInterpose: true, VMMGeneration: "generation", VMMStateDigest: validDigest,
		}},
		{name: "enabled without digest", cuda: CUDAManifest{
			PIDs: []int{1}, VMMInterpose: true, VMMGeneration: validGeneration,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.cuda.validate(); err == nil {
				t.Fatal("validation accepted inconsistent CUDA VMM metadata")
			}
		})
	}
}

func TestManifestIORejectsInconsistentVMMMetadata(t *testing.T) {
	manifest := &CheckpointManifest{CheckpointID: "checkpoint", CUDA: CUDAManifest{
		VMMGeneration: "00112233445566778899aabbccddeeff",
	}}
	if err := WriteManifest(t.TempDir(), manifest); err == nil {
		t.Fatal("WriteManifest accepted CUDA VMM metadata without opt-in")
	}

	dir := t.TempDir()
	if err := os.WriteFile(
		filepath.Join(dir, manifestFilename),
		[]byte(
			"checkpointId: checkpoint\ncudaRestore:\n  vmmGeneration: invalid\n",
		),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadManifest(dir); err == nil {
		t.Fatal("ReadManifest accepted CUDA VMM metadata without opt-in")
	}
}

func TestNewCRIUDumpManifest(t *testing.T) {
	t.Run("nil CriuOpts does not panic", func(t *testing.T) {
		m := NewCRIUDumpManifest(nil, CRIUSettings{LogLevel: 2})
		if m.CRIU.LogLevel != 2 {
			t.Errorf("LogLevel = %d, want 2", m.CRIU.LogLevel)
		}
		if m.ExtMnt != nil {
			t.Errorf("ExtMnt should be nil, got %v", m.ExtMnt)
		}
	})

	t.Run("extracts ExtMnt from protobuf correctly", func(t *testing.T) {
		opts := &criurpc.CriuOpts{
			ExtMnt: []*criurpc.ExtMountMap{
				{Key: proto.String("/etc/hosts"), Val: proto.String("/etc/hosts")},
				{Key: proto.String("/proc/acpi"), Val: proto.String("/dev/null")},
				// nil entry and empty key should be skipped
				nil,
				{Key: proto.String(""), Val: proto.String("ignored")},
			},
			External: []string{"net[1234]:extNetNs"},
			SkipMnt:  []string{"/proc/kcore", "/sys/firmware"},
		}

		m := NewCRIUDumpManifest(opts, CRIUSettings{})
		if len(m.ExtMnt) != 2 {
			t.Fatalf("ExtMnt count = %d, want 2; got %v", len(m.ExtMnt), m.ExtMnt)
		}
		if m.ExtMnt["/etc/hosts"] != "/etc/hosts" {
			t.Errorf("ExtMnt[/etc/hosts] = %q", m.ExtMnt["/etc/hosts"])
		}
		if m.ExtMnt["/proc/acpi"] != "/dev/null" {
			t.Errorf("ExtMnt[/proc/acpi] = %q", m.ExtMnt["/proc/acpi"])
		}
		if len(m.External) != 1 {
			t.Errorf("External = %v", m.External)
		}
		if len(m.SkipMnt) != 2 {
			t.Errorf("SkipMnt = %v", m.SkipMnt)
		}
	})

	t.Run("empty ExtMnt entries results in nil map", func(t *testing.T) {
		opts := &criurpc.CriuOpts{
			ExtMnt: []*criurpc.ExtMountMap{
				nil,
				{Key: proto.String(""), Val: proto.String("x")},
			},
		}
		m := NewCRIUDumpManifest(opts, CRIUSettings{})
		if m.ExtMnt != nil {
			t.Errorf("expected nil ExtMnt when all entries are empty/nil, got %v", m.ExtMnt)
		}
	})
}

func TestWriteManifestRejectsMissingCheckpointID(t *testing.T) {
	dir := t.TempDir()

	err := WriteManifest(dir, &CheckpointManifest{})
	if err == nil || err.Error() != "checkpoint manifest is missing checkpointId" {
		t.Fatalf("expected missing checkpointId error, got %v", err)
	}
}

func TestReadManifestRejectsMissingCheckpointID(t *testing.T) {
	dir := t.TempDir()

	content := []byte("createdAt: 2026-03-31T00:00:00Z\n")
	if err := os.WriteFile(filepath.Join(dir, manifestFilename), content, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := ReadManifest(dir)
	if err == nil || err.Error() != "checkpoint manifest is missing checkpointId" {
		t.Fatalf("expected missing checkpointId error, got %v", err)
	}
}
