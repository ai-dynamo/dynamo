package runtime

import (
	"os"
	"path/filepath"
	"testing"

	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const testCRIOContainerID = "1122334455667788990011223344556677889900aabbccddeeff00112233"

// TestParseCRIOStatusInfoPath loads a real crictl-shaped payload and asserts
// every field downstream callers actually read survives the parse step.
func TestParseCRIOStatusInfoPath(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "crio_info.json"))
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}

	pid, spec, err := parseCRIOStatus(testCRIOContainerID, map[string]string{"info": string(raw)})
	if err != nil {
		t.Fatalf("parseCRIOStatus: %v", err)
	}

	if pid != 54321 {
		t.Errorf("pid = %d, want 54321", pid)
	}
	if len(spec.Mounts) != 3 {
		t.Fatalf("Mounts len = %d, want 3", len(spec.Mounts))
	}
	var sawHostname bool
	for _, m := range spec.Mounts {
		if m.Destination == "/etc/hostname" && m.Type == "bind" {
			sawHostname = true
		}
	}
	if !sawHostname {
		t.Error("expected /etc/hostname bind mount to survive parse")
	}
	if spec.Linux == nil {
		t.Fatal("expected Linux section to be populated")
	}
	if len(spec.Linux.MaskedPaths) != 3 {
		t.Errorf("MaskedPaths len = %d, want 3", len(spec.Linux.MaskedPaths))
	}
	if len(spec.Linux.ReadonlyPaths) != 4 {
		t.Errorf("ReadonlyPaths len = %d, want 4", len(spec.Linux.ReadonlyPaths))
	}
}

func TestParseCRIOStatusErrors(t *testing.T) {
	cases := map[string]map[string]string{
		"missing info key":    {},
		"malformed JSON":      {"info": "{{{bad"},
		"missing pid":         {"info": `{"runtimeSpec":{}}`},
		"missing runtimeSpec": {"info": `{"pid":42}`},
	}
	for name, info := range cases {
		t.Run(name, func(t *testing.T) {
			if _, _, err := parseCRIOStatus(testCRIOContainerID, info); err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

// TestCollectOCIManagedPathsPartialSpec confirms the helper doesn't panic on
// partial specs the CRI-O path can produce.
func TestCollectOCIManagedPathsPartialSpec(t *testing.T) {
	set := collectOCIManagedPaths(&specs.Spec{
		Mounts: []specs.Mount{{Destination: "/data"}},
	}, "")
	if _, ok := set["/data"]; !ok || len(set) != 1 {
		t.Errorf("want just /data, got %v", set)
	}
}

func TestStripCRIScheme(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"containerd scheme", "containerd://abc123", "abc123"},
		{"hyphenated cri-o scheme", "cri-o://abc123", "abc123"},
		{"non-hyphenated crio scheme", "crio://abc123", "abc123"},
		{"no scheme", "abc123", "abc123"},
		{"empty string", "", ""},
		{"unknown scheme left intact", "docker://abc123", "docker://abc123"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := StripCRIScheme(tc.in); got != tc.want {
				t.Errorf("StripCRIScheme(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
