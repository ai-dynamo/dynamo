package common

import (
	"os"
	"path/filepath"
	"testing"

	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

func TestClassifyMounts(t *testing.T) {
	tests := []struct {
		name    string
		mounts  []types.MountInfo
		ociSpec *specs.Spec
		rootFS  string
		want    map[string]bool // mountpoint → expected IsOCIManaged
	}{
		{
			name: "mount matching OCI destination",
			mounts: []types.MountInfo{
				{MountPoint: "/etc/hostname"},
			},
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{{Destination: "/etc/hostname"}},
			},
			want: map[string]bool{"/etc/hostname": true},
		},
		{
			name: "mount with no OCI match",
			mounts: []types.MountInfo{
				{MountPoint: "/some/random/path"},
			},
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{{Destination: "/etc/hostname"}},
			},
			want: map[string]bool{"/some/random/path": false},
		},
		{
			name: "/run/ mount aliased to /var/run/ in OCI spec",
			mounts: []types.MountInfo{
				{MountPoint: "/run/secrets"},
			},
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{{Destination: "/var/run/secrets"}},
			},
			want: map[string]bool{"/run/secrets": true},
		},
		{
			name: "/var/run/ mount aliased to /run/ in OCI spec",
			mounts: []types.MountInfo{
				{MountPoint: "/var/run/secrets"},
			},
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{{Destination: "/run/secrets"}},
			},
			want: map[string]bool{"/var/run/secrets": true},
		},
		{
			name: "/run/ prefix without alias match stays unmanaged",
			mounts: []types.MountInfo{
				{MountPoint: "/run/other"},
			},
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{{Destination: "/var/run/different"}},
			},
			want: map[string]bool{"/run/other": false},
		},
		{
			name:   "nil OCI spec classifies nothing",
			mounts: []types.MountInfo{{MountPoint: "/etc/hostname"}},
			want:   map[string]bool{"/etc/hostname": false},
		},
		{
			name: "masked and readonly paths are OCI-managed",
			mounts: []types.MountInfo{
				{MountPoint: "/proc/acpi"},
				{MountPoint: "/proc/sys"},
			},
			ociSpec: &specs.Spec{
				Linux: &specs.Linux{
					MaskedPaths:   []string{"/proc/acpi"},
					ReadonlyPaths: []string{"/proc/sys"},
				},
			},
			want: map[string]bool{
				"/proc/acpi": true,
				"/proc/sys":  true,
			},
		},
		{
			name:   "empty mounts slice",
			mounts: []types.MountInfo{},
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{{Destination: "/etc/hostname"}},
			},
			want: map[string]bool{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := ClassifyMounts(tc.mounts, tc.ociSpec, tc.rootFS)
			for _, m := range result {
				expected, ok := tc.want[m.MountPoint]
				if !ok {
					continue
				}
				if m.IsOCIManaged != expected {
					t.Errorf("mount %q: IsOCIManaged = %v, want %v", m.MountPoint, m.IsOCIManaged, expected)
				}
			}
		})
	}
}

func TestBuildMountPolicy(t *testing.T) {
	tests := []struct {
		name         string
		mounts       []types.MountInfo
		rootFS       string
		maskedPaths  []string
		wantExt      map[string]string // expected entries in extMap
		wantSkipped  []string          // expected entries in skipped
		wantNotInExt []string          // keys that must NOT be in extMap
	}{
		{
			name: "non-OCI /proc submount is skipped",
			mounts: []types.MountInfo{
				{MountPoint: "/proc/kcore", IsOCIManaged: false},
			},
			wantSkipped:  []string{"/proc/kcore"},
			wantNotInExt: []string{"/proc/kcore"},
		},
		{
			name: "non-OCI /sys submount is skipped",
			mounts: []types.MountInfo{
				{MountPoint: "/sys/firmware", IsOCIManaged: false},
			},
			wantSkipped:  []string{"/sys/firmware"},
			wantNotInExt: []string{"/sys/firmware"},
		},
		{
			name: "non-OCI /run submount is skipped",
			mounts: []types.MountInfo{
				{MountPoint: "/run/some-daemon", IsOCIManaged: false},
			},
			wantSkipped:  []string{"/run/some-daemon"},
			wantNotInExt: []string{"/run/some-daemon"},
		},
		{
			name: "OCI-managed /proc submount is externalized, not skipped",
			mounts: []types.MountInfo{
				{MountPoint: "/proc/acpi", IsOCIManaged: true},
			},
			wantExt: map[string]string{"/proc/acpi": "/proc/acpi"},
		},
		{
			name: "/dev/shm tmpfs is not externalized",
			mounts: []types.MountInfo{
				{MountPoint: "/dev/shm", FSType: "tmpfs"},
			},
			wantNotInExt: []string{"/dev/shm"},
		},
		{
			name: "/dev/shm non-tmpfs is externalized",
			mounts: []types.MountInfo{
				{MountPoint: "/dev/shm", FSType: "bind"},
			},
			wantExt: map[string]string{"/dev/shm": "/dev/shm"},
		},
		{
			name: "normal mount is externalized",
			mounts: []types.MountInfo{
				{MountPoint: "/etc/hostname", IsOCIManaged: true},
			},
			wantExt: map[string]string{"/etc/hostname": "/etc/hostname"},
		},
		{
			name: "empty mount point is ignored",
			mounts: []types.MountInfo{
				{MountPoint: ""},
			},
			wantExt: map[string]string{},
		},
		{
			name: "masked path non-dir file maps to /dev/null",
			mounts: []types.MountInfo{},
			rootFS: func() string {
				dir := t.TempDir()
				os.WriteFile(filepath.Join(dir, "proc"), []byte("x"), 0644)
				return dir
			}(),
			maskedPaths: []string{"/proc"},
			wantExt:     map[string]string{"/proc": "/dev/null"},
		},
		{
			name: "masked path directory is ignored",
			mounts: []types.MountInfo{},
			rootFS: func() string {
				dir := t.TempDir()
				os.MkdirAll(filepath.Join(dir, "proc"), 0755)
				return dir
			}(),
			maskedPaths:  []string{"/proc"},
			wantNotInExt: []string{"/proc"},
		},
		{
			name:         "masked path that doesn't exist is ignored",
			mounts:       []types.MountInfo{},
			rootFS:       t.TempDir(),
			maskedPaths:  []string{"/nonexistent"},
			wantNotInExt: []string{"/nonexistent"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			extMap, skipped := BuildMountPolicy(tc.mounts, tc.rootFS, tc.maskedPaths)

			for k, v := range tc.wantExt {
				got, ok := extMap[k]
				if !ok {
					t.Errorf("expected extMap[%q] to exist", k)
					continue
				}
				if got != v {
					t.Errorf("extMap[%q] = %q, want %q", k, got, v)
				}
			}
			for _, k := range tc.wantNotInExt {
				if _, ok := extMap[k]; ok {
					t.Errorf("extMap should not contain %q", k)
				}
			}
			if tc.wantSkipped != nil {
				skippedSet := make(map[string]struct{}, len(skipped))
				for _, s := range skipped {
					skippedSet[s] = struct{}{}
				}
				for _, want := range tc.wantSkipped {
					if _, ok := skippedSet[want]; !ok {
						t.Errorf("expected %q in skipped list, got %v", want, skipped)
					}
				}
			}
		})
	}
}

func TestNormalizeOCIPath(t *testing.T) {
	tests := []struct {
		name   string
		raw    string
		rootFS string
		want   string
	}{
		{name: "normal absolute path", raw: "/etc/hostname", want: "/etc/hostname"},
		{name: "empty string", raw: "", want: ""},
		{name: "whitespace only", raw: "   ", want: ""},
		{name: "dot path", raw: ".", want: ""},
		{name: "path with trailing slashes cleaned", raw: "/etc/hostname///", want: "/etc/hostname"},
		{
			name: "with rootFS strips prefix via securejoin",
			raw:  "/etc/hostname",
			// SecureJoin(rootFS, "/etc/hostname") → rootFS+"/etc/hostname", then strip rootFS prefix
			rootFS: "/tmp/fakefs",
			want:   "/etc/hostname",
		},
		{
			name:   "root path with rootFS returns /",
			raw:    "/",
			rootFS: "/tmp/fakefs",
			want:   "/",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := normalizeOCIPath(tc.raw, tc.rootFS)
			if got != tc.want {
				t.Errorf("normalizeOCIPath(%q, %q) = %q, want %q", tc.raw, tc.rootFS, got, tc.want)
			}
		})
	}
}
