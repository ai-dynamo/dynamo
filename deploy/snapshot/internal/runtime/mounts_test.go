package runtime

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestClassifyMounts(t *testing.T) {
	tests := []struct {
		name    string
		mounts  []types.MountInfo
		ociSpec *specs.Spec
		rootFS  string
		setup   func(*testing.T) string
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
			setup: rootFSWithSymlink("var/run", "../run"),
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
			setup: rootFSWithSymlink("run", "var/run"),
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
			setup: rootFSWithSymlink("var/run", "../run"),
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{{Destination: "/var/run/different"}},
			},
			want: map[string]bool{"/run/other": false},
		},
		{
			name: "arbitrary nested rootfs symlink alias",
			mounts: []types.MountInfo{
				{MountPoint: "/srv/volumes/model/cache"},
			},
			setup: rootFSWithSymlink(
				"opt/app/current",
				"../../../srv/volumes/model",
			),
			ociSpec: &specs.Spec{
				Mounts: []specs.Mount{
					{Destination: "/opt/app/current/cache"},
				},
			},
			want: map[string]bool{"/srv/volumes/model/cache": true},
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
			rootFS := tc.rootFS
			if tc.setup != nil {
				rootFS = tc.setup(t)
			}
			result, err := ClassifyMounts(tc.mounts, tc.ociSpec, rootFS)
			if err != nil {
				t.Fatal(err)
			}
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
			name:   "masked path non-dir file maps to /dev/null",
			mounts: []types.MountInfo{},
			rootFS: func() string {
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, "proc"), []byte("x"), 0644); err != nil {
					t.Fatalf("write masked file: %v", err)
				}
				return dir
			}(),
			maskedPaths: []string{"/proc"},
			wantExt:     map[string]string{"/proc": "/dev/null"},
		},
		{
			name:   "masked path directory is ignored",
			mounts: []types.MountInfo{},
			rootFS: func() string {
				dir := t.TempDir()
				if err := os.MkdirAll(filepath.Join(dir, "proc"), 0755); err != nil {
					t.Fatalf("mkdir masked dir: %v", err)
				}
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
		name string
		raw  string
		want string
	}{
		{name: "normal absolute path", raw: "/etc/hostname", want: "/etc/hostname"},
		{name: "path with trailing slashes cleaned", raw: "/etc/hostname///", want: "/etc/hostname"},
		{name: "root path with rootFS returns root", raw: "/", want: "/"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeOCIPath(tc.raw, t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Errorf("normalizeOCIPath(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}

	for _, raw := range []string{"", "   ", ".", "relative", "/a/../b", "/a/./b"} {
		t.Run("invalid_"+strings.ReplaceAll(raw, "/", "_"), func(t *testing.T) {
			if _, err := normalizeOCIPath(raw, t.TempDir()); err == nil {
				t.Fatalf("normalizeOCIPath(%q) succeeded", raw)
			}
		})
	}
}

func TestBuildRootfsMountExclusionsAliases(t *testing.T) {
	tests := []struct {
		name        string
		setup       func(*testing.T) string
		destination string
		mountPoint  string
		want        []string
	}{
		{
			name:        "var run resolves to run",
			setup:       rootFSWithSymlink("var/run", "../run"),
			destination: "/var/run/secrets",
			mountPoint:  "/run/secrets",
			want:        []string{"/run/secrets", "/var/run/secrets"},
		},
		{
			name:        "run resolves to var run",
			setup:       rootFSWithSymlink("run", "var/run"),
			destination: "/run/secrets",
			mountPoint:  "/var/run/secrets",
			want:        []string{"/run/secrets", "/var/run/secrets"},
		},
		{
			name: "nested alias",
			setup: rootFSWithSymlink(
				"opt/app/current",
				"../../../srv/volumes/model",
			),
			destination: "/opt/app/current/cache",
			mountPoint:  "/srv/volumes/model/cache",
			want: []string{
				"/opt/app/current/cache",
				"/srv/volumes/model/cache",
			},
		},
		{
			name:        "raw path without symlink",
			setup:       func(t *testing.T) string { return t.TempDir() },
			destination: "/data",
			mountPoint:  "/data",
			want:        []string{"/data"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rootFS := test.setup(t)
			spec := &specs.Spec{Mounts: []specs.Mount{{
				Destination: test.destination,
				Type:        "bind",
			}}}
			mounts, err := ClassifyMounts(
				[]types.MountInfo{{MountPoint: test.mountPoint}},
				spec,
				rootFS,
			)
			if err != nil {
				t.Fatal(err)
			}
			got, err := BuildRootfsMountExclusions(spec, mounts, rootFS)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.All, test.want) {
				t.Fatalf("All = %v, want %v", got.All, test.want)
			}
		})
	}
}

func TestBuildRootfsMountExclusionsRequiresCanonicalMountinfo(t *testing.T) {
	rootFS := rootFSWithSymlink("var/run", "../run")(t)
	spec := &specs.Spec{Mounts: []specs.Mount{{
		Destination: "/var/run/secrets",
		Type:        "bind",
	}}}

	for _, test := range []struct {
		name   string
		mounts []types.MountInfo
	}{
		{name: "missing"},
		{
			name: "disagrees",
			mounts: []types.MountInfo{{
				MountPoint:   "/var/run/secrets",
				IsOCIManaged: true,
			}},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := BuildRootfsMountExclusions(spec, test.mounts, rootFS)
			if err == nil {
				t.Fatal("BuildRootfsMountExclusions succeeded")
			}
			if want := `canonical OCI bind mountpoint "/run/secrets"`; !strings.Contains(
				err.Error(),
				want,
			) {
				t.Fatalf("error = %q, want %q", err, want)
			}
		})
	}

	t.Run("matching classification disagreement", func(t *testing.T) {
		got, err := BuildRootfsMountExclusions(
			spec,
			[]types.MountInfo{{
				MountPoint:   "/run/secrets",
				IsOCIManaged: false,
			}},
			rootFS,
		)
		if err != nil {
			t.Fatal(err)
		}
		if want := []string{
			"/run/secrets",
			"/var/run/secrets",
		}; !reflect.DeepEqual(got.All, want) {
			t.Fatalf("All = %v, want %v", got.All, want)
		}
	})
}

func TestBuildRootfsMountExclusionsOnlyIncludesCanonicalBinds(t *testing.T) {
	spec := &specs.Spec{
		Mounts: []specs.Mount{{
			Destination: "/data",
			Type:        "bind",
		}},
		Linux: &specs.Linux{
			MaskedPaths:   []string{"/proc/acpi"},
			ReadonlyPaths: []string{"/proc/sys"},
		},
	}
	got, err := BuildRootfsMountExclusions(
		spec,
		[]types.MountInfo{
			{MountPoint: "/data"},
			{MountPoint: "/run/nvidia", IsOCIManaged: true},
			{MountPoint: "/proc/acpi", IsOCIManaged: true},
			{MountPoint: "/proc/sys", IsOCIManaged: true},
		},
		t.TempDir(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"/data"}; !reflect.DeepEqual(got.All, want) {
		t.Fatalf("All = %v, want %v", got.All, want)
	}
}

func TestBuildRootfsMountExclusionsDeduplicatesCanonicalAliases(t *testing.T) {
	rootFS := rootFSWithSymlink("var/run", "../run")(t)
	spec := &specs.Spec{Mounts: []specs.Mount{
		{Destination: "/var/run/secrets", Type: "bind"},
		{Destination: "/run/secrets", Type: "bind"},
		{Destination: "/var/run/secrets", Type: "bind"},
	}}
	got, err := BuildRootfsMountExclusions(
		spec,
		[]types.MountInfo{{MountPoint: "/run/secrets"}},
		rootFS,
	)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"/run/secrets"}; !reflect.DeepEqual(
		got.Effective,
		want,
	) {
		t.Fatalf("Effective = %v, want %v", got.Effective, want)
	}
	if want := []string{
		"/run/secrets",
		"/var/run/secrets",
	}; !reflect.DeepEqual(got.All, want) {
		t.Fatalf("All = %v, want %v", got.All, want)
	}
}

func TestBuildRootfsMountExclusionsFailsClosedOnSymlinkChange(t *testing.T) {
	rootFS := rootFSWithSymlink("var/run", "../run")(t)
	spec := &specs.Spec{Mounts: []specs.Mount{{
		Destination: "/var/run/secrets",
		Type:        "bind",
	}}}
	mounts, err := ClassifyMounts(
		[]types.MountInfo{{MountPoint: "/run/secrets"}},
		spec,
		rootFS,
	)
	if err != nil {
		t.Fatal(err)
	}
	linkPath := filepath.Join(rootFS, "var", "run")
	if err := os.Remove(linkPath); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../other", linkPath); err != nil {
		t.Fatal(err)
	}

	_, err = BuildRootfsMountExclusions(spec, mounts, rootFS)
	if err == nil {
		t.Fatal("BuildRootfsMountExclusions succeeded")
	}
	if want := `canonical OCI bind mountpoint "/other/secrets"`; !strings.Contains(
		err.Error(),
		want,
	) {
		t.Fatalf("error = %q, want %q", err, want)
	}
}

func TestBuildRootfsMountExclusionsDeterministic(t *testing.T) {
	rootFS := rootFSWithSymlink("var/run", "../run")(t)
	spec := &specs.Spec{Mounts: []specs.Mount{
		{Destination: "/var/run/a", Type: "bind"},
		{Destination: "/plain", Type: "bind"},
		{Destination: "/run/a", Type: "bind"},
		{Destination: "/var/run/a", Type: "bind"},
	}}
	mounts, err := ClassifyMounts([]types.MountInfo{
		{MountPoint: "/plain"},
		{MountPoint: "/run/a"},
	}, spec, rootFS)
	if err != nil {
		t.Fatal(err)
	}
	got, err := BuildRootfsMountExclusions(spec, mounts, rootFS)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"/plain", "/run/a", "/var/run/a"}; !reflect.DeepEqual(got.All, want) {
		t.Fatalf("All = %v, want %v", got.All, want)
	}
	if want := []string{"/plain", "/run/a", "/var/run/a"}; !reflect.DeepEqual(got.Raw, want) {
		t.Fatalf("Raw = %v, want %v", got.Raw, want)
	}
	if want := []string{"/plain", "/run/a"}; !reflect.DeepEqual(got.Effective, want) {
		t.Fatalf("Effective = %v, want %v", got.Effective, want)
	}
}

func TestBuildRootfsMountExclusionsRejectsInvalidDestinations(t *testing.T) {
	for _, destination := range []string{
		"",
		"relative",
		".",
		"..",
		"/",
		"/data/../escape",
		"/data/./ambiguous",
		"/data\x00suffix",
		"/" + strings.Repeat("a", maxRootfsPathLength),
	} {
		t.Run(strings.ReplaceAll(destination, "/", "_"), func(t *testing.T) {
			_, err := BuildRootfsMountExclusions(
				&specs.Spec{Mounts: []specs.Mount{{
					Destination: destination,
					Type:        "bind",
				}}},
				nil,
				t.TempDir(),
			)
			if err == nil {
				t.Fatalf("destination %q succeeded", destination)
			}
		})
	}

	t.Run("symlink resolution failure", func(t *testing.T) {
		rootFS := t.TempDir()
		if err := os.Symlink("loop", filepath.Join(rootFS, "loop")); err != nil {
			t.Fatal(err)
		}
		_, err := BuildRootfsMountExclusions(
			&specs.Spec{Mounts: []specs.Mount{{
				Destination: "/loop/data",
				Type:        "bind",
			}}},
			nil,
			rootFS,
		)
		if err == nil {
			t.Fatal("symlink loop succeeded")
		}
	})
}

func TestBuildRootfsMountExclusionsForBindLikeVolumes(t *testing.T) {
	destinations := []string{
		"/projected",
		"/config-map",
		"/secret",
		"/pvc",
		"/empty-dir",
		"/sub-path",
	}
	spec := &specs.Spec{}
	mounts := make([]types.MountInfo, 0, len(destinations))
	rootFS := t.TempDir()
	for _, destination := range destinations {
		spec.Mounts = append(spec.Mounts, specs.Mount{
			Destination: destination,
			Type:        "bind",
		})
		mounts = append(mounts, types.MountInfo{MountPoint: destination})
	}
	classified, err := ClassifyMounts(mounts, spec, rootFS)
	if err != nil {
		t.Fatal(err)
	}
	got, err := BuildRootfsMountExclusions(spec, classified, rootFS)
	if err != nil {
		t.Fatal(err)
	}
	want := append([]string(nil), destinations...)
	sort.Strings(want)
	if !reflect.DeepEqual(got.All, want) {
		t.Fatalf("All = %v, want %v", got.All, want)
	}
}

func rootFSWithSymlink(link, target string) func(*testing.T) string {
	return func(t *testing.T) string {
		t.Helper()
		rootFS := t.TempDir()
		linkPath := filepath.Join(rootFS, link)
		if err := os.MkdirAll(filepath.Dir(linkPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, linkPath); err != nil {
			t.Fatal(err)
		}
		return rootFS
	}
}
