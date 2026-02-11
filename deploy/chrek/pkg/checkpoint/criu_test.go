package checkpoint

import (
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

func TestBuildCRIUOpts(t *testing.T) {
	tests := []struct {
		name string
		cfg  CRIUConfig
	}{
		{
			name: "basic config",
			cfg: CRIUConfig{
				PID:        12345,
				ImageDirFD: 5,
				RootFS:     "/",
				GhostLimit: 0,
				Timeout:    0,
			},
		},
		{
			name: "config with ghost limit",
			cfg: CRIUConfig{
				PID:        99999,
				ImageDirFD: 10,
				RootFS:     "/container/root",
				GhostLimit: 1048576,
				Timeout:    0,
			},
		},
		{
			name: "config with timeout",
			cfg: CRIUConfig{
				PID:        54321,
				ImageDirFD: 3,
				RootFS:     "/",
				GhostLimit: 0,
				Timeout:    120,
			},
		},
		{
			name: "config with both ghost limit and timeout",
			cfg: CRIUConfig{
				PID:        11111,
				ImageDirFD: 7,
				RootFS:     "/rootfs",
				GhostLimit: 2097152,
				Timeout:    300,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := BuildCRIUOpts(tt.cfg)

			// Check required fields
			if opts.GetPid() != int32(tt.cfg.PID) {
				t.Errorf("Pid = %v, want %v", opts.GetPid(), tt.cfg.PID)
			}
			if opts.GetImagesDirFd() != tt.cfg.ImageDirFD {
				t.Errorf("ImagesDirFd = %v, want %v", opts.GetImagesDirFd(), tt.cfg.ImageDirFD)
			}
			if opts.GetRoot() != tt.cfg.RootFS {
				t.Errorf("Root = %v, want %v", opts.GetRoot(), tt.cfg.RootFS)
			}

			// Check common K8s flags
			common.AssertCommonK8sFlags(t, opts)

			// Check checkpoint-specific flags
			if !opts.GetLeaveRunning() {
				t.Error("LeaveRunning should be true")
			}
			if !opts.GetOrphanPtsMaster() {
				t.Error("OrphanPtsMaster should be true")
			}
			if !opts.GetLinkRemap() {
				t.Error("LinkRemap should be true")
			}
			if !opts.GetExtMasters() {
				t.Error("ExtMasters should be true")
			}

			// Check optional fields
			if tt.cfg.GhostLimit > 0 {
				if opts.GetGhostLimit() != tt.cfg.GhostLimit {
					t.Errorf("GhostLimit = %v, want %v", opts.GetGhostLimit(), tt.cfg.GhostLimit)
				}
			}
			if tt.cfg.Timeout > 0 {
				if opts.GetTimeout() != tt.cfg.Timeout {
					t.Errorf("Timeout = %v, want %v", opts.GetTimeout(), tt.cfg.Timeout)
				}
			}

			// Check log settings
			if opts.GetLogLevel() != 4 {
				t.Errorf("LogLevel = %v, want 4", opts.GetLogLevel())
			}
			if opts.GetLogFile() != "dump.log" {
				t.Errorf("LogFile = %v, want dump.log", opts.GetLogFile())
			}

			// Check cgroup mode is SOFT (for external restore)
			if opts.GetManageCgroupsMode() != criurpc.CriuCgMode_SOFT {
				t.Errorf("ManageCgroupsMode = %v, want SOFT", opts.GetManageCgroupsMode())
			}
		})
	}
}

func TestAddExternalMounts(t *testing.T) {
	tests := []struct {
		name   string
		mounts []AllMountInfo
		want   int // number of external mounts
	}{
		{
			name:   "empty mounts",
			mounts: []AllMountInfo{},
			want:   0,
		},
		{
			name: "single mount",
			mounts: []AllMountInfo{
				{MountPoint: "/data", FSType: "ext4"},
			},
			want: 1,
		},
		{
			name: "multiple mounts",
			mounts: []AllMountInfo{
				{MountPoint: "/data", FSType: "ext4"},
				{MountPoint: "/var/cache", FSType: "tmpfs"},
				{MountPoint: "/mnt/nfs", FSType: "nfs4"},
			},
			want: 3,
		},
		{
			name: "duplicate mount points",
			mounts: []AllMountInfo{
				{MountPoint: "/data", FSType: "ext4"},
				{MountPoint: "/data", FSType: "ext4"},
				{MountPoint: "/cache", FSType: "tmpfs"},
			},
			want: 2, // duplicates should be deduplicated
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			criuOpts := &criurpc.CriuOpts{}
			AddExternalMounts(criuOpts, tt.mounts)

			if len(criuOpts.ExtMnt) != tt.want {
				t.Errorf("len(ExtMnt) = %v, want %v", len(criuOpts.ExtMnt), tt.want)
			}

			// Verify mount points are correctly added (Key == Val == MountPoint)
			for i, extMnt := range criuOpts.ExtMnt {
				if extMnt.GetKey() != extMnt.GetVal() {
					t.Errorf("ExtMnt[%d].Key = %v, want %v", i, extMnt.GetKey(), extMnt.GetVal())
				}
			}
		})
	}
}

func TestAddExternalPaths(t *testing.T) {
	tests := []struct {
		name         string
		existingMnts []string
		newPaths     []string
		wantTotal    int
	}{
		{
			name:         "add to empty",
			existingMnts: []string{},
			newPaths:     []string{"/proc/sys", "/sys/firmware"},
			wantTotal:    2,
		},
		{
			name:         "add without duplicates",
			existingMnts: []string{"/data"},
			newPaths:     []string{"/proc/sys", "/sys/firmware"},
			wantTotal:    3,
		},
		{
			name:         "skip duplicates",
			existingMnts: []string{"/proc/sys", "/data"},
			newPaths:     []string{"/proc/sys", "/sys/firmware"},
			wantTotal:    3, // /proc/sys already exists, only /sys/firmware added
		},
		{
			name:         "all duplicates",
			existingMnts: []string{"/proc/sys", "/sys/firmware"},
			newPaths:     []string{"/proc/sys", "/sys/firmware"},
			wantTotal:    2, // nothing added
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			criuOpts := &criurpc.CriuOpts{}

			// Add existing mounts
			for _, path := range tt.existingMnts {
				criuOpts.ExtMnt = append(criuOpts.ExtMnt, &criurpc.ExtMountMap{
					Key: proto.String(path),
					Val: proto.String(path),
				})
			}

			// Add new paths
			AddExternalPaths(criuOpts, tt.newPaths)

			if len(criuOpts.ExtMnt) != tt.wantTotal {
				t.Errorf("len(ExtMnt) = %v, want %v", len(criuOpts.ExtMnt), tt.wantTotal)
			}
		})
	}
}

func TestAddExternalNamespace(t *testing.T) {
	tests := []struct {
		name   string
		nsType NamespaceType
		inode  uint64
		key    string
		want   string
	}{
		{
			name:   "network namespace",
			nsType: NamespaceNet,
			inode:  4026531956,
			key:    "extRootNetNS",
			want:   "net[4026531956]:extRootNetNS",
		},
		{
			name:   "pid namespace",
			nsType: NamespacePID,
			inode:  4026531957,
			key:    "extRootPidNS",
			want:   "pid[4026531957]:extRootPidNS",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			criuOpts := &criurpc.CriuOpts{}
			AddExternalNamespace(criuOpts, tt.nsType, tt.inode, tt.key)

			if len(criuOpts.External) != 1 {
				t.Fatalf("len(External) = %v, want 1", len(criuOpts.External))
			}

			if criuOpts.External[0] != tt.want {
				t.Errorf("External[0] = %v, want %v", criuOpts.External[0], tt.want)
			}
		})
	}
}

func TestConfigureExternalNamespaces(t *testing.T) {
	tests := []struct {
		name           string
		namespaces     map[NamespaceType]*NamespaceInfo
		externalMounts []string
		wantExtCount   int
		wantNetNsInode uint64
	}{
		{
			name: "network namespace with external mounts",
			namespaces: map[NamespaceType]*NamespaceInfo{
				NamespaceNet: {
					Type:  NamespaceNet,
					Inode: 4026531956,
				},
			},
			externalMounts: []string{"extfile:/dev/nvidia0", "extfile:/dev/nvidiactl"},
			wantExtCount:   3, // 1 net NS + 2 external mounts
			wantNetNsInode: 4026531956,
		},
		{
			name: "network namespace without external mounts",
			namespaces: map[NamespaceType]*NamespaceInfo{
				NamespaceNet: {
					Type:  NamespaceNet,
					Inode: 4026531957,
				},
			},
			externalMounts: []string{},
			wantExtCount:   1, // only net NS
			wantNetNsInode: 4026531957,
		},
		{
			name:           "no network namespace",
			namespaces:     map[NamespaceType]*NamespaceInfo{},
			externalMounts: []string{"extfile:/dev/nvidia0"},
			wantExtCount:   1, // only external mounts
			wantNetNsInode: 0,
		},
		{
			name:           "empty everything",
			namespaces:     map[NamespaceType]*NamespaceInfo{},
			externalMounts: []string{},
			wantExtCount:   0,
			wantNetNsInode: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			criuOpts := &criurpc.CriuOpts{}
			netNsInode := ConfigureExternalNamespaces(criuOpts, tt.namespaces, tt.externalMounts)

			if len(criuOpts.External) != tt.wantExtCount {
				t.Errorf("len(External) = %v, want %v", len(criuOpts.External), tt.wantExtCount)
			}

			if netNsInode != tt.wantNetNsInode {
				t.Errorf("netNsInode = %v, want %v", netNsInode, tt.wantNetNsInode)
			}

			// Verify format of external strings
			for _, ext := range criuOpts.External {
				if ext == "" {
					t.Error("External string should not be empty")
				}
			}
		})
	}
}
