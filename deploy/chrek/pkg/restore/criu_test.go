package restore

import (
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

func TestBuildRestoreCRIUOpts(t *testing.T) {
	tests := []struct {
		name string
		cfg  CRIURestoreConfig
	}{
		{
			name: "basic restore config",
			cfg: CRIURestoreConfig{
				ImageDirFD: 5,
				RootPath:   "/",
				LogLevel:   4,
				LogFile:    "restore.log",
				WorkDirFD:  -1,
				NetNsFD:    -1,
			},
		},
		{
			name: "restore with work dir",
			cfg: CRIURestoreConfig{
				ImageDirFD: 10,
				RootPath:   "/container/root",
				LogLevel:   2,
				LogFile:    "criu-restore.log",
				WorkDirFD:  8,
				NetNsFD:    -1,
			},
		},
		{
			name: "restore with network namespace",
			cfg: CRIURestoreConfig{
				ImageDirFD: 7,
				RootPath:   "/",
				LogLevel:   4,
				LogFile:    "restore.log",
				WorkDirFD:  -1,
				NetNsFD:    12,
			},
		},
		{
			name: "restore with external mounts",
			cfg: CRIURestoreConfig{
				ImageDirFD: 5,
				RootPath:   "/",
				LogLevel:   4,
				LogFile:    "restore.log",
				WorkDirFD:  -1,
				NetNsFD:    -1,
				ExtMountMaps: []*criurpc.ExtMountMap{
					{Key: proto.String("/proc/sys"), Val: proto.String("/proc/sys")},
					{Key: proto.String("/data"), Val: proto.String("/var/lib/data")},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := BuildRestoreCRIUOpts(tt.cfg)

			// Check required fields
			if opts.GetImagesDirFd() != tt.cfg.ImageDirFD {
				t.Errorf("ImagesDirFd = %v, want %v", opts.GetImagesDirFd(), tt.cfg.ImageDirFD)
			}
			if opts.GetRoot() != tt.cfg.RootPath {
				t.Errorf("Root = %v, want %v", opts.GetRoot(), tt.cfg.RootPath)
			}
			if opts.GetLogLevel() != tt.cfg.LogLevel {
				t.Errorf("LogLevel = %v, want %v", opts.GetLogLevel(), tt.cfg.LogLevel)
			}
			if opts.GetLogFile() != tt.cfg.LogFile {
				t.Errorf("LogFile = %v, want %v", opts.GetLogFile(), tt.cfg.LogFile)
			}

			// Check common K8s flags
			common.AssertCommonK8sFlags(t, opts)

			// Check restore-specific flags
			if !opts.GetRstSibling() {
				t.Error("RstSibling should be true (detached restore)")
			}
			if !opts.GetMntnsCompatMode() {
				t.Error("MntnsCompatMode should be true (cross-container restore)")
			}

			// Check cgroup mode is SOFT (for external restore)
			if opts.GetManageCgroupsMode() != criurpc.CriuCgMode_SOFT {
				t.Errorf("ManageCgroupsMode = %v, want SOFT", opts.GetManageCgroupsMode())
			}

			// Check optional work dir FD
			if tt.cfg.WorkDirFD > 0 {
				if opts.GetWorkDirFd() != tt.cfg.WorkDirFD {
					t.Errorf("WorkDirFd = %v, want %v", opts.GetWorkDirFd(), tt.cfg.WorkDirFD)
				}
			}

			// Check optional network namespace FD
			if tt.cfg.NetNsFD >= 0 {
				if len(opts.InheritFd) != 1 {
					t.Errorf("len(InheritFd) = %v, want 1 when NetNsFD is provided", len(opts.InheritFd))
				} else {
					if opts.InheritFd[0].GetKey() != "extNetNs" {
						t.Errorf("InheritFd[0].Key = %v, want extNetNs", opts.InheritFd[0].GetKey())
					}
					if opts.InheritFd[0].GetFd() != tt.cfg.NetNsFD {
						t.Errorf("InheritFd[0].Fd = %v, want %v", opts.InheritFd[0].GetFd(), tt.cfg.NetNsFD)
					}
				}
			}

			// Check external mount maps
			if len(tt.cfg.ExtMountMaps) > 0 {
				if len(opts.ExtMnt) != len(tt.cfg.ExtMountMaps) {
					t.Errorf("len(ExtMnt) = %v, want %v", len(opts.ExtMnt), len(tt.cfg.ExtMountMaps))
				}
			}
		})
	}
}
