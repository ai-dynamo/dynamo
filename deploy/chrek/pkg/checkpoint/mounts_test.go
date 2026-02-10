package checkpoint

import (
	"reflect"
	"testing"
)

func TestParseMountInfoLine(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		want     MountMapping
		wantSkip bool
	}{
		{
			name: "kubernetes pvc bind mount",
			line: "200 199 8:1 /volumes/pvc-123/data /data rw,relatime shared:1 - ext4 /dev/sda1 rw,data=ordered",
			want: MountMapping{
				InsidePath:  "/data",
				OutsidePath: "/volumes/pvc-123/data",
				FSType:      "ext4",
				Source:      "/dev/sda1",
				Options:     "rw,relatime,rw,data=ordered",
			},
			wantSkip: false,
		},
		{
			name:     "kubernetes emptyDir tmpfs",
			line:     "300 299 0:50 / /var/lib/kubelet/pods/abc/volumes/kubernetes.io~empty-dir/cache rw,relatime shared:2 - tmpfs tmpfs rw,size=1024k",
			want:     MountMapping{},
			wantSkip: true, // tmpfs is filtered out
		},
		{
			name: "bind mount with root as /",
			line: "400 399 8:1 / /mnt/host rw,relatime shared:3 - ext4 /dev/sda1 rw",
			want: MountMapping{
				InsidePath:  "/mnt/host",
				OutsidePath: "/dev/sda1",
				FSType:      "ext4",
				Source:      "/dev/sda1",
				Options:     "rw,relatime,rw",
			},
			wantSkip: false,
		},
		{
			name:     "proc mount (should skip)",
			line:     "100 99 0:5 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name:     "sys mount (should skip)",
			line:     "101 99 0:6 / /sys ro,nosuid,nodev,noexec,relatime - sysfs sysfs ro",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name:     "dev mount (should skip)",
			line:     "102 99 0:7 / /dev rw,nosuid - devtmpfs devtmpfs rw,size=1024k",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name:     "cgroup mount (should skip)",
			line:     "103 99 0:25 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:10 - cgroup cgroup rw,memory",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name:     "overlay root mount (should skip)",
			line:     "500 499 0:100 / / rw,relatime - overlay overlay rw,lowerdir=/lower,upperdir=/upper,workdir=/work",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name:     "sys subdirectory (should skip)",
			line:     "600 599 0:30 / /sys/kernel/debug rw,relatime - debugfs debugfs rw",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name:     "proc subdirectory (should skip)",
			line:     "601 599 0:31 / /proc/sys/fs/binfmt_misc rw,relatime - binfmt_misc binfmt_misc rw",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name: "nfs mount",
			line: "700 699 0:40 / /mnt/nfs rw,relatime shared:5 - nfs4 server.example.com:/export rw,vers=4.1",
			want: MountMapping{
				InsidePath:  "/mnt/nfs",
				OutsidePath: "server.example.com:/export", // When root is "/", use source
				FSType:      "nfs4",
				Source:      "server.example.com:/export",
				Options:     "rw,relatime,rw,vers=4.1",
			},
			wantSkip: false,
		},
		{
			name:     "malformed line - too few fields",
			line:     "36 35 98:0",
			want:     MountMapping{},
			wantSkip: true,
		},
		{
			name:     "malformed line - no separator",
			line:     "36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 ext3 /dev/root",
			want:     MountMapping{},
			wantSkip: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, skip := parseMountInfoLine(tt.line)
			if skip != tt.wantSkip {
				t.Errorf("parseMountInfoLine() skip = %v, want %v", skip, tt.wantSkip)
				return
			}
			if !tt.wantSkip && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseMountInfoLine() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestParseAllMountInfoLine(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		want    AllMountInfo
		wantErr bool
	}{
		{
			name: "valid mount with all fields",
			line: "36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue",
			want: AllMountInfo{
				MountID:      "36",
				ParentID:     "35",
				Root:         "/mnt1",
				MountPoint:   "/mnt2",
				Options:      "rw,noatime",
				FSType:       "ext3",
				Source:       "/dev/root",
				SuperOptions: "rw,errors=continue",
			},
			wantErr: false,
		},
		{
			name: "proc mount",
			line: "100 99 0:5 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw",
			want: AllMountInfo{
				MountID:      "100",
				ParentID:     "99",
				Root:         "/",
				MountPoint:   "/proc",
				Options:      "rw,nosuid,nodev,noexec,relatime",
				FSType:       "proc",
				Source:       "proc",
				SuperOptions: "rw",
			},
			wantErr: false,
		},
		{
			name: "overlay root mount",
			line: "500 499 0:100 / / rw,relatime - overlay overlay rw,lowerdir=/lower,upperdir=/upper,workdir=/work",
			want: AllMountInfo{
				MountID:      "500",
				ParentID:     "499",
				Root:         "/",
				MountPoint:   "/",
				Options:      "rw,relatime",
				FSType:       "overlay",
				Source:       "overlay",
				SuperOptions: "rw,lowerdir=/lower,upperdir=/upper,workdir=/work",
			},
			wantErr: false,
		},
		{
			name: "tmpfs mount",
			line: "200 199 0:50 / /tmp rw,nosuid,nodev shared:1 - tmpfs tmpfs rw,size=65536k",
			want: AllMountInfo{
				MountID:      "200",
				ParentID:     "199",
				Root:         "/",
				MountPoint:   "/tmp",
				Options:      "rw,nosuid,nodev",
				FSType:       "tmpfs",
				Source:       "tmpfs",
				SuperOptions: "rw,size=65536k",
			},
			wantErr: false,
		},
		{
			name: "mount without super options",
			line: "300 299 0:60 / /dev/shm rw,nosuid,nodev shared:2 - tmpfs tmpfs",
			want: AllMountInfo{
				MountID:      "300",
				ParentID:     "299",
				Root:         "/",
				MountPoint:   "/dev/shm",
				Options:      "rw,nosuid,nodev",
				FSType:       "tmpfs",
				Source:       "tmpfs",
				SuperOptions: "",
			},
			wantErr: false,
		},
		{
			name: "cgroup mount",
			line: "400 399 0:25 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:10 - cgroup cgroup rw,memory",
			want: AllMountInfo{
				MountID:      "400",
				ParentID:     "399",
				Root:         "/",
				MountPoint:   "/sys/fs/cgroup/memory",
				Options:      "rw,nosuid,nodev,noexec,relatime",
				FSType:       "cgroup",
				Source:       "cgroup",
				SuperOptions: "rw,memory",
			},
			wantErr: false,
		},
		{
			name:    "malformed line - too few fields",
			line:    "36 35 98:0",
			wantErr: true,
		},
		{
			name:    "malformed line - no separator",
			line:    "36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 ext3 /dev/root rw",
			wantErr: true,
		},
		{
			name:    "malformed line - separator at end",
			line:    "36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseAllMountInfoLine(tt.line)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseAllMountInfoLine() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseAllMountInfoLine() = %+v, want %+v", got, tt.want)
			}
		})
	}
}
