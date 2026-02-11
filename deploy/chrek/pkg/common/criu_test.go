package common

import (
	"reflect"
	"testing"
)

func TestParseCRIUMountInfoLine(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		want    CRIUMountPoint
		wantErr bool
	}{
		{
			name: "valid mount with all fields",
			line: "36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue",
			want: CRIUMountPoint{
				MountID:   "36",
				ParentID:  "35",
				Root:      "/mnt1",
				Path:      "/mnt2",
				Options:   "rw,noatime",
				FSType:    "ext3",
				Source:    "/dev/root",
				SuperOpts: "rw,errors=continue",
			},
			wantErr: false,
		},
		{
			name: "valid mount without super options",
			line: "100 99 0:22 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:10 - cgroup cgroup",
			want: CRIUMountPoint{
				MountID:   "100",
				ParentID:  "99",
				Root:      "/",
				Path:      "/sys/fs/cgroup/memory",
				Options:   "rw,nosuid,nodev,noexec,relatime",
				FSType:    "cgroup",
				Source:    "cgroup",
				SuperOpts: "",
			},
			wantErr: false,
		},
		{
			name: "kubernetes volume mount",
			line: "200 199 0:50 / /var/lib/kubelet/pods/abc-123/volumes/kubernetes.io~empty-dir/cache rw,relatime shared:1 - tmpfs tmpfs rw,size=1024k",
			want: CRIUMountPoint{
				MountID:   "200",
				ParentID:  "199",
				Root:      "/",
				Path:      "/var/lib/kubelet/pods/abc-123/volumes/kubernetes.io~empty-dir/cache",
				Options:   "rw,relatime",
				FSType:    "tmpfs",
				Source:    "tmpfs",
				SuperOpts: "rw,size=1024k",
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
			line:    "36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 ext3 /dev/root rw,errors=continue",
			wantErr: true,
		},
		{
			name:    "malformed line - separator at end",
			line:    "36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 -",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseCRIUMountInfoLine(tt.line)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseCRIUMountInfoLine() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseCRIUMountInfoLine() = %+v, want %+v", got, tt.want)
			}
		})
	}
}
