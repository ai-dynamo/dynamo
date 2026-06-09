package executor

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-logr/logr/testr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestExecNSRestoreRejectsRelativeContainerCheckpointLocation(t *testing.T) {
	_, err := execNSRestore(
		context.Background(),
		testr.New(t),
		RestoreRequest{
			ContainerCheckpointLocation: "relative/checkpoint",
			NSRestorePath:               "/usr/local/bin/nsrestore",
		},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/host/checkpoints/abc123",
			PlaceholderPID: 1,
		},
	)
	if err == nil {
		t.Fatal("expected relative container checkpoint location to be rejected")
	}
	if !strings.Contains(err.Error(), "absolute") {
		t.Fatalf("expected absolute-path validation error, got: %v", err)
	}
}

func TestPrepareKubeletMountpointsForRestore(t *testing.T) {
	hostRoot := t.TempDir()
	targetCSIPath := filepath.Join(hostRoot, "var", "lib", "kubelet", "pods", "target-pod", "volumes", "kubernetes.io~csi", "pvc-abc", "mount")
	targetProjectedPath := filepath.Join(hostRoot, "var", "lib", "kubelet", "pods", "target-pod", "volumes", "kubernetes.io~projected", "kube-api-access-live")
	targetSubpath := filepath.Join(hostRoot, "var", "lib", "kubelet", "pods", "target-pod", "volume-subpaths", "config", "main", "7")
	for _, path := range []string{targetCSIPath, targetProjectedPath, targetSubpath} {
		if err := os.MkdirAll(path, 0755); err != nil {
			t.Fatalf("mkdir target path %s: %v", path, err)
		}
	}

	m := &types.CheckpointManifest{
		CRIUDump: types.CRIUDumpManifest{
			ExtMnt: map[string]string{
				"/checkpoints": "/host/var/lib/kubelet/pods/source-pod/volumes/kubernetes.io~csi/pvc-abc/mount",
				"/projected":   "/host/var/lib/kubelet/pods/source-pod/volumes/kubernetes.io~projected/kube-api-access-old",
				"/subpath":     "/host/var/lib/kubelet/pods/source-pod/volume-subpaths/config/main/2",
			},
		},
	}
	err := prepareKubeletMountpointsForRestore(
		testr.New(t),
		RestoreRequest{PodUID: "target-pod", ContainerName: "main"},
		m,
		hostRoot,
	)
	if err != nil {
		t.Fatalf("prepareKubeletMountpointsForRestore: %v", err)
	}

	sourceCSIPath := filepath.Join(hostRoot, "var", "lib", "kubelet", "pods", "source-pod", "volumes", "kubernetes.io~csi", "pvc-abc", "mount")
	if info, err := os.Stat(sourceCSIPath); err != nil || !info.IsDir() {
		t.Fatalf("source CSI mountpoint was not created: info=%v err=%v", info, err)
	}
	sourceProjectedPath := filepath.Join(hostRoot, "var", "lib", "kubelet", "pods", "source-pod", "volumes", "kubernetes.io~projected", "kube-api-access-old")
	if info, err := os.Stat(sourceProjectedPath); err != nil || !info.IsDir() {
		t.Fatalf("source projected mountpoint was not created: info=%v err=%v", info, err)
	}
	sourceSubpath := filepath.Join(hostRoot, "var", "lib", "kubelet", "pods", "source-pod", "volume-subpaths", "config", "main", "2")
	if info, err := os.Stat(sourceSubpath); err != nil || !info.IsDir() {
		t.Fatalf("source subpath mountpoint was not created: info=%v err=%v", info, err)
	}
}
