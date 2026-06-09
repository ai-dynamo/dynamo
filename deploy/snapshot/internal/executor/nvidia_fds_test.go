package executor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCollectNVIDIAOpenFDs(t *testing.T) {
	procRoot := t.TempDir()
	pid := 101
	procDir := filepath.Join(procRoot, "101")
	fdDir := filepath.Join(procDir, "fd")
	fdInfoDir := filepath.Join(procDir, "fdinfo")
	if err := os.MkdirAll(fdDir, 0755); err != nil {
		t.Fatalf("MkdirAll(fdDir): %v", err)
	}
	if err := os.MkdirAll(fdInfoDir, 0755); err != nil {
		t.Fatalf("MkdirAll(fdInfoDir): %v", err)
	}
	if err := os.WriteFile(filepath.Join(procDir, "status"), []byte("Name:\tpython3\nPPid:\t1\nNSpid:\t101 7\n"), 0644); err != nil {
		t.Fatalf("WriteFile(status): %v", err)
	}
	if err := os.WriteFile(filepath.Join(procDir, "cmdline"), []byte("python3\x00-m\x00dynamo.vllm\x00"), 0644); err != nil {
		t.Fatalf("WriteFile(cmdline): %v", err)
	}
	if err := os.Symlink("/dev/nvidiactl", filepath.Join(fdDir, "10")); err != nil {
		t.Fatalf("Symlink(nvidia): %v", err)
	}
	if err := os.Symlink("/tmp/regular", filepath.Join(fdDir, "11")); err != nil {
		t.Fatalf("Symlink(regular): %v", err)
	}
	if err := os.WriteFile(filepath.Join(fdInfoDir, "10"), []byte("pos:\t0\nflags:\t0100002\nmnt_id:\t123\nino:\t99\n"), 0644); err != nil {
		t.Fatalf("WriteFile(fdinfo): %v", err)
	}

	fds := collectNVIDIAOpenFDs(procRoot, []int{pid}, []int{pid})
	if len(fds) != 1 {
		t.Fatalf("len(fds) = %d, want 1: %#v", len(fds), fds)
	}
	if fds[0].HostPID != pid || fds[0].InnermostPID != 7 || !fds[0].CUDAPID || fds[0].FD != 10 {
		t.Fatalf("unexpected fd entry: %#v", fds[0])
	}
	if got := formatIntSlice(fds[0].NamespacePIDs); got != "101,7" {
		t.Fatalf("NamespacePIDs = %q, want 101,7", got)
	}
	if fds[0].Target != "/dev/nvidiactl" {
		t.Fatalf("target = %q, want /dev/nvidiactl", fds[0].Target)
	}
	if fds[0].FDInfo != "pos:=0;flags:=0100002;mnt_id:=123" {
		t.Fatalf("FDInfo = %q, want selected fdinfo fields", fds[0].FDInfo)
	}
	if !strings.Contains(fds[0].Cmdline, "dynamo.vllm") {
		t.Fatalf("cmdline = %q, want dynamo.vllm", fds[0].Cmdline)
	}
}

func TestFormatNVIDIAOpenFDs(t *testing.T) {
	got := formatNVIDIAOpenFDs(nil, []int{101}, []nvidiaOpenFD{
		{
			HostPID:       101,
			InnermostPID:  7,
			NamespacePIDs: []int{101, 7},
			CUDAPID:       true,
			Cmdline:       "python3 -m dynamo.vllm",
			FD:            10,
			Target:        "/dev/nvidiactl",
			Major:         195,
			Minor:         255,
			FDInfo:        "pos:=0;flags:=0100002;mnt_id:=123",
		},
	})
	for _, want := range []string{
		"phase=post_cuda_checkpoint_pre_criu_dump",
		"process_tree_host_pids=101",
		"count=1",
		"host_pid\tinnermost_pid\tnspid\tcuda_pid\tfd\ttarget\tmajor\tminor\tfdinfo\tcmdline",
		"101\t7\t101,7\ttrue\t10\t/dev/nvidiactl\t195\t255\tpos:=0;flags:=0100002;mnt_id:=123\tpython3 -m dynamo.vllm",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatted output missing %q:\n%s", want, got)
		}
	}
}
