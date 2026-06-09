package executor

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const nvidiaOpenFDLogFilename = "nvidia-open-fds.log"

type nvidiaOpenFD struct {
	HostPID       int
	InnermostPID  int
	NamespacePIDs []int
	CUDAPID       bool
	Cmdline       string
	FD            int
	Target        string
	Major         int64
	Minor         int64
	FDInfo        string
}

func writeNVIDIAOpenFDDiagnostic(state *types.CheckpointContainerSnapshot, checkpointDir string, log logr.Logger) {
	if state == nil || state.PID <= 0 {
		return
	}

	pids := snapshotruntime.ProcessTreePIDs(state.PID)
	fds := collectNVIDIAOpenFDs(snapshotruntime.HostProcPath, pids, state.CUDAHostPIDs)
	sort.Slice(fds, func(i, j int) bool {
		if fds[i].HostPID == fds[j].HostPID {
			return fds[i].FD < fds[j].FD
		}
		return fds[i].HostPID < fds[j].HostPID
	})

	path := filepath.Join(checkpointDir, nvidiaOpenFDLogFilename)
	if err := os.WriteFile(path, []byte(formatNVIDIAOpenFDs(state, pids, fds)), 0644); err != nil {
		log.Error(err, "Failed to write NVIDIA open FD diagnostic", "path", path)
		return
	}
	log.Info("Captured NVIDIA open FD diagnostic before CRIU dump", "count", len(fds), "path", path)
}

func collectNVIDIAOpenFDs(procRoot string, pids []int, cudaHostPIDs []int) []nvidiaOpenFD {
	cudaPID := make(map[int]bool, len(cudaHostPIDs))
	for _, pid := range cudaHostPIDs {
		cudaPID[pid] = true
	}

	var out []nvidiaOpenFD
	for _, pid := range pids {
		fdDir := filepath.Join(procRoot, strconv.Itoa(pid), "fd")
		entries, err := os.ReadDir(fdDir)
		if err != nil {
			continue
		}
		process := snapshotruntime.ReadProcessDetailsOrDefault(procRoot, pid)
		for _, entry := range entries {
			fd, err := strconv.Atoi(entry.Name())
			if err != nil {
				continue
			}
			fdPath := filepath.Join(fdDir, entry.Name())
			target, err := os.Readlink(fdPath)
			if err != nil {
				continue
			}

			major, minor := statDeviceMajorMinor(fdPath)
			if !isNVIDIAFD(target, major) {
				continue
			}
			out = append(out, nvidiaOpenFD{
				HostPID:       pid,
				InnermostPID:  process.InnermostPID,
				NamespacePIDs: append([]int(nil), process.NamespacePIDs...),
				CUDAPID:       cudaPID[pid],
				Cmdline:       process.Cmdline,
				FD:            fd,
				Target:        target,
				Major:         major,
				Minor:         minor,
				FDInfo:        readFDInfo(procRoot, pid, fd),
			})
		}
	}
	return out
}

func statDeviceMajorMinor(path string) (int64, int64) {
	var stat unix.Stat_t
	if err := unix.Stat(path, &stat); err != nil {
		return -1, -1
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFCHR {
		return -1, -1
	}
	return int64(unix.Major(uint64(stat.Rdev))), int64(unix.Minor(uint64(stat.Rdev)))
}

func isNVIDIAFD(target string, major int64) bool {
	if strings.Contains(target, "nvidia") {
		return true
	}
	return major == 195
}

func readFDInfo(procRoot string, pid int, fd int) string {
	data, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "fdinfo", strconv.Itoa(fd)))
	if err != nil {
		return ""
	}
	var fields []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "pos:") ||
			strings.HasPrefix(line, "flags:") ||
			strings.HasPrefix(line, "mnt_id:") {
			fields = append(fields, strings.Join(strings.Fields(line), "="))
		}
	}
	return strings.Join(fields, ";")
}

func formatNVIDIAOpenFDs(state *types.CheckpointContainerSnapshot, pids []int, fds []nvidiaOpenFD) string {
	var b strings.Builder
	fmt.Fprintln(&b, "phase=post_cuda_checkpoint_pre_criu_dump")
	if state != nil {
		fmt.Fprintf(&b, "root_host_pid=%d\n", state.PID)
		fmt.Fprintf(&b, "cuda_host_pids=%s\n", formatIntSlice(state.CUDAHostPIDs))
		fmt.Fprintf(&b, "cuda_namespace_pids=%s\n", formatIntSlice(state.CUDANSPIDs))
	}
	fmt.Fprintf(&b, "process_tree_host_pids=%s\n", formatIntSlice(pids))
	fmt.Fprintf(&b, "count=%d\n", len(fds))
	fmt.Fprintln(&b, "host_pid\tinnermost_pid\tnspid\tcuda_pid\tfd\ttarget\tmajor\tminor\tfdinfo\tcmdline")
	for _, fd := range fds {
		fmt.Fprintf(
			&b,
			"%d\t%d\t%s\t%t\t%d\t%s\t%d\t%d\t%s\t%s\n",
			fd.HostPID,
			fd.InnermostPID,
			formatIntSlice(fd.NamespacePIDs),
			fd.CUDAPID,
			fd.FD,
			sanitizeDiagnosticField(fd.Target),
			fd.Major,
			fd.Minor,
			sanitizeDiagnosticField(fd.FDInfo),
			sanitizeDiagnosticField(fd.Cmdline),
		)
	}
	return b.String()
}

func formatIntSlice(values []int) string {
	if len(values) == 0 {
		return ""
	}
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, strconv.Itoa(value))
	}
	return strings.Join(out, ",")
}

func sanitizeDiagnosticField(value string) string {
	value = strings.ReplaceAll(value, "\t", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	return value
}
