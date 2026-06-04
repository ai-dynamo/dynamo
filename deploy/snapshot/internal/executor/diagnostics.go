package executor

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	checkpointDiagnosticsEnv       = "DYN_SNAPSHOT_DIAGNOSTICS"
	checkpointKeepFailedStagingEnv = "DYN_SNAPSHOT_KEEP_FAILED_CHECKPOINT"
)

func checkpointDiagnosticsEnabled() bool {
	return envTruthy(checkpointDiagnosticsEnv) ||
		envTruthy(checkpointKeepFailedStagingEnv)
}

func checkpointKeepFailedStagingEnabled() bool {
	return checkpointDiagnosticsEnabled()
}

func envTruthy(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "t", "true", "y", "yes", "on":
		return true
	default:
		return false
	}
}

func writeCheckpointProcessDiagnostics(
	ctx context.Context,
	checkpointDir string,
	phase string,
	state *types.CheckpointContainerSnapshot,
	log logr.Logger,
) {
	if state == nil || state.PID <= 0 {
		return
	}

	diagnosticDir := filepath.Join(checkpointDir, "diagnostics")
	if err := os.MkdirAll(diagnosticDir, 0700); err != nil {
		log.Error(err, "Failed to create checkpoint diagnostics directory",
			"phase", phase,
			"diagnostic_dir", diagnosticDir,
		)
		return
	}

	diagnosticPath := filepath.Join(diagnosticDir, sanitizeDiagnosticName(phase)+".txt")
	var out bytes.Buffer
	cudaPIDs := intSet(state.CUDAHostPIDs)
	processTreePIDs := snapshotruntime.ProcessTreePIDs(state.PID)
	diagnosticPIDs := collectDiagnosticPIDs(state.PID, processTreePIDs)

	fmt.Fprintf(&out, "phase: %s\n", phase)
	fmt.Fprintf(&out, "timestamp: %s\n", time.Now().Format(time.RFC3339Nano))
	fmt.Fprintf(&out, "root_host_pid: %d\n", state.PID)
	fmt.Fprintf(&out, "process_tree_pids: %v\n", processTreePIDs)
	fmt.Fprintf(&out, "cuda_checkpoint_host_pids: %v\n", state.CUDAHostPIDs)
	fmt.Fprintf(&out, "cuda_checkpoint_namespace_pids: %v\n", state.CUDANSPIDs)
	fmt.Fprintf(&out, "diagnostic_pids: %v\n\n", diagnosticPIDs)

	writeCommandSection(ctx, &out, "nvidia-smi compute apps",
		5*time.Second,
		"nvidia-smi",
		"--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
		"--format=csv,noheader,nounits",
	)
	writeCommandSection(ctx, &out, "cuda-checkpoint version",
		5*time.Second,
		"/usr/local/sbin/cuda-checkpoint",
		"--version",
	)
	writeCommandSection(ctx, &out, "criu version",
		5*time.Second,
		"/usr/local/sbin/criu",
		"--version",
	)

	for _, pid := range diagnosticPIDs {
		writeOneProcessDiagnostics(ctx, &out, pid, cudaPIDs)
	}

	if err := os.WriteFile(diagnosticPath, out.Bytes(), 0600); err != nil {
		log.Error(err, "Failed to write checkpoint diagnostics",
			"phase", phase,
			"diagnostic_path", diagnosticPath,
		)
		return
	}

	log.Info("Wrote checkpoint process diagnostics",
		"phase", phase,
		"diagnostic_path", diagnosticPath,
	)
}

func sanitizeDiagnosticName(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

func collectDiagnosticPIDs(rootPID int, processTreePIDs []int) []int {
	pids := intSet(processTreePIDs)
	if rootPID > 0 {
		pids[rootPID] = struct{}{}
	}

	rootPIDNS, _ := os.Readlink(hostProcPath(rootPID, "ns", "pid"))
	rootCgroup := readHostProcFile(rootPID, "cgroup")

	entries, err := os.ReadDir(snapshotruntime.HostProcPath)
	if err != nil {
		return sortedIntSet(pids)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || pid <= 0 {
			continue
		}
		if rootPIDNS != "" {
			if pidNS, err := os.Readlink(hostProcPath(pid, "ns", "pid")); err == nil && pidNS == rootPIDNS {
				pids[pid] = struct{}{}
				continue
			}
		}
		if rootCgroup != "" && readHostProcFile(pid, "cgroup") == rootCgroup {
			pids[pid] = struct{}{}
		}
	}

	return sortedIntSet(pids)
}

func writeOneProcessDiagnostics(
	ctx context.Context,
	out *bytes.Buffer,
	pid int,
	cudaPIDs map[int]struct{},
) {
	fmt.Fprintf(out, "\n===== process host_pid=%d =====\n", pid)
	if _, ok := cudaPIDs[pid]; ok {
		fmt.Fprintln(out, "tracked_by_cuda_checkpoint: true")
	} else {
		fmt.Fprintln(out, "tracked_by_cuda_checkpoint: false")
	}

	process, err := snapshotruntime.ReadProcessDetails(snapshotruntime.HostProcPath, pid)
	if err != nil {
		fmt.Fprintf(out, "process_details_error: %v\n", err)
		return
	}
	fmt.Fprintf(out, "parent_host_pid: %d\n", process.ParentPID)
	fmt.Fprintf(out, "outermost_pid: %d\n", process.OutermostPID)
	fmt.Fprintf(out, "innermost_pid: %d\n", process.InnermostPID)
	fmt.Fprintf(out, "namespace_pids: %v\n", process.NamespacePIDs)
	fmt.Fprintf(out, "cmdline: %s\n", process.Cmdline)

	writeSelectedStatus(out, pid)
	writeProcLink(out, pid, "exe")
	writeProcLink(out, pid, "cwd")
	fmt.Fprintf(out, "pid_namespace: %s\n", readProcLink(pid, "ns", "pid"))
	fmt.Fprintf(out, "mnt_namespace: %s\n", readProcLink(pid, "ns", "mnt"))
	fmt.Fprintf(out, "net_namespace: %s\n", readProcLink(pid, "ns", "net"))
	writeCgroup(out, pid)

	writeCommandSection(ctx, out, "cuda restore tid",
		3*time.Second,
		"/usr/local/bin/cuda-checkpoint-helper",
		"--get-restore-tid",
		"--pid",
		strconv.Itoa(pid),
	)
	writeCommandSection(ctx, out, "cuda state",
		3*time.Second,
		"/usr/local/bin/cuda-checkpoint-helper",
		"--get-state",
		"--pid",
		strconv.Itoa(pid),
	)

	writeThreads(out, pid)
	writeFDs(out, pid)
	writeMatchingSocketTable(out, pid)
	writeDeviceVMAs(out, pid)
}

func writeCommandSection(
	ctx context.Context,
	out *bytes.Buffer,
	section string,
	timeout time.Duration,
	name string,
	args ...string,
) {
	fmt.Fprintf(out, "\n--- %s ---\n", section)
	result := runDiagnosticCommand(ctx, timeout, name, args...)
	fmt.Fprintln(out, result)
}

func runDiagnosticCommand(
	ctx context.Context,
	timeout time.Duration,
	name string,
	args ...string,
) string {
	commandCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(commandCtx, name, args...)
	output, err := cmd.CombinedOutput()
	trimmed := strings.TrimSpace(string(output))
	if commandCtx.Err() != nil {
		return fmt.Sprintf("timeout after %s output=%q", timeout, trimmed)
	}
	if err != nil {
		return fmt.Sprintf("error=%v output=%q", err, trimmed)
	}
	return fmt.Sprintf("ok output=%q", trimmed)
}

func writeSelectedStatus(out *bytes.Buffer, pid int) {
	fmt.Fprintln(out, "\n--- selected status ---")
	status := readHostProcFile(pid, "status")
	if status == "" {
		fmt.Fprintln(out, "<unavailable>")
		return
	}
	prefixes := []string{
		"Name:",
		"State:",
		"Tgid:",
		"Pid:",
		"PPid:",
		"NSpid:",
		"Threads:",
		"VmPeak:",
		"VmSize:",
		"VmRSS:",
		"voluntary_ctxt_switches:",
		"nonvoluntary_ctxt_switches:",
	}
	for _, line := range strings.Split(status, "\n") {
		for _, prefix := range prefixes {
			if strings.HasPrefix(line, prefix) {
				fmt.Fprintln(out, line)
				break
			}
		}
	}
}

func writeProcLink(out *bytes.Buffer, pid int, name string) {
	fmt.Fprintf(out, "%s: %s\n", name, readProcLink(pid, name))
}

func readProcLink(pid int, parts ...string) string {
	target, err := os.Readlink(hostProcPath(pid, parts...))
	if err != nil {
		return fmt.Sprintf("<error: %v>", err)
	}
	return target
}

func writeCgroup(out *bytes.Buffer, pid int) {
	fmt.Fprintln(out, "\n--- cgroup ---")
	cgroup := readHostProcFile(pid, "cgroup")
	if cgroup == "" {
		fmt.Fprintln(out, "<unavailable>")
		return
	}
	fmt.Fprint(out, cgroup)
	if !strings.HasSuffix(cgroup, "\n") {
		fmt.Fprintln(out)
	}
}

func writeThreads(out *bytes.Buffer, pid int) {
	fmt.Fprintln(out, "\n--- threads ---")
	taskDir := hostProcPath(pid, "task")
	entries, err := os.ReadDir(taskDir)
	if err != nil {
		fmt.Fprintf(out, "<error reading %s: %v>\n", taskDir, err)
		return
	}
	sort.Slice(entries, func(i, j int) bool {
		left, _ := strconv.Atoi(entries[i].Name())
		right, _ := strconv.Atoi(entries[j].Name())
		return left < right
	})
	const maxThreads = 256
	for i, entry := range entries {
		if i == maxThreads {
			fmt.Fprintf(out, "... truncated after %d threads ...\n", maxThreads)
			return
		}
		tid := entry.Name()
		comm := strings.TrimSpace(readHostProcPath(filepath.Join(taskDir, tid, "comm")))
		children := strings.TrimSpace(readHostProcPath(filepath.Join(taskDir, tid, "children")))
		fmt.Fprintf(out, "tid=%s comm=%q children=%q\n", tid, comm, children)
	}
}

func writeFDs(out *bytes.Buffer, pid int) {
	fmt.Fprintln(out, "\n--- selected fds ---")
	fdDir := hostProcPath(pid, "fd")
	entries, err := os.ReadDir(fdDir)
	if err != nil {
		fmt.Fprintf(out, "<error reading %s: %v>\n", fdDir, err)
		return
	}
	sort.Slice(entries, func(i, j int) bool {
		left, _ := strconv.Atoi(entries[i].Name())
		right, _ := strconv.Atoi(entries[j].Name())
		return left < right
	})

	wrote := false
	for _, entry := range entries {
		target, err := os.Readlink(filepath.Join(fdDir, entry.Name()))
		if err != nil {
			continue
		}
		if !interestingFDTarget(target) {
			continue
		}
		wrote = true
		fmt.Fprintf(out, "fd=%s target=%s\n", entry.Name(), target)
		if strings.Contains(target, "nvidia") || strings.Contains(target, "uvm") {
			fdinfo := readHostProcFile(pid, "fdinfo", entry.Name())
			if fdinfo != "" {
				fmt.Fprintf(out, "fd=%s fdinfo:\n%s", entry.Name(), fdinfo)
				if !strings.HasSuffix(fdinfo, "\n") {
					fmt.Fprintln(out)
				}
			}
		}
	}
	if !wrote {
		fmt.Fprintln(out, "<none>")
	}
}

func interestingFDTarget(target string) bool {
	return strings.Contains(target, "nvidia") ||
		strings.Contains(target, "cuda") ||
		strings.Contains(target, "uvm") ||
		strings.HasPrefix(target, "socket:[")
}

func writeMatchingSocketTable(out *bytes.Buffer, pid int) {
	fmt.Fprintln(out, "\n--- tcp sockets matching process fd inodes ---")
	inodes := socketInodes(pid)
	if len(inodes) == 0 {
		fmt.Fprintln(out, "<none>")
		return
	}
	wrote := false
	for _, table := range []string{"tcp", "tcp6"} {
		path := hostProcPath(pid, "net", table)
		content := readHostProcPath(path)
		if content == "" {
			continue
		}
		lines := strings.Split(content, "\n")
		for _, line := range lines[1:] {
			fields := strings.Fields(line)
			if len(fields) < 10 {
				continue
			}
			if _, ok := inodes[fields[9]]; !ok {
				continue
			}
			wrote = true
			fmt.Fprintf(out, "%s %s\n", table, strings.TrimSpace(line))
		}
	}
	if !wrote {
		fmt.Fprintf(out, "<socket fds present but no matching tcp/tcp6 rows: %v>\n",
			sortedStringSet(inodes))
	}
}

func socketInodes(pid int) map[string]struct{} {
	inodes := map[string]struct{}{}
	fdDir := hostProcPath(pid, "fd")
	entries, err := os.ReadDir(fdDir)
	if err != nil {
		return inodes
	}
	for _, entry := range entries {
		target, err := os.Readlink(filepath.Join(fdDir, entry.Name()))
		if err != nil || !strings.HasPrefix(target, "socket:[") {
			continue
		}
		inode := strings.TrimSuffix(strings.TrimPrefix(target, "socket:["), "]")
		if inode != "" {
			inodes[inode] = struct{}{}
		}
	}
	return inodes
}

func writeDeviceVMAs(out *bytes.Buffer, pid int) {
	fmt.Fprintln(out, "\n--- device vmas ---")
	maps := readHostProcFile(pid, "maps")
	if maps == "" {
		fmt.Fprintln(out, "<unavailable>")
		return
	}
	wrote := false
	for _, line := range strings.Split(maps, "\n") {
		if !interestingVMALine(line) {
			continue
		}
		wrote = true
		fmt.Fprintln(out, line)
	}
	if !wrote {
		fmt.Fprintln(out, "<none>")
	}
}

func interestingVMALine(line string) bool {
	return strings.Contains(line, "/dev/nvidia") ||
		strings.Contains(line, "/dev/dri/") ||
		strings.Contains(line, "/dev/nv")
}

func hostProcPath(pid int, parts ...string) string {
	allParts := append([]string{snapshotruntime.HostProcPath, strconv.Itoa(pid)}, parts...)
	return filepath.Join(allParts...)
}

func readHostProcFile(pid int, parts ...string) string {
	return readHostProcPath(hostProcPath(pid, parts...))
}

func readHostProcPath(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(data)
}

func intSet(values []int) map[int]struct{} {
	set := make(map[int]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return set
}

func sortedIntSet(set map[int]struct{}) []int {
	values := make([]int, 0, len(set))
	for value := range set {
		values = append(values, value)
	}
	sort.Ints(values)
	return values
}

func sortedStringSet(set map[string]struct{}) []string {
	values := make([]string, 0, len(set))
	for value := range set {
		values = append(values, value)
	}
	sort.Strings(values)
	return values
}
