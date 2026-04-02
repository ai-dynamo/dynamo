package common

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"

	"github.com/go-logr/logr"
	"github.com/prometheus/procfs"
)

// HostProcPath is the mount point for the host's /proc in DaemonSet pods.
const HostProcPath = "/host/proc"

// ProcessDetails captures the parent link plus the observed, outermost, and innermost
// PID views for one proc entry. ObservedPID is relative to the proc root being read.
type ProcessDetails struct {
	ObservedPID   int
	ParentPID     int
	OutermostPID  int
	InnermostPID  int
	NamespacePIDs []int
	Cmdline       string
}

// ReadProcessDetails reads one proc entry from a proc root.
func ReadProcessDetails(procRoot string, pid int) (ProcessDetails, error) {
	if pid <= 0 {
		return ProcessDetails{}, fmt.Errorf("invalid PID %d", pid)
	}

	statusPath := filepath.Join(procRoot, strconv.Itoa(pid), "status")
	statusBytes, err := os.ReadFile(statusPath)
	if err != nil {
		return ProcessDetails{}, fmt.Errorf("failed to read %s: %w", statusPath, err)
	}
	status := string(statusBytes)

	parentPID := 0
	parentPIDFound := false
	for _, line := range strings.Split(status, "\n") {
		if strings.HasPrefix(line, "PPid:") {
			value := strings.TrimSpace(strings.TrimPrefix(line, "PPid:"))
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return ProcessDetails{}, fmt.Errorf("failed to parse PPid value %q: %w", value, err)
			}
			parentPID = parsed
			parentPIDFound = true
			break
		}
	}
	if !parentPIDFound {
		return ProcessDetails{}, fmt.Errorf("missing PPid in process status")
	}

	var nspids []int
	for _, line := range strings.Split(status, "\n") {
		if !strings.HasPrefix(line, "NSpid:") {
			continue
		}

		fields := strings.Fields(strings.TrimPrefix(line, "NSpid:"))
		if len(fields) == 0 {
			break
		}

		nspids = make([]int, 0, len(fields))
		for _, field := range fields {
			value, err := strconv.Atoi(field)
			if err != nil {
				return ProcessDetails{}, fmt.Errorf("failed to parse NSpid %q: %w", field, err)
			}
			nspids = append(nspids, value)
		}
		break
	}
	if len(nspids) == 0 {
		return ProcessDetails{}, fmt.Errorf("missing NSpid in process status")
	}

	cmdline := ""
	if data, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "cmdline")); err == nil {
		cmdline = strings.TrimSpace(strings.ReplaceAll(string(data), "\x00", " "))
	}
	if cmdline == "" {
		comm, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "comm"))
		if err == nil {
			cmdline = strings.TrimSpace(string(comm))
		}
	}

	return ProcessDetails{
		ObservedPID:   pid,
		ParentPID:     parentPID,
		OutermostPID:  nspids[0],
		InnermostPID:  nspids[len(nspids)-1],
		NamespacePIDs: nspids,
		Cmdline:       cmdline,
	}, nil
}

// ReadProcessDetailsOrDefault preserves pid-scoped logging even when proc parsing fails.
func ReadProcessDetailsOrDefault(procRoot string, pid int) ProcessDetails {
	details := ProcessDetails{
		ObservedPID:   pid,
		OutermostPID:  pid,
		InnermostPID:  pid,
		NamespacePIDs: []int{pid},
	}
	if process, err := ReadProcessDetails(procRoot, pid); err == nil {
		details = process
	}
	return details
}

// ReadProcessTable snapshots every numeric proc entry under procRoot.
// Used by restore-side PID remap and diagnostics after CRIU restore.
func ReadProcessTable(procRoot string) ([]ProcessDetails, error) {
	entries, err := os.ReadDir(procRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to read proc root %s: %w", procRoot, err)
	}

	processes := make([]ProcessDetails, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		process, err := ReadProcessDetails(procRoot, pid)
		if err != nil {
			continue
		}
		processes = append(processes, process)
	}

	// Keep restore diagnostics deterministic by ordering on the manifest-facing PID view first.
	sort.Slice(processes, func(i, j int) bool {
		leftPID := processes[i].ObservedPID
		if processes[i].InnermostPID > 0 {
			leftPID = processes[i].InnermostPID
		}
		rightPID := processes[j].ObservedPID
		if processes[j].InnermostPID > 0 {
			rightPID = processes[j].InnermostPID
		}

		if leftPID == rightPID {
			return processes[i].ObservedPID < processes[j].ObservedPID
		}
		return leftPID < rightPID
	})

	return processes, nil
}

// ResolveManifestPIDsToObservedPIDs is the restore-side remap from checkpoint-time
// innermost namespace PIDs onto the current observed PIDs in the restored subtree rooted at restoredPID.
func ResolveManifestPIDsToObservedPIDs(processes []ProcessDetails, restoredPID int, manifestPIDs []int) ([]int, error) {
	processByObservedPID := make(map[int]ProcessDetails, len(processes))
	childrenByParentPID := make(map[int][]int, len(processes))
	for _, process := range processes {
		processByObservedPID[process.ObservedPID] = process
		childrenByParentPID[process.ParentPID] = append(childrenByParentPID[process.ParentPID], process.ObservedPID)
	}

	if _, ok := processByObservedPID[restoredPID]; !ok {
		return nil, fmt.Errorf("restored root pid %d not found in process table", restoredPID)
	}

	innermostToObservedPID := map[int]int{}
	queue := []int{restoredPID}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]

		process, ok := processByObservedPID[pid]
		if !ok {
			continue
		}
		if len(process.NamespacePIDs) != 2 {
			return nil, fmt.Errorf("restored process %d has namespace depth %d, want 2", pid, len(process.NamespacePIDs))
		}
		if existingPID, ok := innermostToObservedPID[process.InnermostPID]; ok {
			return nil, fmt.Errorf("multiple restored processes map to innermost pid %d: %d and %d", process.InnermostPID, existingPID, process.ObservedPID)
		}

		innermostToObservedPID[process.InnermostPID] = process.ObservedPID
		queue = append(queue, childrenByParentPID[pid]...)
	}

	restorePIDs := make([]int, 0, len(manifestPIDs))
	for _, manifestPID := range manifestPIDs {
		observedPID, ok := innermostToObservedPID[manifestPID]
		if !ok {
			return nil, fmt.Errorf("manifest cuda pid %d not found under restored subtree rooted at %d", manifestPID, restoredPID)
		}
		restorePIDs = append(restorePIDs, observedPID)
	}

	return restorePIDs, nil
}

// ProcessTreePIDs walks the process tree rooted at rootPID and returns all PIDs.
// Used during checkpoint to enumerate the source process tree before CUDA filtering.
func ProcessTreePIDs(rootPID int) []int {
	if rootPID <= 0 {
		return nil
	}

	queue := []int{rootPID}
	seen := map[int]struct{}{}
	all := make([]int, 0, 16)

	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if _, ok := seen[pid]; ok {
			continue
		}
		seen[pid] = struct{}{}
		if _, err := os.Stat(fmt.Sprintf("/proc/%d", pid)); err != nil {
			continue
		}
		all = append(all, pid)

		// Iterate all threads — child processes can be spawned from any thread, not just the main thread (tid==pid).
		taskDir := fmt.Sprintf("/proc/%d/task", pid)
		tids, err := os.ReadDir(taskDir)
		if err != nil {
			continue
		}
		for _, tid := range tids {
			children, err := os.ReadFile(fmt.Sprintf("%s/%s/children", taskDir, tid.Name()))
			if err != nil {
				continue
			}
			for _, child := range strings.Fields(string(children)) {
				childPID, err := strconv.Atoi(child)
				if err != nil {
					continue
				}
				queue = append(queue, childPID)
			}
		}
	}

	return all
}

// ValidateProcessState checks that a process is alive and not a zombie.
func ValidateProcessState(procRoot string, pid int) error {
	if pid <= 0 {
		return fmt.Errorf("invalid restored PID %d", pid)
	}

	fs, err := procfs.NewFS(procRoot)
	if err != nil {
		return fmt.Errorf("failed to open procfs at %s: %w", procRoot, err)
	}
	proc, err := fs.Proc(pid)
	if err != nil {
		return fmt.Errorf("process %d exited", pid)
	}
	stat, err := proc.Stat()
	if err != nil {
		return fmt.Errorf("failed to inspect process %d: %w", pid, err)
	}
	if stat.State == "Z" {
		return fmt.Errorf("process %d became zombie", pid)
	}
	return nil
}

// ParseProcExitCode extracts and decodes the exit_code field (field 52) from a /proc/<pid>/stat line.
func ParseProcExitCode(statLine string) (syscall.WaitStatus, error) {
	statLine = strings.TrimSpace(statLine)
	paren := strings.LastIndex(statLine, ")")
	if paren < 0 || paren+2 > len(statLine) {
		return 0, fmt.Errorf("malformed stat line")
	}
	fields := strings.Fields(statLine[paren+2:])
	if len(fields) == 0 {
		return 0, fmt.Errorf("malformed stat fields")
	}
	raw, err := strconv.Atoi(fields[len(fields)-1])
	if err != nil {
		return 0, err
	}
	return syscall.WaitStatus(raw), nil
}

// SendSignalToPID sends a signal to a host-visible PID via syscall.Kill.
func SendSignalToPID(log logr.Logger, pid int, sig syscall.Signal, reason string) error {
	signalID := int(sig)
	if pid <= 0 {
		return fmt.Errorf("invalid PID %d for signal %d", pid, signalID)
	}
	if err := syscall.Kill(pid, sig); err != nil {
		return fmt.Errorf("failed to signal PID %d with signal %d (%s): %w", pid, signalID, reason, err)
	}
	log.Info("Signaled runtime process", "pid", pid, "signal", signalID, "reason", reason)
	return nil
}
