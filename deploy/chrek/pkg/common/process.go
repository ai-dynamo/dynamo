package common

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/go-logr/logr"
	"github.com/prometheus/procfs"
)

// HostProcPath is the mount point for the host's /proc in DaemonSet pods.
const HostProcPath = "/host/proc"

// ProcessTreePIDs walks the process tree rooted at rootPID and returns all PIDs.
// Used by nsrestore for in-namespace CUDA PID discovery.
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
