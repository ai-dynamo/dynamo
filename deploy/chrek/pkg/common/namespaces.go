package common

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"
)

// ResolveNamespacePIDs translates host-visible PIDs to their innermost namespace-relative PIDs
// by parsing the NSpid line from /proc/<pid>/status. The returned slice preserves the input order.
// This runs on the DaemonSet (hostPID: true), so /proc is the host's /proc.
func ResolveNamespacePIDs(hostPIDs []int) ([]int, error) {
	nsPIDs := make([]int, len(hostPIDs))
	for i, hpid := range hostPIDs {
		status, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", hpid))
		if err != nil {
			return nil, fmt.Errorf("failed to read /proc/%d/status: %w", hpid, err)
		}
		nsPID, err := parseInnermostNSpid(string(status), hpid)
		if err != nil {
			return nil, err
		}
		nsPIDs[i] = nsPID
	}
	return nsPIDs, nil
}

// parseInnermostNSpid extracts the innermost (last) PID from the NSpid line in /proc/status.
// Format: "NSpid:\t<host_pid>\t<ns_pid_1>\t...\t<innermost_ns_pid>"
func parseInnermostNSpid(status string, hostPID int) (int, error) {
	for _, line := range strings.Split(status, "\n") {
		if !strings.HasPrefix(line, "NSpid:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("malformed NSpid line for PID %d: %q", hostPID, line)
		}
		// Last field is the innermost namespace PID
		nsPID, err := strconv.Atoi(fields[len(fields)-1])
		if err != nil {
			return 0, fmt.Errorf("failed to parse namespace PID for host PID %d: %w", hostPID, err)
		}
		return nsPID, nil
	}
	return 0, fmt.Errorf("NSpid line not found in /proc/%d/status", hostPID)
}

// GetNetNSInode returns the network namespace inode for a container process via /host/proc.
func GetNetNSInode(pid int) (uint64, error) {
	nsPath := fmt.Sprintf("%s/%d/ns/net", HostProcPath, pid)
	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return 0, fmt.Errorf("failed to stat %s: %w", nsPath, err)
	}
	return stat.Ino, nil
}

// SendSignalViaPIDNamespace sends a signal to a namespace-relative PID by entering the
// PID namespace of referenceHostPID via nsenter.
func SendSignalViaPIDNamespace(ctx context.Context, log logr.Logger, referenceHostPID, targetNamespacePID int, sig syscall.Signal, reason string) error {
	if referenceHostPID <= 0 {
		return fmt.Errorf("invalid reference host PID %d for signal %d", referenceHostPID, int(sig))
	}
	if targetNamespacePID <= 0 {
		return fmt.Errorf("invalid namespace PID %d for signal %d", targetNamespacePID, int(sig))
	}

	cmd := exec.CommandContext(
		ctx,
		"nsenter",
		"-t", strconv.Itoa(referenceHostPID),
		"-p",
		"--",
		"kill",
		fmt.Sprintf("-%d", int(sig)),
		strconv.Itoa(targetNamespacePID),
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf(
			"failed to signal namespace PID %d via reference host PID %d with signal %d (%s): %w (output: %s)",
			targetNamespacePID, referenceHostPID, int(sig), reason, err, strings.TrimSpace(string(output)),
		)
	}

	log.Info("Signaled runtime process in PID namespace",
		"reference_host_pid", referenceHostPID,
		"namespace_pid", targetNamespacePID,
		"signal", int(sig),
		"reason", reason,
	)
	return nil
}

