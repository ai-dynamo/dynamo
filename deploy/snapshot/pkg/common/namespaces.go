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

// GetNetNSInode returns the network namespace inode for a container process via /host/proc.
func GetNetNSInode(pid int) (uint64, error) {
	nsPath := fmt.Sprintf("%s/%d/ns/net", HostProcPath, pid)
	var stat unix.Stat_t
	if err := unix.Stat(nsPath, &stat); err != nil {
		return 0, fmt.Errorf("failed to stat %s: %w", nsPath, err)
	}
	return stat.Ino, nil
}

// NamespacePIDs returns the nested NSpid chain from outermost to innermost for a host PID.
func NamespacePIDs(hostPID int) ([]int, error) {
	if hostPID <= 0 {
		return nil, fmt.Errorf("invalid host PID %d", hostPID)
	}

	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", hostPID))
	if err != nil {
		return nil, fmt.Errorf("failed to read status for pid %d: %w", hostPID, err)
	}

	return parseNSPIDs(string(data), hostPID)
}

func parseNSPIDs(status string, hostPID int) ([]int, error) {
	for _, line := range strings.Split(status, "\n") {
		if !strings.HasPrefix(line, "NSpid:") {
			continue
		}

		fields := strings.Fields(strings.TrimPrefix(line, "NSpid:"))
		if len(fields) == 0 {
			break
		}

		nspids := make([]int, 0, len(fields))
		for _, field := range fields {
			value, err := strconv.Atoi(field)
			if err != nil {
				return nil, fmt.Errorf("failed to parse NSpid %q for pid %d: %w", field, hostPID, err)
			}
			nspids = append(nspids, value)
		}
		return nspids, nil
	}

	return nil, fmt.Errorf("pid %d is missing NSpid data", hostPID)
}

// NamespacePID returns the innermost namespace PID for a host-visible PID.
func NamespacePID(hostPID int) (int, error) {
	nspids, err := NamespacePIDs(hostPID)
	if err != nil {
		return 0, err
	}
	return nspids[len(nspids)-1], nil
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
