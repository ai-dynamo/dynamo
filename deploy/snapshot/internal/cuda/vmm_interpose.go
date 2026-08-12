package cuda

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
)

const (
	vmmInterposeEnv = "DYN_SNAPSHOT_CUDA_VMM_INTERPOSE"
	vmmCoordinator  = "/usr/local/bin/snapshot-cuda-vmm"
	vmmStateFile    = "cuda-vmm.state"
)

func DetectVMMInterpose(procRoot string, pids []int) (bool, error) {
	enabled := 0
	for _, pid := range pids {
		content, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "environ"))
		if err != nil {
			return false, fmt.Errorf("read CUDA process %d environment: %w", pid, err)
		}
		for _, entry := range strings.Split(string(content), "\x00") {
			if entry == vmmInterposeEnv+"=1" {
				enabled++
				break
			}
		}
	}
	if enabled != 0 && enabled != len(pids) {
		return false, fmt.Errorf(
			"CUDA VMM interposer is enabled for %d of %d CUDA processes",
			enabled,
			len(pids),
		)
	}
	return enabled != 0, nil
}

func PrepareVMM(
	ctx context.Context,
	checkpointDir string,
	procRoot string,
	observedPIDs []int,
	namespacePIDs []int,
	log logr.Logger,
) error {
	args, err := vmmArgs("prepare", checkpointDir, procRoot, observedPIDs, namespacePIDs)
	if err != nil {
		return err
	}
	output, err := exec.CommandContext(ctx, vmmCoordinator, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s failed: %w (output: %s)", vmmCoordinator, err, strings.TrimSpace(string(output)))
	}
	log.Info("Prepared CUDA VMM interpose state")
	return nil
}

func RestoreVMM(
	ctx context.Context,
	checkpointDir string,
	observedPIDs []int,
	namespacePIDs []int,
) error {
	args, err := vmmArgs("restore", checkpointDir, "", observedPIDs, namespacePIDs)
	if err != nil {
		return err
	}
	output, err := exec.CommandContext(ctx, vmmCoordinator, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s failed: %w (output: %s)", vmmCoordinator, err, strings.TrimSpace(string(output)))
	}
	return nil
}

func HasVMMState(checkpointDir string) (bool, error) {
	path := filepath.Join(checkpointDir, vmmStateFile)
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("stat CUDA VMM state: %w", err)
	}
	if !info.Mode().IsRegular() {
		return false, fmt.Errorf("CUDA VMM state %q is not a regular file", path)
	}
	return true, nil
}

func vmmArgs(operation, checkpointDir, procRoot string, observedPIDs, namespacePIDs []int) ([]string, error) {
	if len(observedPIDs) != len(namespacePIDs) {
		return nil, fmt.Errorf(
			"CUDA VMM PID mapping count mismatch: observed=%d namespace=%d",
			len(observedPIDs),
			len(namespacePIDs),
		)
	}
	args := []string{
		"--" + operation,
		"--proc-root",
		procRoot,
		"--checkpoint-dir",
		checkpointDir,
	}
	for index, observedPID := range observedPIDs {
		args = append(
			args,
			"--process",
			strconv.Itoa(observedPID),
			strconv.Itoa(namespacePIDs[index]),
		)
	}
	return args, nil
}
