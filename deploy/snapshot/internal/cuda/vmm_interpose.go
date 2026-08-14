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
)

func DetectVMMInterpose(procRoot string, observedPIDs, namespacePIDs []int) (bool, error) {
	if len(observedPIDs) != len(namespacePIDs) {
		return false, fmt.Errorf(
			"CUDA VMM PID mapping count mismatch: observed=%d namespace=%d",
			len(observedPIDs),
			len(namespacePIDs),
		)
	}
	markers := 0
	endpoints := 0
	validEndpoints := 0
	for index, observedPID := range observedPIDs {
		content, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(observedPID), "environ"))
		if err != nil {
			return false, fmt.Errorf("read CUDA process %d environment: %w", observedPID, err)
		}
		for _, entry := range strings.Split(string(content), "\x00") {
			if entry == vmmInterposeEnv+"=1" {
				markers++
				break
			}
		}
		endpoint := filepath.Join(
			procRoot,
			strconv.Itoa(observedPID),
			"root",
			"snapshot-control",
			fmt.Sprintf("cuda-vmm-%d.sock", namespacePIDs[index]),
		)
		info, err := os.Lstat(endpoint)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return false, fmt.Errorf("stat CUDA VMM endpoint %q: %w", endpoint, err)
		}
		endpoints++
		if info.Mode()&os.ModeSocket != 0 {
			validEndpoints++
		}
	}
	if markers == 0 && endpoints == 0 {
		return false, nil
	}
	if validEndpoints != len(observedPIDs) {
		return false, fmt.Errorf(
			"CUDA VMM interposer endpoint missing or invalid for %d of %d CUDA processes",
			len(observedPIDs)-validEndpoints,
			len(observedPIDs),
		)
	}
	return true, nil
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
