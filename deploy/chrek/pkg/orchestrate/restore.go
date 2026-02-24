package orchestrate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/containerd/containerd"
	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/logging"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

// RestoreRequest holds the parameters for a restore operation.
type RestoreRequest struct {
	CheckpointHash string
	CheckpointBase string
	NSRestorePath  string
	PodName        string
	PodNamespace   string
	ContainerName  string
}

// Restore performs external restore for the given request.
// Returns the namespace-relative PID of the restored process.
// The DaemonSet side inspects the placeholder and launches nsrestore,
// which handles rootfs application, CRIU restore, and CUDA restore inside the namespace.
func Restore(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req RestoreRequest) (int, error) {
	restoreStart := time.Now()
	log.Info("=== Starting external restore ===",
		"checkpoint_hash", req.CheckpointHash,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)

	// Phase 1: Inspect — resolve placeholder, discover target GPUs, build device map
	snap, err := inspectRestore(ctx, ctrd, log, req)
	if err != nil {
		return 0, err
	}

	// Phase 2: Execute — nsrestore handles rootfs, CRIU restore, and CUDA restore inside namespace
	restoredPID, err := execNSRestore(ctx, log, req, snap)
	if err != nil {
		return 0, fmt.Errorf("nsrestore failed: %w", err)
	}
	log.Info("nsrestore completed", "restored_pid", restoredPID)

	// Validate restored process from the host side
	procRoot := filepath.Join(snap.TargetRoot, "proc")
	if err := common.ValidateProcessState(procRoot, restoredPID); err != nil {
		restoreLogPath := filepath.Join(snap.TargetRoot, "var", "criu-work", criu.RestoreLogFilename)
		logging.LogProcessDiagnostics(procRoot, restoredPID, restoreLogPath, log)
		return 0, fmt.Errorf("restored process failed post-restore validation: %w", err)
	}

	log.Info("=== External restore completed ===", "total_duration", time.Since(restoreStart))

	return restoredPID, nil
}

func inspectRestore(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req RestoreRequest) (*types.RestoreContainerSnapshot, error) {
	checkpointPath := filepath.Join(req.CheckpointBase, req.CheckpointHash)
	baseAbs, err := filepath.Abs(req.CheckpointBase)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve checkpoint base path: %w", err)
	}
	checkpointAbs, err := filepath.Abs(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve checkpoint path: %w", err)
	}
	if checkpointAbs != baseAbs && !strings.HasPrefix(checkpointAbs, baseAbs+string(os.PathSeparator)) {
		return nil, fmt.Errorf("invalid checkpoint hash %q", req.CheckpointHash)
	}

	m, err := types.ReadManifest(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}

	placeholderPID, _, err := common.ResolveContainerByPod(ctx, ctrd, req.PodName, req.PodNamespace, containerName)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve placeholder container: %w", err)
	}
	log.Info("Resolved placeholder container", "pid", placeholderPID)

	cgroupRoot, err := common.ResolveCgroupRootFromHostPID(placeholderPID)
	if err != nil {
		log.Error(err, "Failed to resolve placeholder cgroup root; proceeding without explicit cgroup remap")
		cgroupRoot = ""
	}

	cudaDeviceMap := ""
	if !m.CUDA.IsEmpty() {
		if len(m.CUDA.SourceGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
		}
		targetGPUUUIDs, err := cuda.GetPodGPUUUIDs(ctx, req.PodName, req.PodNamespace, containerName)
		if err != nil {
			return nil, fmt.Errorf("failed to get target GPU UUIDs: %w", err)
		}
		if len(targetGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", req.PodNamespace, req.PodName, containerName)
		}
		cudaDeviceMap, err = cuda.BuildDeviceMap(m.CUDA.SourceGPUUUIDs, targetGPUUUIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to build CUDA device map: %w", err)
		}
	}

	return &types.RestoreContainerSnapshot{
		CheckpointPath: checkpointPath,
		PlaceholderPID: placeholderPID,
		TargetRoot:     fmt.Sprintf("%s/%d/root", common.HostProcPath, placeholderPID),
		CgroupRoot:     cgroupRoot,
		CUDADeviceMap:  cudaDeviceMap,
	}, nil
}

// execNSRestore launches the nsrestore binary inside the placeholder container's
// namespaces via nsenter and parses the restored PID from stdout JSON.
func execNSRestore(ctx context.Context, log logr.Logger, req RestoreRequest, snap *types.RestoreContainerSnapshot) (int, error) {
	args := []string{
		"-t", strconv.Itoa(snap.PlaceholderPID),
		// Intentionally exclude cgroup namespace (-C): CRIU must manage cgroups
		// from the host-visible hierarchy so --cgroup-root remap works.
		"-m", "-u", "-i", "-n", "-p",
		"--", req.NSRestorePath,
		"--checkpoint-path", snap.CheckpointPath,
	}
	if snap.CUDADeviceMap != "" {
		args = append(args, "--cuda-device-map", snap.CUDADeviceMap)
	}
	if snap.CgroupRoot != "" {
		args = append(args, "--cgroup-root", snap.CgroupRoot)
	}

	cmd := exec.CommandContext(ctx, "nsenter", args...)
	log.V(1).Info("Executing nsenter + nsrestore", "cmd", cmd.String())

	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf("nsrestore failed: %w\nstdout: %s", err, stdout.String())
	}

	var result struct {
		RestoredPID int `json:"restoredPID"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return 0, fmt.Errorf("failed to parse nsrestore result: %w\nstdout: %s", err, stdout.String())
	}
	if result.RestoredPID <= 0 {
		return 0, fmt.Errorf("nsrestore returned invalid PID %d", result.RestoredPID)
	}

	return result.RestoredPID, nil
}

