package executor

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
	"k8s.io/client-go/kubernetes"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// RestoreRequest holds the parameters for a restore operation.
type RestoreRequest struct {
	CheckpointID       string
	CheckpointLocation string
	StartedAt          time.Time
	NSRestorePath      string
	PodName            string
	PodNamespace       string
	ContainerName      string
	Clientset          kubernetes.Interface
}

// Restore performs external restore for the given request.
// Returns the namespace-relative PID of the restored process.
// The DaemonSet side inspects the placeholder and launches nsrestore,
// which handles rootfs application, CRIU restore, and CUDA restore inside the namespace.
func Restore(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req RestoreRequest) (int, error) {
	restoreStart := time.Now()
	log.Info("=== Starting external restore ===",
		"checkpoint_id", req.CheckpointID,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)

	// Phase 1: Host inspect — resolve placeholder, discover target GPUs, build device map
	hostInspectStart := time.Now()
	snap, err := inspectRestore(ctx, ctrd, log, req)
	if err != nil {
		return 0, err
	}
	hostInspectDuration := time.Since(hostInspectStart)

	// Phase 2: Execute — nsrestore handles rootfs, CRIU restore, and CUDA restore inside namespace
	result, err := execNSRestore(ctx, log, req, snap)
	if err != nil {
		return 0, fmt.Errorf("nsrestore failed: %w", err)
	}
	restoreDuration := hostInspectDuration + result.NSRestoreSetupDuration + result.CRIURestoreDuration + result.CUDADuration
	log.Info("Restore timing summary",
		"restore", map[string]any{
			"duration": restoreDuration.String(),
			"phases": map[string]string{
				"host_inspect_duration":    hostInspectDuration.String(),
				"nsrestore_setup_duration": result.NSRestoreSetupDuration.String(),
				"criu_restore_duration":    result.CRIURestoreDuration.String(),
				"cuda_duration":            result.CUDADuration.String(),
			},
		},
	)
	if !req.StartedAt.IsZero() {
		log.Info("Restore wall time from agent detection",
			"started_to_restore_complete", time.Since(req.StartedAt),
		)
	}

	// Validate restored process from the host side
	validationStart := time.Now()
	procRoot := filepath.Join(snap.TargetRoot, "proc")
	if err := snapshotruntime.ValidateProcessState(procRoot, result.RestoredPID); err != nil {
		restoreLogPath := filepath.Join(snap.TargetRoot, "var", "criu-work", criu.RestoreLogFilename)
		logging.LogProcessDiagnostics(procRoot, result.RestoredPID, restoreLogPath, log)
		return 0, fmt.Errorf("restored process failed post-restore validation: %w", err)
	}

	log.Info("=== External restore completed ===",
		"restored_pid", result.RestoredPID,
		"validation_duration", time.Since(validationStart),
		"total_duration", time.Since(restoreStart),
	)

	return result.RestoredPID, nil
}

func inspectRestore(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req RestoreRequest) (*types.RestoreContainerSnapshot, error) {
	if req.CheckpointLocation == "" {
		return nil, fmt.Errorf("checkpoint location is required")
	}

	checkpointPath := req.CheckpointLocation
	baseAbs, err := filepath.Abs(filepath.Dir(checkpointPath))
	if err != nil {
		return nil, fmt.Errorf("failed to resolve checkpoint base path: %w", err)
	}
	checkpointAbs, err := filepath.Abs(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve checkpoint path: %w", err)
	}
	if checkpointAbs != baseAbs && !strings.HasPrefix(checkpointAbs, baseAbs+string(os.PathSeparator)) {
		return nil, fmt.Errorf("invalid checkpoint id %q", req.CheckpointID)
	}

	m, err := types.ReadManifest(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}

	placeholderPID, _, err := snapshotruntime.ResolveContainerByPod(ctx, ctrd, req.PodName, req.PodNamespace, containerName)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve placeholder container: %w", err)
	}
	log.V(1).Info("Resolved placeholder container", "pid", placeholderPID)

	cgroupRoot, err := snapshotruntime.ResolveCgroupRootFromHostPID(placeholderPID)
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
			log.Info("PodResources API returned no target GPU UUIDs, falling back to nvidia-smi", "pid", placeholderPID)
			targetGPUUUIDs, err = cuda.GetGPUUUIDsViaNvidiaSmi(ctx, snapshotruntime.HostProcPath, placeholderPID)
			if err != nil {
				return nil, fmt.Errorf("nvidia-smi GPU UUID fallback failed for restore target: %w", err)
			}
			log.Info("nvidia-smi fallback discovered target GPU UUIDs", "uuids", targetGPUUUIDs)
		}
		if len(targetGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", req.PodNamespace, req.PodName, containerName)
		}
		cudaDeviceMap, err = cuda.BuildDeviceMap(m.CUDA.SourceGPUUUIDs, targetGPUUUIDs, log)
		if err != nil {
			return nil, fmt.Errorf("failed to build CUDA device map: %w", err)
		}
		log.V(1).Info("GPU UUIDs for device map",
			"source_uuids", m.CUDA.SourceGPUUUIDs,
			"target_uuids", targetGPUUUIDs,
			"device_map", cudaDeviceMap,
		)
	}

	return &types.RestoreContainerSnapshot{
		CheckpointPath: checkpointPath,
		PlaceholderPID: placeholderPID,
		TargetRoot:     fmt.Sprintf("%s/%d/root", snapshotruntime.HostProcPath, placeholderPID),
		CgroupRoot:     cgroupRoot,
		CUDADeviceMap:  cudaDeviceMap,
	}, nil
}

// execNSRestore launches the nsrestore binary inside the placeholder container's
// namespaces via nsenter and parses the restored PID from stdout JSON.
func execNSRestore(ctx context.Context, log logr.Logger, req RestoreRequest, snap *types.RestoreContainerSnapshot) (*RestoreInNamespaceResult, error) {
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
	// Inherit the agent environment so nsrestore uses the same logger settings.
	cmd.Env = os.Environ()
	log.V(1).Info("Executing nsenter + nsrestore", "cmd", cmd.String())

	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("nsrestore failed: %w\nstdout: %s", err, stdout.String())
	}

	var result RestoreInNamespaceResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("failed to parse nsrestore result: %w\nstdout: %s", err, stdout.String())
	}
	if result.RestoredPID <= 0 {
		return nil, fmt.Errorf("nsrestore returned invalid PID %d", result.RestoredPID)
	}

	return &result, nil
}
