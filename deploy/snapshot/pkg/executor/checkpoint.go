// Package executor provides the top-level checkpoint and restore executors.
// These wire together the lib packages (criu, cuda, etc.) into multi-step workflows.
package executor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/containerd/containerd"
	"github.com/go-logr/logr"
	"github.com/google/uuid"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/types"
)

// CheckpointRequest holds per-checkpoint identifiers for a checkpoint operation.
type CheckpointRequest struct {
	ContainerID           string
	ContainerName         string
	CheckpointHash        string
	CheckpointLocation    string
	CheckpointStorageType string
	NodeName              string
	PodName               string
	PodNamespace          string
}

// Checkpoint performs a CRIU dump of a container.
// The operation has three phases: inspect, configure, capture.
//
// The checkpoint directory is staged under tmp/<uuid> during the operation.
// On success, the previous checkpoint is removed and the staged directory is
// renamed into place at the base path root.
func Checkpoint(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req CheckpointRequest, cfg *types.AgentConfig) error {
	checkpointStart := time.Now()
	log.Info("=== Starting checkpoint operation ===")

	if req.CheckpointStorageType != "pvc" {
		return fmt.Errorf("checkpoint storage type %q is not supported", req.CheckpointStorageType)
	}
	if req.CheckpointLocation == "" {
		return fmt.Errorf("checkpoint location is required")
	}

	finalDir := req.CheckpointLocation
	tmpRoot := filepath.Join(filepath.Dir(finalDir), "tmp")
	if err := os.MkdirAll(tmpRoot, 0700); err != nil {
		return fmt.Errorf("failed to create checkpoint staging root: %w", err)
	}
	tmpDir := filepath.Join(tmpRoot, uuid.NewString())
	if err := os.Mkdir(tmpDir, 0700); err != nil {
		return fmt.Errorf("failed to create checkpoint staging directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Phase 1: Inspect container state
	state, err := inspectContainer(ctx, ctrd, log, req)
	if err != nil {
		return err
	}

	// Phase 2: Configure CRIU options and build checkpoint manifest
	criuOpts, data, err := configureCheckpoint(log, state, req, cfg, tmpDir)
	if err != nil {
		return err
	}

	// Phase 3: Capture — CRIU dump, rootfs diff
	criuDumpDuration, err := captureCheckpoint(ctx, criuOpts, &cfg.CRIU, data, state, tmpDir, log)
	if err != nil {
		return err
	}

	// Remove any previous checkpoint with the same identity hash before finalizing
	if err := os.RemoveAll(finalDir); err != nil {
		return fmt.Errorf("failed to remove previous checkpoint directory: %w", err)
	}
	if err := os.Rename(tmpDir, finalDir); err != nil {
		return fmt.Errorf("failed to finalize checkpoint directory: %w", err)
	}

	totalDuration := time.Since(checkpointStart)
	log.Info("=== Checkpoint operation completed ===",
		"total_duration", totalDuration,
		"criu_dump_duration", criuDumpDuration,
	)

	return nil
}

func inspectContainer(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req CheckpointRequest) (*types.CheckpointContainerSnapshot, error) {
	containerID := req.ContainerID
	pid, ociSpec, err := common.ResolveContainer(ctx, ctrd, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container: %w", err)
	}

	var hostCgroupPath string
	if cgPath, err := common.ResolveCgroupRootFromHostPID(pid); err == nil && cgPath != "" {
		hostCgroupPath = filepath.Join(common.HostCgroupPath, cgPath)
	}

	rootFS, err := common.GetRootFS(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get rootfs: %w", err)
	}

	upperDir, err := common.GetOverlayUpperDir(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get overlay upperdir: %w", err)
	}

	mountInfo, err := common.ReadMountInfo(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo: %w", err)
	}
	mounts := common.ClassifyMounts(mountInfo, ociSpec, rootFS)

	netNSInode, err := common.GetNetNSInode(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get net namespace inode: %w", err)
	}

	// Read stdio FD targets (like runc's getPipeFds / descriptors.json).
	stdioFDs := make([]string, 3)
	for i := range 3 {
		target, err := os.Readlink(fmt.Sprintf("%s/%d/fd/%d", common.HostProcPath, pid, i))
		if err != nil {
			log.V(1).Info("Failed to readlink stdio FD", "fd", i, "error", err)
			continue
		}
		stdioFDs[i] = target
	}

	// Discover CUDA processes and GPU UUIDs
	allPIDs := common.ProcessTreePIDs(pid)
	cudaHostPIDs := cuda.FilterProcesses(ctx, allPIDs, log)
	cudaNamespacePIDs := make([]int, 0, len(cudaHostPIDs))
	for _, cudaHostPID := range cudaHostPIDs {
		process, err := common.ReadProcessDetails(common.HostProcPath, cudaHostPID)
		if err != nil {
			return nil, fmt.Errorf("failed to read process details for CUDA process %d: %w", cudaHostPID, err)
		}
		if len(process.NamespacePIDs) != 2 {
			return nil, fmt.Errorf("CUDA process %d has namespace depth %d, want 2", cudaHostPID, len(process.NamespacePIDs))
		}
		cudaNamespacePIDs = append(cudaNamespacePIDs, process.InnermostPID)
	}
	if len(cudaHostPIDs) > 0 {
		log.Info("Resolved checkpoint CUDA PID mapping", "host_pids", cudaHostPIDs, "namespace_pids", cudaNamespacePIDs)
	}
	var gpuUUIDs []string
	if len(cudaHostPIDs) > 0 {
		gpuUUIDs, err = cuda.GetPodGPUUUIDs(ctx, req.PodName, req.PodNamespace, req.ContainerName)
		if err != nil {
			return nil, fmt.Errorf("failed to discover source GPU UUIDs: %w", err)
		}
	}

	return &types.CheckpointContainerSnapshot{
		PID:            pid,
		RootFS:         rootFS,
		UpperDir:       upperDir,
		OCISpec:        ociSpec,
		Mounts:         mounts,
		NetNSInode:     netNSInode,
		StdioFDs:       stdioFDs,
		HostCgroupPath: hostCgroupPath,
		CUDAHostPIDs:   cudaHostPIDs,
		CUDANSPIDs:     cudaNamespacePIDs,
		GPUUUIDs:       gpuUUIDs,
	}, nil
}

func configureCheckpoint(
	log logr.Logger,
	state *types.CheckpointContainerSnapshot,
	req CheckpointRequest,
	cfg *types.AgentConfig,
	checkpointDir string,
) (*criurpc.CriuOpts, *types.CheckpointManifest, error) {
	criuOpts, err := criu.BuildDumpOptions(state, &cfg.CRIU, checkpointDir, log)
	if err != nil {
		return nil, nil, err
	}

	m := types.NewCheckpointManifest(
		req.CheckpointHash,
		types.NewCRIUDumpManifest(criuOpts, cfg.CRIU),
		types.NewSourcePodManifest(req.ContainerID, state.PID, req.NodeName, req.PodName, req.PodNamespace, state.StdioFDs),
		types.NewOverlayManifest(cfg.Overlay, state.UpperDir, state.OCISpec),
	)
	if len(state.CUDANSPIDs) > 0 {
		m.CUDA = types.NewCUDAManifest(state.CUDANSPIDs, state.GPUUUIDs)
	}

	if err := types.WriteManifest(checkpointDir, m); err != nil {
		return nil, nil, fmt.Errorf("failed to write checkpoint manifest: %w", err)
	}

	return criuOpts, m, nil
}

func captureCheckpoint(ctx context.Context, criuOpts *criurpc.CriuOpts, criuSettings *types.CRIUSettings, data *types.CheckpointManifest, state *types.CheckpointContainerSnapshot, checkpointDir string, log logr.Logger) (time.Duration, error) {
	// CUDA lock+checkpoint must happen before CRIU dump
	if len(state.CUDAHostPIDs) > 0 {
		if err := cuda.LockAndCheckpointProcessTree(ctx, state.CUDAHostPIDs, log); err != nil {
			return 0, fmt.Errorf("CUDA checkpoint failed: %w", err)
		}
	}

	criuDumpDuration, err := criu.ExecuteDump(criuOpts, checkpointDir, criuSettings, log)
	if err != nil {
		return 0, err
	}

	// Overlay rootfs diff capture is best-effort. Failures are logged but not
	// propagated — a checkpoint without overlay diffs is still valid for restore
	// (the base container image provides the filesystem).
	if state.UpperDir != "" {
		if _, err := common.CaptureRootfsDiff(state.UpperDir, checkpointDir, data.Overlay.Exclusions, data.Overlay.BindMountDests); err != nil {
			log.Error(err, "Failed to capture rootfs diff")
		}
		if _, err := common.CaptureDeletedFiles(state.UpperDir, checkpointDir); err != nil {
			log.Error(err, "Failed to capture deleted files")
		}
	}

	return criuDumpDuration, nil
}
