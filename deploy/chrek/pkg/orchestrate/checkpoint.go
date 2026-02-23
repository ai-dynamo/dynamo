// Package orchestrate provides the top-level checkpoint and restore orchestrators.
// These wire together the lib packages (criu, cuda, etc.) into multi-step workflows.
package orchestrate

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/containerd/containerd"
	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

// CheckpointRequest holds per-checkpoint identifiers for a checkpoint operation.
type CheckpointRequest struct {
	ContainerID    string
	ContainerName  string
	CheckpointHash string
	CheckpointDir  string
	NodeName       string
	PodName        string
	PodNamespace   string
}

// Checkpoint performs a CRIU dump of a container.
// The operation has three phases: inspect, configure, capture.
//
// The checkpoint directory is staged under tmp/<hash> during the operation.
// On success, it is atomically renamed to <hash> at the base path root.
func Checkpoint(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req CheckpointRequest, cfg *types.AgentConfig) error {
	checkpointStart := time.Now()
	log.Info("=== Starting checkpoint operation ===")

	finalDir := filepath.Join(req.CheckpointDir, req.CheckpointHash)
	tmpDir := filepath.Join(req.CheckpointDir, "tmp", req.CheckpointHash)
	if err := os.RemoveAll(tmpDir); err != nil {
		return fmt.Errorf("failed to clean checkpoint staging directory: %w", err)
	}
	if err := os.MkdirAll(tmpDir, 0700); err != nil {
		return fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

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
	os.RemoveAll(finalDir)
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
	cudaPIDs := cuda.FilterProcesses(ctx, allPIDs, log)
	var gpuUUIDs []string
	if len(cudaPIDs) > 0 {
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
		CUDAPIDs:       cudaPIDs,
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
	if len(state.CUDAPIDs) > 0 {
		m.CUDA = types.NewCUDAManifest(state.CUDAPIDs, state.GPUUUIDs)
	}

	if err := types.WriteManifest(checkpointDir, m); err != nil {
		return nil, nil, fmt.Errorf("failed to write checkpoint manifest: %w", err)
	}

	return criuOpts, m, nil
}

func captureCheckpoint(ctx context.Context, criuOpts *criurpc.CriuOpts, criuSettings *types.CRIUSettings, data *types.CheckpointManifest, state *types.CheckpointContainerSnapshot, checkpointDir string, log logr.Logger) (time.Duration, error) {
	// CUDA lock+checkpoint must happen before CRIU dump
	if len(state.CUDAPIDs) > 0 {
		if err := cuda.LockAndCheckpointProcessTree(ctx, state.CUDAPIDs, log); err != nil {
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
