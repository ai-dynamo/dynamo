// Package executor provides the top-level checkpoint and restore executors.
// These wire together the lib packages (criu, cuda, etc.) into multi-step workflows.
package executor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/containerd/containerd"
	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// CheckpointRequest holds per-checkpoint identifiers for a checkpoint operation.
type CheckpointRequest struct {
	ContainerID        string
	ContainerName      string
	CheckpointID       string
	CheckpointLocation string
	StartedAt          time.Time
	NodeName           string
	PodName            string
	PodNamespace       string
	Clientset          kubernetes.Interface
}

type checkpointPhaseTimings struct {
	PrepareDuration        time.Duration
	CUDADuration           time.Duration
	CRIUDumpDuration       time.Duration
	OverlayCaptureDuration time.Duration
	FinalizeDuration       time.Duration
}

// Checkpoint performs a CRIU dump of a container.
// The operation has three phases: inspect, configure, capture.
//
// The checkpoint directory is staged under tmp/<uuid> during the operation.
// On success, the previous checkpoint is removed and the staged directory is
// renamed into place at the base path root.
func Checkpoint(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req CheckpointRequest, cfg *types.AgentConfig) error {
	checkpointStart := time.Now()
	phaseTimings := checkpointPhaseTimings{}
	prepareStart := time.Now()
	log.Info("=== Starting checkpoint operation ===")

	if strings.TrimSpace(req.CheckpointID) == "" {
		return fmt.Errorf("checkpoint ID is required")
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
	phaseTimings.PrepareDuration = time.Since(prepareStart)

	// Phase 3: Capture — CRIU dump, rootfs diff
	captureTimings, err := captureCheckpoint(ctx, criuOpts, &cfg.CRIU, data, state, tmpDir, log)
	if err != nil {
		return err
	}
	phaseTimings.CUDADuration = captureTimings.CUDADuration
	phaseTimings.CRIUDumpDuration = captureTimings.CRIUDumpDuration
	phaseTimings.OverlayCaptureDuration = captureTimings.OverlayCaptureDuration

	// Remove any previous checkpoint with the same identity hash, then
	// promote the staged checkpoint directory into place.
	finalizeStart := time.Now()
	if err := os.RemoveAll(finalDir); err != nil {
		return fmt.Errorf("failed to remove previous checkpoint directory: %w", err)
	}
	if err := os.Rename(tmpDir, finalDir); err != nil {
		return fmt.Errorf("failed to finalize checkpoint directory: %w", err)
	}
	phaseTimings.FinalizeDuration = time.Since(finalizeStart)

	totalDuration := time.Since(checkpointStart)
	log.Info("Checkpoint timing summary",
		"checkpoint", map[string]any{
			"duration": totalDuration.String(),
			"phases": map[string]string{
				"prepare_duration":         phaseTimings.PrepareDuration.String(),
				"cuda_duration":            phaseTimings.CUDADuration.String(),
				"criu_dump_duration":       phaseTimings.CRIUDumpDuration.String(),
				"overlay_capture_duration": phaseTimings.OverlayCaptureDuration.String(),
				"finalize_duration":        phaseTimings.FinalizeDuration.String(),
			},
		},
	)
	if !req.StartedAt.IsZero() {
		log.Info("Checkpoint wall time from agent detection",
			"started_to_checkpoint_complete", time.Since(req.StartedAt),
		)
	}

	return nil
}

func inspectContainer(ctx context.Context, ctrd *containerd.Client, log logr.Logger, req CheckpointRequest) (*types.CheckpointContainerSnapshot, error) {
	containerID := req.ContainerID
	pid, ociSpec, err := snapshotruntime.ResolveContainer(ctx, ctrd, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container: %w", err)
	}

	var hostCgroupPath string
	if cgPath, err := snapshotruntime.ResolveCgroupRootFromHostPID(pid); err == nil && cgPath != "" {
		hostCgroupPath = filepath.Join(snapshotruntime.HostCgroupPath, cgPath)
	}

	rootFS, err := snapshotruntime.GetRootFS(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get rootfs: %w", err)
	}

	upperDir, err := snapshotruntime.GetOverlayUpperDir(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get overlay upperdir: %w", err)
	}

	mountInfo, err := snapshotruntime.ReadMountInfo(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo: %w", err)
	}
	mounts := snapshotruntime.ClassifyMounts(mountInfo, ociSpec, rootFS)

	netNSInode, err := snapshotruntime.GetNetNSInode(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get net namespace inode: %w", err)
	}

	// Read stdio FD targets (like runc's getPipeFds / descriptors.json).
	stdioFDs := make([]string, 3)
	for i := range 3 {
		target, err := os.Readlink(fmt.Sprintf("%s/%d/fd/%d", snapshotruntime.HostProcPath, pid, i))
		if err != nil {
			log.V(1).Info("Failed to readlink stdio FD", "fd", i, "error", err)
			continue
		}
		stdioFDs[i] = target
	}

	// Discover CUDA processes and GPU UUIDs
	allPIDs := snapshotruntime.ProcessTreePIDs(pid)
	cudaHostPIDs := cuda.FilterProcesses(ctx, allPIDs, log)
	cudaNamespacePIDs := make([]int, 0, len(cudaHostPIDs))
	for _, cudaHostPID := range cudaHostPIDs {
		process, err := snapshotruntime.ReadProcessDetails(snapshotruntime.HostProcPath, cudaHostPID)
		if err != nil {
			return nil, fmt.Errorf("failed to read process details for CUDA process %d: %w", cudaHostPID, err)
		}
		if len(process.NamespacePIDs) != 2 {
			return nil, fmt.Errorf("CUDA process %d has namespace depth %d, want 2", cudaHostPID, len(process.NamespacePIDs))
		}
		cudaNamespacePIDs = append(cudaNamespacePIDs, process.InnermostPID)
	}
	if len(cudaHostPIDs) > 0 {
		log.V(1).Info("Resolved checkpoint CUDA PID mapping", "host_pids", cudaHostPIDs, "namespace_pids", cudaNamespacePIDs)
	}
	var gpuUUIDs []string
	if len(cudaHostPIDs) > 0 {
		gpuUUIDs, err = cuda.GetPodGPUUUIDs(ctx, req.PodName, req.PodNamespace, req.ContainerName)
		if err != nil {
			return nil, fmt.Errorf("failed to discover source GPU UUIDs: %w", err)
		}
		if len(gpuUUIDs) == 0 {
			log.Info("PodResources API returned no GPU UUIDs, trying DRA API lookup", "pod", req.PodNamespace+"/"+req.PodName)
			gpuUUIDs, err = cuda.GetGPUUUIDsViaDRAAPI(ctx, req.Clientset, req.PodName, req.PodNamespace, log)
			if err != nil {
				return nil, fmt.Errorf("DRA API GPU UUID lookup failed: %w", err)
			}
		}
		if len(gpuUUIDs) == 0 {
			log.Info("DRA API returned no GPU UUIDs, falling back to nvidia-smi", "pid", pid)
			gpuUUIDs, err = cuda.GetGPUUUIDsViaNvidiaSmi(ctx, snapshotruntime.HostProcPath, pid)
			if err != nil {
				return nil, fmt.Errorf("nvidia-smi GPU UUID fallback failed: %w", err)
			}
			log.Info("nvidia-smi fallback discovered GPU UUIDs", "uuids", gpuUUIDs)
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
		req.CheckpointID,
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

func captureCheckpoint(ctx context.Context, criuOpts *criurpc.CriuOpts, criuSettings *types.CRIUSettings, data *types.CheckpointManifest, state *types.CheckpointContainerSnapshot, checkpointDir string, log logr.Logger) (*checkpointPhaseTimings, error) {
	timings := &checkpointPhaseTimings{}

	// CUDA lock+checkpoint must happen before CRIU dump
	if len(state.CUDAHostPIDs) > 0 {
		cudaTimings, err := cuda.LockAndCheckpointProcessTree(ctx, state.CUDAHostPIDs, log)
		if err != nil {
			return nil, fmt.Errorf("CUDA checkpoint failed: %w", err)
		}
		timings.CUDADuration = cudaTimings.TotalDuration
	}

	criuDumpDuration, err := criu.ExecuteDump(criuOpts, checkpointDir, criuSettings, log)
	if err != nil {
		return nil, err
	}
	timings.CRIUDumpDuration = criuDumpDuration

	// Overlay rootfs diff capture is best-effort. Failures are logged but not
	// propagated — a checkpoint without overlay diffs is still valid for restore
	// (the base container image provides the filesystem).
	if state.UpperDir != "" {
		overlayCaptureStart := time.Now()
		if _, err := snapshotruntime.CaptureRootfsDiff(state.UpperDir, checkpointDir, data.Overlay.Exclusions, data.Overlay.BindMountDests); err != nil {
			log.Error(err, "Failed to capture rootfs diff")
		}
		if _, err := snapshotruntime.CaptureDeletedFiles(state.UpperDir, checkpointDir); err != nil {
			log.Error(err, "Failed to capture deleted files")
		}
		timings.OverlayCaptureDuration = time.Since(overlayCaptureStart)
	}

	return timings, nil
}
