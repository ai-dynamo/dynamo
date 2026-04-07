package executor

import (
	"context"
	"fmt"
	"syscall"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// RestoreOptions holds configuration for an in-namespace restore.
type RestoreOptions struct {
	CheckpointPath string
	CUDADeviceMap  string
	CgroupRoot     string
}

type RestoreInNamespaceResult struct {
	RestoredPID            int           `json:"restoredPID"`
	NSRestoreSetupDuration time.Duration `json:"nsrestoreSetupDuration"`
	CRIURestoreDuration    time.Duration `json:"criuRestoreDuration"`
	CUDADuration           time.Duration `json:"cudaDuration"`
}

// RestoreInNamespace performs a full restore from inside the target container's namespaces.
func RestoreInNamespace(ctx context.Context, opts RestoreOptions, log logr.Logger) (*RestoreInNamespaceResult, error) {
	restoreStart := time.Now()
	log.Info("Starting nsrestore workflow",
		"checkpoint_path", opts.CheckpointPath,
		"has_cuda_map", opts.CUDADeviceMap != "",
		"cgroup_root", opts.CgroupRoot,
	)

	manifestReadStart := time.Now()
	m, err := types.ReadManifest(opts.CheckpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read manifest: %w", err)
	}
	manifestReadDuration := time.Since(manifestReadStart)
	log.V(1).Info("Loaded checkpoint manifest",
		"ext_mounts", len(m.CRIUDump.ExtMnt),
		"criu_log_level", m.CRIUDump.CRIU.LogLevel,
		"manage_cgroups_mode", m.CRIUDump.CRIU.ManageCgroupsMode,
		"checkpoint_has_cuda", !m.CUDA.IsEmpty(),
	)

	// Phase 1: Configure — build CRIU opts from manifest
	configureStart := time.Now()
	criuOpts, err := criu.BuildRestoreOpts(m, opts.CheckpointPath, opts.CgroupRoot, log)
	if err != nil {
		return nil, err
	}
	configureDuration := time.Since(configureStart)

	// Phase 2: Execute — rootfs, CRIU restore, CUDA restore
	executeTimings, restoredPID, err := executeRestore(ctx, criuOpts, m, opts, log)
	if err != nil {
		return nil, err
	}

	result := &RestoreInNamespaceResult{
		RestoredPID:            restoredPID,
		NSRestoreSetupDuration: manifestReadDuration + configureDuration + executeTimings.nsrestoreSetupDuration,
		CRIURestoreDuration:    executeTimings.criuRestoreDuration,
		CUDADuration:           executeTimings.cudaDuration,
	}
	log.V(1).Info("nsrestore timing summary",
		"restored_pid", restoredPID,
		"nsrestore_setup_duration", result.NSRestoreSetupDuration,
		"criu_restore_duration", result.CRIURestoreDuration,
		"cuda_duration", result.CUDADuration,
		"total_duration", time.Since(restoreStart),
	)
	return result, nil
}

type nsrestorePhaseTimings struct {
	nsrestoreSetupDuration time.Duration
	criuRestoreDuration    time.Duration
	cudaDuration           time.Duration
}

func executeRestore(ctx context.Context, criuOpts *criurpc.CriuOpts, m *types.CheckpointManifest, opts RestoreOptions, log logr.Logger) (*nsrestorePhaseTimings, int, error) {
	timings := &nsrestorePhaseTimings{}

	// Apply rootfs diff inside the namespace (target root is /)
	nsrestoreSetupStart := time.Now()
	if err := snapshotruntime.ApplyRootfsDiff(opts.CheckpointPath, "/", log); err != nil {
		return nil, 0, fmt.Errorf("rootfs diff failed: %w", err)
	}

	if err := snapshotruntime.ApplyDeletedFiles(opts.CheckpointPath, "/", log); err != nil {
		log.Error(err, "Failed to apply deleted files")
	}

	// Unmount placeholder's /dev/shm so CRIU can recreate tmpfs with checkpointed content
	if err := syscall.Unmount("/dev/shm", 0); err != nil {
		return nil, 0, fmt.Errorf("failed to unmount /dev/shm before restore: %w", err)
	}

	if err := snapshotruntime.RemountProcSys(true); err != nil {
		return nil, 0, fmt.Errorf("failed to remount /proc/sys read-write for restore: %w", err)
	}
	timings.nsrestoreSetupDuration = time.Since(nsrestoreSetupStart)
	defer func() {
		if err := snapshotruntime.RemountProcSys(false); err != nil {
			log.Error(err, "Failed to remount /proc/sys read-only after restore")
		}
	}()

	// CRIU restore
	criuRestoreStart := time.Now()
	restoredPID, err := criu.ExecuteRestore(criuOpts, m, opts.CheckpointPath, log)
	if err != nil {
		return nil, 0, err
	}
	timings.criuRestoreDuration = time.Since(criuRestoreStart)

	cudaStart := time.Now()
	processes, err := snapshotruntime.ReadProcessTable("/proc")
	if err != nil {
		return nil, 0, fmt.Errorf("failed to read restored process table: %w", err)
	}
	log.V(1).Info("Restored process table snapshot",
		"proc_root", "/proc",
		"criu_callback_pid", restoredPID,
		"process_count", len(processes),
		"manifest_cuda_pids", m.CUDA.PIDs,
	)
	for _, process := range processes {
		log.V(1).Info("Restored process entry",
			"observed_pid", process.ObservedPID,
			"parent_pid", process.ParentPID,
			"outermost_pid", process.OutermostPID,
			"innermost_pid", process.InnermostPID,
			"namespace_pids", process.NamespacePIDs,
			"cmdline", process.Cmdline,
		)
	}

	// CUDA restore — remap checkpoint-time innermost namespace PIDs onto the
	// current visible restored PIDs before invoking cuda-checkpoint.
	if !m.CUDA.IsEmpty() {
		restorePIDs, err := snapshotruntime.ResolveManifestPIDsToObservedPIDs(processes, int(restoredPID), m.CUDA.PIDs)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to resolve restored CUDA PIDs: %w", err)
		}
		log.V(1).Info("Resolved manifest CUDA PIDs to current restore PIDs",
			"manifest_cuda_pids", m.CUDA.PIDs,
			"restored_cuda_pids", restorePIDs,
			"criu_callback_pid", restoredPID,
		)
		_, err = cuda.RestoreAndUnlockProcessTree(ctx, restorePIDs, opts.CUDADeviceMap, log)
		if err != nil {
			return nil, 0, fmt.Errorf("CUDA restore failed: %w", err)
		}
	}
	timings.cudaDuration = time.Since(cudaStart)

	return timings, int(restoredPID), nil
}
