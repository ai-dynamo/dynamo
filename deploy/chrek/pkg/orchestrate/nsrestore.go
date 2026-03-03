package orchestrate

import (
	"context"
	"fmt"
	"os"
	"strings"
	"syscall"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criu"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

// RestoreOptions holds configuration for an in-namespace restore.
type RestoreOptions struct {
	CheckpointPath string
	CUDADeviceMap  string
	CgroupRoot     string
}

// RestoreInNamespace performs a full restore from inside the target container's namespaces.
func RestoreInNamespace(ctx context.Context, opts RestoreOptions, log logr.Logger) (int, error) {
	restoreStart := time.Now()
	log.Info("Starting nsrestore workflow",
		"checkpoint_path", opts.CheckpointPath,
		"has_cuda_map", opts.CUDADeviceMap != "",
		"cgroup_root", opts.CgroupRoot,
	)

	m, err := types.ReadManifest(opts.CheckpointPath)
	if err != nil {
		return 0, fmt.Errorf("failed to read manifest: %w", err)
	}
	log.Info("Loaded checkpoint manifest",
		"ext_mounts", len(m.CRIUDump.ExtMnt),
		"criu_log_level", m.CRIUDump.CRIU.LogLevel,
		"manage_cgroups_mode", m.CRIUDump.CRIU.ManageCgroupsMode,
		"checkpoint_has_cuda", !m.CUDA.IsEmpty(),
	)
	// Phase 1: Configure — build CRIU opts from manifest
	criuOpts, err := criu.BuildRestoreOpts(m, opts.CgroupRoot, log)
	if err != nil {
		return 0, err
	}

	// Phase 2: Execute — rootfs, CRIU restore, CUDA restore
	restoredPID, err := executeRestore(ctx, criuOpts, m, opts, log)
	if err != nil {
		return 0, err
	}

	log.Info("nsrestore completed", "restored_pid", restoredPID, "duration", time.Since(restoreStart))
	return restoredPID, nil
}


func executeRestore(ctx context.Context, criuOpts *criurpc.CriuOpts, m *types.CheckpointManifest, opts RestoreOptions, log logr.Logger) (int, error) {
	// Apply rootfs diff inside the namespace (target root is /)
	if err := common.ApplyRootfsDiff(opts.CheckpointPath, "/", log); err != nil {
		return 0, fmt.Errorf("rootfs diff failed: %w", err)
	}
	if err := common.ApplyDeletedFiles(opts.CheckpointPath, "/", log); err != nil {
		log.Error(err, "Failed to apply deleted files")
	}

	// Unmount placeholder's /dev/shm so CRIU can recreate tmpfs with checkpointed content
	if err := syscall.Unmount("/dev/shm", 0); err != nil {
		return 0, fmt.Errorf("failed to unmount /dev/shm before restore: %w", err)
	}

	if err := common.RemountProcSys(true); err != nil {
		return 0, fmt.Errorf("failed to remount /proc/sys read-write for restore: %w", err)
	}
	defer func() {
		if err := common.RemountProcSys(false); err != nil {
			log.Error(err, "Failed to remount /proc/sys read-only after restore")
		}
	}()

	// CRIU restore
	restoredPID, err := criu.ExecuteRestore(criuOpts, m, opts.CheckpointPath, log)
	if err != nil {
		return 0, err
	}

	// CUDA restore — use manifest namespace PIDs directly.
	// With rstSibling=false, CRIU creates a nested PID namespace that preserves the
	// original namespace-relative PIDs. FilterProcesses validates they hold CUDA contexts.
	if !m.CUDA.IsEmpty() {
		cudaPIDs := cuda.FilterProcesses(ctx, m.CUDA.NamespacePIDs, log)
		log.Info("CUDA restore using manifest namespace PIDs",
			"manifest_nspids", m.CUDA.NamespacePIDs, "validated_count", len(cudaPIDs))

		// Debug pause: wait for /tmp/chrek-debug-continue before running cuda-checkpoint restore.
		// Set CHREK_DEBUG_PAUSE_BEFORE_CUDA_RESTORE=1 on the agent to enable.
		if strings.TrimSpace(os.Getenv("CHREK_DEBUG_PAUSE_BEFORE_CUDA_RESTORE")) != "" {
			log.Info("DEBUG PAUSE: CRIU restore done, waiting before cuda-checkpoint restore",
				"restored_pid", restoredPID, "cuda_pids", cudaPIDs,
				"nsrestore_pid", os.Getpid(),
				"signal_file", "/tmp/chrek-debug-continue")
			for {
				if _, err := os.Stat("/tmp/chrek-debug-continue"); err == nil {
					os.Remove("/tmp/chrek-debug-continue")
					log.Info("DEBUG PAUSE: continue signal received, proceeding with cuda-checkpoint restore")
					break
				}
				time.Sleep(1 * time.Second)
			}
		}

		if len(cudaPIDs) != len(m.CUDA.NamespacePIDs) {
			return 0, fmt.Errorf("CUDA PID mismatch: manifest has %d namespace PIDs, but only %d hold CUDA contexts after restore",
				len(m.CUDA.NamespacePIDs), len(cudaPIDs))
		}
		if err := cuda.RestoreAndUnlockProcessTree(ctx, cudaPIDs, opts.CUDADeviceMap, log); err != nil {
			return 0, fmt.Errorf("CUDA restore failed: %w", err)
		}
	}

	return int(restoredPID), nil
}
