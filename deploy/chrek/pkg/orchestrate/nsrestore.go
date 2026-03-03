package orchestrate

import (
	"context"
	"fmt"
	"os"
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

	// CUDA restore — CRIU creates a nested PID namespace, so the manifest's namespace
	// PIDs (e.g. 1, 1091, 1313, 1314) only exist inside that nested namespace. From
	// nsrestore's level (the parent namespace), the restored processes have different
	// PIDs. We discover those parent-visible PIDs, then map them to inner namespace PIDs
	// via /proc/<pid>/status NSpid to match the manifest ordering.
	if !m.CUDA.IsEmpty() {
		// Discover all PIDs in the restored process tree as seen from our namespace
		candidates := common.ProcessTreePIDs(int(restoredPID))
		cudaPIDs := cuda.FilterProcesses(ctx, candidates, log)

		// Build inner→outer PID mapping via NSpid field
		nsPIDs, err := common.ResolveNamespacePIDs(cudaPIDs)
		if err != nil {
			return 0, fmt.Errorf("failed to resolve namespace PIDs for restored CUDA processes: %w", err)
		}
		innerToOuter := make(map[int]int, len(cudaPIDs))
		for i, outerPID := range cudaPIDs {
			innerToOuter[nsPIDs[i]] = outerPID
		}

		// Reorder outer PIDs to match the manifest's namespace PID ordering
		orderedCUDAPids := make([]int, 0, len(m.CUDA.NamespacePIDs))
		for _, manifestNsPID := range m.CUDA.NamespacePIDs {
			if outerPID, ok := innerToOuter[manifestNsPID]; ok {
				orderedCUDAPids = append(orderedCUDAPids, outerPID)
			}
		}

		log.Info("CUDA PID mapping resolved",
			"manifest_nspids", m.CUDA.NamespacePIDs,
			"discovered_outer_pids", cudaPIDs,
			"inner_to_outer_map", innerToOuter,
			"ordered_cuda_pids", orderedCUDAPids)

		// Debug pause: set DYN_CHREK_CUDA_DEBUG_SIGNAL_FILE to a file path to pause
		// CUDA restore until that file appears on disk (e.g. `touch /tmp/resume`).
		if signalFile := os.Getenv("DYN_CHREK_CUDA_DEBUG_SIGNAL_FILE"); signalFile != "" {
			log.Info("Debug pause: waiting for signal file before CUDA restore",
				"signal_file", signalFile,
				"restored_pid", restoredPID, "ordered_cuda_pids", orderedCUDAPids)
			for {
				select {
				case <-ctx.Done():
					return 0, ctx.Err()
				default:
				}
				if _, err := os.Stat(signalFile); err == nil {
					os.Remove(signalFile)
					log.Info("Debug pause: signal file detected, proceeding with CUDA restore")
					break
				}
				time.Sleep(1 * time.Second)
			}
		}

		if len(orderedCUDAPids) != len(m.CUDA.NamespacePIDs) {
			return 0, fmt.Errorf("CUDA PID mismatch: manifest has %d namespace PIDs but only %d matched after NSpid mapping (discovered %d CUDA PIDs total)",
				len(m.CUDA.NamespacePIDs), len(orderedCUDAPids), len(cudaPIDs))
		}
		if err := cuda.RestoreAndUnlockProcessTree(ctx, orderedCUDAPids, opts.CUDADeviceMap, log); err != nil {
			return 0, fmt.Errorf("CUDA restore failed: %w", err)
		}
	}

	return int(restoredPID), nil
}
