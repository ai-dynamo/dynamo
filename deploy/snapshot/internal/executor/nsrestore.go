package executor

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/profile"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// RestoreOptions holds configuration for an in-namespace restore.
type RestoreOptions struct {
	CheckpointPath string
	CUDADeviceMap  string
	CgroupRoot     string
	TargetPodIP    string
	RootfsMountFD  int
	WorkspaceFD    int
}

type RestoreInNamespaceResult struct {
	RestoredPID int `json:"restoredPID"`
}

// RestoreInNamespace performs a full restore from inside the target container's namespaces.
func RestoreInNamespace(ctx context.Context, opts RestoreOptions, log logr.Logger) (*RestoreInNamespaceResult, error) {
	opts.CheckpointPath = canonicalizeCheckpointPath(opts.CheckpointPath)
	restoreStart := time.Now()
	operation := profile.NewOperation("nsrestore", "checkpoint_path", opts.CheckpointPath)
	restoreSpan := operation.Start(log, "nsrestore", "nsrestore_total")
	var retErr error
	defer func() {
		restoreSpan.EndStatus(retErr)
	}()
	log.Info("Starting nsrestore workflow",
		"checkpoint_path", opts.CheckpointPath,
		"has_cuda_map", opts.CUDADeviceMap != "",
		"cgroup_root", opts.CgroupRoot,
		"target_pod_ip_present", opts.TargetPodIP != "",
	)

	manifestReadStart := time.Now()
	manifestSpan := operation.Start(log, "nsrestore", "manifest_read")
	m, err := types.ReadManifest(opts.CheckpointPath)
	manifestSpan.EndStatus(err)
	if err != nil {
		retErr = fmt.Errorf("failed to read manifest: %w", err)
		return nil, retErr
	}
	manifestReadDuration := time.Since(manifestReadStart)
	log.V(1).Info("Loaded checkpoint manifest",
		"ext_mounts", len(m.CRIUDump.ExtMnt),
		"criu_log_level", m.CRIUDump.CRIU.LogLevel,
		"manage_cgroups_mode", m.CRIUDump.CRIU.ManageCgroupsMode,
		"checkpoint_has_cuda", !m.CUDA.IsEmpty(),
	)

	// Phase 1: Configure checkpoint metadata that does not depend on the staged root.
	configureStart := time.Now()
	configureSpan := operation.Start(log, "nsrestore", "configure", "checkpoint_id", m.CheckpointID)
	if err := criu.ConfigureInetRemap(m, opts.TargetPodIP, log); err != nil {
		configureSpan.EndStatus(err)
		retErr = err
		return nil, retErr
	}
	configureSpan.EndStatus(nil)
	configureDuration := time.Since(configureStart)

	// Phase 2: Execute — rootfs, CRIU restore, CUDA restore
	executeTimings, restoredPID, err := executeRestore(ctx, m, opts, log, operation)
	if err != nil {
		retErr = err
		return nil, retErr
	}

	result := &RestoreInNamespaceResult{
		RestoredPID: restoredPID,
	}
	log.Info("nsrestore timing summary",
		"restored_pid", restoredPID,
		"manifest_read_duration", manifestReadDuration,
		"configure_duration", configureDuration,
		"root_setup_duration", executeTimings.nsrestoreSetupDuration,
		"criu_restore_duration", executeTimings.criuRestoreDuration,
		"cuda_duration", executeTimings.cudaDuration,
		"total_duration", time.Since(restoreStart),
	)
	return result, nil
}

func canonicalizeCheckpointPath(checkpointPath string) string {
	const procSelfFDPrefix = "/proc/self/fd/"

	fd, err := strconv.Atoi(strings.TrimPrefix(checkpointPath, procSelfFDPrefix))
	if err != nil || fd < 0 || !strings.HasPrefix(checkpointPath, procSelfFDPrefix) {
		return checkpointPath
	}
	return fmt.Sprintf("/proc/%d/fd/%d", os.Getpid(), fd)
}

type nsrestorePhaseTimings struct {
	nsrestoreSetupDuration time.Duration
	criuRestoreDuration    time.Duration
	cudaDuration           time.Duration
}

func executeRestore(ctx context.Context, m *types.CheckpointManifest, opts RestoreOptions, log logr.Logger, operation profile.Operation) (timings *nsrestorePhaseTimings, resultPID int, retErr error) {
	timings = &nsrestorePhaseTimings{}

	nsrestoreSetupStart := time.Now()
	setupSpan := operation.Start(log, "nsrestore", "root_setup", "checkpoint_id", m.CheckpointID)
	deletedFile, err := snapshotruntime.OpenDeletedFiles(opts.CheckpointPath)
	if err != nil {
		setupSpan.EndStatus(err)
		return nil, 0, err
	}
	deletedFD := -1
	if deletedFile != nil {
		deletedFD = int(deletedFile.Fd())
		defer deletedFile.Close()
	}
	composition, err := snapshotruntime.ComposeRoot(
		opts.RootfsMountFD,
		opts.WorkspaceFD,
		deletedFD,
	)
	if err != nil {
		setupSpan.EndStatus(err)
		return nil, 0, fmt.Errorf("compose rootfs: %w", err)
	}
	defer func() {
		retErr = errors.Join(retErr, composition.Close())
	}()
	log.V(1).Info(
		"Root composition timing",
		"root_path", composition.RootPath,
		"mount_attach_duration", composition.MountAttachDuration,
		"overlay_setup_duration", composition.OverlaySetupDuration,
	)

	// Unmount placeholder's /dev/shm so CRIU can recreate tmpfs with checkpointed content.
	if err := syscall.Unmount("/dev/shm", 0); err != nil {
		setupSpan.EndStatus(err)
		return nil, 0, fmt.Errorf("failed to unmount /dev/shm before restore: %w", err)
	}

	if err := snapshotruntime.RemountProcSys(true); err != nil {
		setupSpan.EndStatus(err)
		return nil, 0, fmt.Errorf("failed to remount /proc/sys read-write for restore: %w", err)
	}
	defer func() {
		if err := snapshotruntime.RemountProcSys(false); err != nil {
			log.Error(err, "Failed to remount /proc/sys read-only after restore")
		}
	}()
	criuOpts, err := criu.BuildRestoreOpts(
		m,
		opts.CheckpointPath,
		composition.RootPath,
		opts.CgroupRoot,
		log,
	)
	if err != nil {
		setupSpan.EndStatus(err)
		return nil, 0, err
	}
	timings.nsrestoreSetupDuration = time.Since(nsrestoreSetupStart)
	setupSpan.EndStatus(nil)

	// CRIU restore
	criuRestoreStart := time.Now()
	criuSpan := operation.Start(log, "nsrestore", "criu_restore", "checkpoint_id", m.CheckpointID)
	restoredPID, err := criu.ExecuteRestore(criuOpts, m, opts.CheckpointPath, log)
	criuSpan.EndStatus(err)
	if err != nil {
		return nil, 0, err
	}
	timings.criuRestoreDuration = time.Since(criuRestoreStart)

	cudaStart := time.Now()
	processTableSpan := operation.Start(log, "nsrestore", "process_table_read", "checkpoint_id", m.CheckpointID)
	processes, err := snapshotruntime.ReadProcessTable("/proc")
	processTableSpan.EndStatus(err)
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
		cudaSpan := operation.Start(log, "nsrestore", "cuda_restore", "checkpoint_id", m.CheckpointID)
		restorePIDs, err := snapshotruntime.ResolveManifestPIDsToObservedPIDs(processes, int(restoredPID), m.CUDA.PIDs)
		if err != nil {
			cudaSpan.EndStatus(err)
			return nil, 0, fmt.Errorf("failed to resolve restored CUDA PIDs: %w", err)
		}
		log.V(1).Info("Resolved manifest CUDA PIDs to current restore PIDs",
			"manifest_cuda_pids", m.CUDA.PIDs,
			"restored_cuda_pids", restorePIDs,
			"criu_callback_pid", restoredPID,
		)
		_, err = cuda.RestoreAndUnlockProcessTree(ctx, restorePIDs, opts.CUDADeviceMap, log)
		cudaSpan.EndStatus(err)
		if err != nil {
			return nil, 0, fmt.Errorf("CUDA restore failed: %w", err)
		}
	}
	timings.cudaDuration = time.Since(cudaStart)

	return timings, int(restoredPID), nil
}
