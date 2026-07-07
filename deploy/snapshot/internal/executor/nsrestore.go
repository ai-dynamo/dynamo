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
	TargetPodIP    string
	ProcRoot       string
}

type RestoreInNamespaceResult struct {
	RestoredPID            int                   `json:"restoredPID"`
	NSRestoreSetupDuration time.Duration         `json:"nsrestoreSetupDuration"`
	NSRestoreSetupTimings  NSRestoreSetupTimings `json:"nsrestoreSetupTimings"`
	CRIURestoreDuration    time.Duration         `json:"criuRestoreDuration"`
	CUDADuration           time.Duration         `json:"cudaDuration"`
}

// NSRestoreSetupTimings breaks down NSRestoreSetupDuration without changing it.
type NSRestoreSetupTimings struct {
	ManifestReadDuration          time.Duration `json:"manifestReadDuration"`
	InetRemapDuration             time.Duration `json:"inetRemapDuration"`
	BuildRestoreOptsDuration      time.Duration `json:"buildRestoreOptsDuration"`
	RootfsDiffStatDuration        time.Duration `json:"rootfsDiffStatDuration"`
	RootfsDiffSizeBytes           int64         `json:"rootfsDiffSizeBytes"`
	RootfsDiffExtractDuration     time.Duration `json:"rootfsDiffExtractDuration"`
	DeletedFilesReadDuration      time.Duration `json:"deletedFilesReadDuration"`
	DeletedFilesParseDuration     time.Duration `json:"deletedFilesParseDuration"`
	DeletedFilesRemoveDuration    time.Duration `json:"deletedFilesRemoveDuration"`
	DeletedFilesDuration          time.Duration `json:"deletedFilesDuration"`
	DeletedFilesSizeBytes         int64         `json:"deletedFilesSizeBytes"`
	DeletedFilesEntries           int           `json:"deletedFilesEntries"`
	DeletedFilesRemoved           int           `json:"deletedFilesRemoved"`
	DeletedFilesSkipped           int           `json:"deletedFilesSkipped"`
	DevShmUnmountDuration         time.Duration `json:"devShmUnmountDuration"`
	ProcSysRemountReadWrite       time.Duration `json:"procSysRemountReadWriteDuration"`
	SetupSubphasesDuration        time.Duration `json:"setupSubphasesDuration"`
	SetupUnaccountedDuration      time.Duration `json:"setupUnaccountedDuration"`
	ProcSysRemountReadOnlyOutside time.Duration `json:"procSysRemountReadOnlyOutsideSetupDuration"`
}

func (t *NSRestoreSetupTimings) finalize(total time.Duration) {
	t.SetupSubphasesDuration = t.ManifestReadDuration +
		t.InetRemapDuration +
		t.BuildRestoreOptsDuration +
		t.RootfsDiffStatDuration +
		t.RootfsDiffExtractDuration +
		t.DeletedFilesReadDuration +
		t.DeletedFilesParseDuration +
		t.DeletedFilesRemoveDuration +
		t.DevShmUnmountDuration +
		t.ProcSysRemountReadWrite
	t.DeletedFilesDuration = t.DeletedFilesReadDuration +
		t.DeletedFilesParseDuration +
		t.DeletedFilesRemoveDuration
	t.SetupUnaccountedDuration = total - t.SetupSubphasesDuration
}

// RestoreInNamespace performs a full restore from inside the target container's namespaces.
func RestoreInNamespace(ctx context.Context, opts RestoreOptions, log logr.Logger) (*RestoreInNamespaceResult, error) {
	if opts.ProcRoot == "" {
		opts.ProcRoot = "/proc"
	}
	restoreStart := time.Now()
	log.Info("Starting nsrestore workflow",
		"checkpoint_path", opts.CheckpointPath,
		"has_cuda_map", opts.CUDADeviceMap != "",
		"cgroup_root", opts.CgroupRoot,
		"target_pod_ip_present", opts.TargetPodIP != "",
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

	// Phase 1: Configure — build CRIU opts from manifest.
	configureStart := time.Now()
	inetRemapStart := time.Now()
	if err := criu.ConfigureInetRemap(m, opts.TargetPodIP, log); err != nil {
		return nil, err
	}
	inetRemapDuration := time.Since(inetRemapStart)
	buildRestoreOptsStart := time.Now()
	criuOpts, err := criu.BuildRestoreOpts(m, opts.CheckpointPath, opts.CgroupRoot, log)
	if err != nil {
		return nil, err
	}
	buildRestoreOptsDuration := time.Since(buildRestoreOptsStart)
	configureDuration := time.Since(configureStart)

	// Phase 2: Execute — rootfs, CRIU restore, CUDA restore
	executeTimings, restoredPID, err := executeRestore(ctx, criuOpts, m, opts, log)
	if err != nil {
		return nil, err
	}

	setupDuration := manifestReadDuration + configureDuration + executeTimings.nsrestoreSetupDuration
	setupTimings := executeTimings.nsrestoreSetupTimings
	setupTimings.ManifestReadDuration = manifestReadDuration
	setupTimings.InetRemapDuration = inetRemapDuration
	setupTimings.BuildRestoreOptsDuration = buildRestoreOptsDuration
	setupTimings.finalize(setupDuration)
	result := &RestoreInNamespaceResult{
		RestoredPID:            restoredPID,
		NSRestoreSetupDuration: setupDuration,
		NSRestoreSetupTimings:  setupTimings,
		CRIURestoreDuration:    executeTimings.criuRestoreDuration,
		CUDADuration:           executeTimings.cudaDuration,
	}
	log.Info("nsrestore setup timing breakdown",
		"nsrestore_setup", map[string]any{
			"duration":                      result.NSRestoreSetupDuration.String(),
			"subphases_duration":            setupTimings.SetupSubphasesDuration.String(),
			"unaccounted_duration":          setupTimings.SetupUnaccountedDuration.String(),
			"manifest_read_duration":        setupTimings.ManifestReadDuration.String(),
			"inet_remap_duration":           setupTimings.InetRemapDuration.String(),
			"build_restore_opts_duration":   setupTimings.BuildRestoreOptsDuration.String(),
			"rootfs_diff_stat_duration":     setupTimings.RootfsDiffStatDuration.String(),
			"rootfs_diff_size_bytes":        setupTimings.RootfsDiffSizeBytes,
			"rootfs_diff_extract_duration":  setupTimings.RootfsDiffExtractDuration.String(),
			"deleted_files_read_duration":   setupTimings.DeletedFilesReadDuration.String(),
			"deleted_files_parse_duration":  setupTimings.DeletedFilesParseDuration.String(),
			"deleted_files_remove_duration": setupTimings.DeletedFilesRemoveDuration.String(),
			"deleted_files_duration":        setupTimings.DeletedFilesDuration.String(),
			"deleted_files_size_bytes":      setupTimings.DeletedFilesSizeBytes,
			"deleted_files_entries":         setupTimings.DeletedFilesEntries,
			"deleted_files_removed":         setupTimings.DeletedFilesRemoved,
			"deleted_files_skipped":         setupTimings.DeletedFilesSkipped,
			"dev_shm_unmount_duration":      setupTimings.DevShmUnmountDuration.String(),
			"proc_sys_remount_rw_duration":  setupTimings.ProcSysRemountReadWrite.String(),
			"proc_sys_remount_ro_duration":  setupTimings.ProcSysRemountReadOnlyOutside.String(),
			"proc_sys_remount_ro_scope":     "outside_setup",
		},
	)
	log.Info("nsrestore timing summary",
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
	nsrestoreSetupTimings  NSRestoreSetupTimings
	criuRestoreDuration    time.Duration
	cudaDuration           time.Duration
}

func executeRestore(ctx context.Context, criuOpts *criurpc.CriuOpts, m *types.CheckpointManifest, opts RestoreOptions, log logr.Logger) (*nsrestorePhaseTimings, int, error) {
	timings := &nsrestorePhaseTimings{}

	// Apply rootfs diff inside the namespace (target root is /)
	nsrestoreSetupStart := time.Now()
	rootfsStats, err := snapshotruntime.ApplyRootfsDiffWithStats(opts.CheckpointPath, "/", log)
	timings.nsrestoreSetupTimings.RootfsDiffStatDuration = rootfsStats.StatDuration
	timings.nsrestoreSetupTimings.RootfsDiffSizeBytes = rootfsStats.SizeBytes
	timings.nsrestoreSetupTimings.RootfsDiffExtractDuration = rootfsStats.ExtractDuration
	if err != nil {
		return nil, 0, fmt.Errorf("rootfs diff failed: %w", err)
	}

	deletedStats, err := snapshotruntime.ApplyDeletedFilesWithStats(opts.CheckpointPath, "/", log)
	timings.nsrestoreSetupTimings.DeletedFilesReadDuration = deletedStats.ReadDuration
	timings.nsrestoreSetupTimings.DeletedFilesParseDuration = deletedStats.ParseDuration
	timings.nsrestoreSetupTimings.DeletedFilesRemoveDuration = deletedStats.RemoveDuration
	timings.nsrestoreSetupTimings.DeletedFilesSizeBytes = deletedStats.SizeBytes
	timings.nsrestoreSetupTimings.DeletedFilesEntries = deletedStats.Entries
	timings.nsrestoreSetupTimings.DeletedFilesRemoved = deletedStats.Removed
	timings.nsrestoreSetupTimings.DeletedFilesSkipped = deletedStats.Skipped
	if err != nil {
		log.Error(err, "Failed to apply deleted files")
	}

	// Unmount placeholder's /dev/shm so CRIU can recreate tmpfs with checkpointed content
	devShmUnmountStart := time.Now()
	if err := syscall.Unmount("/dev/shm", 0); err != nil {
		return nil, 0, fmt.Errorf("failed to unmount /dev/shm before restore: %w", err)
	}
	timings.nsrestoreSetupTimings.DevShmUnmountDuration = time.Since(devShmUnmountStart)

	procSysRemountReadWriteStart := time.Now()
	if err := snapshotruntime.RemountProcSys(true); err != nil {
		return nil, 0, fmt.Errorf("failed to remount /proc/sys read-write for restore: %w", err)
	}
	timings.nsrestoreSetupTimings.ProcSysRemountReadWrite = time.Since(procSysRemountReadWriteStart)
	timings.nsrestoreSetupDuration = time.Since(nsrestoreSetupStart)
	defer func() {
		procSysRemountReadOnlyStart := time.Now()
		if err := snapshotruntime.RemountProcSys(false); err != nil {
			log.Error(err, "Failed to remount /proc/sys read-only after restore")
		}
		timings.nsrestoreSetupTimings.ProcSysRemountReadOnlyOutside = time.Since(procSysRemountReadOnlyStart)
		log.Info("Deferred /proc/sys read-only remount timing",
			"duration", timings.nsrestoreSetupTimings.ProcSysRemountReadOnlyOutside,
			"scope", "outside_nsrestore_setup",
		)
	}()

	// CRIU restore
	criuRestoreStart := time.Now()
	restoredPID, err := criu.ExecuteRestore(criuOpts, m, opts.CheckpointPath, log)
	if err != nil {
		return nil, 0, err
	}
	timings.criuRestoreDuration = time.Since(criuRestoreStart)

	cudaStart := time.Now()
	processes, err := snapshotruntime.ReadProcessTable(opts.ProcRoot)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to read restored process table: %w", err)
	}
	log.V(1).Info("Restored process table snapshot",
		"proc_root", opts.ProcRoot,
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
		_, err = cuda.RestoreAndUnlockProcessTree(
			ctx,
			restorePIDs,
			opts.CUDADeviceMap,
			opts.ProcRoot,
			log,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("CUDA restore failed: %w", err)
		}
	}
	timings.cudaDuration = time.Since(cudaStart)

	return timings, int(restoredPID), nil
}
