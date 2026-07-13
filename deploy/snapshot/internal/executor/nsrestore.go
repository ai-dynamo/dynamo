package executor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
	CheckpointPath       string
	CUDADeviceMap        string
	CgroupRoot           string
	TargetPodIP          string
	RootfsGMSReadyFile   string
	RootfsGMSReleaseFile string
	RootfsGMSReleaseMode string
	RootfsGMSWaitTimeout time.Duration
}

type RestoreInNamespaceResult struct {
	RestoredPID            int                   `json:"restoredPID"`
	PreRootfsWaitDuration  time.Duration         `json:"preRootfsWaitDuration"`
	NSRestoreSetupDuration time.Duration         `json:"nsrestoreSetupDuration"`
	NSRestoreSetupTimings  NSRestoreSetupTimings `json:"nsrestoreSetupTimings"`
	CRIURestoreDuration    time.Duration         `json:"criuRestoreDuration"`
	CUDADuration           time.Duration         `json:"cudaDuration"`
}

// NSRestoreSetupTimings breaks down NSRestoreSetupDuration without changing it.
type NSRestoreSetupTimings struct {
	ManifestReadDuration          time.Duration                          `json:"manifestReadDuration"`
	InetRemapDuration             time.Duration                          `json:"inetRemapDuration"`
	BuildRestoreOptsDuration      time.Duration                          `json:"buildRestoreOptsDuration"`
	RootfsDiffStatDuration        time.Duration                          `json:"rootfsDiffStatDuration"`
	RootfsDiffSizeBytes           int64                                  `json:"rootfsDiffSizeBytes"`
	RootfsDiffExtractDuration     time.Duration                          `json:"rootfsDiffExtractDuration"`
	RootfsDiffChildRusage         snapshotruntime.ChildRusageStats       `json:"rootfsDiffChildRusage"`
	RootfsDiffCgroupBefore        snapshotruntime.CgroupResourceSnapshot `json:"rootfsDiffCgroupBefore"`
	RootfsDiffCgroupAfter         snapshotruntime.CgroupResourceSnapshot `json:"rootfsDiffCgroupAfter"`
	RootfsDiffCgroupDelta         snapshotruntime.CgroupResourceDelta    `json:"rootfsDiffCgroupDelta"`
	RootfsDiffCgroupReadErrors    []string                               `json:"rootfsDiffCgroupReadErrors,omitempty"`
	RootfsReleaseMarkerDuration   time.Duration                          `json:"rootfsReleaseMarkerDuration"`
	DeletedFilesReadDuration      time.Duration                          `json:"deletedFilesReadDuration"`
	DeletedFilesParseDuration     time.Duration                          `json:"deletedFilesParseDuration"`
	DeletedFilesRemoveDuration    time.Duration                          `json:"deletedFilesRemoveDuration"`
	DeletedFilesDuration          time.Duration                          `json:"deletedFilesDuration"`
	DeletedFilesSizeBytes         int64                                  `json:"deletedFilesSizeBytes"`
	DeletedFilesEntries           int                                    `json:"deletedFilesEntries"`
	DeletedFilesRemoved           int                                    `json:"deletedFilesRemoved"`
	DeletedFilesSkipped           int                                    `json:"deletedFilesSkipped"`
	DevShmUnmountDuration         time.Duration                          `json:"devShmUnmountDuration"`
	ProcSysRemountReadWrite       time.Duration                          `json:"procSysRemountReadWriteDuration"`
	SetupSubphasesDuration        time.Duration                          `json:"setupSubphasesDuration"`
	SetupUnaccountedDuration      time.Duration                          `json:"setupUnaccountedDuration"`
	ProcSysRemountReadOnlyOutside time.Duration                          `json:"procSysRemountReadOnlyOutsideSetupDuration"`
}

func (t *NSRestoreSetupTimings) finalize(total time.Duration) {
	t.SetupSubphasesDuration = t.ManifestReadDuration +
		t.InetRemapDuration +
		t.BuildRestoreOptsDuration +
		t.RootfsDiffStatDuration +
		t.RootfsDiffExtractDuration +
		t.RootfsReleaseMarkerDuration +
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
		PreRootfsWaitDuration:  executeTimings.preRootfsWaitDuration,
		NSRestoreSetupDuration: setupDuration,
		NSRestoreSetupTimings:  setupTimings,
		CRIURestoreDuration:    executeTimings.criuRestoreDuration,
		CUDADuration:           executeTimings.cudaDuration,
	}
	log.Info("nsrestore setup timing breakdown",
		"nsrestore_setup", map[string]any{
			"duration":                       result.NSRestoreSetupDuration.String(),
			"subphases_duration":             setupTimings.SetupSubphasesDuration.String(),
			"unaccounted_duration":           setupTimings.SetupUnaccountedDuration.String(),
			"manifest_read_duration":         setupTimings.ManifestReadDuration.String(),
			"inet_remap_duration":            setupTimings.InetRemapDuration.String(),
			"build_restore_opts_duration":    setupTimings.BuildRestoreOptsDuration.String(),
			"rootfs_diff_stat_duration":      setupTimings.RootfsDiffStatDuration.String(),
			"rootfs_diff_size_bytes":         setupTimings.RootfsDiffSizeBytes,
			"rootfs_diff_extract_duration":   setupTimings.RootfsDiffExtractDuration.String(),
			"rootfs_diff_child_rusage":       setupTimings.RootfsDiffChildRusage,
			"rootfs_diff_cgroup_before":      setupTimings.RootfsDiffCgroupBefore,
			"rootfs_diff_cgroup_after":       setupTimings.RootfsDiffCgroupAfter,
			"rootfs_diff_cgroup_delta":       setupTimings.RootfsDiffCgroupDelta,
			"rootfs_diff_cgroup_read_errors": setupTimings.RootfsDiffCgroupReadErrors,
			"rootfs_release_marker_duration": setupTimings.RootfsReleaseMarkerDuration.String(),
			"deleted_files_read_duration":    setupTimings.DeletedFilesReadDuration.String(),
			"deleted_files_parse_duration":   setupTimings.DeletedFilesParseDuration.String(),
			"deleted_files_remove_duration":  setupTimings.DeletedFilesRemoveDuration.String(),
			"deleted_files_duration":         setupTimings.DeletedFilesDuration.String(),
			"deleted_files_size_bytes":       setupTimings.DeletedFilesSizeBytes,
			"deleted_files_entries":          setupTimings.DeletedFilesEntries,
			"deleted_files_removed":          setupTimings.DeletedFilesRemoved,
			"deleted_files_skipped":          setupTimings.DeletedFilesSkipped,
			"dev_shm_unmount_duration":       setupTimings.DevShmUnmountDuration.String(),
			"proc_sys_remount_rw_duration":   setupTimings.ProcSysRemountReadWrite.String(),
			"proc_sys_remount_ro_duration":   setupTimings.ProcSysRemountReadOnlyOutside.String(),
			"proc_sys_remount_ro_scope":      "outside_setup",
		},
	)
	log.Info("nsrestore timing summary",
		"restored_pid", restoredPID,
		"pre_rootfs_wait_duration", result.PreRootfsWaitDuration,
		"nsrestore_setup_duration", result.NSRestoreSetupDuration,
		"criu_restore_duration", result.CRIURestoreDuration,
		"cuda_duration", result.CUDADuration,
		"total_duration", time.Since(restoreStart),
	)
	return result, nil
}

type nsrestorePhaseTimings struct {
	preRootfsWaitDuration  time.Duration
	nsrestoreSetupDuration time.Duration
	nsrestoreSetupTimings  NSRestoreSetupTimings
	criuRestoreDuration    time.Duration
	cudaDuration           time.Duration
}

func executeRestore(ctx context.Context, criuOpts *criurpc.CriuOpts, m *types.CheckpointManifest, opts RestoreOptions, log logr.Logger) (*nsrestorePhaseTimings, int, error) {
	timings := &nsrestorePhaseTimings{}

	gateEnabled, err := validateRootfsGMSGate(opts)
	if err != nil {
		return nil, 0, err
	}
	if gateEnabled {
		waitStart := time.Now()
		if err := waitForMarker(ctx, opts.RootfsGMSReadyFile, opts.RootfsGMSWaitTimeout); err != nil {
			return nil, 0, fmt.Errorf("wait for GMS transfer-ready marker: %w", err)
		}
		timings.preRootfsWaitDuration = time.Since(waitStart)
		log.Info("GMS transfer-ready marker observed",
			"path", opts.RootfsGMSReadyFile,
			"duration", timings.preRootfsWaitDuration,
		)
	}

	// Apply rootfs diff inside the namespace (target root is /)
	nsrestoreSetupStart := time.Now()
	var beforeRootfsExtract func() error
	if gateEnabled && opts.RootfsGMSReleaseMode == "control" {
		beforeRootfsExtract = func() error {
			markerStart := time.Now()
			if err := writeMarkerAtomically(opts.RootfsGMSReleaseFile); err != nil {
				return fmt.Errorf("write control GMS release marker: %w", err)
			}
			timings.nsrestoreSetupTimings.RootfsReleaseMarkerDuration = time.Since(markerStart)
			log.Info("Control GMS release marker created immediately before rootfs tar",
				"path", opts.RootfsGMSReleaseFile,
				"duration", timings.nsrestoreSetupTimings.RootfsReleaseMarkerDuration,
			)
			return nil
		}
	}
	rootfsStats, err := snapshotruntime.ApplyRootfsDiffWithStatsBeforeExtract(opts.CheckpointPath, "/", log, beforeRootfsExtract)
	timings.nsrestoreSetupTimings.RootfsDiffStatDuration = rootfsStats.StatDuration
	timings.nsrestoreSetupTimings.RootfsDiffSizeBytes = rootfsStats.SizeBytes
	timings.nsrestoreSetupTimings.RootfsDiffExtractDuration = rootfsStats.ExtractDuration
	timings.nsrestoreSetupTimings.RootfsDiffChildRusage = rootfsStats.ChildRusage
	timings.nsrestoreSetupTimings.RootfsDiffCgroupBefore = rootfsStats.CgroupBefore
	timings.nsrestoreSetupTimings.RootfsDiffCgroupAfter = rootfsStats.CgroupAfter
	timings.nsrestoreSetupTimings.RootfsDiffCgroupDelta = rootfsStats.CgroupDelta
	timings.nsrestoreSetupTimings.RootfsDiffCgroupReadErrors = rootfsStats.CgroupReadErrors
	if err != nil {
		return nil, 0, fmt.Errorf("rootfs diff failed: %w", err)
	}
	if gateEnabled && opts.RootfsGMSReleaseMode == "treatment" {
		markerStart := time.Now()
		if err := writeMarkerAtomically(opts.RootfsGMSReleaseFile); err != nil {
			return nil, 0, fmt.Errorf("write treatment GMS release marker: %w", err)
		}
		timings.nsrestoreSetupTimings.RootfsReleaseMarkerDuration = time.Since(markerStart)
		log.Info("Treatment GMS release marker created immediately after rootfs",
			"path", opts.RootfsGMSReleaseFile,
			"duration", timings.nsrestoreSetupTimings.RootfsReleaseMarkerDuration,
		)
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
		_, err = cuda.RestoreAndUnlockProcessTree(
			ctx,
			restorePIDs,
			opts.CUDADeviceMap,
			m.CUDA.CanRestoreConcurrently(),
			log,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("CUDA restore failed: %w", err)
		}
	}
	timings.cudaDuration = time.Since(cudaStart)

	return timings, int(restoredPID), nil
}

func validateRootfsGMSGate(opts RestoreOptions) (bool, error) {
	values := []string{
		opts.RootfsGMSReadyFile,
		opts.RootfsGMSReleaseFile,
		opts.RootfsGMSReleaseMode,
	}
	configured := 0
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			configured++
		}
	}
	if configured == 0 {
		return false, nil
	}
	if configured != len(values) {
		return false, fmt.Errorf("rootfs/GMS gate requires ready file, release file, and release mode")
	}
	if opts.RootfsGMSReleaseMode != "control" && opts.RootfsGMSReleaseMode != "treatment" {
		return false, fmt.Errorf("invalid rootfs/GMS release mode %q", opts.RootfsGMSReleaseMode)
	}
	if opts.RootfsGMSWaitTimeout <= 0 {
		return false, fmt.Errorf("rootfs/GMS wait timeout must be positive")
	}
	ready := filepath.Clean(opts.RootfsGMSReadyFile)
	release := filepath.Clean(opts.RootfsGMSReleaseFile)
	if !filepath.IsAbs(ready) || !filepath.IsAbs(release) {
		return false, fmt.Errorf("rootfs/GMS marker paths must be absolute")
	}
	if ready == release {
		return false, fmt.Errorf("rootfs/GMS ready and release marker paths must differ")
	}
	checkpoint := filepath.Clean(opts.CheckpointPath)
	for _, marker := range []string{ready, release} {
		if marker == checkpoint || strings.HasPrefix(marker, checkpoint+string(os.PathSeparator)) {
			return false, fmt.Errorf("rootfs/GMS marker path %q is inside immutable checkpoint %q", marker, checkpoint)
		}
	}
	return true, nil
}

func waitForMarker(ctx context.Context, path string, timeout time.Duration) error {
	waitCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		if _, err := os.Stat(path); err == nil {
			return nil
		} else if !os.IsNotExist(err) {
			return err
		}
		select {
		case <-waitCtx.Done():
			return fmt.Errorf("timed out waiting for %s: %w", path, waitCtx.Err())
		case <-ticker.C:
		}
	}
}

func writeMarkerAtomically(path string) error {
	parent := filepath.Dir(path)
	if info, err := os.Stat(parent); err != nil {
		return fmt.Errorf("marker directory %s: %w", parent, err)
	} else if !info.IsDir() {
		return fmt.Errorf("marker parent %s is not a directory", parent)
	}
	temp, err := os.CreateTemp(parent, "."+filepath.Base(path)+".")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if _, err := fmt.Fprintf(temp, "%s\n", time.Now().UTC().Format(time.RFC3339Nano)); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	return os.Link(tempPath, path)
}
