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

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsmount"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/pagebroker"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// Mounter mounts the agent binary bundle into a placeholder container's mount namespace.
type Mounter interface {
	Mount(ctx context.Context, pid int) (nsmount.MountPoint, error)
}

// RestoreRequest holds the parameters for a restore operation.
type RestoreRequest struct {
	CheckpointID                string
	CheckpointLocation          string
	ContainerCheckpointLocation string
	ContainerID                 string
	StartedAt                   time.Time
	PodName                     string
	PodNamespace                string
	TargetPodIP                 string
	ContainerName               string
	Clientset                   kubernetes.Interface
	PageBrokerRequested         bool
	PageBrokerEnabled           bool
	PageBrokerControlSocketPath string
}

// Restore performs external restore for the given request.
// Returns the namespace-relative PID of the restored process.
// The DaemonSet side inspects the placeholder and launches nsrestore,
// which handles rootfs application, CRIU restore, and CUDA restore inside the namespace.
//
// Returns the placeholder container's host PID so callers can reach into the
// container's mount namespace (e.g. to write sentinels under /snapshot-control)
// without re-resolving via the runtime.
func Restore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest, mounter Mounter) (placeholderPID int, retErr error) {
	restoreStart := time.Now()
	log.Info("=== Starting external restore ===",
		"checkpoint_id", req.CheckpointID,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)
	transactionID := ""
	var broker pagebroker.Client
	committed := false
	brokered := req.PageBrokerRequested && req.PageBrokerEnabled
	if brokered {
		transactionID = uuid.NewString()
		broker = pagebroker.Client{ControlSocketPath: req.PageBrokerControlSocketPath}
		defer func() {
			if !committed {
				abortCtx, cancel := context.WithTimeout(context.Background(), pageBrokerAbortTimeout)
				defer cancel()
				_ = broker.Abort(abortCtx, transactionID)
			}
		}()
		staged, err := broker.StagedRestore(ctx, transactionID, req.CheckpointLocation)
		if err != nil {
			return 0, fmt.Errorf("stage PageBroker restore: %w", err)
		}
		req.CheckpointLocation = staged
	}

	// Phase 1: Host inspect — resolve placeholder, discover target GPUs, build device map.
	hostInspectStart := time.Now()
	snap, err := inspectRestore(ctx, rt, log, req)
	if err != nil {
		return 0, err
	}
	hostInspectDuration := time.Since(hostInspectStart)

	// Phase 2: Mount agent binaries into the placeholder's namespace so nsrestore is reachable.
	injectStart := time.Now()
	mp, err := mountBundle(ctx, mounter, snap.PlaceholderPID)
	if err != nil {
		return 0, err
	}
	injectDuration := time.Since(injectStart)
	defer func() {
		// Pass a background context: mp.Unmount has its own internal timeout
		// (nsmount.unmountTimeout) around the ns-bind-mount subprocess.
		if cleanupErr := mp.Unmount(context.Background()); cleanupErr != nil {
			// Deliberately not promoted to retErr. The controller treats any
			// error from Restore as a failed restore and SIGKILLs the placeholder,
			// so surfacing a cleanup failure here would destroy a workload that
			// already restored successfully. Log it and let the pod continue.
			log.Error(cleanupErr, "failed to unmount agent bundle from placeholder namespace")
		}
	}()

	var mountedStaging nsmount.MountPoint
	if brokered {
		stagingMounter, err := nsmount.New(req.CheckpointLocation, nsmount.PageBrokerDst, log)
		if err != nil {
			return 0, fmt.Errorf("create PageBroker staging mount: %w", err)
		}
		mountedStaging, err = stagingMounter.Mount(ctx, snap.PlaceholderPID)
		if err != nil {
			return 0, fmt.Errorf("mount PageBroker staging: %w", err)
		}
		req.ContainerCheckpointLocation = nsmount.PageBrokerDst
	}

	// Phase 3: Execute — nsrestore handles rootfs, CRIU restore, and CUDA restore inside namespace.
	result, err := execNSRestore(ctx, log, req, snap, mp)
	if mountedStaging != nil {
		if cleanupErr := mountedStaging.Unmount(context.Background()); cleanupErr != nil {
			log.Error(cleanupErr, "failed to unmount PageBroker staging from placeholder namespace")
		}
	}
	if err != nil {
		return 0, fmt.Errorf("nsrestore failed: %w", err)
	}
	if brokered {
		if err := broker.Commit(ctx, transactionID); err != nil {
			log.Error(err, "failed to commit PageBroker restore")
		} else {
			committed = true
		}
	}
	restoreDuration := hostInspectDuration + injectDuration + result.TotalDuration()
	log.Info("Restore timing summary",
		"restore", map[string]any{
			"duration": restoreDuration.String(),
			"phases": map[string]string{
				"host_inspect_duration":    hostInspectDuration.String(),
				"inject_duration":          injectDuration.String(),
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

	validationStart := time.Now()
	if err := validateRestoredProcess(snap.TargetRoot, result.RestoredPID, log); err != nil {
		return 0, err
	}

	log.Info("=== External restore completed ===",
		"restored_pid", result.RestoredPID,
		"placeholder_host_pid", snap.PlaceholderPID,
		"validation_duration", time.Since(validationStart),
		"total_duration", time.Since(restoreStart),
	)

	return snap.PlaceholderPID, nil
}

func mountBundle(ctx context.Context, mounter Mounter, pid int) (nsmount.MountPoint, error) {
	mp, err := mounter.Mount(ctx, pid)
	if err != nil {
		return nil, fmt.Errorf("mount agent bundle into placeholder: %w", err)
	}
	return mp, nil
}

func validateRestoredProcess(targetRoot string, restoredPID int, log logr.Logger) error {
	procRoot := filepath.Join(targetRoot, "proc")
	if err := snapshotruntime.ValidateProcessState(procRoot, restoredPID); err != nil {
		restoreLogPath := filepath.Join(targetRoot, "var", "criu-work", criu.RestoreLogFilename)
		logging.LogProcessDiagnostics(procRoot, restoredPID, restoreLogPath, log)
		return fmt.Errorf("restored process failed post-restore validation: %w", err)
	}
	return nil
}

func inspectRestore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest) (*types.RestoreContainerSnapshot, error) {
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

	var placeholderPID int
	if req.ContainerID != "" {
		placeholderPID, _, err = rt.ResolveContainer(ctx, req.ContainerID)
	} else {
		placeholderPID, _, err = rt.ResolveContainerByPod(ctx, req.PodName, req.PodNamespace, containerName)
	}
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
		targetGPUUUIDs, err := cuda.DiscoverGPUUUIDs(
			ctx,
			req.Clientset,
			req.PodName,
			req.PodNamespace,
			containerName,
			snapshotruntime.HostProcPath,
			placeholderPID,
			log,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to get target GPU UUIDs: %w", err)
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
//
// Security hardening in place:
//
//  1. Mount-namespace pinning: mp.NsFd() is the /proc/<pid>/ns/mnt fd opened at
//     mount time. Passing it via --mount=/proc/self/fd/N to nsenter pins the mount
//     namespace against PID reuse. The remaining four namespaces (uts, ipc, net,
//     pid) are still resolved via -t <pid> and are not protected against reuse.
//
//  2. nsrestore binary fd: we open nsrestore from the agent host side (SnapshotBinSrc)
//     before entering any namespace and exec it via /proc/self/fd/N. This protects
//     the nsrestore binary itself against path-based substitution inside the
//     container. Binaries that nsrestore subsequently loads (criu, ip, tar, .so
//     files) are still resolved by PATH/LD_LIBRARY_PATH inside the container's
//     mount namespace.
func execNSRestore(ctx context.Context, log logr.Logger, req RestoreRequest, snap *types.RestoreContainerSnapshot, mp nsmount.MountPoint) (*RestoreInNamespaceResult, error) {
	checkpointPath := req.ContainerCheckpointLocation
	if checkpointPath != "" && !filepath.IsAbs(checkpointPath) {
		return nil, fmt.Errorf("container checkpoint location must be absolute: %q", checkpointPath)
	}
	if checkpointPath == "" {
		checkpointPath = snap.CheckpointPath
	}

	// Open nsrestore from the agent host side before entering the container
	// namespace, so the binary fd is immune to rename attacks inside the container.
	binaryFile, err := os.Open(filepath.Join(nsmount.SnapshotBinSrc, "nsrestore"))
	if err != nil {
		return nil, fmt.Errorf("open nsrestore from agent bundle: %w", err)
	}
	defer binaryFile.Close()

	// ExtraFiles[0] → child fd 3, ExtraFiles[1] → child fd 4.
	// These constants mirror nsFdChildNum in mount.go (ExtraFiles[0] = fd 3).
	const (
		nsFdChild     = 3 // mp.NsFd() passed as ExtraFiles[0]
		binaryFdChild = 4 // binaryFile passed as ExtraFiles[1]
	)

	bundleDir := nsmount.SnapshotBinDst // bundle root as seen inside the container
	var args []string

	nsFd := mp.NsFd()
	if nsFd != nil {
		// Use the pinned ns fd for the mount namespace; keep -t for the other
		// namespaces (user, ipc, net, pid). This decouples mount-ns entry from
		// PID liveness.
		args = []string{
			fmt.Sprintf("--mount=/proc/self/fd/%d", nsFdChild),
			"-t", strconv.Itoa(snap.PlaceholderPID),
			// Intentionally exclude cgroup namespace (-C): CRIU must manage cgroups
			// from the host-visible hierarchy so --cgroup-root remap works.
			"-u", "-i", "-n", "-p",
			"--", fmt.Sprintf("/proc/self/fd/%d", binaryFdChild),
		}
	} else {
		return nil, fmt.Errorf("execNSRestore: mp.NsFd() is nil; mount point was not properly initialized")
	}
	args = append(args,
		"--checkpoint-path", checkpointPath,
		"--bundle-dir", bundleDir,
	)
	if snap.CUDADeviceMap != "" {
		args = append(args, "--cuda-device-map", snap.CUDADeviceMap)
	}
	if snap.CgroupRoot != "" {
		args = append(args, "--cgroup-root", snap.CgroupRoot)
	}
	if req.TargetPodIP != "" {
		args = append(args, "--target-pod-ip", req.TargetPodIP)
	}

	cmd := exec.CommandContext(ctx, "nsenter", args...)
	// Inherit the agent environment so nsrestore uses the same logger settings.
	cmd.Env = os.Environ()
	cmd.ExtraFiles = []*os.File{nsFd, binaryFile}
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
