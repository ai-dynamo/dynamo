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
	"time"

	"github.com/go-logr/logr"
	"k8s.io/client-go/kubernetes"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsmount"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	// invalidPID is returned alongside an error when no placeholder container
	// has been resolved, so callers can tell "no target" from a real PID.
	invalidPID = -1
	// nsRestoreBinary is the executable the agent runs inside the target. It is
	// opened on the agent side and exec'd through its fd, never by name inside
	// the target: see execNSRestore.
	nsRestoreBinary = "nsrestore"
)

// Mounter mounts src at dst inside a placeholder container's mount namespace.
// The attributes are the mounter's own: each implementation carries one policy,
// so the restore path names paths and never repeats mount flags.
type Mounter interface {
	Mount(ctx context.Context, pid int, src, dst string) (nsmount.MountPoint, error)
}

// Mounters carries one Mounter per policy the restore path needs. They are
// independent, so the caller unwinds them individually and in reverse order.
type Mounters struct {
	// Bundle mounts the agent binaries, which must stay executable.
	Bundle Mounter
	// Artifact mounts the checkpoint, which must not be executable.
	Artifact Mounter
}

// RestoreRequest holds the parameters for a restore operation.
type RestoreRequest struct {
	CheckpointID    string
	ArtifactVersion string
	BasePath        string
	ContainerID     string
	StartedAt       time.Time
	PodName         string
	PodNamespace    string
	TargetPodIP     string
	ContainerName   string
	Clientset       kubernetes.Interface
	// OnArtifactReady runs after the descriptor-pinned artifact and manifest
	// have been validated, but before any target namespace is modified.
	OnArtifactReady func() error
}

// Restore performs external restore for the given request.
// Returns the namespace-relative PID of the restored process.
// The DaemonSet side inspects the placeholder and launches nsrestore,
// which handles rootfs application, CRIU restore, and CUDA restore inside the namespace.
//
// Returns the placeholder container's host PID so callers can reach into the
// container's mount namespace (e.g. to write sentinels under /snapshot-control)
// without re-resolving via the runtime.
func Restore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest, mounts Mounters) (placeholderPID int, retErr error) {
	restoreStart := time.Now()
	log.Info("=== Starting external restore ===",
		"checkpoint_id", req.CheckpointID,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)

	artifactPath, err := nsmount.ResolveArtifact(req.BasePath, req.CheckpointID, req.ArtifactVersion)
	if err != nil {
		return invalidPID, fmt.Errorf("resolve checkpoint artifact: %w", err)
	}
	manifest, err := types.ReadManifest(artifactPath)
	if err != nil {
		return invalidPID, fmt.Errorf("read checkpoint manifest: %w", err)
	}
	if manifest.CheckpointID != req.CheckpointID {
		return invalidPID, fmt.Errorf("checkpoint manifest ID %q does not match requested ID %q", manifest.CheckpointID, req.CheckpointID)
	}
	// Phase 1: Host inspect — resolve placeholder, discover target GPUs, build device map.
	hostInspectStart := time.Now()
	snap, err := inspectRestore(ctx, rt, log, req, manifest)
	if err != nil {
		return invalidPID, err
	}
	if req.OnArtifactReady != nil {
		if err := req.OnArtifactReady(); err != nil {
			return snap.PlaceholderPID, fmt.Errorf("publish artifact-ready restore state: %w", err)
		}
	}
	hostInspectDuration := time.Since(hostInspectStart)

	// Phase 2: Mount agent binaries then the checkpoint artifact into the
	// placeholder's namespace so nsrestore can reach both. Deferred cleanup
	// unwinds in reverse: artifact first (LIFO), then bundle.
	injectStart := time.Now()
	// Named once and reused: the flag nsrestore receives must be the place the
	// artifact was actually mounted.
	checkpointDst := nsmount.CheckpointDst
	bundleMount, err := mounts.Bundle.Mount(ctx, snap.PlaceholderPID, nsmount.SnapshotBinSrc, nsmount.SnapshotBinDst)
	if err != nil {
		return snap.PlaceholderPID, fmt.Errorf("mount restore resources: %w", err)
	}
	// Pass a background context: Unmount has its own internal timeout
	// (nsmount.unmountTimeout) around the ns-bind-mount subprocess.
	//
	// Cleanup failures are deliberately not promoted to retErr. The controller
	// treats any error from Restore as a failed restore and SIGKILLs the
	// placeholder, so surfacing one here would destroy a workload that already
	// restored successfully. Log it and let the pod continue.
	defer func() {
		if cleanupErr := bundleMount.Unmount(context.Background(), true); cleanupErr != nil {
			log.Error(cleanupErr, "failed to clean bundle mount from placeholder namespace")
		}
	}()

	artifactMount, err := mounts.Artifact.Mount(ctx, snap.PlaceholderPID, artifactPath, checkpointDst)
	if err != nil {
		return snap.PlaceholderPID, fmt.Errorf("mount restore resources: %w", err)
	}
	injectDuration := time.Since(injectStart)
	defer func() {
		if cleanupErr := artifactMount.Unmount(context.Background(), true); cleanupErr != nil {
			log.Error(cleanupErr, "failed to clean artifact mount from placeholder namespace")
		}
	}()

	// Phase 3: Execute — nsrestore handles rootfs, CRIU restore, and CUDA restore inside namespace.
	result, err := execNSRestore(ctx, log, req, snap, bundleMount, checkpointDst)
	if err != nil {
		return snap.PlaceholderPID, fmt.Errorf("nsrestore failed: %w", err)
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
		return snap.PlaceholderPID, err
	}

	log.Info("=== External restore completed ===",
		"restored_pid", result.RestoredPID,
		"placeholder_host_pid", snap.PlaceholderPID,
		"validation_duration", time.Since(validationStart),
		"total_duration", time.Since(restoreStart),
	)

	return snap.PlaceholderPID, nil
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

func inspectRestore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest, m *types.CheckpointManifest) (*types.RestoreContainerSnapshot, error) {
	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}

	var (
		placeholderPID int
		err            error
	)
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
func execNSRestore(ctx context.Context, log logr.Logger, req RestoreRequest, snap *types.RestoreContainerSnapshot, mp nsmount.MountPoint, checkpointPath string) (*RestoreInNamespaceResult, error) {
	// Open nsrestore from the agent host side before entering the container
	// namespace, so the binary fd is immune to rename attacks inside the container.
	binaryFile, err := os.Open(filepath.Join(nsmount.SnapshotBinSrc, nsRestoreBinary))
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
