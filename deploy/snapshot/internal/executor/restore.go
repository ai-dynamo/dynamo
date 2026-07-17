package executor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"
	"k8s.io/client-go/kubernetes"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/profile"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// RestoreRequest holds the parameters for a restore operation.
type RestoreRequest struct {
	CheckpointID                string
	CheckpointLocation          string
	ContainerCheckpointLocation string
	ContainerID                 string
	StartedAt                   time.Time
	NSRestorePath               string
	PodName                     string
	PodNamespace                string
	TargetPodIP                 string
	ContainerName               string
	Clientset                   kubernetes.Interface
}

// Restore performs external restore for the given request.
// Returns the namespace-relative PID of the restored process.
// The DaemonSet side inspects the placeholder and launches nsrestore,
// which handles rootfs application, CRIU restore, and CUDA restore inside the namespace.
//
// Returns the placeholder container's host PID so callers can reach into the
// container's mount namespace (e.g. to write sentinels under /snapshot-control)
// without re-resolving via the runtime.
func Restore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest) (placeholderHostPID int, retErr error) {
	restoreStart := time.Now()
	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}
	operation := profile.NewOperation(
		"restore",
		"checkpoint_id", req.CheckpointID,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", containerName,
	)
	restoreSpan := operation.Start(log, "snapshot-agent", "restore_total")
	defer func() {
		restoreSpan.EndStatus(retErr)
	}()
	log.Info("=== Starting external restore ===",
		"checkpoint_id", req.CheckpointID,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)

	// Phase 1: Host inspect — resolve placeholder, discover target GPUs, build device map
	hostInspectStart := time.Now()
	inspectSpan := operation.Start(log, "snapshot-agent", "host_inspect_total")
	snap, err := inspectRestore(ctx, rt, log, req, operation)
	inspectSpan.EndStatus(err)
	if err != nil {
		return 0, err
	}
	hostInspectDuration := time.Since(hostInspectStart)

	image, err := snapshotruntime.OpenValidatedRootfs(snap.CheckpointPath, req.CheckpointID)
	if err != nil {
		return 0, fmt.Errorf("validate rootfs artifact: %w", err)
	}
	defer image.Close()
	rootfsMount, err := snapshotruntime.PrepareDetachedRootfsMount(image)
	if err != nil {
		return 0, fmt.Errorf("prepare rootfs mount: %w", err)
	}
	defer rootfsMount.Close()
	workspace, err := openRootfsBackingWorkspace(snap.PlaceholderPID)
	if err != nil {
		return 0, err
	}
	defer workspace.Close()
	checkpoint, err := openCheckpointDirectory(snap.CheckpointPath)
	if err != nil {
		return 0, err
	}
	defer checkpoint.Close()

	// Phase 2: Execute — nsrestore handles rootfs, CRIU restore, and CUDA restore inside namespace
	result, err := execNSRestore(
		ctx,
		log,
		req,
		snap,
		rootfsMount,
		workspace,
		checkpoint,
	)
	if err != nil {
		return 0, fmt.Errorf("nsrestore failed: %w", err)
	}
	log.Info("Restore timing summary",
		"restore", map[string]any{
			"duration": time.Since(restoreStart).String(),
			"phases": map[string]string{
				"host_inspect_duration": hostInspectDuration.String(),
			},
		},
	)
	if !req.StartedAt.IsZero() {
		log.Info("Restore wall time from agent detection",
			"started_to_restore_complete", time.Since(req.StartedAt),
		)
	}

	// Validate restored process from the host side
	validationStart := time.Now()
	procRoot := filepath.Join(snap.TargetRoot, "proc")
	if err := snapshotruntime.ValidateProcessState(procRoot, result.RestoredPID); err != nil {
		restoreLogPath := filepath.Join(snap.TargetRoot, "var", "criu-work", criu.RestoreLogFilename)
		logging.LogProcessDiagnostics(procRoot, result.RestoredPID, restoreLogPath, log)
		return 0, fmt.Errorf("restored process failed post-restore validation: %w", err)
	}

	log.Info("=== External restore completed ===",
		"restored_pid", result.RestoredPID,
		"placeholder_host_pid", snap.PlaceholderPID,
		"validation_duration", time.Since(validationStart),
		"total_duration", time.Since(restoreStart),
	)

	return snap.PlaceholderPID, nil
}

func inspectRestore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest, operation profile.Operation) (*types.RestoreContainerSnapshot, error) {
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

	manifestSpan := operation.Start(log, "snapshot-agent", "manifest_read")
	m, err := types.ReadManifest(checkpointPath)
	manifestSpan.EndStatus(err)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}

	var placeholderPID int
	containerSpan := operation.Start(log, "snapshot-agent", "container_resolution")
	if req.ContainerID != "" {
		placeholderPID, _, err = rt.ResolveContainer(ctx, req.ContainerID)
	} else {
		placeholderPID, _, err = rt.ResolveContainerByPod(ctx, req.PodName, req.PodNamespace, containerName)
	}
	containerSpan.EndStatus(err)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve placeholder container: %w", err)
	}
	log.V(1).Info("Resolved placeholder container", "pid", placeholderPID)

	cgroupSpan := operation.Start(log, "snapshot-agent", "cgroup_inspection", "pid", placeholderPID)
	cgroupRoot, err := snapshotruntime.ResolveCgroupRootFromHostPID(placeholderPID)
	cgroupSpan.EndStatus(err)
	if err != nil {
		log.Error(err, "Failed to resolve placeholder cgroup root; proceeding without explicit cgroup remap")
		cgroupRoot = ""
	}

	cudaDeviceMap := ""
	if !m.CUDA.IsEmpty() {
		if len(m.CUDA.SourceGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing source GPU UUIDs in checkpoint manifest")
		}
		discoverySpan := operation.Start(log, "snapshot-agent", "gpu_discovery", "pid", placeholderPID)
		targetGPUUUIDs, err := cuda.DiscoverGPUUUIDs(
			ctx,
			req.Clientset,
			req.PodName,
			req.PodNamespace,
			containerName,
			snapshotruntime.HostProcPath,
			placeholderPID,
			log,
			operation,
		)
		discoverySpan.EndStatus(err)
		if err != nil {
			return nil, fmt.Errorf("failed to get target GPU UUIDs: %w", err)
		}
		if len(targetGPUUUIDs) == 0 {
			return nil, fmt.Errorf("missing target GPU UUIDs for %s/%s container %s", req.PodNamespace, req.PodName, containerName)
		}
		deviceMapSpan := operation.Start(log, "snapshot-agent", "device_map_build")
		cudaDeviceMap, err = cuda.BuildDeviceMap(m.CUDA.SourceGPUUUIDs, targetGPUUUIDs, log)
		deviceMapSpan.EndStatus(err)
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
func execNSRestore(
	ctx context.Context,
	log logr.Logger,
	req RestoreRequest,
	snap *types.RestoreContainerSnapshot,
	rootfsMount, workspace, checkpoint *os.File,
) (*RestoreInNamespaceResult, error) {
	if rootfsMount == nil || workspace == nil || checkpoint == nil {
		return nil, fmt.Errorf("rootfs mount, backing workspace, and checkpoint are required")
	}
	checkpointPath := "/proc/self/fd/5"
	args := append(
		nsenterTargetArgs(snap.PlaceholderPID),
		"--", req.NSRestorePath,
		"--checkpoint-path", checkpointPath,
		"--rootfs-mount-fd", "3",
		"--rootfs-workspace-fd", "4",
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
	cmd.ExtraFiles = []*os.File{rootfsMount, workspace, checkpoint}
	// Inherit the agent environment so nsrestore uses the same logger settings.
	cmd.Env = os.Environ()
	log.V(1).Info("Executing nsenter + nsrestore", "cmd", cmd.String())

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = io.MultiWriter(os.Stderr, &stderr)

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf(
			"nsrestore failed: %w\nstdout: %s\nstderr tail: %s",
			err,
			stdout.String(),
			stderrTail(stderr.Bytes()),
		)
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

func nsenterTargetArgs(targetPID int) []string {
	return []string{
		"-t", strconv.Itoa(targetPID),
		// Intentionally exclude cgroup namespace (-C): CRIU must manage cgroups
		// from the host-visible hierarchy so --cgroup-root remap works.
		"-m", "-u", "-i", "-n", "-p",
	}
}

// Kubernetes Event messages are limited to 1 KiB. Leave room for the
// surrounding restore error while preserving the final nsrestore log lines.
const nsRestoreStderrTailLimit = 768

func stderrTail(stderr []byte) []byte {
	if len(stderr) <= nsRestoreStderrTailLimit {
		return stderr
	}
	tail := stderr[len(stderr)-nsRestoreStderrTailLimit:]
	for len(tail) > 0 && !utf8.RuneStart(tail[0]) {
		tail = tail[1:]
	}
	return tail
}

func openRootfsBackingWorkspace(placeholderPID int) (*os.File, error) {
	path := filepath.Join(
		snapshotruntime.HostProcPath,
		strconv.Itoa(placeholderPID),
		"root",
		strings.TrimPrefix(snapshotruntime.RootfsBackingWorkspaceMountPath, "/"),
	)
	fd, err := unix.Open(
		path,
		unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return nil, fmt.Errorf("open rootfs backing workspace: %w", err)
	}
	return os.NewFile(uintptr(fd), "rootfs-backing-workspace"), nil
}

func openCheckpointDirectory(path string) (*os.File, error) {
	fd, err := unix.Open(
		path,
		unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return nil, fmt.Errorf("open checkpoint directory: %w", err)
	}
	return os.NewFile(uintptr(fd), "checkpoint-directory"), nil
}
