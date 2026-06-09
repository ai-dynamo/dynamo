package executor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/client-go/kubernetes"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/criu"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// RestoreRequest holds the parameters for a restore operation.
type RestoreRequest struct {
	CheckpointID                string
	CheckpointLocation          string
	ContainerCheckpointLocation string
	StartedAt                   time.Time
	NSRestorePath               string
	PodName                     string
	PodNamespace                string
	PodUID                      string
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
func Restore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest) (int, error) {
	restoreStart := time.Now()
	log.Info("=== Starting external restore ===",
		"checkpoint_id", req.CheckpointID,
		"pod", req.PodName,
		"namespace", req.PodNamespace,
		"container", req.ContainerName,
	)

	// Phase 1: Host inspect — resolve placeholder, discover target GPUs, build device map
	hostInspectStart := time.Now()
	snap, err := inspectRestore(ctx, rt, log, req)
	if err != nil {
		return 0, err
	}
	hostInspectDuration := time.Since(hostInspectStart)

	m, err := types.ReadManifest(snap.CheckpointPath)
	if err != nil {
		return 0, fmt.Errorf("failed to read checkpoint manifest for restore preflight: %w", err)
	}
	if err := prepareKubeletMountpointsForRestore(
		log,
		req,
		m,
		filepath.Join(snapshotruntime.HostProcPath, "1", "root"),
	); err != nil {
		return 0, err
	}

	// Phase 2: Execute — nsrestore handles rootfs, CRIU restore, and CUDA restore inside namespace
	result, err := execNSRestore(ctx, log, req, snap)
	if err != nil {
		return 0, fmt.Errorf("nsrestore failed: %w", err)
	}
	restoreDuration := hostInspectDuration + result.NSRestoreSetupDuration + result.CRIURestoreDuration + result.CUDADuration
	log.Info("Restore timing summary",
		"restore", map[string]any{
			"duration": restoreDuration.String(),
			"phases": map[string]string{
				"host_inspect_duration":    hostInspectDuration.String(),
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

	placeholderPID, _, err := rt.ResolveContainerByPod(ctx, req.PodName, req.PodNamespace, containerName)
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

func prepareKubeletMountpointsForRestore(log logr.Logger, req RestoreRequest, m *types.CheckpointManifest, hostRoot string) error {
	if m == nil || strings.TrimSpace(req.PodUID) == "" || len(m.CRIUDump.ExtMnt) == 0 {
		return nil
	}

	prepared := 0
	for _, val := range m.CRIUDump.ExtMnt {
		sourcePath, targetPath, ok := kubeletMountpointPaths(val, req.PodUID, req.ContainerName, hostRoot)
		if !ok || sourcePath == targetPath {
			continue
		}
		info, err := os.Stat(targetPath)
		if err != nil {
			continue
		}
		if info.IsDir() {
			if err := os.MkdirAll(sourcePath, 0755); err != nil {
				return fmt.Errorf("failed to create checkpoint-time kubelet mountpoint %s: %w", sourcePath, err)
			}
			prepared++
			continue
		}
		if err := os.MkdirAll(filepath.Dir(sourcePath), 0755); err != nil {
			return fmt.Errorf("failed to create checkpoint-time kubelet mountpoint parent %s: %w", filepath.Dir(sourcePath), err)
		}
		f, err := os.OpenFile(sourcePath, os.O_CREATE|os.O_WRONLY, info.Mode().Perm())
		if err != nil {
			return fmt.Errorf("failed to create checkpoint-time kubelet mountpoint file %s: %w", sourcePath, err)
		}
		if err := f.Close(); err != nil {
			return fmt.Errorf("failed to close checkpoint-time kubelet mountpoint file %s: %w", sourcePath, err)
		}
		prepared++
	}
	if prepared > 0 {
		log.Info("Prepared checkpoint-time kubelet mountpoints for CRIU restore",
			"count", prepared,
			"restore_pod_uid", req.PodUID,
		)
	}
	return nil
}

func kubeletMountpointPaths(path string, restorePodUID string, restoreContainerName string, hostRoot string) (string, string, bool) {
	restorePodUID = strings.TrimSpace(restorePodUID)
	if restorePodUID == "" {
		return "", "", false
	}
	cleanPath := filepath.Clean(path)
	rel, ok := strings.CutPrefix(cleanPath, "/host/var/lib/kubelet/pods"+string(os.PathSeparator))
	if !ok {
		return "", "", false
	}
	parts := strings.Split(rel, string(os.PathSeparator))
	if len(parts) < 3 {
		return "", "", false
	}

	sourceParts := append([]string{hostRoot, "var", "lib", "kubelet", "pods"}, parts...)
	sourcePath := filepath.Join(sourceParts...)
	switch parts[1] {
	case "volumes":
		targetPath, ok := kubeletVolumeTargetPath(parts, restorePodUID, hostRoot)
		return sourcePath, targetPath, ok
	case "volume-subpaths":
		targetPath, ok := kubeletVolumeSubpathTargetPath(parts, restorePodUID, restoreContainerName, hostRoot)
		return sourcePath, targetPath, ok
	default:
		return "", "", false
	}
}

func kubeletVolumeTargetPath(parts []string, restorePodUID string, hostRoot string) (string, bool) {
	if len(parts) < 4 {
		return "", false
	}
	targetParts := append([]string{hostRoot, "var", "lib", "kubelet", "pods", restorePodUID, "volumes"}, parts[2:]...)
	targetPath := filepath.Join(targetParts...)
	if _, err := os.Stat(targetPath); err == nil {
		return targetPath, true
	}
	if parts[2] != "kubernetes.io~projected" || !strings.HasPrefix(parts[3], "kube-api-access-") {
		return targetPath, true
	}

	matches, err := filepath.Glob(filepath.Join(
		hostRoot,
		"var",
		"lib",
		"kubelet",
		"pods",
		restorePodUID,
		"volumes",
		parts[2],
		"kube-api-access-*",
	))
	if err != nil || len(matches) == 0 {
		return "", false
	}
	sort.Strings(matches)
	for _, match := range matches {
		targetPath := filepath.Join(append([]string{match}, parts[4:]...)...)
		if _, err := os.Stat(targetPath); err == nil {
			return targetPath, true
		}
	}
	return "", false
}

func kubeletVolumeSubpathTargetPath(parts []string, restorePodUID string, restoreContainerName string, hostRoot string) (string, bool) {
	if len(parts) < 5 {
		return "", false
	}
	if strings.TrimSpace(restoreContainerName) == "" {
		restoreContainerName = parts[3]
	}
	targetParts := append(
		[]string{hostRoot, "var", "lib", "kubelet", "pods", restorePodUID, "volume-subpaths", parts[2], restoreContainerName},
		parts[4:]...,
	)
	targetPath := filepath.Join(targetParts...)
	if _, err := os.Stat(targetPath); err == nil {
		return targetPath, true
	}

	matches, err := filepath.Glob(filepath.Join(
		hostRoot,
		"var",
		"lib",
		"kubelet",
		"pods",
		restorePodUID,
		"volume-subpaths",
		parts[2],
		restoreContainerName,
		"*",
	))
	if err != nil || len(matches) != 1 {
		return "", false
	}
	targetPath = filepath.Join(append([]string{matches[0]}, parts[5:]...)...)
	if _, err := os.Stat(targetPath); err == nil {
		return targetPath, true
	}
	return "", false
}

// execNSRestore launches the nsrestore binary inside the placeholder container's
// namespaces via nsenter and parses the restored PID from stdout JSON.
func execNSRestore(ctx context.Context, log logr.Logger, req RestoreRequest, snap *types.RestoreContainerSnapshot) (*RestoreInNamespaceResult, error) {
	checkpointPath := req.ContainerCheckpointLocation
	if checkpointPath != "" && !filepath.IsAbs(checkpointPath) {
		return nil, fmt.Errorf("container checkpoint location must be absolute: %q", checkpointPath)
	}
	if checkpointPath == "" {
		checkpointPath = snap.CheckpointPath
	}
	args := []string{
		"-t", strconv.Itoa(snap.PlaceholderPID),
		// Intentionally exclude cgroup namespace (-C): CRIU must manage cgroups
		// from the host-visible hierarchy so --cgroup-root remap works.
		"-m", "-u", "-i", "-n", "-p",
		"--", req.NSRestorePath,
		"--checkpoint-path", checkpointPath,
	}
	if snap.CUDADeviceMap != "" {
		args = append(args, "--cuda-device-map", snap.CUDADeviceMap)
	}
	if snap.CgroupRoot != "" {
		args = append(args, "--cgroup-root", snap.CgroupRoot)
	}
	if req.PodUID != "" {
		args = append(args, "--restore-pod-uid", req.PodUID)
	}
	if req.ContainerName != "" {
		args = append(args, "--restore-container-name", req.ContainerName)
	}

	cmd := exec.CommandContext(ctx, "nsenter", args...)
	// Inherit the agent environment so nsrestore uses the same logger settings.
	cmd.Env = os.Environ()
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
