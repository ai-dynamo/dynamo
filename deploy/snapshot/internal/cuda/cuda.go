// Package cuda provides CUDA checkpoint and restore operations.
package cuda

import (
	"context"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/client-go/kubernetes"
	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

const (
	nvidiaGPUResource  = "nvidia.com/gpu"
	nvidiaGPUDRADriver = "gpu.nvidia.com"
)

var podResourcesSocketPath = "/var/lib/kubelet/pod-resources/kubelet.sock"

var gpuUUIDPattern = regexp.MustCompile(`^GPU-[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$`)

type CheckpointPhaseTimings struct {
	TotalDuration time.Duration
}

type RestorePhaseTimings struct {
	TotalDuration time.Duration
}

// GetPodGPUUUIDs resolves GPU UUIDs for a pod/container from kubelet
// PodResources (nvidia.com/gpu entries in GetDevices()).
func GetPodGPUUUIDs(ctx context.Context, podName, podNamespace, containerName string) ([]string, error) {
	if podName == "" || podNamespace == "" {
		return nil, nil
	}

	conn, err := grpc.NewClient(
		"unix://"+podResourcesSocketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := podresourcesv1.NewPodResourcesListerClient(conn)
	resp, err := client.List(ctx, &podresourcesv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, err
	}

	var uuids []string
	for _, pod := range resp.GetPodResources() {
		if pod.GetName() != podName || pod.GetNamespace() != podNamespace {
			continue
		}
		for _, container := range pod.GetContainers() {
			if containerName != "" && container.GetName() != containerName {
				continue
			}
			for _, device := range container.GetDevices() {
				if device.GetResourceName() == nvidiaGPUResource {
					uuids = append(uuids, device.GetDeviceIds()...)
				}
			}

		}
	}

	return uuids, nil
}

// GetGPUUUIDsViaNvidiaSmi discovers GPU UUIDs by running nvidia-smi inside the
// container's mount namespace. This is the fallback path when the kubelet
// PodResources API does not report GPU devices (e.g. when GPUs are allocated
// via DRA instead of the NVIDIA device plugin).
func GetGPUUUIDsViaNvidiaSmi(ctx context.Context, hostProcPath string, pid int) ([]string, error) {
	mountPath := fmt.Sprintf("%s/%d/ns/mnt", strings.TrimRight(hostProcPath, "/"), pid)
	cmd := exec.CommandContext(
		ctx,
		"nsenter",
		fmt.Sprintf("--mount=%s", mountPath),
		"--",
		"nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader",
	)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi via nsenter (pid %d) failed: %w", pid, err)
	}
	var uuids []string
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			uuids = append(uuids, line)
		}
	}
	return uuids, nil
}

// DiscoverGPUUUIDs resolves GPU UUIDs according to the pod's allocation mode:
// DRA-backed pods use the DRA API, classic nvidia.com/gpu pods use PodResources,
// and nvidia-smi remains the last fallback for either path.
func DiscoverGPUUUIDs(ctx context.Context, clientset kubernetes.Interface, podName, podNamespace, containerName, hostProcPath string, pid int, log logr.Logger) ([]string, error) {
	gpuUUIDs, hasNVIDIADRAAllocation, err := GetGPUUUIDsViaDRAAPI(ctx, clientset, podName, podNamespace, log)
	fallbackReason := "DRA API returned no GPU UUIDs"
	if err != nil {
		log.Error(
			err,
			"DRA API GPU UUID lookup failed, trying other discovery paths",
			"pod", podNamespace+"/"+podName,
			"has_nvidia_dra_allocation", hasNVIDIADRAAllocation,
		)
		gpuUUIDs = nil
		fallbackReason = "DRA API GPU UUID lookup failed"
	}
	if len(gpuUUIDs) > 0 {
		return gpuUUIDs, nil
	}
	if !hasNVIDIADRAAllocation {
		gpuUUIDs, err = GetPodGPUUUIDs(ctx, podName, podNamespace, containerName)
		if err != nil {
			return nil, fmt.Errorf("PodResources GPU UUID lookup failed: %w", err)
		}
		if len(gpuUUIDs) > 0 {
			return gpuUUIDs, nil
		}
		fallbackReason = "PodResources API returned no GPU UUIDs"
	}

	log.Info(fallbackReason+", falling back to nvidia-smi", "pid", pid)
	gpuUUIDs, err = GetGPUUUIDsViaNvidiaSmi(ctx, hostProcPath, pid)
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi GPU UUID fallback failed: %w", err)
	}
	log.Info("nvidia-smi fallback discovered GPU UUIDs", "uuids", gpuUUIDs)
	return gpuUUIDs, nil
}

// FilterProcesses returns the subset of candidate PIDs that hold actual CUDA contexts.
// Uses --get-restore-tid (the same technique as the CRIU CUDA plugin) instead of
// --get-state, because --get-state incorrectly matches coordinator processes like
// cuda-checkpoint --launch-job that share a /proc namespace with CUDA processes but
// don't hold CUDA contexts themselves.
func FilterProcesses(ctx context.Context, allPIDs []int, log logr.Logger) []int {
	cudaPIDs := make([]int, 0, len(allPIDs))
	for _, pid := range allPIDs {
		if pid <= 0 {
			continue
		}
		cmd := exec.CommandContext(ctx, cudaCheckpointHelperBinary, "--get-restore-tid", "--pid", strconv.Itoa(pid))
		output, err := cmd.CombinedOutput()
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			log.V(1).Info("CUDA restore-tid probe negative", "pid", pid)
			continue
		}
		tid := strings.TrimSpace(string(output))
		log.V(1).Info("CUDA restore-tid probe positive", "pid", pid, "tid", tid)
		cudaPIDs = append(cudaPIDs, pid)
	}
	return cudaPIDs
}

// BuildDeviceMap creates a cuda-checkpoint-helper --device-map value from source and target GPU UUID lists.
// When a source UUID exists in the target set, it maps to itself (identity mapping) to avoid
// unnecessary cross-GPU restore on same-node restores where kubelet returns GPUs in different order.
// Remaining unmatched source UUIDs are paired with remaining unmatched target UUIDs positionally.
func BuildDeviceMap(sourceUUIDs, targetUUIDs []string, log logr.Logger) (string, error) {
	if len(sourceUUIDs) != len(targetUUIDs) {
		return "", fmt.Errorf("GPU count mismatch: source has %d, target has %d", len(sourceUUIDs), len(targetUUIDs))
	}
	if len(sourceUUIDs) == 0 {
		return "", fmt.Errorf("GPU UUID list is empty")
	}
	log.V(1).Info("BuildDeviceMap inputs", "source_uuids", sourceUUIDs, "target_uuids", targetUUIDs)

	targetSet := make(map[string]bool, len(targetUUIDs))
	for _, t := range targetUUIDs {
		targetSet[t] = true
	}

	// First pass: identity-map any source UUID that exists in the target set
	mapping := make(map[string]string, len(sourceUUIDs))
	usedTargets := make(map[string]bool, len(targetUUIDs))
	for _, src := range sourceUUIDs {
		if targetSet[src] {
			mapping[src] = src
			usedTargets[src] = true
		}
	}

	// Second pass: pair remaining source UUIDs with remaining target UUIDs positionally
	var remainingTargets []string
	for _, t := range targetUUIDs {
		if !usedTargets[t] {
			remainingTargets = append(remainingTargets, t)
		}
	}
	idx := 0
	for _, src := range sourceUUIDs {
		if _, ok := mapping[src]; !ok {
			mapping[src] = remainingTargets[idx]
			idx++
		}
	}

	pairs := make([]string, len(sourceUUIDs))
	for i, src := range sourceUUIDs {
		pairs[i] = src + "=" + mapping[src]
	}
	return strings.Join(pairs, ","), nil
}

// LockAndCheckpointProcessTree locks and checkpoints CUDA state for all given PIDs.
// On failure, the caller is expected to fail the operation and terminate the workload.
func LockAndCheckpointProcessTree(ctx context.Context, cudaPIDs []int, log logr.Logger) (CheckpointPhaseTimings, error) {
	var timings CheckpointPhaseTimings

	start := time.Now()
	for _, pid := range cudaPIDs {
		if err := lock(ctx, pid, log); err != nil {
			timings.TotalDuration = time.Since(start)
			return timings, err
		}
	}

	for _, pid := range cudaPIDs {
		if err := checkpoint(ctx, pid, log); err != nil {
			timings.TotalDuration = time.Since(start)
			return timings, err
		}
	}
	timings.TotalDuration = time.Since(start)

	return timings, nil
}

// RestoreAndUnlockProcessTree restores and unlocks CUDA state for the given PIDs.
func RestoreAndUnlockProcessTree(ctx context.Context, cudaPIDs []int, deviceMap string, log logr.Logger) (RestorePhaseTimings, error) {
	var timings RestorePhaseTimings

	start := time.Now()
	for _, pid := range cudaPIDs {
		if err := restoreProcess(ctx, pid, deviceMap, log); err != nil {
			timings.TotalDuration = time.Since(start)
			return timings, err
		}
	}

	for _, pid := range cudaPIDs {
		if err := unlock(ctx, pid, log); err != nil {
			timings.TotalDuration = time.Since(start)
			state, stateErr := getState(ctx, pid)
			if stateErr == nil && state == "running" {
				log.Info("cuda-checkpoint-helper unlock returned error but process is already running", "pid", pid)
				continue
			}
			return timings, err
		}
	}
	timings.TotalDuration = time.Since(start)

	return timings, nil
}
