// Package cuda provides CUDA checkpoint and restore operations.
package cuda

import (
	"context"
	"errors"
	"fmt"
	"math"
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

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	nvidiaGPUResource  = "nvidia.com/gpu"
	nvidiaGPUDRADriver = "gpu.nvidia.com"
)

var podResourcesSocketPath = "/var/lib/kubelet/pod-resources/kubelet.sock"

var gpuUUIDPattern = regexp.MustCompile(`^GPU-[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$`)
var normalizedGPUUUIDPattern = regexp.MustCompile(`^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$`)

type CheckpointPhaseTimings struct {
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
// container's mount and PID namespaces. This is the fallback path when the kubelet
// PodResources API does not report GPU devices (e.g. when GPUs are allocated
// via DRA instead of the NVIDIA device plugin).
func GetGPUUUIDsViaNvidiaSmi(ctx context.Context, hostProcPath string, pid int) ([]string, error) {
	mountPath := fmt.Sprintf("%s/%d/ns/mnt", strings.TrimRight(hostProcPath, "/"), pid)
	pidPath := fmt.Sprintf("%s/%d/ns/pid", strings.TrimRight(hostProcPath, "/"), pid)
	cmd := exec.CommandContext(
		ctx,
		"nsenter",
		fmt.Sprintf("--mount=%s", mountPath),
		fmt.Sprintf("--pid=%s", pidPath),
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

type visibleGPUDiscovery func(context.Context, string, int) ([]string, error)

// DiscoverGPUUUIDs resolves GPU UUIDs in the container's runtime ordinal order.
func DiscoverGPUUUIDs(ctx context.Context, clientset kubernetes.Interface, podName, podNamespace, containerName, hostProcPath string, pid int, log logr.Logger) ([]string, error) {
	return discoverGPUUUIDs(
		ctx,
		clientset,
		podName,
		podNamespace,
		containerName,
		hostProcPath,
		pid,
		GetGPUUUIDsViaNvidiaSmi,
		log,
	)
}

func discoverGPUUUIDs(
	ctx context.Context,
	clientset kubernetes.Interface,
	podName,
	podNamespace,
	containerName,
	hostProcPath string,
	pid int,
	discoverVisibleGPUs visibleGPUDiscovery,
	log logr.Logger,
) ([]string, error) {
	gpuUUIDs, hasNVIDIADRAAllocation, err := GetGPUUUIDsViaDRAAPI(ctx, clientset, podName, podNamespace, containerName, log)
	if err != nil {
		if hasNVIDIADRAAllocation {
			return nil, fmt.Errorf("DRA GPU UUID lookup failed: %w", err)
		}
		log.Error(
			err,
			"DRA API GPU UUID lookup failed, trying other discovery paths",
			"pod", podNamespace+"/"+podName,
		)
		gpuUUIDs = nil
	}

	if hasNVIDIADRAAllocation {
		if len(gpuUUIDs) == 0 {
			return nil, errors.New(
				"DRA GPU allocation has no resolvable UUIDs",
			)
		}
		visibleGPUUUIDs, err := discoverVisibleGPUs(ctx, hostProcPath, pid)
		if err != nil {
			return nil, fmt.Errorf(
				"discover DRA GPUs in container ordinal order: %w",
				err,
			)
		}
		orderedUUIDs, err := orderDRAUUIDsByRuntime(gpuUUIDs, visibleGPUUUIDs)
		if err != nil {
			return nil, err
		}
		log.Info(
			"resolved DRA GPU UUIDs in container ordinal order",
			"uuids", orderedUUIDs,
		)
		return orderedUUIDs, nil
	}

	gpuUUIDs, err = GetPodGPUUUIDs(ctx, podName, podNamespace, containerName)
	if err != nil {
		return nil, fmt.Errorf("PodResources GPU UUID lookup failed: %w", err)
	}
	if len(gpuUUIDs) > 0 {
		return gpuUUIDs, nil
	}

	log.Info("PodResources API returned no GPU UUIDs, falling back to nvidia-smi", "pid", pid)
	gpuUUIDs, err = discoverVisibleGPUs(ctx, hostProcPath, pid)
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi GPU UUID fallback failed: %w", err)
	}
	log.Info("nvidia-smi fallback discovered GPU UUIDs", "uuids", gpuUUIDs)
	return gpuUUIDs, nil
}

func orderDRAUUIDsByRuntime(allocatedUUIDs, visibleUUIDs []string) ([]string, error) {
	if len(allocatedUUIDs) != len(visibleUUIDs) {
		return nil, fmt.Errorf(
			"DRA allocation and container-visible GPU count differ: allocated=%d visible=%d",
			len(allocatedUUIDs),
			len(visibleUUIDs),
		)
	}

	allocated := make(map[string]struct{}, len(allocatedUUIDs))
	for _, uuid := range allocatedUUIDs {
		if !gpuUUIDPattern.MatchString(uuid) {
			return nil, fmt.Errorf("DRA allocation contains invalid GPU UUID %q", uuid)
		}
		if _, duplicate := allocated[uuid]; duplicate {
			return nil, fmt.Errorf("DRA allocation contains duplicate GPU UUID %q", uuid)
		}
		allocated[uuid] = struct{}{}
	}

	seen := make(map[string]struct{}, len(visibleUUIDs))
	for _, uuid := range visibleUUIDs {
		if !gpuUUIDPattern.MatchString(uuid) {
			return nil, fmt.Errorf("container reports invalid GPU UUID %q", uuid)
		}
		if _, duplicate := seen[uuid]; duplicate {
			return nil, fmt.Errorf("container reports duplicate GPU UUID %q", uuid)
		}
		if _, ok := allocated[uuid]; !ok {
			return nil, fmt.Errorf(
				"container-visible GPU %q is not in the DRA allocation",
				uuid,
			)
		}
		seen[uuid] = struct{}{}
	}

	return append([]string(nil), visibleUUIDs...), nil
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

type gpuPlacement struct {
	source        string
	target        string
	targetOrdinal int32
}

func normalizeGPUUUID(value string) (string, error) {
	value = strings.TrimPrefix(value, "GPU-")
	if !normalizedGPUUUIDPattern.MatchString(value) {
		return "", fmt.Errorf("invalid GPU UUID %q", value)
	}
	return strings.ToLower(value), nil
}

func buildGPUPlacements(sourceUUIDs, targetUUIDs []string) ([]gpuPlacement, error) {
	if len(sourceUUIDs) != len(targetUUIDs) {
		return nil, fmt.Errorf("GPU count mismatch: source has %d, target has %d", len(sourceUUIDs), len(targetUUIDs))
	}
	if len(sourceUUIDs) == 0 {
		return nil, fmt.Errorf("GPU UUID list is empty")
	}

	sources := make([]string, len(sourceUUIDs))
	sourceSet := make(map[string]struct{}, len(sourceUUIDs))
	for index, value := range sourceUUIDs {
		uuid, err := normalizeGPUUUID(value)
		if err != nil {
			return nil, fmt.Errorf("source GPU UUID at index %d: %w", index, err)
		}
		if _, duplicate := sourceSet[uuid]; duplicate {
			return nil, fmt.Errorf("duplicate source GPU UUID %q", value)
		}
		sourceSet[uuid] = struct{}{}
		sources[index] = uuid
	}

	targets := make([]string, len(targetUUIDs))
	targetOrdinals := make(map[string]int32, len(targetUUIDs))
	for index, value := range targetUUIDs {
		uuid, err := normalizeGPUUUID(value)
		if err != nil {
			return nil, fmt.Errorf("target GPU UUID at index %d: %w", index, err)
		}
		if _, duplicate := targetOrdinals[uuid]; duplicate {
			return nil, fmt.Errorf("duplicate target GPU UUID %q", value)
		}
		if index > math.MaxInt32 {
			return nil, errors.New("target GPU ordinal cannot be represented")
		}
		targetOrdinals[uuid] = int32(index)
		targets[index] = uuid
	}

	mapping := make(map[string]string, len(sources))
	usedTargets := make(map[string]struct{}, len(targets))
	for _, source := range sources {
		if _, present := targetOrdinals[source]; present {
			mapping[source] = source
			usedTargets[source] = struct{}{}
		}
	}

	remainingTargets := make([]string, 0, len(targets)-len(usedTargets))
	for _, target := range targets {
		if _, used := usedTargets[target]; !used {
			remainingTargets = append(remainingTargets, target)
		}
	}
	remainingIndex := 0
	for _, source := range sources {
		if _, identity := mapping[source]; !identity {
			mapping[source] = remainingTargets[remainingIndex]
			remainingIndex++
		}
	}

	placements := make([]gpuPlacement, len(sources))
	for index, source := range sources {
		target := mapping[source]
		placements[index] = gpuPlacement{
			source:        source,
			target:        target,
			targetOrdinal: targetOrdinals[target],
		}
	}
	return placements, nil
}

// BuildDeviceMap creates a cuda-checkpoint-helper --device-map value from source and target GPU UUID lists.
// Identity matches are selected first, then unmatched sources are paired with
// unmatched targets in target order. An all-identity mapping remains omitted.
func BuildDeviceMap(sourceUUIDs, targetUUIDs []string, log logr.Logger) (string, error) {
	log.V(1).Info("BuildDeviceMap inputs", "source_uuids", sourceUUIDs, "target_uuids", targetUUIDs)
	placements, err := buildGPUPlacements(sourceUUIDs, targetUUIDs)
	if err != nil {
		return "", err
	}
	allIdentity := true
	for _, placement := range placements {
		if placement.source != placement.target {
			allIdentity = false
			break
		}
	}
	if allIdentity {
		return "", nil
	}

	pairs := make([]string, len(placements))
	for index, placement := range placements {
		pairs[index] = "GPU-" + placement.source + "=GPU-" + placement.target
	}
	return strings.Join(pairs, ","), nil
}

// BuildVMMPlacement creates the complete authoritative VMM placement plan.
func BuildVMMPlacement(sourceUUIDs, targetUUIDs []string) ([]types.VMMPlacement, error) {
	placements, err := buildGPUPlacements(sourceUUIDs, targetUUIDs)
	if err != nil {
		return nil, err
	}
	result := make([]types.VMMPlacement, len(placements))
	for index, placement := range placements {
		result[index] = types.VMMPlacement{
			SourceGPUUUID: placement.source,
			TargetGPUUUID: placement.target,
			TargetOrdinal: placement.targetOrdinal,
		}
	}
	return result, nil
}

// ValidateVMMPlacement verifies a transient plan against manifest source UUIDs.
func ValidateVMMPlacement(
	sourceUUIDs []string,
	placements []types.VMMPlacement,
) error {
	if len(sourceUUIDs) != len(placements) {
		return fmt.Errorf(
			"CUDA VMM placement count is %d, want %d",
			len(placements),
			len(sourceUUIDs),
		)
	}
	targetUUIDs := make([]string, len(placements))
	for _, placement := range placements {
		if placement.TargetOrdinal < 0 ||
			int64(placement.TargetOrdinal) >= int64(len(placements)) {
			return fmt.Errorf(
				"CUDA VMM placement has invalid target ordinal %d",
				placement.TargetOrdinal,
			)
		}
		if targetUUIDs[placement.TargetOrdinal] != "" {
			return fmt.Errorf(
				"CUDA VMM placement has duplicate target ordinal %d",
				placement.TargetOrdinal,
			)
		}
		if normalized, err := normalizeGPUUUID(placement.SourceGPUUUID); err != nil ||
			normalized != placement.SourceGPUUUID {
			return fmt.Errorf(
				"CUDA VMM placement has non-canonical source GPU UUID %q",
				placement.SourceGPUUUID,
			)
		}
		if normalized, err := normalizeGPUUUID(placement.TargetGPUUUID); err != nil ||
			normalized != placement.TargetGPUUUID {
			return fmt.Errorf(
				"CUDA VMM placement has non-canonical target GPU UUID %q",
				placement.TargetGPUUUID,
			)
		}
		targetUUIDs[placement.TargetOrdinal] = placement.TargetGPUUUID
	}
	expected, err := BuildVMMPlacement(sourceUUIDs, targetUUIDs)
	if err != nil {
		return err
	}
	if len(expected) != len(placements) {
		return errors.New("CUDA VMM placement is incomplete")
	}
	for index := range expected {
		if expected[index] != placements[index] {
			return fmt.Errorf(
				"CUDA VMM placement entry %d does not match the device mapping policy",
				index,
			)
		}
	}
	return nil
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

// RestoreProcessTree restores CUDA state while keeping every process locked.
func RestoreProcessTree(ctx context.Context, cudaPIDs []int, deviceMap string, log logr.Logger) error {
	for _, pid := range cudaPIDs {
		if err := restoreProcess(ctx, pid, deviceMap, log); err != nil {
			return err
		}
	}
	return nil
}

// UnlockProcessTree resumes all restored CUDA processes.
func UnlockProcessTree(ctx context.Context, cudaPIDs []int, log logr.Logger) error {
	for _, pid := range cudaPIDs {
		if err := unlock(ctx, pid, log); err != nil {
			state, stateErr := getState(ctx, pid)
			if stateErr == nil && state == "running" {
				log.Info("cuda-checkpoint-helper unlock returned error but process is already running", "pid", pid)
				continue
			}
			return err
		}
	}
	return nil
}
