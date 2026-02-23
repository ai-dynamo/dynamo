// Package cuda provides CUDA checkpoint and restore operations.
package cuda

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

const (
	podResourcesSocket = "/var/lib/kubelet/pod-resources/kubelet.sock"
	nvidiaGPUResource  = "nvidia.com/gpu"
)

// GetPodGPUUUIDs resolves GPU UUIDs for a pod/container from the kubelet PodResources API.
func GetPodGPUUUIDs(ctx context.Context, podName, podNamespace, containerName string) ([]string, error) {
	if podName == "" || podNamespace == "" {
		return nil, nil
	}

	conn, err := grpc.DialContext(
		ctx,
		"unix://"+podResourcesSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
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
					return device.GetDeviceIds(), nil
				}
			}
		}
	}

	return nil, nil
}

// FilterProcesses returns the subset of candidate PIDs that report CUDA state.
func FilterProcesses(ctx context.Context, allPIDs []int, log logr.Logger) []int {
	cudaPIDs := make([]int, 0, len(allPIDs))
	for _, pid := range allPIDs {
		if pid <= 0 {
			continue
		}
		cmd := exec.CommandContext(ctx, cudaCheckpointBinary, "--get-state", "--pid", strconv.Itoa(pid))
		if err := cmd.Run(); err != nil {
			if ctx.Err() != nil {
				break
			}
			log.V(1).Info("CUDA state probe failed", "pid", pid, "error", err)
			continue
		}
		cudaPIDs = append(cudaPIDs, pid)
	}
	return cudaPIDs
}

// BuildDeviceMap creates a cuda-checkpoint --device-map value from source and target GPU UUID lists.
func BuildDeviceMap(sourceUUIDs, targetUUIDs []string) (string, error) {
	if len(sourceUUIDs) != len(targetUUIDs) {
		return "", fmt.Errorf("GPU count mismatch: source has %d, target has %d", len(sourceUUIDs), len(targetUUIDs))
	}
	if len(sourceUUIDs) == 0 {
		return "", fmt.Errorf("GPU UUID list is empty")
	}
	pairs := make([]string, len(sourceUUIDs))
	for i := range sourceUUIDs {
		pairs[i] = sourceUUIDs[i] + "=" + targetUUIDs[i]
	}
	return strings.Join(pairs, ","), nil
}

// LockAndCheckpointProcessTree locks and checkpoints CUDA state for all given PIDs.
// On partial failure, already-checkpointed PIDs are restored+unlocked.
func LockAndCheckpointProcessTree(ctx context.Context, cudaPIDs []int, log logr.Logger) error {
	locked := make([]int, 0, len(cudaPIDs))
	for _, pid := range cudaPIDs {
		if err := lock(ctx, pid, log); err != nil {
			bulkUnlock(context.Background(), locked, log)
			return fmt.Errorf("cuda lock failed for PID %d: %w", pid, err)
		}
		locked = append(locked, pid)
	}

	checkpointed := make([]int, 0, len(cudaPIDs))
	for _, pid := range cudaPIDs {
		if err := checkpoint(ctx, pid, log); err != nil {
			recoverCheckpointed(context.Background(), checkpointed, locked, log)
			return fmt.Errorf("cuda checkpoint failed for PID %d: %w", pid, err)
		}
		checkpointed = append(checkpointed, pid)
	}

	return nil
}

// RestoreAndUnlockProcessTree restores and unlocks CUDA state for the given PIDs.
func RestoreAndUnlockProcessTree(ctx context.Context, cudaPIDs []int, deviceMap string, log logr.Logger) error {
	for _, pid := range cudaPIDs {
		if err := restoreProcess(ctx, pid, deviceMap, log); err != nil {
			return fmt.Errorf("cuda restore failed for PID %d: %w", pid, err)
		}
	}
	for _, pid := range cudaPIDs {
		if err := unlock(ctx, pid, log); err != nil {
			state, stateErr := getState(ctx, pid)
			if stateErr == nil && state == "running" {
				log.Info("cuda-checkpoint unlock returned error but process is already running", "pid", pid)
				continue
			}
			return fmt.Errorf("failed to unlock CUDA process %d: %w", pid, err)
		}
	}
	return nil
}

// bulkUnlock unlocks a list of CUDA PIDs (best-effort).
func bulkUnlock(ctx context.Context, pids []int, log logr.Logger) {
	for _, pid := range pids {
		if err := unlock(ctx, pid, log); err != nil {
			log.Error(err, "Failed to unlock CUDA process", "pid", pid)
		}
	}
}

// recoverCheckpointed is best-effort cleanup when checkpoint fails partway.
// Checkpointed PIDs need restore+unlock; locked-only PIDs just need unlock.
func recoverCheckpointed(ctx context.Context, checkpointed, locked []int, log logr.Logger) {
	checkpointedSet := make(map[int]struct{}, len(checkpointed))
	for _, pid := range checkpointed {
		checkpointedSet[pid] = struct{}{}
	}
	for _, pid := range checkpointed {
		if err := restoreProcess(ctx, pid, "", log); err != nil {
			log.Error(err, "Failed to restore CUDA process during cleanup", "pid", pid)
			continue
		}
		if err := unlock(ctx, pid, log); err != nil {
			log.Error(err, "Failed to unlock CUDA process after restore during cleanup", "pid", pid)
		}
	}
	for _, pid := range locked {
		if _, ok := checkpointedSet[pid]; ok {
			continue
		}
		if err := unlock(ctx, pid, log); err != nil {
			log.Error(err, "Failed to unlock CUDA process during cleanup", "pid", pid)
		}
	}
}
