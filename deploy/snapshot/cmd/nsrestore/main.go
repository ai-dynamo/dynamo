package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
)

func main() {
	// Logs go to stderr so stdout is reserved for the structured result.
	log := logging.ConfigureLogger("stderr").WithName("nsrestore")

	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	cudaDeviceMap := flag.String("cuda-device-map", "", "CUDA device map for cuda-checkpoint-helper restore")
	cgroupRoot := flag.String("cgroup-root", "", "CRIU cgroup root remap path")
	targetPodIP := flag.String("target-pod-ip", "", "Restore pod IP for CRIU TCP socket remapping")
	rootfsMountFD := flag.Int("rootfs-mount-fd", -1, "Inherited detached SquashFS mount FD")
	rootfsWorkspaceFD := flag.Int("rootfs-workspace-fd", -1, "Inherited rootfs backing mount FD")
	flag.Parse()

	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}
	if *rootfsMountFD < 3 || *rootfsWorkspaceFD < 3 {
		fatal(log, nil, "--rootfs-mount-fd and --rootfs-workspace-fd are required")
	}

	opts := executor.RestoreOptions{
		CheckpointPath: *checkpointPath,
		CUDADeviceMap:  *cudaDeviceMap,
		CgroupRoot:     *cgroupRoot,
		TargetPodIP:    *targetPodIP,
		RootfsMountFD:  *rootfsMountFD,
		WorkspaceFD:    *rootfsWorkspaceFD,
	}

	result, err := executor.RestoreInNamespace(context.Background(), opts, log)
	if err != nil {
		fatal(log, err, "restore failed")
	}
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fatal(log, err, "Failed to write restore result")
	}
}

func fatal(log logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	if err != nil {
		log.Error(err, msg, keysAndValues...)
	} else {
		log.Info(msg, keysAndValues...)
	}
	os.Exit(1)
}
