package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func main() {
	// Logs go to stderr so stdout is reserved for the structured result.
	log := logging.ConfigureLogger("stderr").WithName("nsrestore")

	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	cudaDeviceMap := flag.String("cuda-device-map", "", "CUDA device map for cuda-checkpoint-helper restore")
	cudaVMMPlacement := flag.String("cuda-vmm-placement", "", "Transient CUDA VMM placement plan")
	cgroupRoot := flag.String("cgroup-root", "", "CRIU cgroup root remap path")
	targetPodIP := flag.String("target-pod-ip", "", "Restore pod IP for CRIU TCP socket remapping")
	flag.Parse()

	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}

	var placement []types.VMMPlacement
	if *cudaVMMPlacement != "" {
		if err := json.Unmarshal([]byte(*cudaVMMPlacement), &placement); err != nil {
			fatal(log, err, "invalid --cuda-vmm-placement")
		}
	}
	opts := executor.RestoreOptions{
		CheckpointPath:   *checkpointPath,
		CUDADeviceMap:    *cudaDeviceMap,
		CUDAVMMPlacement: placement,
		CgroupRoot:       *cgroupRoot,
		TargetPodIP:      *targetPodIP,
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
