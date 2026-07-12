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
	cudaTransferBufferCount := flag.Int(
		"cuda-transfer-buffer-count",
		types.DefaultCUDATransferBufferCount,
		"Pinned transfer buffer count per CUDA device",
	)
	cudaTransferChunkBytes := flag.Uint64(
		"cuda-transfer-chunk-bytes",
		types.DefaultCUDATransferChunkBytes,
		"Pinned transfer chunk size in bytes",
	)
	cgroupRoot := flag.String("cgroup-root", "", "CRIU cgroup root remap path")
	targetPodIP := flag.String("target-pod-ip", "", "Restore pod IP for CRIU TCP socket remapping")
	deferCUDA := flag.Bool("defer-cuda", false, "Return restored CUDA process identities without restoring CUDA")
	checkDeferCUDACapability := flag.Bool(
		"check-defer-cuda-capability",
		false,
		"Report support for host-side deferred CUDA restore and exit",
	)
	flag.Parse()

	if *checkDeferCUDACapability {
		if flag.NArg() != 0 {
			fatal(log, nil, "--check-defer-cuda-capability takes no arguments")
		}
		_, _ = os.Stdout.WriteString("defer-cuda-v1\n")
		return
	}
	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}
	transferSettings := types.CUDATransferSettings{
		BufferCount: *cudaTransferBufferCount,
		ChunkBytes:  *cudaTransferChunkBytes,
	}
	if err := transferSettings.Validate(); err != nil {
		fatal(log, err, "invalid CUDA transfer settings")
	}

	opts := executor.RestoreOptions{
		CheckpointPath: *checkpointPath,
		CUDADeviceMap:  *cudaDeviceMap,
		CUDATransfer:   transferSettings,
		CgroupRoot:     *cgroupRoot,
		TargetPodIP:    *targetPodIP,
		DeferCUDA:      *deferCUDA,
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
