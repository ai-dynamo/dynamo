package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/logging"
)

func main() {
	// Logs go to stderr so stdout is reserved for the structured result.
	log := logging.ConfigureLogger("stderr").WithName("nsrestore")

	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	cudaDeviceMap := flag.String("cuda-device-map", "", "CUDA device map for cuda-checkpoint restore")
	cgroupRoot := flag.String("cgroup-root", "", "CRIU cgroup root remap path")
	debugPauseCUDARestore := flag.Bool("debug-pause-cuda-restore", false, "Pause each cuda-checkpoint restore child with SIGSTOP until continued")
	debugResumeMode := flag.String("debug-resume-mode", "", "Resume mode for paused cuda-checkpoint restore children: file or signal")
	debugContinueFile := flag.String("debug-continue-file", "", "File path whose creation resumes a paused cuda-checkpoint restore child")
	flag.Parse()

	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}

	opts := executor.RestoreOptions{
		CheckpointPath:        *checkpointPath,
		CUDADeviceMap:         *cudaDeviceMap,
		CgroupRoot:            *cgroupRoot,
		DebugPauseCUDARestore: *debugPauseCUDARestore,
		DebugResumeMode:       *debugResumeMode,
		DebugContinueFile:     *debugContinueFile,
	}

	restoredPID, err := executor.RestoreInNamespace(context.Background(), opts, log)
	if err != nil {
		fatal(log, err, "restore failed")
	}

	result := struct {
		RestoredPID int `json:"restoredPID"`
	}{RestoredPID: restoredPID}
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
