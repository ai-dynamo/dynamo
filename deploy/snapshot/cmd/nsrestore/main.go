package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsmountinjector"
)

// useInjectedBundle points every binary and library lookup at the agent bundle
// mounted into this namespace. The placeholder ships no restore tooling, so
// criu, its shared libraries, and the binaries criu forks (ip, iptables) must
// all resolve from the mount.
//
// These are set on nsrestore's own environment rather than per-command: criu is
// launched by go-criu, and criu in turn forks ip/iptables, so neither child is
// reachable through an exec.Cmd we control. Both inherit this environment.
// nsrestore itself is a static binary, so LD_LIBRARY_PATH does not affect it.
func useInjectedBundle() error {
	if err := os.Setenv("LD_LIBRARY_PATH", filepath.Join(nsmountinjector.SnapshotBinDir, "lib")); err != nil {
		return err
	}
	if err := os.Setenv("PATH", nsmountinjector.SnapshotBinDir+":"+os.Getenv("PATH")); err != nil {
		return err
	}
	cuda.CUDACheckpointHelperBinary = filepath.Join(nsmountinjector.SnapshotBinDir, "cuda-checkpoint-helper")
	return nil
}

func main() {
	// Logs go to stderr so stdout is reserved for the structured result.
	log := logging.ConfigureLogger("stderr").WithName("nsrestore")

	if err := useInjectedBundle(); err != nil {
		fatal(log, err, "failed to point lookups at the injected bundle")
	}

	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	cudaDeviceMap := flag.String("cuda-device-map", "", "CUDA device map for cuda-checkpoint-helper restore")
	cgroupRoot := flag.String("cgroup-root", "", "CRIU cgroup root remap path")
	targetPodIP := flag.String("target-pod-ip", "", "Restore pod IP for CRIU TCP socket remapping")
	flag.Parse()

	if *checkpointPath == "" {
		fatal(log, nil, "--checkpoint-path is required")
	}

	opts := executor.RestoreOptions{
		CheckpointPath: *checkpointPath,
		CUDADeviceMap:  *cudaDeviceMap,
		CgroupRoot:     *cgroupRoot,
		TargetPodIP:    *targetPodIP,
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
