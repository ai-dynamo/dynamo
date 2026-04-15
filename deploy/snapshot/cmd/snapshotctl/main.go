package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
)

var snapshotctlLog = logging.ConfigureLogger("stderr").WithName("snapshotctl")

func main() {
	if err := run(os.Args[1:]); err != nil {
		snapshotctlLog.Error(err, "snapshotctl failed")
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage()
		return nil
	}

	switch args[0] {
	case "checkpoint":
		return runCheckpoint(args[1:])
	case "restore":
		return runRestore(args[1:])
	case "help", "-h", "--help":
		printUsage()
		return nil
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func runCheckpoint(args []string) error {
	flags := flag.NewFlagSet("checkpoint", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)

	manifest := flags.String("manifest", "", "Path to a worker Pod manifest")
	namespace := flags.String("namespace", "", "Namespace override; defaults to the manifest namespace or current kube context namespace")
	kubeContext := flags.String("kube-context", "", "Kubernetes context override")
	checkpointID := flags.String("checkpoint-id", "", "Explicit checkpoint ID; defaults to a generated value")
	disableCudaCheckpointJobFile := flags.Bool("disable-cuda-checkpoint-job-file", false, "Preserve the manifest command instead of wrapping it with cuda-checkpoint --launch-job")
	timeout := flags.Duration("timeout", 45*time.Minute, "Maximum time to wait for checkpoint completion")

	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(flags.Args()) != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	if *manifest == "" {
		return fmt.Errorf("--manifest is required")
	}

	snapshotctlLog.Info("Running checkpoint", "manifest", *manifest, "namespace", *namespace)
	result, err := runCheckpointFlow(context.Background(), checkpointOptions{
		ManifestPath:                 *manifest,
		Namespace:                    *namespace,
		KubeContext:                  *kubeContext,
		CheckpointID:                 *checkpointID,
		DisableCudaCheckpointJobFile: *disableCudaCheckpointJobFile,
		Timeout:                      *timeout,
	})
	if err != nil {
		return err
	}
	snapshotctlLog.Info("Checkpoint completed", "job", result.CheckpointJob, "checkpoint_id", result.CheckpointID)

	fmt.Printf("status=%s\n", result.Status)
	fmt.Printf("namespace=%s\n", result.Namespace)
	fmt.Printf("name=%s\n", result.Name)
	fmt.Printf("checkpoint_job=%s\n", result.CheckpointJob)
	fmt.Printf("checkpoint_id=%s\n", result.CheckpointID)
	fmt.Printf("checkpoint_location=%s\n", result.CheckpointLocation)
	return nil
}

func runRestore(args []string) error {
	flags := flag.NewFlagSet("restore", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)

	manifest := flags.String("manifest", "", "Path to a worker Pod manifest used to create a new restore pod")
	podName := flags.String("pod", "", "Existing restore target pod name")
	namespace := flags.String("namespace", "", "Namespace override; defaults to the manifest namespace or current kube context namespace")
	kubeContext := flags.String("kube-context", "", "Kubernetes context override")
	checkpointID := flags.String("checkpoint-id", "", "Checkpoint ID to restore")
	timeout := flags.Duration("timeout", 45*time.Minute, "Maximum time to wait for restore completion")

	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(flags.Args()) != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	if (*manifest == "") == (*podName == "") {
		return fmt.Errorf("must specify exactly one of --manifest or --pod")
	}

	snapshotctlLog.Info("Running restore", "manifest", *manifest, "pod", *podName, "namespace", *namespace, "checkpoint_id", *checkpointID)
	result, err := runRestoreFlow(context.Background(), restoreOptions{
		ManifestPath: *manifest,
		PodName:      *podName,
		Namespace:    *namespace,
		KubeContext:  *kubeContext,
		CheckpointID: *checkpointID,
		Timeout:      *timeout,
	})
	if err != nil {
		return err
	}
	snapshotctlLog.Info("Restore completed", "pod", result.RestorePod, "checkpoint_id", result.CheckpointID)

	fmt.Printf("status=%s\n", result.Status)
	fmt.Printf("namespace=%s\n", result.Namespace)
	fmt.Printf("name=%s\n", result.Name)
	fmt.Printf("restore_pod=%s\n", result.RestorePod)
	fmt.Printf("checkpoint_id=%s\n", result.CheckpointID)
	fmt.Printf("checkpoint_location=%s\n", result.CheckpointLocation)
	return nil
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `snapshotctl runs snapshot checkpoint and restore flows from a worker Pod manifest.

Subcommands:
  checkpoint
  restore

Examples:
  snapshotctl checkpoint --manifest /tmp/vllm-worker-pod.yaml
  snapshotctl restore --manifest /tmp/sglang-worker-pod.yaml --checkpoint-id manual-snapshot-123
  snapshotctl restore --pod existing-restore-target --checkpoint-id manual-snapshot-123
`)
}
