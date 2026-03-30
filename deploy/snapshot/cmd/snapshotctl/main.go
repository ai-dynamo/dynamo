package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
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
	checkpointHash := flags.String("checkpoint-hash", "", "Explicit checkpoint hash; defaults to a generated value")
	disableCudaCheckpointJobFile := flags.Bool("disable-cuda-checkpoint-job-file", false, "Preserve the manifest command instead of wrapping it with cuda-checkpoint --launch-job")
	timeout := flags.Duration("timeout", 45*time.Minute, "Maximum time to wait for checkpoint completion")

	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(flags.Args()) != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}

	result, err := runCheckpointFlow(context.Background(), checkpointOptions{
		ManifestPath:                 *manifest,
		Namespace:                    *namespace,
		CheckpointHash:               *checkpointHash,
		DisableCudaCheckpointJobFile: *disableCudaCheckpointJobFile,
		Timeout:                      *timeout,
	})
	if err != nil {
		return err
	}

	fmt.Printf("status=%s\n", result.Status)
	fmt.Printf("namespace=%s\n", result.Namespace)
	fmt.Printf("name=%s\n", result.Name)
	fmt.Printf("checkpoint_job=%s\n", result.CheckpointJob)
	fmt.Printf("checkpoint_hash=%s\n", result.CheckpointHash)
	fmt.Printf("checkpoint_location=%s\n", result.CheckpointLocation)
	return nil
}

func runRestore(args []string) error {
	flags := flag.NewFlagSet("restore", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)

	manifest := flags.String("manifest", "", "Path to a worker Pod manifest")
	namespace := flags.String("namespace", "", "Namespace override; defaults to the manifest namespace or current kube context namespace")
	checkpointHash := flags.String("checkpoint-hash", "", "Checkpoint hash to restore")
	timeout := flags.Duration("timeout", 45*time.Minute, "Maximum time to wait for restore completion")

	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(flags.Args()) != 0 {
		return fmt.Errorf("unexpected arguments: %v", flags.Args())
	}

	result, err := runRestoreFlow(context.Background(), restoreOptions{
		ManifestPath:   *manifest,
		Namespace:      *namespace,
		CheckpointHash: *checkpointHash,
		Timeout:        *timeout,
	})
	if err != nil {
		return err
	}

	fmt.Printf("status=%s\n", result.Status)
	fmt.Printf("namespace=%s\n", result.Namespace)
	fmt.Printf("name=%s\n", result.Name)
	fmt.Printf("restore_pod=%s\n", result.RestorePod)
	fmt.Printf("checkpoint_hash=%s\n", result.CheckpointHash)
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
  snapshotctl restore --manifest /tmp/sglang-worker-pod.yaml --checkpoint-hash manual-snapshot-123
`)
}
