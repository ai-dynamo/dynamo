// Package main provides the restore-entrypoint binary for self-restoring placeholder containers.
// This binary replaces the shell script restore-entrypoint.sh with a Go implementation
// that uses the go-criu library for CRIU operations.
package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/restore"
)

// logGPUDiagnostics logs nvidia-smi output and /dev/nvidia* devices for debugging GPU visibility.
func logGPUDiagnostics(label string) {
	fmt.Printf("=== GPU DIAGNOSTICS [%s] ===\n", label)

	// nvidia-smi
	if out, err := exec.Command("nvidia-smi", "-L").CombinedOutput(); err != nil {
		fmt.Printf("nvidia-smi -L: error: %v\n", err)
	} else {
		fmt.Printf("nvidia-smi -L:\n%s", out)
	}

	// GPU memory usage
	if out, err := exec.Command("nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,memory.free", "--format=csv,noheader").CombinedOutput(); err != nil {
		fmt.Printf("nvidia-smi memory query: error: %v\n", err)
	} else {
		fmt.Printf("nvidia-smi memory:\n%s", out)
	}

	// /dev/nvidia* devices
	matches, _ := filepath.Glob("/dev/nvidia*")
	fmt.Printf("/dev/nvidia* devices: %s\n", strings.Join(matches, ", "))

	// NVIDIA_VISIBLE_DEVICES env
	fmt.Printf("NVIDIA_VISIBLE_DEVICES=%s\n", os.Getenv("NVIDIA_VISIBLE_DEVICES"))
	fmt.Printf("CUDA_VISIBLE_DEVICES=%s\n", os.Getenv("CUDA_VISIBLE_DEVICES"))

	// Linux namespaces for PID 1
	for _, ns := range []string{"mnt", "pid", "ipc", "net", "uts", "cgroup"} {
		link, err := os.Readlink(fmt.Sprintf("/proc/1/ns/%s", ns))
		if err != nil {
			link = err.Error()
		}
		fmt.Printf("ns/%s: %s\n", ns, link)
	}

	fmt.Printf("=== END GPU DIAGNOSTICS [%s] ===\n", label)
}

func main() {
	// Log GPU diagnostics BEFORE anything else (gated on DEBUG for production quietness)
	if os.Getenv("DEBUG") == "1" {
		logGPUDiagnostics("PRE-RESTORE")
	}

	// Set up logging
	log := logrus.New()
	log.SetOutput(os.Stdout)
	log.SetFormatter(&logrus.TextFormatter{
		FullTimestamp:   true,
		TimestampFormat: "2006-01-02 15:04:05",
	})

	// Load configuration from hardcoded defaults + operator-injected env vars.
	// os.Args[1:] are the cold start command args (passed by the operator via pod spec).
	cfg, err := restore.NewRestoreRequest(os.Args[1:])
	if err != nil {
		log.WithError(err).Fatal("Failed to load restore configuration")
	}

	// Set log level based on DEBUG flag
	if cfg.Debug {
		log.SetLevel(logrus.DebugLevel)
	} else {
		log.SetLevel(logrus.InfoLevel)
	}

	entry := log.WithField("component", "restore-entrypoint")

	// Set up context with signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		sig := <-sigChan
		entry.WithField("signal", sig).Info("Received shutdown signal")
		cancel()
	}()

	// Run the restore entrypoint
	if err := restore.Run(ctx, cfg, entry); err != nil {
		entry.WithError(err).Fatal("Restore entrypoint failed")
	}
}
