// Package main provides the snapshot-agent DaemonSet entrypoint.
// The agent runs the node-local snapshot controller and delegates CRIU/CUDA
// execution to the snapshot executor workflows.
package main

import (
	"cmp"
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/controller"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
)

func main() {
	runtimeType := flag.String("runtime", cmp.Or(os.Getenv("RUNTIME_TYPE"), snapshotruntime.RuntimeContainerd),
		"Container runtime backend: containerd or crio")
	runtimeSocket := flag.String("runtime-socket", os.Getenv("RUNTIME_SOCKET"),
		"Path to the container runtime socket (defaults to per-runtime convention)")
	flag.Parse()

	rootLog := logging.ConfigureLogger("stdout")
	agentLog := rootLog.WithName("agent")

	cfg, err := LoadConfigOrDefault(ConfigMapPath)
	if err != nil {
		fatal(agentLog, err, "Failed to load configuration")
	}
	if err := cfg.Validate(); err != nil {
		fatal(agentLog, err, "Invalid configuration")
	}

	rt, err := snapshotruntime.New(*runtimeType, *runtimeSocket)
	if err != nil {
		fatal(agentLog, err, "Failed to initialize container runtime",
			"runtime", *runtimeType, "socket", *runtimeSocket)
	}
	defer func() {
		if closeErr := rt.Close(); closeErr != nil {
			agentLog.Error(closeErr, "Failed to close runtime client")
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	agentLog.Info("Starting snapshot agent",
		"node", cfg.NodeName,
		"restricted_namespace", cfg.RestrictedNamespace,
		"runtime", *runtimeType,
	)

	nodeController, err := controller.NewNodeController(cfg, rt, rootLog.WithName("controller"))
	if err != nil {
		fatal(agentLog, err, "Failed to create snapshot node controller")
	}

	// Run the node-local controller in the background.
	controllerDone := make(chan error, 1)
	go func() {
		agentLog.Info("Snapshot node controller started")
		controllerDone <- nodeController.Run(ctx)
	}()

	// Wait for signal or controller exit.
	select {
	case <-sigChan:
		agentLog.Info("Shutting down")
		cancel()
		select {
		case err := <-controllerDone:
			if err != nil {
				agentLog.Error(err, "Snapshot node controller exited with error during shutdown")
			}
		default:
		}
	case err := <-controllerDone:
		if err != nil {
			fatal(agentLog, err, "Snapshot node controller exited with error")
		}
	}

	agentLog.Info("Agent stopped")
}

func fatal(log logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	if err != nil {
		log.Error(err, msg, keysAndValues...)
	} else {
		log.Info(msg, keysAndValues...)
	}
	os.Exit(1)
}
