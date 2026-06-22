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

	// rootCtx is cancelled on signal. The single node controller drives both the
	// restore (pod informer) and capture (PodSnapshotContent informer) paths and shuts
	// down when rootCtx is cancelled.
	rootCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		agentLog.Info("Shutting down")
		cancel()
	}()

	agentLog.Info("Starting snapshot agent",
		"node", cfg.NodeName,
		"restricted_namespace", cfg.RestrictedNamespace,
		"runtime", *runtimeType,
	)

	// The node controller handles both restore and capture paths.
	nodeController, err := controller.NewNodeController(cfg, rt, rootLog.WithName("controller"))
	if err != nil {
		fatal(agentLog, err, "Failed to create snapshot node controller")
	}
	if runErr := nodeController.Run(rootCtx); runErr != nil {
		fatal(agentLog, runErr, "Snapshot node controller exited with error")
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
