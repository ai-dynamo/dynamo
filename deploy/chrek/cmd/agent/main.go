// Package main provides the CRIU node agent with HTTP API and/or pod watching.
// The agent supports two modes that can be enabled independently:
// - HTTP API mode: Exposes REST endpoints for checkpoint/restore operations
// - Watcher mode: Automatically checkpoints pods with nvidia.com/checkpoint-source=true label
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	httpApiServer "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/http_api_server"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/watcher"
)

func main() {
	// Load configuration from ConfigMap (or use defaults if not found)
	cfg, err := LoadConfigOrDefault(ConfigMapPath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Validate configuration
	if err := cfg.Agent.Validate(); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	// Create discovery client
	discoveryClient, err := checkpoint.NewDiscoveryClient()
	if err != nil {
		log.Fatalf("Failed to create discovery client: %v", err)
	}
	defer discoveryClient.Close()

	// Create checkpointer
	checkpointer := checkpoint.NewCheckpointer(discoveryClient)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Printf("CRIU Node Agent starting (node: %s)", cfg.Agent.NodeName)
	log.Printf("Checkpoint directory: %s", cfg.Checkpoint.BasePath)
	log.Printf("Signal source: %s", cfg.Agent.SignalSource)

	switch cfg.Agent.GetSignalSource() {
	case SignalFromHTTP:
		serverCfg := httpApiServer.ServerConfig{
			ListenAddr:    cfg.Agent.ListenAddr,
			NodeName:      cfg.Agent.NodeName,
			CheckpointCfg: &cfg.Checkpoint,
		}
		srv := httpApiServer.NewServer(serverCfg, checkpointer)

		// Handle graceful shutdown
		go func() {
			<-sigChan
			shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer shutdownCancel()
			if err := srv.Shutdown(shutdownCtx); err != nil {
				log.Printf("HTTP server shutdown error: %v", err)
			}
		}()

		if err := srv.Start(); err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}

	case SignalFromWatcher:
		watcherConfig := watcher.Config{
			NodeName:            cfg.Agent.NodeName,
			ListenAddr:          cfg.Agent.ListenAddr,
			RestrictedNamespace: cfg.Agent.RestrictedNamespace,
			CheckpointConfig:    &cfg.Checkpoint,
		}

		podWatcher, err := watcher.NewWatcher(watcherConfig, discoveryClient, checkpointer)
		if err != nil {
			log.Fatalf("Failed to create pod watcher: %v", err)
		}

		// Handle graceful shutdown
		go func() {
			<-sigChan
			log.Println("Shutting down pod watcher...")
			cancel()
		}()

		log.Printf("Pod watcher started (watching for label: %s=true)", checkpoint.KubeLabelCheckpointSource)
		log.Printf("Health check endpoint: http://0.0.0.0%s/health", cfg.Agent.ListenAddr)
		if err := podWatcher.Start(ctx); err != nil {
			log.Printf("Pod watcher error: %v", err)
		}

	default:
		log.Fatalf("Unknown signal source: %s", cfg.Agent.SignalSource)
	}

	log.Println("Agent stopped")
}
