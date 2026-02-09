// server.go provides the HTTP server for the checkpoint agent.
package httpApiServer

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

// ServerConfig holds the configuration for the HTTP API server.
type ServerConfig struct {
	ListenAddr    string
	NodeName      string
	CheckpointCfg *checkpoint.Config
}

// Server is the HTTP API server for checkpoint operations.
type Server struct {
	cfg        ServerConfig
	handlers   *Handlers
	httpServer *http.Server
}

// NewServer creates a new Server instance.
func NewServer(cfg ServerConfig, checkpointer *checkpoint.Checkpointer) *Server {
	handlers := NewHandlers(cfg, checkpointer)

	// Setup routes
	mux := http.NewServeMux()
	mux.HandleFunc("/health", handlers.HandleHealth)
	mux.HandleFunc("/checkpoint", handlers.HandleCheckpoint)
	mux.HandleFunc("/checkpoints", handlers.HandleListCheckpoints)

	// WriteTimeout must exceed the CRIU checkpoint timeout since /checkpoint
	// blocks until the dump completes. Add 60s buffer for pre/post work.
	writeTimeout := time.Duration(cfg.CheckpointCfg.CRIU.Timeout)*time.Second + 60*time.Second
	if writeTimeout < 300*time.Second {
		writeTimeout = 300 * time.Second
	}

	httpServer := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      LoggingMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: writeTimeout,
		IdleTimeout:  120 * time.Second,
	}

	return &Server{
		cfg:        cfg,
		handlers:   handlers,
		httpServer: httpServer,
	}
}

// Start starts the HTTP server.
// This method blocks until the server is shut down.
func (s *Server) Start() error {
	log.Printf("HTTP API server listening on %s", s.cfg.ListenAddr)
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown(ctx context.Context) error {
	log.Println("Shutting down HTTP server...")
	return s.httpServer.Shutdown(ctx)
}

// Addr returns the server's listen address.
func (s *Server) Addr() string {
	return s.cfg.ListenAddr
}
