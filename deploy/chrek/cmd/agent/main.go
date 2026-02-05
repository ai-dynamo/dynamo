// Package main provides the CRIU node agent with HTTP API and/or pod watching.
// The agent supports two modes that can be enabled independently:
// - HTTP API mode: Exposes REST endpoints for checkpoint/restore operations
// - Watcher mode: Automatically checkpoints pods with nvidia.com/checkpoint-source=true label
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/watcher"
)

// CheckpointSignalSource determines how checkpoint operations are triggered
type CheckpointSignalSource string

const (
	// SignalFromHTTP triggers checkpoints via HTTP API requests
	SignalFromHTTP CheckpointSignalSource = "http"
	// SignalFromWatcher triggers checkpoints automatically when pods become Ready
	SignalFromWatcher CheckpointSignalSource = "watcher"
)

// Config holds the agent configuration
type Config struct {
	// Common settings
	ContainerdSocket    string
	CheckpointDir       string
	HostProc            string
	NodeName            string
	RestrictedNamespace string // Optional: restrict pod watching to this namespace

	// Mode selection
	SignalSource CheckpointSignalSource // "http" or "watcher"

	// HTTP API mode settings (used when SignalSource = "http")
	ListenAddr string

	// CRIU settings (configurable options only; LeaveRunning, ShellJob, etc. are hardcoded in pkg/checkpoint/criu.go)
	CUDAPluginDir  string   // Path to CRIU CUDA plugin directory
	GhostLimit     uint32   // CRIU ghost limit in bytes
	Timeout        uint32   // CRIU timeout in seconds
	ExternalMounts []string // External mount mappings
}

// Server is the HTTP API server
type Server struct {
	config          Config
	discoveryClient *checkpointk8s.DiscoveryClient
	checkpointer    *checkpoint.Checkpointer
}

// CheckpointRequest is the request body for checkpoint operations
type CheckpointRequest struct {
	ContainerID  string `json:"container_id"`
	CheckpointID string `json:"checkpoint_id"`
	PodName      string `json:"pod_name,omitempty"`
	PodNamespace string `json:"pod_namespace,omitempty"`
	DisableCUDA  bool   `json:"disable_cuda,omitempty"` // Disable CUDA plugin for non-GPU workloads
}

// TriggerRestoreRequest is the request body for Option A self-restoring trigger
type TriggerRestoreRequest struct {
	CheckpointID           string `json:"checkpoint_id"`
	PlaceholderContainerID string `json:"placeholder_container_id"`
	SkipImageValidation    bool   `json:"skip_image_validation,omitempty"` // Skip image matching check
}

// TriggerRestoreResponse is the response for trigger restore operations
type TriggerRestoreResponse struct {
	Success          bool   `json:"success"`
	Message          string `json:"message,omitempty"`
	Error            string `json:"error,omitempty"`
	TriggerPath      string `json:"trigger_path,omitempty"`
	CheckpointImage  string `json:"checkpoint_image,omitempty"`
	PlaceholderImage string `json:"placeholder_image,omitempty"`
}

// CheckpointResponse is the response for checkpoint operations
type CheckpointResponse struct {
	Success      bool   `json:"success"`
	CheckpointID string `json:"checkpoint_id,omitempty"`
	Message      string `json:"message,omitempty"`
	Error        string `json:"error,omitempty"`
}

// CheckpointInfo represents information about a checkpoint
type CheckpointInfo struct {
	ID           string    `json:"id"`
	CreatedAt    time.Time `json:"created_at"`
	SourceNode   string    `json:"source_node"`
	ContainerID  string    `json:"container_id"`
	PodName      string    `json:"pod_name"`
	PodNamespace string    `json:"pod_namespace"`
	Image        string    `json:"image"`
}

// ListCheckpointsResponse is the response for list checkpoints
type ListCheckpointsResponse struct {
	Checkpoints []CheckpointInfo `json:"checkpoints"`
}

// HealthResponse is the response for health check
type HealthResponse struct {
	Status   string `json:"status"`
	NodeName string `json:"node_name"`
}

func main() {
	// Parse signal source - default to HTTP for backward compatibility
	signalSource := CheckpointSignalSource(strings.ToLower(getEnv("CHECKPOINT_SIGNAL_FROM", "http")))
	if signalSource != SignalFromHTTP && signalSource != SignalFromWatcher {
		log.Fatalf("Invalid CHECKPOINT_SIGNAL_FROM value: %q (must be 'http' or 'watcher')", signalSource)
	}

	// Parse CRIU settings
	var ghostLimit, timeout uint32
	if v := os.Getenv("CRIU_GHOST_LIMIT"); v != "" {
		if parsed, err := strconv.ParseUint(v, 10, 32); err == nil {
			ghostLimit = uint32(parsed)
		}
	}
	if v := os.Getenv("CRIU_TIMEOUT"); v != "" {
		if parsed, err := strconv.ParseUint(v, 10, 32); err == nil {
			timeout = uint32(parsed)
		}
	}

	// Parse external mounts (comma-separated)
	var externalMounts []string
	if v := os.Getenv("EXTERNAL_MOUNTS"); v != "" {
		externalMounts = strings.Split(v, ",")
	}

	config := Config{
		// Common settings
		ContainerdSocket:    getEnv("CONTAINERD_SOCKET", "/run/containerd/containerd.sock"),
		CheckpointDir:       getEnv("CHECKPOINT_DIR", "/checkpoints"),
		HostProc:            getEnv("HOST_PROC", "/host/proc"),
		NodeName:            getEnv("NODE_NAME", "unknown"),
		RestrictedNamespace: os.Getenv("RESTRICTED_NAMESPACE"), // Optional: empty = cluster-wide watching

		// Mode selection
		SignalSource: signalSource,

		// HTTP API settings
		ListenAddr: getEnv("LISTEN_ADDR", ":8080"),

		// CRIU settings
		CUDAPluginDir:  getEnv("CUDA_PLUGIN_DIR", ""),
		GhostLimit:     ghostLimit,
		Timeout:        timeout,
		ExternalMounts: externalMounts,
	}

	// Create discovery client
	discoveryClient, err := checkpointk8s.NewDiscoveryClient(config.ContainerdSocket)
	if err != nil {
		log.Fatalf("Failed to create discovery client: %v", err)
	}
	defer discoveryClient.Close()

	// Create checkpointer
	checkpointer := checkpoint.NewCheckpointer(discoveryClient, config.HostProc)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Printf("CRIU Node Agent starting (node: %s)", config.NodeName)
	log.Printf("Checkpoint directory: %s", config.CheckpointDir)
	log.Printf("Signal source: %s", config.SignalSource)

	switch config.SignalSource {
	case SignalFromHTTP:
		server := &Server{
			config:          config,
			discoveryClient: discoveryClient,
			checkpointer:    checkpointer,
		}

		// Setup routes
		mux := http.NewServeMux()
		mux.HandleFunc("/health", server.handleHealth)
		mux.HandleFunc("/checkpoint", server.handleCheckpoint)
		mux.HandleFunc("/restore/trigger", server.handleTriggerRestore)
		mux.HandleFunc("/checkpoints", server.handleListCheckpoints)

		httpServer := &http.Server{
			Addr:         config.ListenAddr,
			Handler:      loggingMiddleware(mux),
			ReadTimeout:  30 * time.Second,
			WriteTimeout: 300 * time.Second,
			IdleTimeout:  120 * time.Second,
		}

		// Handle graceful shutdown
		go func() {
			<-sigChan
			log.Println("Shutting down HTTP server...")
			shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer shutdownCancel()
			if err := httpServer.Shutdown(shutdownCtx); err != nil {
				log.Printf("HTTP server shutdown error: %v", err)
			}
		}()

		log.Printf("HTTP API server listening on %s", config.ListenAddr)
		if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}

	case SignalFromWatcher:
		watcherConfig := watcher.Config{
			NodeName:            config.NodeName,
			CheckpointDir:       config.CheckpointDir,
			HostProc:            config.HostProc,
			ListenAddr:          config.ListenAddr, // For health check endpoint
			RestrictedNamespace: config.RestrictedNamespace,
			CUDAPluginDir:       config.CUDAPluginDir,
			GhostLimit:          config.GhostLimit,
			Timeout:             config.Timeout,
			ExternalMounts:      config.ExternalMounts,
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

		log.Printf("Pod watcher started (watching for label: nvidia.com/checkpoint-source=true)")
		log.Printf("Health check endpoint: http://0.0.0.0%s/health", config.ListenAddr)
		if err := podWatcher.Start(ctx); err != nil {
			log.Printf("Pod watcher error: %v", err)
		}
	}

	log.Println("Agent stopped")
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := HealthResponse{
		Status:   "healthy",
		NodeName: s.config.NodeName,
	}

	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleCheckpoint(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CheckpointRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, CheckpointResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid request body: %v", err),
		})
		return
	}

	if req.ContainerID == "" {
		writeJSON(w, http.StatusBadRequest, CheckpointResponse{
			Success: false,
			Error:   "container_id is required",
		})
		return
	}

	if req.CheckpointID == "" {
		req.CheckpointID = fmt.Sprintf("ckpt-%d", time.Now().UnixNano())
	}

	// Determine CUDA plugin directory - only use if not explicitly disabled
	cudaPluginDir := s.config.CUDAPluginDir
	if req.DisableCUDA {
		cudaPluginDir = ""
	}

	// Parse optional CRIU settings from environment
	var ghostLimit, timeout uint32
	if v := os.Getenv("CRIU_GHOST_LIMIT"); v != "" {
		if parsed, err := strconv.ParseUint(v, 10, 32); err == nil {
			ghostLimit = uint32(parsed)
		}
	}
	if v := os.Getenv("CRIU_TIMEOUT"); v != "" {
		if parsed, err := strconv.ParseUint(v, 10, 32); err == nil {
			timeout = uint32(parsed)
		}
	}

	opts := checkpoint.Options{
		ContainerID:   req.ContainerID,
		CheckpointID:  req.CheckpointID,
		CheckpointDir: s.config.CheckpointDir,
		NodeName:      s.config.NodeName,
		PodName:       req.PodName,
		PodNamespace:  req.PodNamespace,
		GhostLimit:    ghostLimit,
		Timeout:       timeout,
		CUDAPluginDir: cudaPluginDir,
	}

	ctx := r.Context()
	result, err := s.checkpointer.Checkpoint(ctx, opts)
	if err != nil {
		log.Printf("Checkpoint failed: %v", err)
		writeJSON(w, http.StatusInternalServerError, CheckpointResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	log.Printf("Checkpoint successful: %s", result.CheckpointID)
	writeJSON(w, http.StatusOK, CheckpointResponse{
		Success:      true,
		CheckpointID: result.CheckpointID,
		Message:      fmt.Sprintf("Checkpoint created successfully at %s", result.CheckpointDir),
	})
}

// handleTriggerRestore implements Option A from RESTORE_ANALYSIS.md
// It triggers a self-restoring placeholder container to start CRIU restore.
// The agent writes a trigger file to the placeholder's filesystem, which
// the placeholder's entrypoint script detects and uses to start restoration.
func (s *Server) handleTriggerRestore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req TriggerRestoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, TriggerRestoreResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid request body: %v", err),
		})
		return
	}

	if req.CheckpointID == "" {
		writeJSON(w, http.StatusBadRequest, TriggerRestoreResponse{
			Success: false,
			Error:   "checkpoint_id is required",
		})
		return
	}

	if req.PlaceholderContainerID == "" {
		writeJSON(w, http.StatusBadRequest, TriggerRestoreResponse{
			Success: false,
			Error:   "placeholder_container_id is required",
		})
		return
	}

	// Verify checkpoint exists and load metadata
	checkpointPath := common.GetCheckpointDir(s.config.CheckpointDir, req.CheckpointID)
	checkpointMeta, err := common.LoadMetadata(checkpointPath)
	if err != nil {
		writeJSON(w, http.StatusNotFound, TriggerRestoreResponse{
			Success: false,
			Error:   fmt.Sprintf("Checkpoint not found: %v", err),
		})
		return
	}

	// Resolve placeholder container to get PID and image
	ctx := r.Context()
	containerInfo, err := s.discoveryClient.ResolveContainer(ctx, req.PlaceholderContainerID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, TriggerRestoreResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to resolve placeholder container: %v", err),
		})
		return
	}

	// Validate that placeholder image matches checkpoint's original image
	// This is critical because rootfs-diff.tar only contains upperdir modifications,
	// not the base image layers (lowerdir). If images differ, CRIU restore will fail.
	if !req.SkipImageValidation && checkpointMeta.Image != "" {
		if !imagesCompatible(checkpointMeta.Image, containerInfo.Image) {
			writeJSON(w, http.StatusBadRequest, TriggerRestoreResponse{
				Success:          false,
				Error:            fmt.Sprintf("Image mismatch: checkpoint was from '%s' but placeholder uses '%s'. The placeholder must use the same base image. Use skip_image_validation=true to override.", checkpointMeta.Image, containerInfo.Image),
				CheckpointImage:  checkpointMeta.Image,
				PlaceholderImage: containerInfo.Image,
			})
			return
		}
		log.Printf("Image validation passed: checkpoint=%s, placeholder=%s", checkpointMeta.Image, containerInfo.Image)
	}

	// Write trigger file to placeholder's filesystem via /proc/<pid>/root
	// The trigger file contains the checkpoint path
	triggerPath := fmt.Sprintf("%s/%d/root/tmp/restore-trigger", s.config.HostProc, containerInfo.PID)

	// Write the checkpoint path to the trigger file
	if err := os.WriteFile(triggerPath, []byte(checkpointPath), 0644); err != nil {
		writeJSON(w, http.StatusInternalServerError, TriggerRestoreResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to write trigger file: %v", err),
		})
		return
	}

	log.Printf("Triggered restore for placeholder %s (PID %d) from checkpoint %s",
		req.PlaceholderContainerID, containerInfo.PID, req.CheckpointID)

	writeJSON(w, http.StatusOK, TriggerRestoreResponse{
		Success:          true,
		Message:          fmt.Sprintf("Restore triggered for checkpoint %s", req.CheckpointID),
		TriggerPath:      triggerPath,
		CheckpointImage:  checkpointMeta.Image,
		PlaceholderImage: containerInfo.Image,
	})
}

func (s *Server) handleListCheckpoints(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	checkpointIDs, err := common.ListCheckpoints(s.config.CheckpointDir)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
		return
	}

	var checkpoints []CheckpointInfo
	for _, id := range checkpointIDs {
		meta, err := common.GetCheckpointInfo(s.config.CheckpointDir, id)
		if err != nil {
			continue
		}
		checkpoints = append(checkpoints, CheckpointInfo{
			ID:           meta.CheckpointID,
			CreatedAt:    meta.CreatedAt,
			SourceNode:   meta.SourceNode,
			ContainerID:  meta.ContainerID,
			PodName:      meta.PodName,
			PodNamespace: meta.PodNamespace,
			Image:        meta.Image,
		})
	}

	writeJSON(w, http.StatusOK, ListCheckpointsResponse{
		Checkpoints: checkpoints,
	})
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		log.Printf("Started %s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
		log.Printf("Completed %s %s in %v", r.Method, r.URL.Path, time.Since(start))
	})
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// imagesCompatible checks if two container images are compatible for CRIU restore.
// The placeholder image must be based on the same image as the checkpoint.
// Handles various image name formats:
//   - nginx:alpine vs nginx:alpine (exact match)
//   - docker.io/library/nginx:alpine vs nginx:alpine (registry prefix)
//   - criu-placeholder-nginx-alpine vs nginx:alpine (placeholder naming convention)
func imagesCompatible(checkpointImage, placeholderImage string) bool {
	// Exact match
	if checkpointImage == placeholderImage {
		return true
	}

	// Normalize images by removing common registry prefixes
	normalize := func(img string) string {
		// Remove docker.io/library/ prefix
		img = strings.TrimPrefix(img, "docker.io/library/")
		// Remove docker.io/ prefix
		img = strings.TrimPrefix(img, "docker.io/")
		return img
	}

	normalizedCheckpoint := normalize(checkpointImage)
	normalizedPlaceholder := normalize(placeholderImage)

	if normalizedCheckpoint == normalizedPlaceholder {
		return true
	}

	// Check if placeholder follows criu-placeholder-<image> naming convention
	// e.g., criu-placeholder-nginx-alpine should match nginx:alpine
	if strings.HasPrefix(normalizedPlaceholder, "criu-placeholder-") {
		// Convert nginx:alpine to nginx-alpine for comparison
		checkpointSanitized := strings.ReplaceAll(normalizedCheckpoint, ":", "-")
		checkpointSanitized = strings.ReplaceAll(checkpointSanitized, "/", "-")
		expectedPlaceholder := "criu-placeholder-" + checkpointSanitized

		if normalizedPlaceholder == expectedPlaceholder ||
			strings.HasPrefix(normalizedPlaceholder, expectedPlaceholder+":") {
			return true
		}
	}

	return false
}
