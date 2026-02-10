// handlers.go provides HTTP handlers for the checkpoint agent server.
package httpApiServer

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

// Handlers holds dependencies for HTTP handlers.
type Handlers struct {
	cfg          ServerConfig
	checkpointer *checkpoint.Checkpointer
}

// NewHandlers creates a new Handlers instance.
func NewHandlers(cfg ServerConfig, checkpointer *checkpoint.Checkpointer) *Handlers {
	return &Handlers{
		cfg:          cfg,
		checkpointer: checkpointer,
	}
}

// HandleHealth handles GET /health requests.
func (h *Handlers) HandleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := HealthResponse{
		Status:   "healthy",
		NodeName: h.cfg.NodeName,
	}

	writeJSON(w, http.StatusOK, resp)
}

// HandleCheckpoint handles POST /checkpoint requests.
func (h *Handlers) HandleCheckpoint(w http.ResponseWriter, r *http.Request) {
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

	// Build checkpoint params
	params := checkpoint.CheckpointRequest{
		ContainerID:   req.ContainerID,
		ContainerName: req.ContainerName,
		CheckpointID:  req.CheckpointID,
		CheckpointDir: h.cfg.CheckpointSpec.BasePath,
		NodeName:      h.cfg.NodeName,
		PodName:       req.PodName,
		PodNamespace:  req.PodNamespace,
	}

	// Copy checkpoint spec and disable CUDA if requested.
	checkpointSpec := *h.cfg.CheckpointSpec
	if req.DisableCUDA {
		checkpointSpec.CRIU.LibDir = ""
	}

	ctx := r.Context()
	result, err := h.checkpointer.Checkpoint(ctx, params, &checkpointSpec)
	if err != nil {
		log.Printf("Checkpoint failed: %v", err)
		writeJSON(w, http.StatusInternalServerError, CheckpointResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// Write checkpoint.done marker so restore-entrypoint can detect this checkpoint
	checkpointDonePath := result.CheckpointDir + "/" + checkpoint.CheckpointDoneFilename
	if err := os.WriteFile(checkpointDonePath, []byte(time.Now().Format(time.RFC3339)), 0644); err != nil {
		log.Printf("Failed to write checkpoint.done marker: %v", err)
		writeJSON(w, http.StatusInternalServerError, CheckpointResponse{
			Success: false,
			Error:   fmt.Sprintf("Checkpoint succeeded but failed to write done marker: %v", err),
		})
		return
	}
	log.Printf("Wrote checkpoint.done marker: %s", checkpointDonePath)

	log.Printf("Checkpoint successful: %s", result.CheckpointID)
	writeJSON(w, http.StatusOK, CheckpointResponse{
		Success:      true,
		CheckpointID: result.CheckpointID,
		Message:      fmt.Sprintf("Checkpoint created successfully at %s", result.CheckpointDir),
	})
}

// HandleListCheckpoints handles GET /checkpoints requests.
func (h *Handlers) HandleListCheckpoints(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	checkpointIDs, err := checkpoint.ListCheckpoints(h.cfg.CheckpointSpec.BasePath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
		return
	}

	var checkpoints []CheckpointInfo
	for _, id := range checkpointIDs {
		meta, err := checkpoint.ReadCheckpointManifest(filepath.Join(h.cfg.CheckpointSpec.BasePath, id))
		if err != nil {
			continue
		}
		checkpoints = append(checkpoints, CheckpointInfo{
			ID:           meta.CheckpointID,
			CreatedAt:    meta.CreatedAt,
			SourceNode:   meta.K8s.SourceNode,
			ContainerID:  meta.K8s.ContainerID,
			PodName:      meta.K8s.PodName,
			PodNamespace: meta.K8s.PodNamespace,
		})
	}

	writeJSON(w, http.StatusOK, ListCheckpointsResponse{
		Checkpoints: checkpoints,
	})
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
