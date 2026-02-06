// Package watcher provides Kubernetes pod watching for automatic checkpointing.
package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
)

const (
	// LabelCheckpointSource is the label that triggers automatic checkpointing
	LabelCheckpointSource = "nvidia.com/checkpoint-source"

	// LabelCheckpointHash is the label specifying the checkpoint identity hash
	LabelCheckpointHash = "nvidia.com/checkpoint-hash"

	// EnvCheckpointSignalFile is the env var in the pod specifying the signal file path
	EnvCheckpointSignalFile = "DYN_CHECKPOINT_SIGNAL_FILE"
)

// SignalFile represents the content of a checkpoint completion signal file
type SignalFile struct {
	CheckpointID   string    `json:"checkpoint_id"`
	CheckpointPath string    `json:"checkpoint_path"`
	Timestamp      time.Time `json:"timestamp"`
	Success        bool      `json:"success"`
	Error          string    `json:"error,omitempty"`
}

// Config holds watcher configuration
type Config struct {
	NodeName            string
	CheckpointDir       string
	HostProc            string
	ListenAddr          string // HTTP server address for health checks (e.g., ":8080")
	RestrictedNamespace string // Optional: restrict watching to this namespace (empty = cluster-wide)

	// GPU/CUDA checkpoint options (passed to checkpoint.Options)
	CUDAPluginDir  string   // Path to CRIU CUDA plugin directory
	GhostLimit     uint32   // Ghost file size limit in bytes (default: 512MB for GPU)
	Timeout        uint32   // CRIU timeout in seconds
	ExternalMounts []string // Additional external mount mappings
}

// Watcher watches for pods with checkpoint labels and triggers checkpoints
type Watcher struct {
	config          Config
	clientset       kubernetes.Interface
	discoveryClient *checkpointk8s.DiscoveryClient
	checkpointer    *checkpoint.Checkpointer
	log             *logrus.Entry

	// Track pods checkpoint status: "in_progress", "completed", or "" (not started/failed)
	checkpointed   map[string]string
	checkpointedMu sync.RWMutex

	stopCh chan struct{}
}

// NewWatcher creates a new pod watcher
func NewWatcher(cfg Config, discoveryClient *checkpointk8s.DiscoveryClient, checkpointer *checkpoint.Checkpointer) (*Watcher, error) {
	// Create in-cluster Kubernetes client
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	return &Watcher{
		config:          cfg,
		clientset:       clientset,
		discoveryClient: discoveryClient,
		checkpointer:    checkpointer,
		log:             logrus.WithField("component", "watcher"),
		checkpointed:    make(map[string]string),
		stopCh:          make(chan struct{}),
	}, nil
}

// Start begins watching for pods and starts the health check server
func (w *Watcher) Start(ctx context.Context) error {
	w.log.WithFields(logrus.Fields{
		"node":            w.config.NodeName,
		"label":           LabelCheckpointSource,
		"signal_file_env": EnvCheckpointSignalFile,
	}).Info("Starting pod watcher")

	// Start health check HTTP server if address is configured
	if w.config.ListenAddr != "" {
		httpServer := w.startHealthServer(ctx)
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			httpServer.Shutdown(shutdownCtx)
		}()
	}

	// Create informer factory with label selector and optional namespace restriction
	labelSelector := labels.SelectorFromSet(labels.Set{
		LabelCheckpointSource: "true",
	}).String()

	factoryOptions := []informers.SharedInformerOption{
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.LabelSelector = labelSelector
		}),
	}

	// If namespace is specified, restrict watching to that namespace
	if w.config.RestrictedNamespace != "" {
		w.log.WithField("namespace", w.config.RestrictedNamespace).Info("Restricting pod watching to namespace")
		factoryOptions = append(factoryOptions, informers.WithNamespace(w.config.RestrictedNamespace))
	} else {
		w.log.Info("Watching pods cluster-wide (all namespaces)")
	}

	factory := informers.NewSharedInformerFactoryWithOptions(
		w.clientset,
		30*time.Second,
		factoryOptions...,
	)

	podInformer := factory.Core().V1().Pods().Informer()

	// Add event handlers
	podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod := obj.(*corev1.Pod)
			w.handlePodEvent(ctx, pod)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			pod := newObj.(*corev1.Pod)
			w.handlePodEvent(ctx, pod)
		},
	})

	// Start informer
	go factory.Start(w.stopCh)

	// Wait for cache sync
	if !cache.WaitForCacheSync(w.stopCh, podInformer.HasSynced) {
		return fmt.Errorf("failed to sync informer cache")
	}

	w.log.Info("Pod watcher started and cache synced")

	// Wait for context cancellation
	<-ctx.Done()
	close(w.stopCh)

	return nil
}

// HealthResponse is the response for health check endpoint
type HealthResponse struct {
	Status   string `json:"status"`
	NodeName string `json:"node_name"`
}

// startHealthServer starts an HTTP server for health checks
func (w *Watcher) startHealthServer(ctx context.Context) *http.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(rw http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(rw, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		json.NewEncoder(rw).Encode(HealthResponse{
			Status:   "healthy",
			NodeName: w.config.NodeName,
		})
	})

	server := &http.Server{
		Addr:         w.config.ListenAddr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		w.log.WithField("addr", w.config.ListenAddr).Info("Starting health check server")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			w.log.WithError(err).Error("Health check server error")
		}
	}()

	return server
}

// Stop stops the watcher
func (w *Watcher) Stop() {
	close(w.stopCh)
}

// handlePodEvent processes a pod event
func (w *Watcher) handlePodEvent(ctx context.Context, pod *corev1.Pod) {
	// Filter to pods on this node
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	// Check if pod is Ready
	if !w.isPodReady(pod) {
		return
	}

	// Check if we've already checkpointed this pod
	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	// Get checkpoint ID from label (uses the checkpoint hash)
	checkpointID, ok := pod.Labels[LabelCheckpointHash]
	if !ok || checkpointID == "" {
		w.log.WithField("pod", podKey).Warn("Pod has checkpoint label but no checkpoint-hash label")
		return
	}

	// Check if checkpoint is already in progress or completed for this pod
	w.checkpointedMu.Lock()
	status := w.checkpointed[podKey]
	if status == "completed" || status == "in_progress" {
		w.checkpointedMu.Unlock()
		return
	}
	// Mark as in_progress to prevent concurrent attempts
	w.checkpointed[podKey] = "in_progress"
	w.checkpointedMu.Unlock()

	// Trigger checkpoint
	w.log.WithFields(logrus.Fields{
		"pod":           podKey,
		"checkpoint_id": checkpointID,
	}).Info("Pod ready, triggering checkpoint")

	go w.doCheckpoint(ctx, pod, checkpointID, podKey)
}

// isPodReady checks if all containers in the pod are ready
func (w *Watcher) isPodReady(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}

	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}

	return false
}

// doCheckpoint performs the checkpoint and writes the signal file
func (w *Watcher) doCheckpoint(ctx context.Context, pod *corev1.Pod, checkpointID, podKey string) {
	log := w.log.WithFields(logrus.Fields{
		"pod":           podKey,
		"checkpoint_id": checkpointID,
	})

	// Find the main container and get signal file path from env
	var containerID string
	var signalFilePath string
	for _, container := range pod.Spec.Containers {
		if container.Name == "main" || len(pod.Spec.Containers) == 1 {
			// Get signal file path from environment
			for _, env := range container.Env {
				if env.Name == EnvCheckpointSignalFile {
					signalFilePath = env.Value
					break
				}
			}
			break
		}
	}

	// Get container ID from status
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == "main" || len(pod.Status.ContainerStatuses) == 1 {
			// Remove containerd:// prefix
			containerID = cs.ContainerID
			if len(containerID) > 13 && containerID[:13] == "containerd://" {
				containerID = containerID[13:]
			}
			break
		}
	}

	if containerID == "" {
		log.Error("Could not find container ID")
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	if signalFilePath == "" {
		log.Warn("No DYN_CHECKPOINT_SIGNAL_FILE env var found, signal file will not be written")
	}

	log.WithFields(logrus.Fields{
		"container_id":     containerID,
		"signal_file_path": signalFilePath,
	}).Info("Found container, starting checkpoint")

	// Resolve container to get PID for signal file writing
	containerInfo, err := w.discoveryClient.ResolveContainer(ctx, containerID)
	if err != nil {
		log.WithError(err).Error("Failed to resolve container")
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	// Perform checkpoint
	opts := checkpoint.Options{
		ContainerID:    containerID,
		CheckpointID:   checkpointID,
		CheckpointDir:  w.config.CheckpointDir,
		NodeName:       w.config.NodeName,
		PodName:        pod.Name,
		PodNamespace:   pod.Namespace,
		CUDAPluginDir:  w.config.CUDAPluginDir,
		GhostLimit:     w.config.GhostLimit,
		Timeout:        w.config.Timeout,
		ExternalMounts: w.config.ExternalMounts,
	}

	result, err := w.checkpointer.Checkpoint(ctx, opts)
	if err != nil {
		log.WithError(err).Error("Checkpoint failed")
		// Write failure marker to PVC so restore pods know checkpoint failed
		checkpointDir := filepath.Join(w.config.CheckpointDir, checkpointID)
		w.writeCheckpointDoneMarker(checkpointDir, checkpointID, false, err.Error(), log)
		if signalFilePath != "" {
			w.writeSignalFileToPod(int(containerInfo.PID), signalFilePath, checkpointID, "", false, err.Error())
		}
		// Clear the in_progress status so checkpoint can be retried
		w.checkpointedMu.Lock()
		delete(w.checkpointed, podKey)
		w.checkpointedMu.Unlock()
		return
	}

	log.WithField("checkpoint_dir", result.CheckpointDir).Info("Checkpoint completed successfully")

	// Write checkpoint.done marker to PVC for cross-node restore detection
	// This is written AFTER rootfs-diff.tar is complete, so it's safe to use as a completion marker
	w.writeCheckpointDoneMarker(result.CheckpointDir, checkpointID, true, "", log)

	// Write signal file to pod's hostPath for checkpoint job pod to exit
	if signalFilePath != "" {
		w.writeSignalFileToPod(int(containerInfo.PID), signalFilePath, checkpointID, result.CheckpointDir, true, "")
	}

	// Mark as completed so we don't checkpoint again
	w.checkpointedMu.Lock()
	w.checkpointed[podKey] = "completed"
	w.checkpointedMu.Unlock()
}

// writeSignalFileToPod writes a signal file to the checkpointed pod's filesystem
// via /proc/<pid>/root to indicate checkpoint completion
func (w *Watcher) writeSignalFileToPod(pid int, signalFilePath, checkpointID, checkpointPath string, success bool, errMsg string) {
	signal := SignalFile{
		CheckpointID:   checkpointID,
		CheckpointPath: checkpointPath,
		Timestamp:      time.Now().UTC(),
		Success:        success,
		Error:          errMsg,
	}

	data, err := json.MarshalIndent(signal, "", "  ")
	if err != nil {
		w.log.WithError(err).Error("Failed to marshal signal file")
		return
	}

	// Write to the pod's filesystem via /proc/<pid>/root
	// signalFilePath is the path inside the pod (e.g., /var/lib/dynamo-checkpoint/signal.done)
	hostSignalPath := fmt.Sprintf("%s/%d/root%s", w.config.HostProc, pid, signalFilePath)

	// Ensure signal directory exists in pod's filesystem
	signalDir := filepath.Dir(hostSignalPath)
	if err := os.MkdirAll(signalDir, 0755); err != nil {
		w.log.WithError(err).WithField("path", signalDir).Error("Failed to create signal directory in pod")
		return
	}

	if err := os.WriteFile(hostSignalPath, data, 0644); err != nil {
		w.log.WithError(err).WithField("path", hostSignalPath).Error("Failed to write signal file to pod")
		return
	}

	w.log.WithFields(logrus.Fields{
		"host_path": hostSignalPath,
		"pod_path":  signalFilePath,
		"pid":       pid,
		"success":   success,
	}).Info("Signal file written to pod filesystem")
}

// writeCheckpointDoneMarker writes a checkpoint.done marker file to the checkpoint directory on shared PVC.
// This file is written AFTER all checkpoint steps complete (including rootfs-diff.tar).
// Restore pods on ANY node check for this file to know the checkpoint is complete and safe to restore.
// This is separate from writeSignalFileToPod which signals the checkpoint job pod to exit.
func (w *Watcher) writeCheckpointDoneMarker(checkpointDir, checkpointID string, success bool, errMsg string, log *logrus.Entry) {
	markerPath := filepath.Join(checkpointDir, "checkpoint.done")

	marker := SignalFile{
		CheckpointID:   checkpointID,
		CheckpointPath: checkpointDir,
		Timestamp:      time.Now().UTC(),
		Success:        success,
		Error:          errMsg,
	}

	data, err := json.MarshalIndent(marker, "", "  ")
	if err != nil {
		log.WithError(err).Error("Failed to marshal checkpoint.done marker")
		return
	}

	if err := os.WriteFile(markerPath, data, 0644); err != nil {
		log.WithError(err).WithField("path", markerPath).Error("Failed to write checkpoint.done marker")
		return
	}

	log.WithFields(logrus.Fields{
		"path":    markerPath,
		"success": success,
	}).Info("checkpoint.done marker written to PVC")
}
