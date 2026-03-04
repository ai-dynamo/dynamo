// Package watcher provides Kubernetes pod watching for automatic checkpoint/restore.
// The watcher is the sole entry point for chrek operations — it detects pods with
// checkpoint/restore labels and calls the orchestrators directly.
package watcher

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/containerd/containerd"
	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/orchestrate"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

const (
	kubeLabelIsCheckpointSource    = "nvidia.com/chrek-is-checkpoint-source"
	kubeLabelCheckpointHash        = "nvidia.com/chrek-checkpoint-hash"
	kubeLabelIsRestoreTarget       = "nvidia.com/chrek-is-restore-target"
	kubeAnnotationCheckpointStatus = "nvidia.com/chrek-checkpoint-status"
	kubeAnnotationRestoreStatus    = "nvidia.com/chrek-restore-status"
)

// Watcher watches for pods with checkpoint/restore labels and triggers operations.
type Watcher struct {
	config     *types.AgentConfig
	clientset  kubernetes.Interface
	containerd *containerd.Client
	log        logr.Logger

	inFlight   map[string]struct{}
	inFlightMu sync.Mutex

	stopCh chan struct{}
}

// NewWatcher creates a new pod watcher.
func NewWatcher(
	cfg *types.AgentConfig,
	containerd *containerd.Client,
	log logr.Logger,
) (*Watcher, error) {
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	return &Watcher{
		config:     cfg,
		clientset:  clientset,
		containerd: containerd,
		log:        log,
		inFlight:        make(map[string]struct{}),
		stopCh:          make(chan struct{}),
	}, nil
}

// Start begins watching for pods and processing checkpoint/restore events.
func (w *Watcher) Start(ctx context.Context) error {
	w.log.Info("Starting pod watcher",
		"node", w.config.NodeName,
		"checkpoint", kubeLabelIsCheckpointSource,
		"restore", kubeLabelIsRestoreTarget,
	)

	var nsOptions []informers.SharedInformerOption
	if w.config.RestrictedNamespace != "" {
		w.log.Info("Restricting pod watching to namespace", "namespace", w.config.RestrictedNamespace)
		nsOptions = append(nsOptions, informers.WithNamespace(w.config.RestrictedNamespace))
	} else {
		w.log.Info("Watching pods cluster-wide (all namespaces)")
	}

	var syncFuncs []cache.InformerSynced

	// Checkpoint informer
	checkpointSelector := labels.SelectorFromSet(labels.Set{
		kubeLabelIsCheckpointSource: "true",
	}).String()

	ckptFactoryOpts := append([]informers.SharedInformerOption{
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.LabelSelector = checkpointSelector
		}),
	}, nsOptions...)

	ckptFactory := informers.NewSharedInformerFactoryWithOptions(
		w.clientset, 30*time.Second, ckptFactoryOpts...,
	)

	ckptInformer := ckptFactory.Core().V1().Pods().Informer()
	ckptInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod, ok := podFromInformerObj(obj)
			if !ok {
				return
			}
			w.handleCheckpointPodEvent(ctx, pod)
		},
		UpdateFunc: func(_, newObj interface{}) {
			pod, ok := podFromInformerObj(newObj)
			if !ok {
				return
			}
			w.handleCheckpointPodEvent(ctx, pod)
		},
	})
	go ckptFactory.Start(w.stopCh)
	syncFuncs = append(syncFuncs, ckptInformer.HasSynced)

	// Restore informer
	restoreSelector := labels.SelectorFromSet(labels.Set{
		kubeLabelIsRestoreTarget: "true",
	}).String()

	restoreFactoryOpts := append([]informers.SharedInformerOption{
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.LabelSelector = restoreSelector
		}),
	}, nsOptions...)

	restoreFactory := informers.NewSharedInformerFactoryWithOptions(
		w.clientset, 30*time.Second, restoreFactoryOpts...,
	)

	restoreInformer := restoreFactory.Core().V1().Pods().Informer()
	restoreInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod, ok := podFromInformerObj(obj)
			if !ok {
				return
			}
			w.handleRestorePodEvent(ctx, pod)
		},
		UpdateFunc: func(_, newObj interface{}) {
			pod, ok := podFromInformerObj(newObj)
			if !ok {
				return
			}
			w.handleRestorePodEvent(ctx, pod)
		},
	})
	go restoreFactory.Start(w.stopCh)
	syncFuncs = append(syncFuncs, restoreInformer.HasSynced)

	if !cache.WaitForCacheSync(w.stopCh, syncFuncs...) {
		return fmt.Errorf("failed to sync informer caches")
	}

	w.log.Info("Pod watcher started and caches synced")
	<-ctx.Done()
	close(w.stopCh)
	return nil
}

func (w *Watcher) handleCheckpointPodEvent(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}
	if !isPodReady(pod) {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	checkpointHash, ok := pod.Labels[kubeLabelCheckpointHash]
	if !ok || checkpointHash == "" {
		w.log.Info("Pod has checkpoint label but no checkpoint-hash label", "pod", podKey)
		return
	}

	annotationStatus := pod.Annotations[kubeAnnotationCheckpointStatus]
	if annotationStatus == "completed" || annotationStatus == "in_progress" {
		return
	}

	if !w.tryAcquire(podKey) {
		return
	}

	w.log.Info("Pod ready, triggering checkpoint", "pod", podKey, "checkpoint_hash", checkpointHash)
	emitPodEvent(ctx, w.clientset, w.log, pod, "chrek", corev1.EventTypeNormal, "CheckpointRequested", fmt.Sprintf("Checkpoint requested: %s", checkpointHash))

	go w.doCheckpoint(ctx, pod, checkpointHash, podKey)
}


func (w *Watcher) handleRestorePodEvent(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	if pod.Status.Phase != corev1.PodRunning {
		return
	}

	annotationStatus := pod.Annotations[kubeAnnotationRestoreStatus]

	if isPodReady(pod) {
		return
	}

	// Restore failures require explicit intervention (new label/update) before retry.
	if annotationStatus == "completed" || annotationStatus == "in_progress" || annotationStatus == "failed" {
		return
	}

	checkpointHash, ok := pod.Labels[kubeLabelCheckpointHash]
	if !ok || checkpointHash == "" {
		w.log.Info("Restore pod has no checkpoint-hash label", "pod", podKey)
		return
	}

	if strings.ContainsAny(checkpointHash, "/\\") || strings.Contains(checkpointHash, "..") || filepath.Clean(checkpointHash) != checkpointHash {
		w.log.Error(fmt.Errorf("invalid checkpoint hash %q", checkpointHash), "Invalid checkpoint hash on restore pod", "pod", podKey)
		return
	}

	checkpointDir := filepath.Join(w.config.BasePath, checkpointHash)
	if _, err := os.Stat(checkpointDir); os.IsNotExist(err) {
		w.log.V(1).Info("Checkpoint not ready on disk, skipping restore", "pod", podKey, "checkpoint_hash", checkpointHash)
		return
	}

	if !w.tryAcquire(podKey) {
		return
	}

	w.log.Info("Restore pod running, triggering external restore", "pod", podKey, "checkpoint_hash", checkpointHash)
	emitPodEvent(ctx, w.clientset, w.log, pod, "chrek", corev1.EventTypeNormal, "RestoreRequested", fmt.Sprintf("Restore requested from checkpoint %s", checkpointHash))

	go w.doRestore(ctx, pod, checkpointHash, podKey)
}

// doCheckpoint runs the full checkpoint workflow for a pod:
//  1. Mark pod as in_progress
//  2. Resolve the container ID and host PID
//  3. Call orchestrate.Checkpoint (inspect → configure → CUDA lock/checkpoint → CRIU dump → rootfs diff)
//  4. SIGUSR1 the process on success (notify workload), SIGKILL on failure (terminate immediately)
//  5. Mark pod as completed or failed
func (w *Watcher) doCheckpoint(ctx context.Context, pod *corev1.Pod, checkpointHash, podKey string) {
	defer w.release(podKey)
	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)

	if err := annotatePod(ctx, w.clientset, log, pod, map[string]string{
		kubeAnnotationCheckpointStatus: "in_progress",
	}); err != nil {
		log.Error(err, "Failed to annotate pod with checkpoint in_progress")
		return
	}

	// Resolve the target container
	containerName := resolveMainContainerName(pod)
	if containerName == "" {
		err := fmt.Errorf("no containers found in pod spec")
		log.Error(err, "Checkpoint failed")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationCheckpointStatus: "failed"})
		return
	}
	var containerID string
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName {
			containerID = strings.TrimPrefix(cs.ContainerID, "containerd://")
			break
		}
	}
	if containerID == "" {
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "CheckpointFailed", "Could not resolve target container ID")
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationCheckpointStatus: "failed"})
		return
	}

	// Resolve the container's host PID (needed for signaling after checkpoint)
	containerPID, _, err := common.ResolveContainer(ctx, w.containerd, containerID)
	if err != nil {
		log.Error(err, "Failed to resolve container")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "CheckpointFailed", fmt.Sprintf("Container resolve failed: %v", err))
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationCheckpointStatus: "failed"})
		return
	}

	// Step 1: Run the checkpoint orchestrator
	req := orchestrate.CheckpointRequest{
		ContainerID:    containerID,
		ContainerName:  containerName,
		CheckpointHash: checkpointHash,
		CheckpointDir:  w.config.BasePath,
		NodeName:       w.config.NodeName,
		PodName:        pod.Name,
		PodNamespace:   pod.Namespace,
	}
	if err := orchestrate.Checkpoint(ctx, w.containerd, log, req, w.config); err != nil {
		log.Error(err, "Checkpoint failed")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		// SIGKILL on failure: process is unrecoverable (CUDA locked), terminate immediately
		if signalErr := common.SendSignalToPID(log, containerPID, syscall.SIGKILL, "checkpoint failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint failure to runtime process")
		}
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationCheckpointStatus: "failed"})
		return
	}

	// Step 2: SIGUSR1 on success: notify the workload that checkpoint completed
	emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeNormal, "CheckpointSucceeded", fmt.Sprintf("Checkpoint completed: %s", checkpointHash))
	if err := common.SendSignalToPID(log, containerPID, syscall.SIGUSR1, "checkpoint complete"); err != nil {
		log.Error(err, "Failed to signal checkpoint completion to runtime process")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationCheckpointStatus: "failed"})
		return
	}

	annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationCheckpointStatus: "completed"})
}

// doRestore runs the full restore workflow for a pod:
//  1. Mark pod as in_progress
//  2. Call orchestrate.Restore (inspect placeholder → nsrestore inside namespace)
//  3. SIGCONT the restored process to wake it up
//  4. Wait for the pod to become Ready
//  5. Mark pod as completed or failed
func (w *Watcher) doRestore(ctx context.Context, pod *corev1.Pod, checkpointHash, podKey string) {
	defer w.release(podKey)
	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)

	if err := annotatePod(ctx, w.clientset, log, pod, map[string]string{
		kubeAnnotationRestoreStatus: "in_progress",
	}); err != nil {
		log.Error(err, "Failed to annotate pod with restore in_progress")
		return
	}

	containerName := resolveMainContainerName(pod)
	if containerName == "" {
		err := fmt.Errorf("no containers found in pod spec")
		log.Error(err, "Restore failed")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationRestoreStatus: "failed"})
		return
	}

	// Step 1: Run the restore orchestrator (inspect + nsrestore)
	req := orchestrate.RestoreRequest{
		CheckpointHash: checkpointHash,
		CheckpointBase: w.config.BasePath,
		NSRestorePath:  w.config.Restore.NSRestorePath,
		PodName:        pod.Name,
		PodNamespace:   pod.Namespace,
		ContainerName:  containerName,
	}
	restoredPID, err := orchestrate.Restore(ctx, w.containerd, log, req)
	if err != nil {
		log.Error(err, "External restore failed")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationRestoreStatus: "failed"})
		return
	}

	// Step 2: SIGCONT the restored process via PID namespace
	placeholderHostPID, _, err := common.ResolveContainerByPod(ctx, w.containerd, pod.Name, pod.Namespace, containerName)
	if err != nil {
		log.Error(err, "Failed to resolve placeholder host PID for signaling")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationRestoreStatus: "failed"})
		return
	}
	if err := common.SendSignalViaPIDNamespace(ctx, log, placeholderHostPID, restoredPID, syscall.SIGCONT, "restore complete"); err != nil {
		log.Error(err, "Failed to signal restored runtime process")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationRestoreStatus: "failed"})
		return
	}

	// Step 3: Wait for the pod to become Ready
	readyCtx := ctx
	if timeout := w.config.Restore.RestoreReadyTimeout(); timeout > 0 {
		var cancel context.CancelFunc
		readyCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	if err := waitForPodReady(readyCtx, w.clientset, pod.Namespace, pod.Name, containerName); err != nil {
		log.Error(err, "Restore post-signal readiness check failed")
		emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationRestoreStatus: "failed"})
		return
	}

	emitPodEvent(ctx, w.clientset, log, pod, "chrek", corev1.EventTypeNormal, "RestoreSucceeded", fmt.Sprintf("Restore completed from checkpoint %s", checkpointHash))
	annotatePod(ctx, w.clientset, log, pod, map[string]string{kubeAnnotationRestoreStatus: "completed"})
}

func (w *Watcher) tryAcquire(podKey string) bool {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	if _, held := w.inFlight[podKey]; held {
		return false
	}
	w.inFlight[podKey] = struct{}{}
	return true
}

func (w *Watcher) release(podKey string) {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	delete(w.inFlight, podKey)
}



