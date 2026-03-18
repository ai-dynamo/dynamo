// Package controller implements the node-local control loop inside snapshot-agent.
// It does not own CRDs or replace the operator. Instead it watches pod, job, and
// lease state on the current node and delegates CRIU/CUDA execution to the
// snapshot executor workflows.
package controller

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
	"github.com/google/uuid"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/types"
)

const (
	kubeLabelIsCheckpointSource         = "nvidia.com/snapshot-is-checkpoint-source"
	kubeLabelCheckpointHash             = "nvidia.com/snapshot-checkpoint-hash"
	kubeLabelIsRestoreTarget            = "nvidia.com/snapshot-is-restore-target"
	kubeAnnotationCheckpointLocation    = "nvidia.com/snapshot-checkpoint-location"
	kubeAnnotationCheckpointStorageType = "nvidia.com/snapshot-checkpoint-storage-type"
	kubeAnnotationCheckpointStatus      = "nvidia.com/snapshot-checkpoint-status"
	kubeAnnotationRestoreStatus         = "nvidia.com/snapshot-restore-status"
	kubeAnnotationRestoreContainerID    = "nvidia.com/snapshot-restore-container-id"
)

// NodeController watches local-node pods with checkpoint metadata and reconciles
// snapshot execution for checkpoint and restore requests.
type NodeController struct {
	config     *types.AgentConfig
	clientset  kubernetes.Interface
	containerd *containerd.Client
	log        logr.Logger
	holderID   string

	inFlight   map[string]struct{}
	inFlightMu sync.Mutex

	stopCh chan struct{}
}

// NewNodeController creates the node-local controller that runs inside snapshot-agent.
func NewNodeController(
	cfg *types.AgentConfig,
	containerd *containerd.Client,
	log logr.Logger,
) (*NodeController, error) {
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	return &NodeController{
		config:     cfg,
		clientset:  clientset,
		containerd: containerd,
		log:        log,
		holderID:   "snapshot-agent/" + uuid.NewString(),
		inFlight:   make(map[string]struct{}),
		stopCh:     make(chan struct{}),
	}, nil
}

// Run starts the local pod informers and processes checkpoint/restore events.
func (w *NodeController) Run(ctx context.Context) error {
	w.log.Info("Starting snapshot node controller",
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
	if _, err := ckptInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod, ok := podFromInformerObj(obj)
			if !ok {
				return
			}
			w.reconcileCheckpointPod(ctx, pod)
		},
		UpdateFunc: func(_, newObj interface{}) {
			pod, ok := podFromInformerObj(newObj)
			if !ok {
				return
			}
			w.reconcileCheckpointPod(ctx, pod)
		},
	}); err != nil {
		return fmt.Errorf("failed to add checkpoint informer handler: %w", err)
	}
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
	if _, err := restoreInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod, ok := podFromInformerObj(obj)
			if !ok {
				return
			}
			w.reconcileRestorePod(ctx, pod)
		},
		UpdateFunc: func(_, newObj interface{}) {
			pod, ok := podFromInformerObj(newObj)
			if !ok {
				return
			}
			w.reconcileRestorePod(ctx, pod)
		},
	}); err != nil {
		return fmt.Errorf("failed to add restore informer handler: %w", err)
	}
	go restoreFactory.Start(w.stopCh)
	syncFuncs = append(syncFuncs, restoreInformer.HasSynced)

	if !cache.WaitForCacheSync(w.stopCh, syncFuncs...) {
		return fmt.Errorf("failed to sync informer caches")
	}

	w.log.Info("Snapshot node controller started and caches synced")
	<-ctx.Done()
	close(w.stopCh)
	return nil
}

func (w *NodeController) reconcileCheckpointPod(ctx context.Context, pod *corev1.Pod) {
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

	job, err := getCheckpointJob(ctx, w.clientset, pod)
	if err != nil {
		w.log.Error(err, "Failed to resolve checkpoint job", "pod", podKey)
		return
	}

	jobStatus := job.Annotations[kubeAnnotationCheckpointStatus]
	if jobStatus == "completed" || jobStatus == "failed" {
		return
	}

	if !w.tryAcquire(podKey) {
		return
	}

	checkpointLocation, checkpointStorageType, err := checkpointStorageFromPod(pod)
	if err != nil {
		w.release(podKey)
		w.log.Error(err, "Checkpoint pod is missing storage metadata", "pod", podKey, "checkpoint_hash", checkpointHash)
		return
	}

	acquiredLease, err := acquireCheckpointLease(ctx, w.clientset, w.log, job, w.holderID)
	if err != nil {
		w.release(podKey)
		w.log.Error(err, "Failed to acquire checkpoint lease", "pod", podKey, "checkpoint_hash", checkpointHash)
		return
	}
	if !acquiredLease {
		w.release(podKey)
		return
	}

	w.log.Info("Pod ready, triggering checkpoint", "pod", podKey, "checkpoint_hash", checkpointHash)
	emitPodEvent(ctx, w.clientset, w.log, pod, "snapshot", corev1.EventTypeNormal, "CheckpointRequested", fmt.Sprintf("Checkpoint requested: %s", checkpointHash))

	go func() {
		if err := w.runCheckpoint(ctx, pod, job, checkpointHash, checkpointLocation, checkpointStorageType, podKey); err != nil {
			opLog := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)
			opLog.Error(err, "Checkpoint controller worker failed")
			emitPodEvent(ctx, w.clientset, opLog, pod, "snapshot", corev1.EventTypeWarning, "CheckpointWorkerFailed", err.Error())
		}
	}()
}

func (w *NodeController) reconcileRestorePod(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	if pod.Status.Phase != corev1.PodRunning {
		return
	}

	if isPodReady(pod) {
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

	checkpointLocation, checkpointStorageType, err := checkpointStorageFromPod(pod)
	if err != nil {
		w.log.Error(err, "Restore pod is missing storage metadata", "pod", podKey, "checkpoint_hash", checkpointHash)
		return
	}
	if _, err := os.Stat(checkpointLocation); os.IsNotExist(err) {
		w.log.V(1).Info("Checkpoint not ready on disk, skipping restore", "pod", podKey, "checkpoint_hash", checkpointHash, "checkpoint_location", checkpointLocation)
		return
	}

	containerName := resolveMainContainerName(pod)
	if containerName == "" {
		w.log.Info("Restore pod has no containers", "pod", podKey)
		return
	}

	containerID := ""
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name != containerName || cs.ContainerID == "" {
			continue
		}
		containerID = strings.TrimPrefix(cs.ContainerID, "containerd://")
		break
	}
	if containerID == "" {
		w.log.V(1).Info("Restore pod has no running main container yet", "pod", podKey, "container", containerName)
		return
	}

	annotationStatus := pod.Annotations[kubeAnnotationRestoreStatus]
	annotationContainerID := pod.Annotations[kubeAnnotationRestoreContainerID]
	if annotationContainerID == containerID && (annotationStatus == "completed" || annotationStatus == "in_progress") {
		return
	}

	restoreAttemptKey := fmt.Sprintf("%s/%s", podKey, containerID)
	if !w.tryAcquire(restoreAttemptKey) {
		return
	}

	w.log.Info("Restore pod running, triggering external restore", "pod", podKey, "checkpoint_hash", checkpointHash)
	emitPodEvent(ctx, w.clientset, w.log, pod, "snapshot", corev1.EventTypeNormal, "RestoreRequested", fmt.Sprintf("Restore requested from checkpoint %s", checkpointHash))

	go func() {
		if err := w.runRestore(ctx, pod, containerName, containerID, checkpointHash, checkpointLocation, checkpointStorageType, restoreAttemptKey); err != nil {
			opLog := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)
			opLog.Error(err, "Restore controller worker failed")
			emitPodEvent(ctx, w.clientset, opLog, pod, "snapshot", corev1.EventTypeWarning, "RestoreWorkerFailed", err.Error())
		}
	}()
}

// runCheckpoint runs the full checkpoint workflow for a pod:
//  1. Hold and renew the checkpoint lease
//  2. Resolve the container ID and host PID
//  3. Call executor.Checkpoint (inspect → configure → CUDA lock/checkpoint → CRIU dump → rootfs diff)
//  4. SIGUSR1 the process on success (notify workload), SIGKILL on failure (terminate immediately)
//  5. Mark job as completed or failed
func (w *NodeController) runCheckpoint(ctx context.Context, pod *corev1.Pod, job *batchv1.Job, checkpointHash, checkpointLocation, checkpointStorageType, podKey string) error {
	releasePodOnExit := true
	defer func() {
		if releasePodOnExit {
			w.release(podKey)
		}
	}()
	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash)
	leaseCtx, stopLease := context.WithCancelCause(ctx)
	defer stopLease(nil)

	releaseLeaseOnExit := true
	defer func() {
		if !releaseLeaseOnExit {
			return
		}
		releaseCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := releaseCheckpointLease(releaseCtx, w.clientset, log, job, w.holderID); err != nil {
			log.Error(err, "Failed to release checkpoint lease")
		}
	}()

	go w.renewCheckpointLease(leaseCtx, log, job, stopLease)

	setCheckpointStatus := func(value string) error {
		if err := annotateJob(ctx, w.clientset, log, job, map[string]string{
			kubeAnnotationCheckpointStatus: value,
		}); err != nil {
			releasePodOnExit = false
			releaseLeaseOnExit = false
			return fmt.Errorf("failed to persist terminal checkpoint status %q: %w", value, err)
		}
		return nil
	}

	// Resolve the target container
	containerName := resolveMainContainerName(pod)
	if containerName == "" {
		err := fmt.Errorf("no containers found in pod spec")
		log.Error(err, "Checkpoint failed")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if statusErr := setCheckpointStatus("failed"); statusErr != nil {
			return statusErr
		}
		return nil
	}
	var containerID string
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName {
			containerID = strings.TrimPrefix(cs.ContainerID, "containerd://")
			break
		}
	}
	if containerID == "" {
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", "Could not resolve target container ID")
		if statusErr := setCheckpointStatus("failed"); statusErr != nil {
			return statusErr
		}
		return nil
	}

	// Resolve the container's host PID (needed for signaling after checkpoint)
	containerPID, _, err := common.ResolveContainer(ctx, w.containerd, containerID)
	if err != nil {
		log.Error(err, "Failed to resolve container")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", fmt.Sprintf("Container resolve failed: %v", err))
		if statusErr := setCheckpointStatus("failed"); statusErr != nil {
			return statusErr
		}
		return nil
	}

	// Step 1: Run the checkpoint orchestrator
	req := executor.CheckpointRequest{
		ContainerID:           containerID,
		ContainerName:         containerName,
		CheckpointHash:        checkpointHash,
		CheckpointLocation:    checkpointLocation,
		CheckpointStorageType: checkpointStorageType,
		NodeName:              w.config.NodeName,
		PodName:               pod.Name,
		PodNamespace:          pod.Namespace,
	}
	if err := executor.Checkpoint(leaseCtx, w.containerd, log, req, w.config); err != nil {
		if cause := context.Cause(leaseCtx); cause != nil && cause != context.Canceled {
			err = fmt.Errorf("checkpoint lease lost: %w", cause)
		}
		log.Error(err, "Checkpoint failed")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		// SIGKILL on failure: process is unrecoverable (CUDA locked), terminate immediately
		if signalErr := common.SendSignalToPID(log, containerPID, syscall.SIGKILL, "checkpoint failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint failure to runtime process")
		}
		if statusErr := setCheckpointStatus("failed"); statusErr != nil {
			return statusErr
		}
		return nil
	}

	info, err := os.Stat(checkpointLocation)
	if err != nil || !info.IsDir() {
		if err == nil {
			err = fmt.Errorf("published checkpoint path %s is not a directory", checkpointLocation)
		} else {
			err = fmt.Errorf("published checkpoint path %s is missing: %w", checkpointLocation, err)
		}
		log.Error(err, "Checkpoint failed verification")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if signalErr := common.SendSignalToPID(log, containerPID, syscall.SIGKILL, "checkpoint verification failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint verification failure to runtime process")
		}
		if statusErr := setCheckpointStatus("failed"); statusErr != nil {
			return statusErr
		}
		return nil
	}

	// Step 2: SIGUSR1 on success: notify the workload that checkpoint completed
	emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeNormal, "CheckpointSucceeded", fmt.Sprintf("Checkpoint completed: %s", checkpointHash))
	if err := common.SendSignalToPID(log, containerPID, syscall.SIGUSR1, "checkpoint complete"); err != nil {
		log.Error(err, "Failed to signal checkpoint completion to runtime process")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if statusErr := setCheckpointStatus("failed"); statusErr != nil {
			return statusErr
		}
		return nil
	}

	if err := setCheckpointStatus("completed"); err != nil {
		return err
	}
	return nil
}

// runRestore runs the full restore workflow for a pod:
//  1. Mark the current container instance as in_progress
//  2. Call executor.Restore (inspect placeholder → nsrestore inside namespace)
//  3. SIGCONT the restored process to wake it up
//  4. Wait for the pod to become Ready
//  5. Mark the container instance as completed
func (w *NodeController) runRestore(ctx context.Context, pod *corev1.Pod, containerName, containerID, checkpointHash, checkpointLocation, checkpointStorageType, restoreAttemptKey string) error {
	releaseOnExit := true
	defer func() {
		if releaseOnExit {
			w.release(restoreAttemptKey)
		}
	}()
	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)
	log := w.log.WithValues("pod", podKey, "checkpoint_hash", checkpointHash, "container_id", containerID)
	setRestoreStatus := func(value string) error {
		annotations := map[string]string{
			kubeAnnotationRestoreStatus:      value,
			kubeAnnotationRestoreContainerID: containerID,
		}
		if err := annotatePod(ctx, w.clientset, log, pod, annotations); err != nil {
			if value == "completed" {
				releaseOnExit = false
				return fmt.Errorf("failed to persist terminal restore status %q: %w", value, err)
			}
			return fmt.Errorf("failed to update restore status %q: %w", value, err)
		}
		return nil
	}

	if err := annotatePod(ctx, w.clientset, log, pod, map[string]string{
		kubeAnnotationRestoreStatus:      "in_progress",
		kubeAnnotationRestoreContainerID: containerID,
	}); err != nil {
		return fmt.Errorf("failed to annotate pod with restore in_progress: %w", err)
	}

	// Step 1: Run the restore orchestrator (inspect + nsrestore)
	req := executor.RestoreRequest{
		CheckpointHash:        checkpointHash,
		CheckpointLocation:    checkpointLocation,
		CheckpointStorageType: checkpointStorageType,
		NSRestorePath:         w.config.Restore.NSRestorePath,
		PodName:               pod.Name,
		PodNamespace:          pod.Namespace,
		ContainerName:         containerName,
	}
	restoredPID, err := executor.Restore(ctx, w.containerd, log, req)
	if err != nil {
		log.Error(err, "External restore failed")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		placeholderHostPID, _, pidErr := common.ResolveContainerByPod(ctx, w.containerd, pod.Name, pod.Namespace, containerName)
		if pidErr != nil {
			releaseOnExit = false
			return fmt.Errorf("restore failed and placeholder PID could not be resolved: %w", pidErr)
		}
		if killErr := common.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore failed"); killErr != nil {
			releaseOnExit = false
			return fmt.Errorf("restore failed and placeholder could not be killed: %w", killErr)
		}
		return nil
	}

	// Step 2: SIGCONT the restored process via PID namespace
	placeholderHostPID, _, err := common.ResolveContainerByPod(ctx, w.containerd, pod.Name, pod.Namespace, containerName)
	if err != nil {
		log.Error(err, "Failed to resolve placeholder host PID for signaling")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		releaseOnExit = false
		return fmt.Errorf("failed to resolve placeholder host PID for signaling: %w", err)
	}
	if err := common.SendSignalViaPIDNamespace(ctx, log, placeholderHostPID, restoredPID, syscall.SIGCONT, "restore complete"); err != nil {
		log.Error(err, "Failed to signal restored runtime process")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		if killErr := common.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore signaling failed"); killErr != nil {
			log.Error(killErr, "Failed to kill placeholder after restore signaling failure")
		}
		releaseOnExit = false
		return fmt.Errorf("failed to signal restored runtime process: %w", err)
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
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		if killErr := common.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore readiness failed"); killErr != nil {
			log.Error(killErr, "Failed to kill placeholder after restore readiness failure")
		}
		releaseOnExit = false
		return fmt.Errorf("restore post-signal readiness check failed: %w", err)
	}

	emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeNormal, "RestoreSucceeded", fmt.Sprintf("Restore completed from checkpoint %s", checkpointHash))
	if err := setRestoreStatus("completed"); err != nil {
		return err
	}
	return nil
}

func (w *NodeController) tryAcquire(podKey string) bool {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	if _, held := w.inFlight[podKey]; held {
		return false
	}
	w.inFlight[podKey] = struct{}{}
	return true
}

func (w *NodeController) release(podKey string) {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	delete(w.inFlight, podKey)
}

func checkpointStorageFromPod(pod *corev1.Pod) (string, string, error) {
	checkpointLocation := strings.TrimSpace(pod.Annotations[kubeAnnotationCheckpointLocation])
	if checkpointLocation == "" {
		return "", "", fmt.Errorf("missing %s annotation", kubeAnnotationCheckpointLocation)
	}

	checkpointStorageType := strings.TrimSpace(pod.Annotations[kubeAnnotationCheckpointStorageType])
	if checkpointStorageType == "" {
		return "", "", fmt.Errorf("missing %s annotation", kubeAnnotationCheckpointStorageType)
	}
	if checkpointStorageType != "pvc" {
		return "", "", fmt.Errorf("checkpoint storage type %q is not supported", checkpointStorageType)
	}

	return checkpointLocation, checkpointStorageType, nil
}
