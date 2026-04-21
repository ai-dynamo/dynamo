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

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
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
		"checkpoint", snapshotprotocol.CheckpointSourceLabel,
		"restore", snapshotprotocol.RestoreTargetLabel,
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
		snapshotprotocol.CheckpointSourceLabel: "true",
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
		snapshotprotocol.RestoreTargetLabel: "true",
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

	checkpointID, ok := pod.Labels[snapshotprotocol.CheckpointIDLabel]
	if !ok || checkpointID == "" {
		w.log.Info("Pod has checkpoint label but no checkpoint-id label", "pod", podKey)
		return
	}

	containerNames, err := snapshotprotocol.ParseCheckpointContainers(pod.Annotations)
	if err != nil {
		w.log.Error(err, "Checkpoint pod has invalid container list annotation", "pod", podKey)
		return
	}

	job, err := getCheckpointJob(ctx, w.clientset, pod)
	if err != nil {
		w.log.Error(err, "Failed to resolve checkpoint job", "pod", podKey)
		return
	}

	jobStatus := job.Annotations[snapshotprotocol.CheckpointJobStatusAnnotation]
	if jobStatus == snapshotprotocol.CheckpointStatusCompleted || jobStatus == snapshotprotocol.CheckpointStatusFailed {
		return
	}

	if !w.tryAcquire(podKey) {
		return
	}

	podCheckpointRoot, err := w.checkpointLocationFromPod(pod, checkpointID)
	if err != nil {
		w.release(podKey)
		w.log.Error(err, "Checkpoint pod is missing storage metadata", "pod", podKey, "checkpoint_id", checkpointID)
		return
	}

	acquiredLease, err := acquireCheckpointLease(ctx, w.clientset, w.log, job, w.holderID)
	if err != nil {
		w.release(podKey)
		w.log.Error(err, "Failed to acquire checkpoint lease", "pod", podKey, "checkpoint_id", checkpointID)
		return
	}
	if !acquiredLease {
		w.release(podKey)
		return
	}

	startedAt := time.Now()
	w.log.Info("Checkpoint target detected, triggering checkpoint",
		"pod", podKey, "checkpoint_id", checkpointID, "containers", containerNames)
	emitPodEvent(ctx, w.clientset, w.log, pod, "snapshot", corev1.EventTypeNormal, "CheckpointRequested", fmt.Sprintf("Checkpoint requested: %s", checkpointID))

	go func() {
		if err := w.runCheckpoint(ctx, pod, job, checkpointID, podCheckpointRoot, containerNames, podKey, startedAt); err != nil {
			opLog := w.log.WithValues("pod", podKey, "checkpoint_id", checkpointID)
			opLog.Error(err, "Checkpoint controller worker failed")
			emitPodEvent(ctx, w.clientset, opLog, pod, "snapshot", corev1.EventTypeWarning, "CheckpointWorkerFailed", err.Error())
		}
	}()
}

func (w *NodeController) reconcileRestorePod(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}
	if pod.Status.Phase != corev1.PodRunning {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	checkpointID, ok := pod.Labels[snapshotprotocol.CheckpointIDLabel]
	if !ok || checkpointID == "" {
		w.log.Info("Restore pod has no checkpoint-id label", "pod", podKey)
		return
	}
	if strings.ContainsAny(checkpointID, "/\\") || strings.Contains(checkpointID, "..") || filepath.Clean(checkpointID) != checkpointID {
		w.log.Error(fmt.Errorf("invalid checkpoint id %q", checkpointID), "Invalid checkpoint id on restore pod", "pod", podKey)
		return
	}

	containerNames, err := snapshotprotocol.ParseCheckpointContainers(pod.Annotations)
	if err != nil {
		w.log.Error(err, "Restore pod has invalid container list annotation", "pod", podKey)
		return
	}

	podCheckpointRoot, err := w.checkpointLocationFromPod(pod, checkpointID)
	if err != nil {
		w.log.Error(err, "Restore pod is missing storage metadata", "pod", podKey, "checkpoint_id", checkpointID)
		return
	}
	if _, err := os.Stat(podCheckpointRoot); os.IsNotExist(err) {
		w.log.V(1).Info("Checkpoint not ready on disk, skipping restore",
			"pod", podKey, "checkpoint_id", checkpointID, "pod_checkpoint_root", podCheckpointRoot)
		return
	}

	containerIDByName := make(map[string]string, len(pod.Status.ContainerStatuses))
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.ContainerID == "" {
			continue
		}
		containerIDByName[cs.Name] = strings.TrimPrefix(cs.ContainerID, "containerd://")
	}

	for _, containerName := range containerNames {
		containerID := containerIDByName[containerName]
		if containerID == "" {
			w.log.V(1).Info("Restore pod container not running yet", "pod", podKey, "container", containerName)
			continue
		}

		statusKey := snapshotprotocol.RestoreStatusAnnotationPrefix + containerName
		idKey := snapshotprotocol.RestoreContainerIDAnnotationPrefix + containerName
		annotationStatus := pod.Annotations[statusKey]
		annotationContainerID := pod.Annotations[idKey]
		if annotationContainerID == containerID &&
			(annotationStatus == snapshotprotocol.RestoreStatusCompleted || annotationStatus == snapshotprotocol.RestoreStatusFailed) {
			continue
		}

		restoreAttemptKey := fmt.Sprintf("%s/%s", podKey, containerID)
		if !w.tryAcquire(restoreAttemptKey) {
			continue
		}

		containerCheckpointPath := snapshotprotocol.ContainerCheckpointPath(podCheckpointRoot, containerName)
		if _, err := os.Stat(containerCheckpointPath); os.IsNotExist(err) {
			w.release(restoreAttemptKey)
			w.log.V(1).Info("Container checkpoint not ready on disk",
				"pod", podKey, "container", containerName, "path", containerCheckpointPath)
			continue
		}

		startedAt := time.Now()
		w.log.Info("Restore target detected, triggering external restore",
			"pod", podKey, "container", containerName, "checkpoint_id", checkpointID)
		emitPodEvent(ctx, w.clientset, w.log, pod, "snapshot", corev1.EventTypeNormal, "RestoreRequested",
			fmt.Sprintf("Restore requested for container %s from checkpoint %s", containerName, checkpointID))

		go func(name, id, path string, startedAt time.Time) {
			if err := w.runRestore(ctx, pod, name, id, checkpointID, path, restoreAttemptKey, startedAt); err != nil {
				opLog := w.log.WithValues("pod", podKey, "container", name, "checkpoint_id", checkpointID)
				opLog.Error(err, "Restore controller worker failed")
				emitPodEvent(ctx, w.clientset, opLog, pod, "snapshot", corev1.EventTypeWarning, "RestoreWorkerFailed", err.Error())
			}
		}(containerName, containerID, containerCheckpointPath, startedAt)
	}
}

// runCheckpoint drives the full checkpoint workflow for a pod:
//  1. Hold and renew the pod-scoped checkpoint lease
//  2. For each listed workload container (sequentially, to avoid CRIU/CUDA
//     contention): resolve containerID + PID, call executor.Checkpoint,
//     write the per-container snapshot-complete sentinel or SIGKILL on
//     failure, annotate the pod with per-container status.
//  3. Write the aggregate job-level status annotation on the batch/v1 Job.
func (w *NodeController) runCheckpoint(
	ctx context.Context,
	pod *corev1.Pod,
	job *batchv1.Job,
	checkpointID string,
	podCheckpointRoot string,
	containerNames []string,
	podKey string,
	startedAt time.Time,
) error {
	releasePodOnExit := true
	defer func() {
		if releasePodOnExit {
			w.release(podKey)
		}
	}()
	log := w.log.WithValues("pod", podKey, "checkpoint_id", checkpointID)
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

	containerIDByName := make(map[string]string, len(pod.Status.ContainerStatuses))
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.ContainerID == "" {
			continue
		}
		containerIDByName[cs.Name] = strings.TrimPrefix(cs.ContainerID, "containerd://")
	}

	allCompleted := true
	for _, containerName := range containerNames {
		containerLog := log.WithValues("container", containerName)
		ok, err := w.runContainerCheckpoint(
			leaseCtx, ctx, pod, checkpointID, podCheckpointRoot,
			containerName, containerIDByName[containerName], startedAt, containerLog,
		)
		if err != nil {
			releasePodOnExit = false
			releaseLeaseOnExit = false
			return err
		}
		if !ok {
			allCompleted = false
		}
	}

	jobStatus := snapshotprotocol.CheckpointStatusCompleted
	if !allCompleted {
		jobStatus = snapshotprotocol.CheckpointStatusFailed
	}
	if err := annotateJob(ctx, w.clientset, log, job, map[string]string{
		snapshotprotocol.CheckpointJobStatusAnnotation: jobStatus,
	}); err != nil {
		releasePodOnExit = false
		releaseLeaseOnExit = false
		return fmt.Errorf("failed to persist terminal job checkpoint status %q: %w", jobStatus, err)
	}
	return nil
}

// runContainerCheckpoint runs the executor for a single workload container.
// Returns (true, nil) on success, (false, nil) on a per-container failure
// that has already been recorded on the pod, or (false, err) if the pod
// annotation itself could not be persisted (caller must keep the lease held
// so the state can converge on retry).
func (w *NodeController) runContainerCheckpoint(
	leaseCtx context.Context,
	ctx context.Context,
	pod *corev1.Pod,
	checkpointID string,
	podCheckpointRoot string,
	containerName string,
	containerID string,
	startedAt time.Time,
	log logr.Logger,
) (bool, error) {
	statusKey := snapshotprotocol.CheckpointStatusAnnotationPrefix + containerName
	setContainerStatus := func(value string) error {
		if err := annotatePod(ctx, w.clientset, log, pod, map[string]string{statusKey: value}); err != nil {
			return fmt.Errorf("failed to persist per-container checkpoint status %q: %w", value, err)
		}
		return nil
	}
	recordFailure := func(reason string, err error) (bool, error) {
		log.Error(err, reason)
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if statusErr := setContainerStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
			return false, statusErr
		}
		return false, nil
	}

	if containerID == "" {
		return recordFailure("Checkpoint failed: could not resolve container ID",
			fmt.Errorf("container %q has no running containerID on pod", containerName))
	}

	containerPID, _, err := snapshotruntime.ResolveContainer(ctx, w.containerd, containerID)
	if err != nil {
		return recordFailure("Failed to resolve container",
			fmt.Errorf("container resolve failed: %w", err))
	}

	containerCheckpointPath := snapshotprotocol.ContainerCheckpointPath(podCheckpointRoot, containerName)
	req := executor.CheckpointRequest{
		ContainerID:        containerID,
		ContainerName:      containerName,
		CheckpointID:       checkpointID,
		CheckpointLocation: containerCheckpointPath,
		PodCheckpointRoot:  podCheckpointRoot,
		StartedAt:          startedAt,
		NodeName:           w.config.NodeName,
		PodName:            pod.Name,
		PodNamespace:       pod.Namespace,
		Clientset:          w.clientset,
	}
	if err := executor.Checkpoint(leaseCtx, w.containerd, log, req, w.config); err != nil {
		if cause := context.Cause(leaseCtx); cause != nil && cause != context.Canceled {
			err = fmt.Errorf("checkpoint lease lost: %w", cause)
		}
		// SIGKILL on failure: process is unrecoverable (CUDA locked), terminate immediately.
		if signalErr := snapshotruntime.SendSignalToPID(log, containerPID, syscall.SIGKILL, "checkpoint failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint failure to runtime process")
		}
		return recordFailure("Checkpoint failed", err)
	}

	info, err := os.Stat(containerCheckpointPath)
	if err != nil || !info.IsDir() {
		if err == nil {
			err = fmt.Errorf("published checkpoint path %s is not a directory", containerCheckpointPath)
		} else {
			err = fmt.Errorf("published checkpoint path %s is missing: %w", containerCheckpointPath, err)
		}
		if signalErr := snapshotruntime.SendSignalToPID(log, containerPID, syscall.SIGKILL, "checkpoint verification failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint verification failure to runtime process")
		}
		return recordFailure("Checkpoint failed verification", err)
	}

	// Sentinel on success. Workload observes via polling on the
	// snapshot-control volume; containerPID is a PID inside the container's
	// mount namespace, which is all the /host/proc/<pid>/root write path
	// requires. The Succeeded event is emitted only after the sentinel has
	// been written so a sentinel-write failure doesn't produce conflicting
	// Succeeded+Failed events for the same operation.
	if err := snapshotruntime.WriteControlSentinel(containerPID, snapshotprotocol.SnapshotCompleteFile); err != nil {
		return recordFailure("Failed to write snapshot-complete sentinel", err)
	}
	emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeNormal, "CheckpointSucceeded",
		fmt.Sprintf("Checkpoint completed for container %s: %s", containerName, checkpointID))

	if err := setContainerStatus(snapshotprotocol.CheckpointStatusCompleted); err != nil {
		return false, err
	}
	return true, nil
}

// runRestore drives the full restore workflow for a single container:
//  1. Annotate per-container in_progress on the pod
//  2. Call executor.Restore for the container's own checkpoint dir
//  3. Write restore-complete sentinel into the container's snapshot-control subPath
//  4. Wait for the container to become Ready within the pod
//  5. Annotate per-container completed / failed
func (w *NodeController) runRestore(
	ctx context.Context,
	pod *corev1.Pod,
	containerName string,
	containerID string,
	checkpointID string,
	containerCheckpointPath string,
	restoreAttemptKey string,
	startedAt time.Time,
) error {
	releaseOnExit := true
	defer func() {
		if releaseOnExit {
			w.release(restoreAttemptKey)
		}
	}()
	restoreCtx := ctx
	if timeout := w.config.Restore.RestoreTimeout(); timeout > 0 {
		var cancel context.CancelFunc
		restoreCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)
	log := w.log.WithValues("pod", podKey, "container", containerName, "checkpoint_id", checkpointID, "container_id", containerID)
	statusKey := snapshotprotocol.RestoreStatusAnnotationPrefix + containerName
	idKey := snapshotprotocol.RestoreContainerIDAnnotationPrefix + containerName
	setRestoreStatus := func(value string) error {
		annotations := map[string]string{
			statusKey: value,
			idKey:     containerID,
		}
		if err := annotatePod(ctx, w.clientset, log, pod, annotations); err != nil {
			if value == snapshotprotocol.RestoreStatusCompleted || value == snapshotprotocol.RestoreStatusFailed {
				releaseOnExit = false
				return fmt.Errorf("failed to persist terminal restore status %q: %w", value, err)
			}
			return fmt.Errorf("failed to update restore status %q: %w", value, err)
		}
		return nil
	}

	if err := setRestoreStatus(snapshotprotocol.RestoreStatusInProgress); err != nil {
		return fmt.Errorf("failed to annotate pod with restore in_progress: %w", err)
	}

	req := executor.RestoreRequest{
		CheckpointID:       checkpointID,
		CheckpointLocation: containerCheckpointPath,
		StartedAt:          startedAt,
		NSRestorePath:      w.config.Restore.NSRestorePath,
		PodName:            pod.Name,
		PodNamespace:       pod.Namespace,
		ContainerName:      containerName,
		Clientset:          w.clientset,
	}
	placeholderHostPID, err := executor.Restore(restoreCtx, w.containerd, log, req)
	if err != nil {
		log.Error(err, "External restore failed")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		if statusErr := setRestoreStatus(snapshotprotocol.RestoreStatusFailed); statusErr != nil {
			return statusErr
		}
		// executor.Restore may have failed before resolving the placeholder; re-resolve.
		placeholderHostPID, _, pidErr := snapshotruntime.ResolveContainerByPod(ctx, w.containerd, pod.Name, pod.Namespace, containerName)
		if pidErr != nil {
			return fmt.Errorf("restore failed and placeholder PID could not be resolved: %w", pidErr)
		}
		if killErr := snapshotruntime.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore failed"); killErr != nil {
			return fmt.Errorf("restore failed and placeholder could not be killed: %w", killErr)
		}
		return nil
	}

	if err := snapshotruntime.WriteControlSentinel(placeholderHostPID, snapshotprotocol.RestoreCompleteFile); err != nil {
		log.Error(err, "Failed to write restore-complete sentinel")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		if statusErr := setRestoreStatus(snapshotprotocol.RestoreStatusFailed); statusErr != nil {
			return statusErr
		}
		if killErr := snapshotruntime.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore sentinel failed"); killErr != nil {
			log.Error(killErr, "Failed to kill placeholder after restore sentinel failure")
		}
		return fmt.Errorf("failed to write restore-complete sentinel: %w", err)
	}

	if err := waitForPodReady(restoreCtx, w.clientset, pod.Namespace, pod.Name, containerName); err != nil {
		log.Error(err, "Restore post-sentinel readiness check failed")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		if statusErr := setRestoreStatus(snapshotprotocol.RestoreStatusFailed); statusErr != nil {
			return statusErr
		}
		if killErr := snapshotruntime.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore readiness failed"); killErr != nil {
			log.Error(killErr, "Failed to kill placeholder after restore readiness failure")
		}
		return fmt.Errorf("restore post-sentinel readiness check failed: %w", err)
	}

	emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeNormal, "RestoreSucceeded",
		fmt.Sprintf("Restore completed for container %s from checkpoint %s", containerName, checkpointID))
	if err := setRestoreStatus(snapshotprotocol.RestoreStatusCompleted); err != nil {
		return err
	}
	return nil
}

func (w *NodeController) tryAcquire(key string) bool {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	if _, held := w.inFlight[key]; held {
		return false
	}
	w.inFlight[key] = struct{}{}
	return true
}

func (w *NodeController) release(key string) {
	w.inFlightMu.Lock()
	defer w.inFlightMu.Unlock()
	delete(w.inFlight, key)
}

// checkpointLocationFromPod returns the pod-scoped checkpoint root on shared
// storage. Per-container artifacts live at ContainerCheckpointPath(root, name).
func (w *NodeController) checkpointLocationFromPod(pod *corev1.Pod, checkpointID string) (string, error) {
	resolvedStorage, err := snapshotprotocol.ResolveCheckpointStorage(
		checkpointID,
		strings.TrimSpace(pod.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation]),
		snapshotprotocol.Storage{
			Type:     w.config.Storage.Type,
			BasePath: w.config.Storage.BasePath,
		},
	)
	if err != nil {
		return "", err
	}
	return resolvedStorage.Location, nil
}
