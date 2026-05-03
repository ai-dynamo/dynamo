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
	config    *types.AgentConfig
	clientset kubernetes.Interface
	runtime   snapshotruntime.Runtime
	log       logr.Logger
	holderID  string

	inFlight   map[string]struct{}
	inFlightMu sync.Mutex

	stopCh chan struct{}
}

const (
	gmsCompletionPollInterval        = 50 * time.Millisecond
	restoreContainerResolveInterval  = 50 * time.Millisecond
	restoreContainerResolveTimeout   = 30 * time.Second
	restoreContainerResolveKeySuffix = "resolve"
)

// NewNodeController creates the node-local controller that runs inside snapshot-agent.
func NewNodeController(
	cfg *types.AgentConfig,
	rt snapshotruntime.Runtime,
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
		config:    cfg,
		clientset: clientset,
		runtime:   rt,
		log:       log,
		holderID:  "snapshot-agent/" + uuid.NewString(),
		inFlight:  make(map[string]struct{}),
		stopCh:    make(chan struct{}),
	}, nil
}

// Run starts the local pod informers and processes checkpoint/restore events.
func (w *NodeController) Run(ctx context.Context) error {
	w.log.Info("Starting snapshot node controller",
		"node", w.config.NodeName,
		"checkpoint_source_label", snapshotprotocol.CheckpointSourceLabel,
		"checkpoint_id_label", snapshotprotocol.CheckpointIDLabel,
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

	// Restore informer: pods that carry a checkpoint-id label but are not
	// themselves the checkpoint source. A single set-based selector
	// expresses this without a dedicated "is restore target" boolean
	// label parallel to the target-containers annotation.
	restoreSel, err := labels.Parse(snapshotprotocol.CheckpointIDLabel + ",!" + snapshotprotocol.CheckpointSourceLabel)
	if err != nil {
		return fmt.Errorf("failed to build restore label selector: %w", err)
	}
	restoreSelector := restoreSel.String()

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

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	checkpointID, ok := pod.Labels[snapshotprotocol.CheckpointIDLabel]
	if !ok || checkpointID == "" {
		w.log.Info("Pod has checkpoint label but no checkpoint-id label", "pod", podKey)
		return
	}

	job, err := getCheckpointJob(ctx, w.clientset, pod)
	if err != nil {
		w.log.Error(err, "Failed to resolve checkpoint job", "pod", podKey)
		return
	}

	jobStatus := job.Annotations[snapshotprotocol.CheckpointStatusAnnotation]
	if jobStatus == snapshotprotocol.CheckpointStatusCompleted || jobStatus == snapshotprotocol.CheckpointStatusFailed {
		return
	}

	// Checkpoint contract: exactly one target container per job.
	targets, err := snapshotprotocol.TargetContainersFromAnnotations(pod.Annotations, 1, 1)
	if err != nil {
		w.log.Error(err, "Checkpoint pod missing target-containers annotation", "pod", podKey)
		return
	}
	containerName := targets[0]
	if !isContainerReady(pod, containerName) {
		return
	}

	if !w.tryAcquire(podKey) {
		return
	}

	checkpointLocation, err := w.checkpointLocationFromPod(pod, checkpointID)
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
	w.log.Info("Checkpoint target detected, triggering checkpoint", "pod", podKey, "checkpoint_id", checkpointID)
	emitPodEvent(ctx, w.clientset, w.log, pod, "snapshot", corev1.EventTypeNormal, "CheckpointRequested", fmt.Sprintf("Checkpoint requested: %s", checkpointID))

	go func() {
		if err := w.runCheckpoint(ctx, pod, job, checkpointID, containerName, checkpointLocation, podKey, startedAt); err != nil {
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

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	if pod.Status.Phase != corev1.PodRunning {
		return
	}

	checkpointID, ok := pod.Labels[snapshotprotocol.CheckpointIDLabel]
	if !ok || checkpointID == "" {
		w.log.Info("Restore pod has no checkpoint-id label", "pod", podKey)
		return
	}

	if strings.ContainsAny(checkpointID, "/\\") || strings.Contains(checkpointID, "..") || filepath.Clean(checkpointID) != checkpointID {
		w.log.Error(fmt.Errorf("invalid checkpoint id %q", checkpointID), "Invalid checkpoint id on restore pod", "pod", podKey)
		return
	}

	checkpointLocation, err := w.checkpointLocationFromPod(pod, checkpointID)
	if err != nil {
		w.log.Error(err, "Restore pod is missing storage metadata", "pod", podKey, "checkpoint_id", checkpointID)
		return
	}
	if _, err := os.Stat(checkpointLocation); os.IsNotExist(err) {
		w.log.V(1).Info("Checkpoint not ready on disk, skipping restore", "pod", podKey, "checkpoint_id", checkpointID, "checkpoint_location", checkpointLocation)
		return
	}

	restoreMode := strings.TrimSpace(pod.Annotations[snapshotprotocol.RestoreModeAnnotation])
	restoreTrigger := strings.TrimSpace(pod.Annotations[snapshotprotocol.RestoreTriggerAnnotation])
	if restoreMode == snapshotprotocol.RestoreModeManual && restoreTrigger == "" {
		w.log.V(1).Info("Restore pod is waiting for manual trigger", "pod", podKey)
		return
	}

	targets, err := snapshotprotocol.TargetContainersFromAnnotations(pod.Annotations, 1, 0)
	if err != nil {
		w.log.Error(err, "Restore pod missing target-containers annotation", "pod", podKey)
		return
	}
	for _, containerName := range targets {
		if _, err := snapshotprotocol.RestoreStatusAnnotationKeysFor(containerName); err != nil {
			w.log.Error(
				err,
				"Restore target container name cannot be used in restore status annotation key",
				"pod", podKey,
				"container", containerName,
			)
			return
		}
	}

	// For each target container: check if kubelet has a running container
	// for it, check the per-container status annotation, and schedule a
	// goroutine when (a) we see a fresh containerd ID and (b) the status is
	// not already terminal for that ID.
	for _, containerName := range targets {
		w.maybeStartRestoreForContainer(ctx, pod, containerName, checkpointID, checkpointLocation, podKey)
	}
}

// maybeStartRestoreForContainer handles one named target container in a
// restore pod. It is a no-op when the container has not been scheduled by
// kubelet yet, or when the agent has already recorded a terminal status for
// the current containerd container ID.
func (w *NodeController) maybeStartRestoreForContainer(
	ctx context.Context,
	pod *corev1.Pod,
	containerName string,
	checkpointID string,
	checkpointLocation string,
	podKey string,
) {
	containerID := restoreContainerIDFromStatus(pod, containerName)
	if containerID == "" {
		w.maybeStartRestoreContainerResolver(ctx, pod.DeepCopy(), containerName, checkpointID, checkpointLocation, podKey)
		return
	}

	w.startRestoreForContainer(ctx, pod, containerName, containerID, checkpointID, checkpointLocation, podKey)
}

func restoreContainerIDFromStatus(pod *corev1.Pod, containerName string) string {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName && cs.ContainerID != "" {
			return snapshotruntime.StripCRIScheme(cs.ContainerID)
		}
	}
	return ""
}

func (w *NodeController) maybeStartRestoreContainerResolver(
	ctx context.Context,
	pod *corev1.Pod,
	containerName string,
	checkpointID string,
	checkpointLocation string,
	podKey string,
) {
	resolveKey := fmt.Sprintf("%s/%s/%s", podKey, containerName, restoreContainerResolveKeySuffix)
	if !w.tryAcquire(resolveKey) {
		return
	}

	w.log.V(1).Info("Restore pod has no running container in Kubernetes status yet; resolving via node runtime",
		"pod", podKey,
		"container", containerName,
	)

	go func() {
		defer w.release(resolveKey)

		resolveCtx, cancel := context.WithTimeout(ctx, restoreContainerResolveTimeout)
		defer cancel()

		ticker := time.NewTicker(restoreContainerResolveInterval)
		defer ticker.Stop()

		for {
			containerID, err := w.runtime.ResolveContainerIDByPod(resolveCtx, pod.Name, pod.Namespace, containerName)
			if err == nil && containerID != "" {
				w.startRestoreForContainer(ctx, pod, containerName, containerID, checkpointID, checkpointLocation, podKey)
				return
			}

			select {
			case <-resolveCtx.Done():
				w.log.V(1).Info("Timed out resolving restore container via node runtime",
					"pod", podKey,
					"container", containerName,
					"err", err,
				)
				return
			case <-ticker.C:
			}
		}
	}()
}

func (w *NodeController) startRestoreForContainer(
	ctx context.Context,
	pod *corev1.Pod,
	containerName string,
	containerID string,
	checkpointID string,
	checkpointLocation string,
	podKey string,
) {
	annotationKeys, err := snapshotprotocol.RestoreStatusAnnotationKeysFor(containerName)
	if err != nil {
		w.log.Error(err, "Restore target container name cannot be used in restore status annotation key", "pod", podKey, "container", containerName)
		return
	}

	restoreTrigger := strings.TrimSpace(pod.Annotations[snapshotprotocol.RestoreTriggerAnnotation])
	if strings.TrimSpace(pod.Annotations[snapshotprotocol.RestoreModeAnnotation]) == snapshotprotocol.RestoreModeManual {
		if restoreTrigger == "" {
			w.log.V(1).Info("Restore target is waiting for manual trigger",
				"pod", podKey,
				"container", containerName,
			)
			return
		}
		if strings.TrimSpace(pod.Annotations[annotationKeys.ProcessedTrigger]) == restoreTrigger {
			return
		}
	}

	annotationStatus := pod.Annotations[annotationKeys.Status]
	annotationContainerID := pod.Annotations[annotationKeys.ContainerID]
	if annotationContainerID == containerID && (annotationStatus == snapshotprotocol.RestoreStatusCompleted || annotationStatus == snapshotprotocol.RestoreStatusFailed) {
		return
	}

	restoreAttemptKey := fmt.Sprintf("%s/%s/%s", podKey, containerName, containerID)
	if !w.tryAcquire(restoreAttemptKey) {
		return
	}

	startedAt := time.Now()
	w.log.Info("Restore target detected, triggering external restore",
		"pod", podKey,
		"checkpoint_id", checkpointID,
		"container", containerName,
		"trigger", restoreTrigger,
	)
	emitPodEvent(ctx, w.clientset, w.log, pod, "snapshot", corev1.EventTypeNormal, "RestoreRequested", fmt.Sprintf("Restore requested from checkpoint %s for container %s", checkpointID, containerName))

	go func() {
		if err := w.runRestore(ctx, pod, containerName, containerID, checkpointID, checkpointLocation, restoreTrigger, restoreAttemptKey, startedAt); err != nil {
			opLog := w.log.WithValues("pod", podKey, "checkpoint_id", checkpointID, "container", containerName)
			opLog.Error(err, "Restore controller worker failed")
			emitPodEvent(ctx, w.clientset, opLog, pod, "snapshot", corev1.EventTypeWarning, "RestoreWorkerFailed", err.Error())
		}
	}()
}

// runCheckpoint runs the full checkpoint workflow for a pod:
//  1. Hold and renew the checkpoint lease
//  2. Resolve the container ID and host PID
//  3. Call executor.Checkpoint (inspect → configure → CUDA lock/checkpoint → CRIU dump → rootfs diff)
//  4. Write a snapshot-complete sentinel into the pod's snapshot-control
//     volume on success (observed by the workload via polling), or SIGKILL
//     on failure (unrecoverable CUDA-locked process)
//  5. Mark job as completed or failed
func (w *NodeController) runCheckpoint(ctx context.Context, pod *corev1.Pod, job *batchv1.Job, checkpointID, containerName, checkpointLocation, podKey string, startedAt time.Time) error {
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

	setCheckpointStatus := func(value string) error {
		if err := annotateJob(ctx, w.clientset, log, job, map[string]string{
			snapshotprotocol.CheckpointStatusAnnotation: value,
		}); err != nil {
			releasePodOnExit = false
			releaseLeaseOnExit = false
			return fmt.Errorf("failed to persist terminal checkpoint status %q: %w", value, err)
		}
		return nil
	}

	// Resolve the target container ID from pod status.
	var containerID string
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName {
			containerID = snapshotruntime.StripCRIScheme(cs.ContainerID)
			break
		}
	}
	if containerID == "" {
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", fmt.Sprintf("Could not resolve container %q ID", containerName))
		if statusErr := setCheckpointStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
			return statusErr
		}
		return nil
	}

	// Resolve the container's host PID (needed for signaling after checkpoint)
	containerPID, _, err := w.runtime.ResolveContainer(ctx, containerID)
	if err != nil {
		log.Error(err, "Failed to resolve container")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", fmt.Sprintf("Container resolve failed: %v", err))
		if statusErr := setCheckpointStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
			return statusErr
		}
		return nil
	}

	// Step 1: Run the checkpoint orchestrator
	req := executor.CheckpointRequest{
		ContainerID:        containerID,
		ContainerName:      containerName,
		CheckpointID:       checkpointID,
		CheckpointLocation: checkpointLocation,
		StartedAt:          startedAt,
		NodeName:           w.config.NodeName,
		PodName:            pod.Name,
		PodNamespace:       pod.Namespace,
		Clientset:          w.clientset,
	}
	if err := executor.Checkpoint(leaseCtx, w.runtime, log, req, w.config); err != nil {
		if cause := context.Cause(leaseCtx); cause != nil && cause != context.Canceled {
			err = fmt.Errorf("checkpoint lease lost: %w", cause)
		}
		log.Error(err, "Checkpoint failed")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		// SIGKILL on failure: process is unrecoverable (CUDA locked), terminate immediately
		if signalErr := snapshotruntime.SendSignalToPID(log, containerPID, syscall.SIGKILL, "checkpoint failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint failure to runtime process")
		}
		if statusErr := setCheckpointStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
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
		if signalErr := snapshotruntime.SendSignalToPID(log, containerPID, syscall.SIGKILL, "checkpoint verification failed"); signalErr != nil {
			log.Error(signalErr, "Failed to signal checkpoint verification failure to runtime process")
		}
		if statusErr := setCheckpointStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
			return statusErr
		}
		return nil
	}
	if gmsDir := strings.TrimSpace(pod.Annotations[snapshotprotocol.GMSCheckpointDirAnnotation]); gmsDir != "" {
		sentinel := filepath.Join(gmsDir, gmsCompletionFileName(pod, snapshotprotocol.GMSSaveCompleteFile))
		log.Info("Waiting for GMS checkpoint saver", "sentinel", sentinel)
		if err := waitForFile(leaseCtx, sentinel, gmsCompletionPollInterval); err != nil {
			log.Error(err, "GMS checkpoint saver did not complete")
			emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
			if signalErr := snapshotruntime.SendSignalToPID(log, containerPID, syscall.SIGKILL, "gms checkpoint save failed"); signalErr != nil {
				log.Error(signalErr, "Failed to signal GMS checkpoint save failure to runtime process")
			}
			if statusErr := setCheckpointStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
				return statusErr
			}
			return nil
		}
	}

	// Step 2: Sentinel on success. Workload observes via polling on the
	// snapshot-control volume; containerPID is a PID inside the container's
	// mount namespace, which is all the /host/proc/<pid>/root write path
	// requires. The Succeeded event is emitted only after the sentinel has
	// been written so a sentinel-write failure doesn't produce conflicting
	// Succeeded+Failed events for the same operation.
	if err := snapshotruntime.WriteControlSentinel(containerPID, snapshotprotocol.SnapshotCompleteFile); err != nil {
		log.Error(err, "Failed to write snapshot-complete sentinel")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if statusErr := setCheckpointStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
			return statusErr
		}
		return nil
	}
	emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeNormal, "CheckpointSucceeded", fmt.Sprintf("Checkpoint completed: %s", checkpointID))

	if err := setCheckpointStatus(snapshotprotocol.CheckpointStatusCompleted); err != nil {
		return err
	}
	return nil
}

// runRestore runs the full restore workflow for a pod:
//  1. Mark the current container instance as in_progress
//  2. Call executor.Restore (inspect placeholder → nsrestore inside namespace)
//  3. Write a restore-complete sentinel into the pod's snapshot-control
//     volume to wake the workload (observed via polling)
//  4. Mark the container instance as completed
//
// Kubernetes readiness is gated separately: each restore-target container's
// startup probe waits on the restore-complete sentinel, then its normal
// readiness probe (if any) decides when the container is ready. The pod only
// becomes Ready once every restored and cold-started container is ready.
func (w *NodeController) runRestore(ctx context.Context, pod *corev1.Pod, containerName, containerID, checkpointID, checkpointLocation, restoreTrigger, restoreAttemptKey string, startedAt time.Time) error {
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
	log := w.log.WithValues("pod", podKey, "checkpoint_id", checkpointID, "container_id", containerID)
	setRestoreStatus := func(value string) error {
		annotations, err := snapshotprotocol.RestoreStatusAnnotations(containerName, value, containerID)
		if err != nil {
			return err
		}
		if restoreTrigger != "" {
			annotationKeys, err := snapshotprotocol.RestoreStatusAnnotationKeysFor(containerName)
			if err != nil {
				return err
			}
			annotations[annotationKeys.ProcessedTrigger] = restoreTrigger
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

	// Step 1: Run the restore orchestrator (inspect + nsrestore)
	req := executor.RestoreRequest{
		CheckpointID:       checkpointID,
		CheckpointLocation: checkpointLocation,
		StartedAt:          startedAt,
		NSRestorePath:      w.config.Restore.NSRestorePath,
		PodName:            pod.Name,
		PodNamespace:       pod.Namespace,
		ContainerName:      containerName,
		Clientset:          w.clientset,
	}
	placeholderHostPID, err := executor.Restore(restoreCtx, w.runtime, log, req)
	if err != nil {
		log.Error(err, "External restore failed")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
		if statusErr := setRestoreStatus(snapshotprotocol.RestoreStatusFailed); statusErr != nil {
			return statusErr
		}
		// Re-resolve: executor.Restore may have failed before resolving the placeholder.
		placeholderHostPID, _, pidErr := w.runtime.ResolveContainerByPod(ctx, pod.Name, pod.Namespace, containerName)
		if pidErr != nil {
			return fmt.Errorf("restore failed and placeholder PID could not be resolved: %w", pidErr)
		}
		if killErr := snapshotruntime.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore failed"); killErr != nil {
			return fmt.Errorf("restore failed and placeholder could not be killed: %w", killErr)
		}
		return nil
	}
	if gmsDir := strings.TrimSpace(pod.Annotations[snapshotprotocol.GMSCheckpointDirAnnotation]); gmsDir != "" {
		sentinel := filepath.Join(gmsDir, gmsCompletionFileName(pod, snapshotprotocol.GMSLoadCompleteFile))
		log.Info("Waiting for GMS checkpoint loader", "sentinel", sentinel)
		if err := waitForFile(restoreCtx, sentinel, gmsCompletionPollInterval); err != nil {
			log.Error(err, "GMS checkpoint loader did not complete")
			emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", err.Error())
			if statusErr := setRestoreStatus(snapshotprotocol.RestoreStatusFailed); statusErr != nil {
				return statusErr
			}
			if killErr := snapshotruntime.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "gms restore load failed"); killErr != nil {
				return fmt.Errorf("gms restore load failed and placeholder could not be killed: %w", killErr)
			}
			return nil
		}
	}

	// Step 2: Write restore-complete sentinel. placeholderHostPID came back
	// from executor.Restore — any PID inside the container's mount namespace
	// reaches /snapshot-control via /host/proc/<pid>/root.
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

	emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeNormal, "RestoreSucceeded", fmt.Sprintf("Restore completed from checkpoint %s", checkpointID))
	if err := setRestoreStatus(snapshotprotocol.RestoreStatusCompleted); err != nil {
		return err
	}
	return nil
}

func gmsCompletionFileName(pod *corev1.Pod, base string) string {
	if pod == nil {
		return base
	}
	switch pod.Annotations[snapshotprotocol.GMSCompletionFileModeAnnotation] {
	case snapshotprotocol.GMSCompletionFileModeShared:
		return base
	}
	uid := strings.TrimSpace(string(pod.UID))
	if uid == "" {
		return base
	}
	return fmt.Sprintf("%s-%s", base, uid)
}

func waitForFile(ctx context.Context, path string, interval time.Duration) error {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		info, err := os.Stat(path)
		switch {
		case err == nil && !info.IsDir():
			return nil
		case err == nil:
			return fmt.Errorf("%s exists but is a directory", path)
		case !os.IsNotExist(err):
			return fmt.Errorf("failed to stat %s: %w", path, err)
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("timed out waiting for %s: %w", path, ctx.Err())
		case <-ticker.C:
		}
	}
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
