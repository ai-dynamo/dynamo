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

type checkpointLocations struct {
	HostPath      string
	ContainerPath string
}

const (
	restoreContainerResolveInterval       = 50 * time.Millisecond
	restoreContainerResolveAttemptTimeout = 1 * time.Second
	restoreContainerResolveTimeout        = 30 * time.Second
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

	// Restore pods carry a checkpoint ID but are not checkpoint sources.
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
		if err := w.runCheckpoint(ctx, pod, job, checkpointID, containerName, podKey, startedAt); err != nil {
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

	for _, containerName := range targets {
		w.maybeStartRestoreForContainer(ctx, pod, containerName, checkpointID, podKey)
	}
}

// maybeStartRestoreForContainer starts one restore worker per fresh container.
// Falls back to polling the OCI runtime when pod.Status hasn't published the
// ContainerID yet (the kubelet status patch can lag exec by 1-5 s).
func (w *NodeController) maybeStartRestoreForContainer(
	ctx context.Context,
	pod *corev1.Pod,
	containerName string,
	checkpointID string,
	podKey string,
) {
	if containerID := restoreContainerIDFromStatus(pod, containerName); containerID != "" {
		w.startRestoreForContainer(ctx, pod, containerName, containerID, checkpointID, podKey)
		return
	}

	resolveKey := fmt.Sprintf("%s/%s/resolve", podKey, containerName)
	if !w.tryAcquire(resolveKey) {
		return
	}
	w.log.V(1).Info("Restore pod has no running container in Kubernetes status yet; polling node runtime",
		"pod", podKey,
		"container", containerName,
	)
	go w.pollForContainerID(ctx, pod.DeepCopy(), containerName, checkpointID, podKey, resolveKey)
}

func restoreContainerIDFromStatus(pod *corev1.Pod, containerName string) string {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName && cs.ContainerID != "" {
			return snapshotruntime.StripCRIScheme(cs.ContainerID)
		}
	}
	return ""
}

func (w *NodeController) pollForContainerID(
	ctx context.Context,
	pod *corev1.Pod,
	containerName, checkpointID, podKey, resolveKey string,
) {
	defer w.release(resolveKey)
	deadlineAt := time.Now().Add(restoreContainerResolveTimeout)
	deadline := time.NewTimer(time.Until(deadlineAt))
	defer deadline.Stop()
	tick := time.NewTicker(restoreContainerResolveInterval)
	defer tick.Stop()
	for {
		resolveCtx, cancel := restoreContainerResolveAttemptContext(ctx, deadlineAt)
		containerID, err := w.runtime.ResolveContainerIDByPod(resolveCtx, pod.Name, pod.Namespace, containerName)
		cancel()
		if err == nil && containerID != "" {
			w.log.V(1).Info("Resolved restore container via node runtime",
				"pod", podKey,
				"container", containerName,
				"container_id", containerID,
			)
			w.startRestoreForContainer(ctx, pod, containerName, containerID, checkpointID, podKey)
			return
		}

		select {
		case <-deadline.C:
			w.log.V(1).Info("Timed out polling node runtime for restore container",
				"pod", podKey,
				"container", containerName,
			)
			return
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}

func restoreContainerResolveAttemptContext(ctx context.Context, deadlineAt time.Time) (context.Context, context.CancelFunc) {
	attemptDeadline := time.Now().Add(restoreContainerResolveAttemptTimeout)
	if deadlineAt.Before(attemptDeadline) {
		attemptDeadline = deadlineAt
	}
	return context.WithDeadline(ctx, attemptDeadline)
}

func (w *NodeController) startRestoreForContainer(
	ctx context.Context,
	pod *corev1.Pod,
	containerName string,
	containerID string,
	checkpointID string,
	podKey string,
) {
	annotationKeys, err := snapshotprotocol.RestoreStatusAnnotationKeysFor(containerName)
	if err != nil {
		w.log.Error(err, "Restore target container name cannot be used in restore status annotation key", "pod", podKey, "container", containerName)
		return
	}
	annotationStatus := pod.Annotations[annotationKeys.Status]
	annotationContainerID := pod.Annotations[annotationKeys.ContainerID]
	if annotationContainerID == containerID && (annotationStatus == snapshotprotocol.RestoreStatusCompleted || annotationStatus == snapshotprotocol.RestoreStatusFailed) {
		return
	}

	placeholderPID := 0
	if strings.TrimSpace(w.config.Storage.AccessMode) == types.StorageAccessModePodMount {
		resolvedPID, _, err := w.runtime.ResolveContainer(ctx, containerID)
		if err != nil {
			w.log.Error(err, "Failed to resolve restore placeholder container", "pod", podKey, "container", containerName)
			return
		}
		placeholderPID = resolvedPID
	}

	checkpointLocation, err := w.checkpointLocationsFromPod(pod, checkpointID, placeholderPID)
	if err != nil {
		w.log.Error(err, "Restore pod is missing storage metadata", "pod", podKey, "checkpoint_id", checkpointID)
		return
	}
	if err := w.validatePodMountContainerPID(ctx, containerID, placeholderPID); err != nil {
		w.log.Error(err, "Restore placeholder container changed before storage access", "pod", podKey, "container", containerName)
		return
	}
	checkpointReady, err := w.restoreCheckpointReady(w.log, podKey, checkpointID, checkpointLocation.HostPath)
	if err != nil {
		w.log.Error(err, "Restore checkpoint path is invalid", "pod", podKey, "checkpoint_id", checkpointID, "checkpoint_location", checkpointLocation.HostPath)
		return
	}
	if !checkpointReady {
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
	)
	emitPodEvent(ctx, w.clientset, w.log, pod, "snapshot", corev1.EventTypeNormal, "RestoreRequested", fmt.Sprintf("Restore requested from checkpoint %s for container %s", checkpointID, containerName))

	go func() {
		if err := w.runRestore(ctx, pod, containerName, containerID, checkpointID, checkpointLocation, restoreAttemptKey, startedAt); err != nil {
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
func (w *NodeController) runCheckpoint(ctx context.Context, pod *corev1.Pod, job *batchv1.Job, checkpointID, containerName, podKey string, startedAt time.Time) error {
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

	checkpointLocation, err := w.checkpointLocationsFromPod(pod, checkpointID, containerPID)
	if err != nil {
		log.Error(err, "Checkpoint pod is missing storage metadata")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
		if statusErr := setCheckpointStatus(snapshotprotocol.CheckpointStatusFailed); statusErr != nil {
			return statusErr
		}
		return nil
	}
	if err := w.validatePodMountContainerPID(ctx, containerID, containerPID); err != nil {
		log.Error(err, "Checkpoint container changed before storage access")
		emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", err.Error())
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
		CheckpointLocation: checkpointLocation.HostPath,
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

	info, err := os.Stat(checkpointLocation.HostPath)
	if err != nil || !info.IsDir() {
		if err == nil {
			err = fmt.Errorf("published checkpoint path %s is not a directory", checkpointLocation.HostPath)
		} else {
			err = fmt.Errorf("published checkpoint path %s is missing: %w", checkpointLocation.HostPath, err)
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

// runRestore runs the full restore workflow for one target container:
//  1. Annotate the pod with restore in_progress
//  2. Call executor.Restore (inspect placeholder → nsrestore inside namespace)
//  3. Write a restore-complete sentinel: the CRIU-restored process resumes
//     inside the polling loop that waits on this file, exits quiescence,
//     and resumes the engine
//  4. Annotate the pod with restore completed
func (w *NodeController) runRestore(ctx context.Context, pod *corev1.Pod, containerName, containerID, checkpointID string, checkpointLocation checkpointLocations, restoreAttemptKey string, startedAt time.Time) error {
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
		if err := annotatePod(ctx, w.clientset, log, pod, annotations); err != nil {
			if value == snapshotprotocol.RestoreStatusCompleted || value == snapshotprotocol.RestoreStatusFailed {
				releaseOnExit = false
				return fmt.Errorf("failed to persist terminal restore status %q: %w", value, err)
			}
			return fmt.Errorf("failed to update restore status %q: %w", value, err)
		}
		if value == snapshotprotocol.RestoreStatusCompleted || value == snapshotprotocol.RestoreStatusFailed {
			// Keep the attempt key for this controller lifetime so a stale
			// runtime resolver cannot start the same container again after
			// terminal status has been persisted.
			releaseOnExit = false
		}
		return nil
	}

	checkpointLocation, err := w.refreshRestoreCheckpointLocation(restoreCtx, pod, containerID, checkpointID, checkpointLocation)
	if err != nil {
		return fmt.Errorf("refresh restore checkpoint location: %w", err)
	}
	checkpointReady, err := w.restoreCheckpointReady(log, podKey, checkpointID, checkpointLocation.HostPath)
	if err != nil {
		return fmt.Errorf("validate refreshed checkpoint location: %w", err)
	}
	if !checkpointReady {
		return nil
	}

	if err := setRestoreStatus(snapshotprotocol.RestoreStatusInProgress); err != nil {
		return fmt.Errorf("failed to annotate pod with restore in_progress: %w", err)
	}

	// Run the restore orchestrator (inspect + nsrestore).
	req := executor.RestoreRequest{
		CheckpointID:                checkpointID,
		CheckpointLocation:          checkpointLocation.HostPath,
		ContainerCheckpointLocation: checkpointLocation.ContainerPath,
		StartedAt:                   startedAt,
		NSRestorePath:               w.config.Restore.NSRestorePath,
		PodName:                     pod.Name,
		PodNamespace:                pod.Namespace,
		ContainerName:               containerName,
		Clientset:                   w.clientset,
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
	// Any PID inside the container mount namespace reaches the control
	// volume through /host/proc/<pid>/root.
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

func (w *NodeController) checkpointLocationsFromPod(pod *corev1.Pod, checkpointID string, hostPID int) (checkpointLocations, error) {
	rawBasePath, hasBasePathAnnotation := pod.Annotations[snapshotprotocol.CheckpointStorageBasePathAnnotation]
	basePath := strings.TrimSpace(rawBasePath)
	if basePath == "" {
		if hasBasePathAnnotation {
			w.log.Info("Ignoring blank checkpoint storage base path annotation", "pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name))
		}
		basePath = strings.TrimSpace(w.config.Storage.BasePath)
	}
	storageType := strings.TrimSpace(pod.Annotations[snapshotprotocol.CheckpointStorageTypeAnnotation])
	if storageType == "" {
		storageType = w.config.Storage.Type
	}
	resolvedStorage, err := snapshotprotocol.ResolveCheckpointStorage(
		checkpointID,
		strings.TrimSpace(pod.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation]),
		snapshotprotocol.Storage{
			Type:     storageType,
			BasePath: basePath,
		},
	)
	if err != nil {
		return checkpointLocations{}, err
	}

	location := resolvedStorage.Location
	if !filepath.IsAbs(location) || filepath.Clean(location) != location {
		return checkpointLocations{}, fmt.Errorf("checkpoint location must be an absolute, clean path: %q", location)
	}
	if strings.TrimSpace(w.config.Storage.AccessMode) == types.StorageAccessModePodMount {
		if hostPID <= 0 {
			return checkpointLocations{}, fmt.Errorf("host PID is required for %s storage access", types.StorageAccessModePodMount)
		}
		hostLocation := filepath.Join(
			snapshotruntime.HostProcPath,
			fmt.Sprintf("%d", hostPID),
			"root",
			strings.TrimPrefix(location, string(os.PathSeparator)),
		)
		return checkpointLocations{HostPath: hostLocation, ContainerPath: location}, nil
	}
	return checkpointLocations{HostPath: location, ContainerPath: location}, nil
}

func (w *NodeController) refreshRestoreCheckpointLocation(ctx context.Context, pod *corev1.Pod, containerID string, checkpointID string, checkpointLocation checkpointLocations) (checkpointLocations, error) {
	if strings.TrimSpace(w.config.Storage.AccessMode) != types.StorageAccessModePodMount {
		return checkpointLocation, nil
	}

	currentHostPID, _, err := w.runtime.ResolveContainer(ctx, containerID)
	if err != nil {
		return checkpointLocations{}, fmt.Errorf("re-resolve restore placeholder container %s before podMount storage access: %w", containerID, err)
	}
	refreshedLocation, err := w.checkpointLocationsFromPod(pod, checkpointID, currentHostPID)
	if err != nil {
		return checkpointLocations{}, err
	}
	if err := w.validatePodMountContainerPID(ctx, containerID, currentHostPID); err != nil {
		return checkpointLocations{}, err
	}
	return refreshedLocation, nil
}

func (w *NodeController) restoreCheckpointReady(log logr.Logger, podKey, checkpointID, checkpointLocation string) (bool, error) {
	info, err := os.Stat(checkpointLocation)
	if err != nil {
		if os.IsNotExist(err) {
			log.V(1).Info("Checkpoint not ready on disk, skipping restore", "pod", podKey, "checkpoint_id", checkpointID, "checkpoint_location", checkpointLocation)
			return false, nil
		}
		return false, fmt.Errorf("stat checkpoint location %s: %w", checkpointLocation, err)
	}
	if !info.IsDir() {
		return false, fmt.Errorf("checkpoint location %s is not a directory", checkpointLocation)
	}
	return true, nil
}

func (w *NodeController) validatePodMountContainerPID(ctx context.Context, containerID string, expectedHostPID int) error {
	if strings.TrimSpace(w.config.Storage.AccessMode) != types.StorageAccessModePodMount {
		return nil
	}
	if expectedHostPID <= 0 {
		return fmt.Errorf("host PID is required for %s storage access", types.StorageAccessModePodMount)
	}

	currentHostPID, _, err := w.runtime.ResolveContainer(ctx, containerID)
	if err != nil {
		return fmt.Errorf("re-resolve container %s before podMount storage access: %w", containerID, err)
	}
	if currentHostPID != expectedHostPID {
		return fmt.Errorf("container %s host PID changed from %d to %d before podMount storage access", containerID, expectedHostPID, currentHostPID)
	}
	if err := snapshotruntime.ValidateProcessState(snapshotruntime.HostProcPath, expectedHostPID); err != nil {
		return fmt.Errorf("validate host PID %d before podMount storage access: %w", expectedHostPID, err)
	}
	return nil
}
