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
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// NodeController watches local-node pods with checkpoint metadata and reconciles
// snapshot execution for checkpoint and restore requests. The restore path is
// driven by a client-go pod informer; the capture path is driven by a dynamic
// informer over PodSnapshotContent work orders filtered to this node, with typed
// reads/writes via an uncached controller-runtime client.
type NodeController struct {
	config       *types.AgentConfig
	clientset    kubernetes.Interface
	client       client.Client
	dynClient    dynamic.Interface
	runtime      snapshotruntime.Runtime
	log          logr.Logger
	holderID     string
	checkpointFn func(ctx context.Context, params CheckpointParams) error

	inFlight   map[string]struct{}
	inFlightMu sync.Mutex

	// contentIndexer is the PodSnapshotContent informer's indexer, indexed by source pod
	// (podRefIndex). The source-pod informer uses it to map a pod event back to its work order.
	contentIndexer cache.Indexer

	stopCh chan struct{}
}

type checkpointLocations struct {
	HostPath      string
	ContainerPath string
}

const (
	containerResolveAttemptTimeout  = 1 * time.Second
	restoreContainerResolveInterval = 50 * time.Millisecond
	restoreContainerResolveTimeout  = 30 * time.Second

	// snapshotContentResyncInterval re-drives every PodSnapshotContent work order so a
	// not-yet-Ready source pod is re-checked for quiesce without a busy loop.
	snapshotContentResyncInterval = 10 * time.Second
)

// podSnapshotContentGVR is the cluster-scoped resource the capture informer watches.
var podSnapshotContentGVR = nvidiacomv1alpha1.GroupVersion.WithResource("podsnapshotcontents")

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

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(nvidiacomv1alpha1.AddToScheme(scheme))

	typedClient, err := client.New(restConfig, client.Options{Scheme: scheme})
	if err != nil {
		return nil, fmt.Errorf("failed to create typed client: %w", err)
	}

	dynClient, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}

	w := &NodeController{
		config:    cfg,
		clientset: clientset,
		client:    typedClient,
		dynClient: dynClient,
		runtime:   rt,
		log:       log,
		holderID:  "snapshot-agent/" + uuid.NewString(),
		inFlight:  make(map[string]struct{}),
		stopCh:    make(chan struct{}),
	}
	w.checkpointFn = w.executorCheckpoint
	return w, nil
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

	// Capture path: a dynamic informer over PodSnapshotContent work orders, filtered at
	// the list/watch level to this node's mirror label. The node-label filter is the
	// node scoping; reconcilePodSnapshotContent keeps a defensive nodeName check.
	nodeContentSelector := labels.SelectorFromSet(labels.Set{snapshotprotocol.SnapshotNodeLabel: w.config.NodeName}).String()
	dynFactory := dynamicinformer.NewFilteredDynamicSharedInformerFactory(
		w.dynClient, snapshotContentResyncInterval, metav1.NamespaceAll,
		func(opts *metav1.ListOptions) {
			opts.LabelSelector = nodeContentSelector
		},
	)
	contentInformer := dynFactory.ForResource(podSnapshotContentGVR).Informer()
	// Index work orders by their source pod so a source-pod event maps back to its
	// PodSnapshotContent in O(1). Must be registered before the informer starts.
	if err := contentInformer.AddIndexers(cache.Indexers{podRefIndex: podRefIndexFunc}); err != nil {
		return fmt.Errorf("failed to add snapshot-content podRef indexer: %w", err)
	}
	w.contentIndexer = contentInformer.GetIndexer()
	if _, err := contentInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if name, ok := contentNameFromInformerObj(obj); ok {
				w.reconcilePodSnapshotContent(ctx, name)
			}
		},
		UpdateFunc: func(_, newObj interface{}) {
			if name, ok := contentNameFromInformerObj(newObj); ok {
				w.reconcilePodSnapshotContent(ctx, name)
			}
		},
	}); err != nil {
		return fmt.Errorf("failed to add snapshot-content informer handler: %w", err)
	}
	go dynFactory.Start(w.stopCh)
	syncFuncs = append(syncFuncs, contentInformer.HasSynced)

	// Source-pod informer: keyed on CaptureEligibleLabel, the promotion label the pre-bind gate
	// (reconcilePodSnapshotContent) adds only after a source pod passes validation. Keying on the
	// gate-applied label (not CheckpointSourceLabel) means only gate-validated pods drive the capture
	// path. A pod status change (a checkpoint container crashing, or the target becoming ready) does
	// not touch the PodSnapshotContent, so without this trigger it would only be acted on at the
	// content informer's resync. It needs its own factory: its selector is disjoint from the restore
	// informer's.
	sourceSelector := labels.SelectorFromSet(labels.Set{snapshotprotocol.CaptureEligibleLabel: "true"}).String()
	sourceFactoryOpts := append([]informers.SharedInformerOption{
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.LabelSelector = sourceSelector
		}),
	}, nsOptions...)
	sourceFactory := informers.NewSharedInformerFactoryWithOptions(
		w.clientset, 30*time.Second, sourceFactoryOpts...,
	)
	sourceInformer := sourceFactory.Core().V1().Pods().Informer()
	if _, err := sourceInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if pod, ok := podFromInformerObj(obj); ok {
				w.reconcileSourcePod(ctx, pod)
			}
		},
		UpdateFunc: func(_, newObj interface{}) {
			if pod, ok := podFromInformerObj(newObj); ok {
				w.reconcileSourcePod(ctx, pod)
			}
		},
	}); err != nil {
		return fmt.Errorf("failed to add source-pod informer handler: %w", err)
	}
	go sourceFactory.Start(w.stopCh)
	syncFuncs = append(syncFuncs, sourceInformer.HasSynced)

	// Close stopCh on cancellation so a stalled cache sync (below) is unblocked by ctx, not only on
	// the happy path.
	var stopOnce sync.Once
	go func() {
		<-ctx.Done()
		stopOnce.Do(func() { close(w.stopCh) })
	}()

	if !cache.WaitForCacheSync(w.stopCh, syncFuncs...) {
		return fmt.Errorf("failed to sync informer caches")
	}

	w.log.Info("PodSnapshot node controller started and caches synced")
	<-ctx.Done()
	stopOnce.Do(func() { close(w.stopCh) })
	return nil
}

func (w *NodeController) reconcileRestorePod(ctx context.Context, pod *corev1.Pod) {
	if pod.Spec.NodeName != w.config.NodeName {
		return
	}

	podKey := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

	if pod.DeletionTimestamp != nil ||
		(pod.Status.Phase != corev1.PodPending && pod.Status.Phase != corev1.PodRunning) {
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

func (w *NodeController) refreshRestorePodForStart(ctx context.Context, pod *corev1.Pod, podKey, containerName string) (*corev1.Pod, bool) {
	livePod, err := w.clientset.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			w.log.V(1).Info("Skipping restore; pod disappeared while polling runtime",
				"pod", podKey,
				"container", containerName,
			)
			return nil, false
		}
		w.log.Error(err, "Failed to refresh restore pod state before starting restore",
			"pod", podKey,
			"container", containerName,
		)
		return nil, false
	}
	if livePod.DeletionTimestamp != nil ||
		(livePod.Status.Phase != corev1.PodPending && livePod.Status.Phase != corev1.PodRunning) {
		w.log.V(1).Info("Skipping restore; pod became ineligible while polling runtime",
			"pod", podKey,
			"container", containerName,
			"phase", livePod.Status.Phase,
		)
		return nil, false
	}
	return livePod, true
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
			livePod, ok := w.refreshRestorePodForStart(ctx, pod, podKey, containerName)
			if !ok {
				return
			}
			w.log.V(1).Info("Resolved restore container via node runtime",
				"pod", podKey,
				"container", containerName,
				"container_id", containerID,
			)
			w.startRestoreForContainer(ctx, livePod, containerName, containerID, checkpointID, podKey)
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
	attemptDeadline := time.Now().Add(containerResolveAttemptTimeout)
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
	if w.config.CRIU.TcpEstablished && pod.Status.PodIP == "" {
		w.log.V(1).Info("Restore pod has no PodIP yet; waiting before TCP-established restore",
			"pod", podKey,
			"container", containerName,
		)
		return
	}

	placeholderPID := 0
	if strings.TrimSpace(w.config.Storage.AccessMode) == types.StorageAccessModePodMount {
		resolvedPID, _, err := w.runtime.ResolveContainer(ctx, containerID)
		if err != nil {
			w.log.Error(err, "Failed to resolve restore standby container", "pod", podKey, "container", containerName)
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
		ContainerID:                 containerID,
		StartedAt:                   startedAt,
		NSRestorePath:               w.config.Restore.NSRestorePath,
		PodName:                     pod.Name,
		PodNamespace:                pod.Namespace,
		TargetPodIP:                 pod.Status.PodIP,
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
		placeholderHostPID, _, pidErr := w.runtime.ResolveContainer(ctx, containerID)
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

// podRefIndex is the PodSnapshotContent informer index keyed by source pod ("<namespace>/<name>").
const podRefIndex = "byPodRef"

// podRefIndexFunc indexes a PodSnapshotContent by its source pod ("<snapshotRef.namespace>/<source.podRef.name>").
// It runs against the dynamic informer's *unstructured.Unstructured objects; an unexpected type or a
// missing field yields no index entry (nil) rather than an error, so it never poisons the index.
func podRefIndexFunc(obj interface{}) ([]string, error) {
	u, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil, nil
	}
	ns, _, _ := unstructured.NestedString(u.Object, "spec", "snapshotRef", "namespace")
	name, _, _ := unstructured.NestedString(u.Object, "spec", "source", "podRef", "name")
	if ns == "" || name == "" {
		return nil, nil
	}
	return []string{ns + "/" + name}, nil
}

// contentFromInformerObj converts a dynamic informer object (or its DeletedFinalStateUnknown
// tombstone) to a typed PodSnapshotContent. It returns false on an unexpected type.
func contentFromInformerObj(obj interface{}) (*nvidiacomv1alpha1.PodSnapshotContent, bool) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	u, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil, false
	}
	content := &nvidiacomv1alpha1.PodSnapshotContent{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(u.Object, content); err != nil {
		return nil, false
	}
	return content, true
}

// chooseActiveContent returns the name of the oldest non-terminal PodSnapshotContent among the indexed
// objects (oldest first by CreationTimestamp, ties broken by Name), or "" when none are active.
// Driving the oldest until it finishes gives deterministic, stable selection across pod events.
func chooseActiveContent(objs []interface{}) string {
	var chosen *nvidiacomv1alpha1.PodSnapshotContent
	for _, obj := range objs {
		content, ok := contentFromInformerObj(obj)
		if !ok || isContentTerminal(content) {
			continue
		}
		if chosen == nil ||
			content.CreationTimestamp.Before(&chosen.CreationTimestamp) ||
			(content.CreationTimestamp.Equal(&chosen.CreationTimestamp) && content.Name < chosen.Name) {
			chosen = content
		}
	}
	if chosen == nil {
		return ""
	}
	return chosen.Name
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
		return checkpointLocations{}, fmt.Errorf("re-resolve restore standby container %s before podMount storage access: %w", containerID, err)
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
