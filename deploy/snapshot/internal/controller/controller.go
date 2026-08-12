// Package controller implements the node-local control loop inside snapshot-agent.
// It does not own CRDs or replace the operator. Instead it watches pod, job, and
// lease state on the current node and delegates CRIU/CUDA execution to the
// snapshot executor workflows.
package controller

import (
	"context"
	"errors"
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
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsmount"
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
	injector     executor.Mounters
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

const (
	containerResolveAttemptTimeout  = 1 * time.Second
	restoreContainerResolveInterval = 50 * time.Millisecond
	restoreContainerResolveTimeout  = 30 * time.Second
	restoreFailureCleanupTimeout    = 30 * time.Second

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

	nsm, err := nsmount.New(log)
	if err != nil {
		return nil, fmt.Errorf("failed to create binary injector: %w", err)
	}
	// One mounter per policy: the bundle stays executable, the artifact does not.
	injector := executor.Mounters{Bundle: nsm.ReadOnly(), Artifact: nsm.ReadOnlyNoExec()}

	w := &NodeController{
		config:    cfg,
		clientset: clientset,
		client:    typedClient,
		dynClient: dynClient,
		runtime:   rt,
		injector:  injector,
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
	// Seed the agent logger onto ctx so the capture path resolves it via log.FromContext.
	ctx = logr.NewContext(ctx, w.log)
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
				if err := w.reconcileSourcePod(ctx, pod); err != nil {
					w.log.Error(err, "Failed to reconcile source pod", "pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name))
				}
			}
		},
		UpdateFunc: func(_, newObj interface{}) {
			if pod, ok := podFromInformerObj(newObj); ok {
				if err := w.reconcileSourcePod(ctx, pod); err != nil {
					w.log.Error(err, "Failed to reconcile source pod", "pod", fmt.Sprintf("%s/%s", pod.Namespace, pod.Name))
				}
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

	artifactPath, err := w.artifactPathForPod(pod, checkpointID)
	if err != nil {
		w.log.Error(err, "Restore pod names an unusable checkpoint artifact", "pod", podKey, "checkpoint_id", checkpointID)
	} else {
		checkpointReady, readyErr := w.restoreCheckpointReady(w.log, podKey, checkpointID, artifactPath)
		if readyErr != nil {
			// Invalid artifacts are terminal and are handled by the restore worker,
			// which persists Failed and terminates the target.
			w.log.Error(readyErr, "Restore checkpoint path is invalid", "pod", podKey, "checkpoint_id", checkpointID, "checkpoint_location", artifactPath)
		} else if !checkpointReady {
			// Missing artifacts retain the existing pending-and-retry behavior.
			return
		}
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
		if err := w.runRestore(ctx, pod, containerName, containerID, checkpointID, restoreAttemptKey, startedAt); err != nil {
			opLog := w.log.WithValues("pod", podKey, "checkpoint_id", checkpointID, "container", containerName)
			opLog.Error(err, "Restore controller worker failed")
			emitPodEvent(ctx, w.clientset, opLog, pod, "snapshot", corev1.EventTypeWarning, "RestoreWorkerFailed", err.Error())
		}
	}()
}

// runRestore runs the full restore workflow for one target container:
//  1. Resolve and validate the exact artifact and target container
//  2. Annotate the pod with restore in_progress before mounting anything
//  3. Call executor.Restore (mount resources → nsrestore inside namespace)
//  4. Write a restore-complete sentinel: the CRIU-restored process resumes
//     inside the polling loop that waits on this file, exits quiescence,
//     and resumes the engine
//  5. Annotate the pod with restore completed
func (w *NodeController) runRestore(ctx context.Context, pod *corev1.Pod, containerName, containerID, checkpointID string, restoreAttemptKey string, startedAt time.Time) error {
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
	setRestoreStatus := func(statusCtx context.Context, value string) error {
		if err := w.writeRestoreStatus(statusCtx, log, pod, containerName, containerID, value); err != nil {
			if isTerminalRestoreStatus(value) {
				releaseOnExit = false
				return fmt.Errorf("failed to persist terminal restore status %q: %w", value, err)
			}
			return fmt.Errorf("failed to update restore status %q: %w", value, err)
		}
		if isTerminalRestoreStatus(value) {
			// Keep the attempt key for this controller lifetime so a stale
			// runtime resolver cannot start the same container again after
			// terminal status has been persisted.
			releaseOnExit = false
		}
		return nil
	}
	failAndKill := func(cause error, placeholderHostPID int) error {
		log.Error(cause, "Restore failed; terminating target container")
		cleanupCtx, cancel := context.WithTimeout(context.Background(), restoreFailureCleanupTimeout)
		defer cancel()
		emitPodEvent(cleanupCtx, w.clientset, log, pod, "snapshot", corev1.EventTypeWarning, "RestoreFailed", cause.Error())

		if placeholderHostPID <= 0 {
			var resolveErr error
			placeholderHostPID, _, resolveErr = w.runtime.ResolveContainer(cleanupCtx, containerID)
			if resolveErr != nil {
				return errors.Join(cause, fmt.Errorf("resolve failed restore target: %w", resolveErr))
			}
		}
		killErr := snapshotruntime.SendSignalToPID(log, placeholderHostPID, syscall.SIGKILL, "restore failed")
		if killErr != nil && !errors.Is(killErr, syscall.ESRCH) {
			// Preserve in_progress until termination can be retried. Publishing a
			// terminal state first creates a crash window in which startup recovery
			// ignores a still-running failed target and its mounts.
			return errors.Join(cause, fmt.Errorf("kill failed restore target: %w", killErr))
		}
		statusErr := setRestoreStatus(cleanupCtx, snapshotprotocol.RestoreStatusFailed)
		return errors.Join(cause, statusErr)
	}

	artifactPath, err := w.artifactPathForPod(pod, checkpointID)
	if err != nil {
		return failAndKill(fmt.Errorf("resolve checkpoint artifact path: %w", err), 0)
	}
	checkpointReady, err := w.restoreCheckpointReady(log, podKey, checkpointID, artifactPath)
	if err != nil {
		return failAndKill(fmt.Errorf("validate checkpoint artifact path: %w", err), 0)
	}
	if !checkpointReady {
		return nil
	}

	// Run the restore orchestrator (inspect + mount agent bundle + nsrestore).
	req := executor.RestoreRequest{
		CheckpointID:    checkpointID,
		ArtifactVersion: artifactVersionFromPod(pod),
		BasePath:        w.config.Storage.BasePath,
		ContainerID:     containerID,
		StartedAt:       startedAt,
		PodName:         pod.Name,
		PodNamespace:    pod.Namespace,
		TargetPodIP:     pod.Status.PodIP,
		ContainerName:   containerName,
		Clientset:       w.clientset,
		OnArtifactReady: func() error {
			return setRestoreStatus(restoreCtx, snapshotprotocol.RestoreStatusInProgress)
		},
	}
	placeholderHostPID, err := executor.Restore(restoreCtx, w.runtime, log, req, w.injector)
	if err != nil {
		return failAndKill(fmt.Errorf("external restore: %w", err), placeholderHostPID)
	}
	// Any PID inside the container mount namespace reaches the control
	// volume through /host/proc/<pid>/root.
	if err := snapshotruntime.WriteControlSentinel(placeholderHostPID, snapshotprotocol.RestoreCompleteFile); err != nil {
		return failAndKill(fmt.Errorf("write restore-complete sentinel: %w", err), placeholderHostPID)
	}

	if err := setRestoreStatus(ctx, snapshotprotocol.RestoreStatusCompleted); err != nil {
		return failAndKill(fmt.Errorf("persist completed restore status: %w", err), placeholderHostPID)
	}
	emitPodEvent(ctx, w.clientset, log, pod, "snapshot", corev1.EventTypeNormal, "RestoreSucceeded", fmt.Sprintf("Restore completed from checkpoint %s", checkpointID))
	return nil
}

func isTerminalRestoreStatus(value string) bool {
	return value == snapshotprotocol.RestoreStatusCompleted || value == snapshotprotocol.RestoreStatusFailed
}

// writeRestoreStatus persists one restore status annotation.
//
// Terminal values are retried. By the time "completed" is written the restored
// container has already been released by the sentinel and is serving traffic,
// and both mounts are gone, so a single transient API error would otherwise
// cost a healthy workload its life: the caller's only alternative to a
// persisted status is to kill the container. The patch replaces a fixed
// annotation map, which makes retrying idempotent. in_progress is not retried,
// because failing fast there simply means the restore does not start.
func (w *NodeController) writeRestoreStatus(
	ctx context.Context,
	log logr.Logger,
	pod *corev1.Pod,
	containerName, containerID, value string,
) error {
	annotations, err := snapshotprotocol.RestoreStatusAnnotations(containerName, value, containerID)
	if err != nil {
		return err
	}
	write := func() error { return annotatePod(ctx, w.clientset, log, pod, annotations) }
	if !isTerminalRestoreStatus(value) {
		return write()
	}
	// Retry anything except a dead context: on agent shutdown the remaining
	// attempts cannot succeed and would only delay exit.
	return retry.OnError(retry.DefaultBackoff, func(error) bool { return ctx.Err() == nil }, write)
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

// artifactVersionFromPod reads the artifact version the pod asks for, falling
// back to the protocol default when the annotation is absent or blank.
func artifactVersionFromPod(pod *corev1.Pod) string {
	return snapshotprotocol.ArtifactVersion(strings.TrimSpace(pod.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation]))
}

// artifactPathForPod resolves the agent-local checkpoint artifact directory for
// a pod's checkpoint. The base path is agent-owned configuration: a pod names
// only the checkpoint and the artifact version, never a filesystem location.
func (w *NodeController) artifactPathForPod(pod *corev1.Pod, checkpointID string) (string, error) {
	return nsmount.ResolveArtifactPath(w.config.Storage.BasePath, checkpointID, artifactVersionFromPod(pod))
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
