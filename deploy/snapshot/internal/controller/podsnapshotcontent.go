// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/executor"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// CheckpointParams carries everything the node driver needs to dump one container.
type CheckpointParams struct {
	// Pod is the live source pod (already provenance-verified by the reconciler).
	Pod *corev1.Pod
	// ContainerName is the single target container to checkpoint.
	ContainerName string
	// ContainerID is the agent-resolved running container ID (CRI scheme stripped).
	ContainerID string
	// ContainerPID is the agent-resolved host PID of the running container.
	ContainerPID int
	// CheckpointID is the stable artifact identity.
	CheckpointID string
	// HostPath is the agent-resolved destination directory for the dump.
	HostPath string
	// ContainerPath is the destination as seen inside the workload container's mount
	// namespace (equal to HostPath under agentMount storage).
	ContainerPath string
	// StartedAt marks when the controller observed the work order, for timing.
	StartedAt time.Time
}

// reconcilePodSnapshotContent is the pre-bind gate for a PodSnapshotContent work order. It validates the
// source pod (existence and provenance) and, when the pod is valid, promotes it by adding
// CaptureEligibleLabel — it never runs the capture flow itself. The source-pod informer (keyed on that
// label) then drives the capture path. Driven by the content informer (Add/Update) and its 10s resync;
// the resync is the backstop that eventually writes a terminal failure for a work order whose source
// pod is gone.
func (w *NodeController) reconcilePodSnapshotContent(ctx context.Context, name string) {
	logger := logr.FromContextOrDiscard(ctx).WithValues("content", name)

	content := &nvidiacomv1alpha1.PodSnapshotContent{}
	if err := w.client.Get(ctx, client.ObjectKey{Name: name}, content); err != nil {
		if apierrors.IsNotFound(err) {
			return
		}
		logger.Error(err, "Failed to get PodSnapshotContent")
		return
	}

	if content.Spec.Source.NodeName != w.config.NodeName {
		return
	}
	if isContentTerminal(content) {
		return
	}

	pod := &corev1.Pod{}
	key := client.ObjectKey{Namespace: content.Spec.PodSnapshotRef.Namespace, Name: content.Spec.Source.PodRef.Name}
	if err := w.client.Get(ctx, key, pod); err != nil {
		if apierrors.IsNotFound(err) {
			// The operator creates the PodSnapshotContent only after the source pod exists, and this
			// is a linearizable (quorum) Get, so NotFound means the pod was deleted, not a
			// creation race: fail the work order terminally.
			w.setSnapshotContentFailed(ctx, content, "SourcePodNotFound", fmt.Errorf("source pod %q not found", key.String()))
			return
		}
		logger.Error(err, "Failed to get source pod", "pod", key.String())
		return
	}
	if reason, msg := classifySourcePod(content, pod); reason != "" {
		w.setSnapshotContentFailed(ctx, content, reason, errors.New(msg))
		return
	}

	// The source-pod informer keys on CaptureEligibleLabel, so this patch is the hand-off that drives
	// the capture path — the gate never calls reconcileSourcePod directly.
	if err := w.labelCaptureEligible(ctx, pod); err != nil {
		logger.Error(err, "Failed to mark source pod capture-eligible", "pod", pod.Name)
	}
}

// reconcileSourcePod is the single capture path. It is driven by source-pod informer events for pods
// the gate promoted with CaptureEligibleLabel. It selects the oldest active work order for
// the pod and drives the unstick + dump. Capture parameters come from the source pod, which is the
// single source of truth; it never mutates spec and writes status via Status().Patch only. The
// triggering content event (if any) may name a different work order than the one chosen here — the
// event is only a trigger; chooseActiveContent picks the oldest active PodSnapshotContent for the pod.
func (w *NodeController) reconcileSourcePod(ctx context.Context, pod *corev1.Pod) error {
	objs, err := w.contentIndexer.ByIndex(podRefIndex, pod.Namespace+"/"+pod.Name)
	if err != nil {
		return fmt.Errorf("look up PodSnapshotContent by source pod %s/%s: %w", pod.Namespace, pod.Name, err)
	}
	name := chooseActiveContent(objs)
	if name == "" {
		return nil
	}
	logger := logr.FromContextOrDiscard(ctx).WithValues("content", name)
	ctx = logr.NewContext(ctx, logger)

	content := &nvidiacomv1alpha1.PodSnapshotContent{}
	if err := w.client.Get(ctx, client.ObjectKey{Name: name}, content); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("get PodSnapshotContent %q: %w", name, err)
	}
	if isContentTerminal(content) {
		return nil
	}

	// Capture parameters come from the source pod, which is the single source of truth. The
	// checkpoint ID is the pod label; the work order name is treated as opaque (never parsed).
	id := strings.TrimSpace(pod.Labels[snapshotprotocol.CheckpointIDLabel])
	if id == "" {
		w.setSnapshotContentFailed(ctx, content, "MissingCheckpointID",
			fmt.Errorf("source pod %q missing %s label", pod.Name, snapshotprotocol.CheckpointIDLabel))
		return nil
	}
	if errs := validation.IsDNS1123Label(id); len(errs) > 0 {
		w.setSnapshotContentFailed(ctx, content, "InvalidCheckpointID",
			fmt.Errorf("checkpoint ID %q is not a valid DNS-1123 label: %s", id, strings.Join(errs, "; ")))
		return nil
	}

	// The checkpoint ID is the artifact identity, so the in-flight guard and lease key on it:
	// a PodSnapshot delete/recreate changes the work-order name but must not admit a second dump
	// into the same artifact path. tryAcquire must stay after the terminal-content check and ID
	// validation so the guard is never held by a terminal work order.
	if !w.tryAcquire(id) {
		return nil
	}
	releaseInFlight := true
	defer func() {
		if releaseInFlight {
			w.release(id)
		}
	}()

	if w.failCheckpointOnContainerExit(ctx, content, pod) {
		return nil
	}
	if reason, msg := classifySourcePod(content, pod); reason != "" {
		w.setSnapshotContentFailed(ctx, content, reason, errors.New(msg))
		w.removeCaptureEligibleLabel(ctx, pod)
		return nil
	}

	containerName, err := snapshotprotocol.TargetContainersFromAnnotations(pod.Annotations, 1, 1)
	if err != nil {
		w.setSnapshotContentFailed(ctx, content, "MissingTargetContainer", err)
		return nil
	}
	if !isContainerReady(pod, containerName[0]) {
		logger.V(1).Info("Source container not ready, awaiting quiesce", "pod", pod.Name, "container", containerName[0])
		return nil
	}

	containerID := containerIDForName(pod, containerName[0])
	if containerID == "" {
		w.setSnapshotContentFailed(ctx, content, "ContainerNotResolved",
			fmt.Errorf("could not resolve container %q ID", containerName[0]))
		return nil
	}
	containerPID, _, err := w.runtime.ResolveContainer(ctx, containerID)
	if err != nil {
		w.setSnapshotContentFailed(ctx, content, "ContainerNotResolved", fmt.Errorf("resolve container %q: %w", containerName[0], err))
		return nil
	}
	loc, err := w.checkpointLocationsFromPod(pod, id, containerPID)
	if err != nil {
		w.setSnapshotContentFailed(ctx, content, "InvalidDestination", err)
		return nil
	}
	if err := w.validatePodMountContainerPID(ctx, containerID, containerPID); err != nil {
		w.setSnapshotContentFailed(ctx, content, "ContainerChanged", err)
		return nil
	}

	// Resume: a present artifact with unwritten status means a prior dump finished but the
	// status write did not. The artifact dir exists only after the executor's atomic rename,
	// so its presence means a completed dump.
	if artifactPresent(loc.HostPath) {
		w.setSnapshotContentSucceeded(ctx, content)
		return nil
	}

	leaseKey := client.ObjectKey{Namespace: content.Spec.PodSnapshotRef.Namespace, Name: checkpointLeaseName(id)}
	acquired, err := w.acquireLease(ctx, leaseKey)
	if err != nil {
		return fmt.Errorf("acquire checkpoint lease %s: %w", leaseKey.String(), err)
	}
	if !acquired {
		return nil
	}

	releaseInFlight = false
	go w.runCheckpoint(ctx, content, pod, containerName[0], containerID, containerPID, id, loc, leaseKey, id)
	return nil
}

// runCheckpoint executes the dump under a renewed lease and writes the terminal status.
// The container ID, host PID, and resolved locations are pre-resolved by the reconciler so
// the dump does not re-resolve them.
func (w *NodeController) runCheckpoint(
	ctx context.Context,
	content *nvidiacomv1alpha1.PodSnapshotContent,
	pod *corev1.Pod,
	containerName, containerID string,
	containerPID int,
	checkpointID string,
	loc checkpointLocations,
	leaseKey client.ObjectKey,
	inFlightKey string,
) {
	logger := logr.FromContextOrDiscard(ctx)
	defer w.release(inFlightKey)

	defer func() {
		releaseCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := w.releaseLease(releaseCtx, leaseKey); err != nil {
			logger.Error(err, "Failed to release checkpoint lease", "lease", leaseKey.String())
		}
	}()

	leaseCtx, stopLease := context.WithCancelCause(ctx)
	defer stopLease(nil)
	go w.renewLease(leaseCtx, leaseKey, stopLease)

	params := CheckpointParams{
		Pod:           pod,
		ContainerName: containerName,
		ContainerID:   containerID,
		ContainerPID:  containerPID,
		CheckpointID:  checkpointID,
		HostPath:      loc.HostPath,
		ContainerPath: loc.ContainerPath,
		StartedAt:     time.Now(),
	}
	if err := w.checkpointFn(leaseCtx, params); err != nil {
		if cause := context.Cause(leaseCtx); cause != nil && !errors.Is(cause, context.Canceled) {
			err = fmt.Errorf("%w; %v", err, cause)
		}
		logger.Error(err, "Checkpoint failed")
		w.setSnapshotContentFailed(ctx, content, "CheckpointFailed", err)
		return
	}

	w.setSnapshotContentSucceeded(ctx, content)
}

// classifySourcePod reports whether the source pod is unusable for capture, returning a terminal
// failure reason and message ("" reason means the pod is valid). It is pure: callers decide whether
// to setSnapshotContentFailed (reconcilePodSnapshotContent, pre-bind) or merely skip capture (reconcileSourcePod
// guard). Pod existence (NotFound) is handled by the caller, which holds the Get error.
func classifySourcePod(content *nvidiacomv1alpha1.PodSnapshotContent, pod *corev1.Pod) (string, string) {
	if content.Spec.Source.PodRef.UID != "" && pod.UID != content.Spec.Source.PodRef.UID {
		return "StalePodReference",
			fmt.Sprintf("source pod %q UID %q does not match work order UID %q", pod.Name, pod.UID, content.Spec.Source.PodRef.UID)
	}
	if pod.DeletionTimestamp != nil || pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded {
		return "SourcePodGone",
			fmt.Sprintf("source pod %q is no longer running (phase %s)", pod.Name, pod.Status.Phase)
	}
	return "", ""
}

// failCheckpointOnContainerExit fails the work order and force-terminates the source pod's
// still-running containers when any checkpoint container has terminated non-zero. It returns
// true when a failure was handled and the caller must stop. Init containers
// (pod.Status.InitContainerStatuses) are intentionally out of scope.
func (w *NodeController) failCheckpointOnContainerExit(ctx context.Context, content *nvidiacomv1alpha1.PodSnapshotContent, pod *corev1.Pod) bool {
	var failed *corev1.ContainerStatus
	for i := range pod.Status.ContainerStatuses {
		cs := &pod.Status.ContainerStatuses[i]
		if cs.State.Terminated != nil && cs.State.Terminated.ExitCode != 0 {
			failed = cs
			break
		}
	}
	if failed == nil {
		return false
	}

	term := failed.State.Terminated
	message := fmt.Sprintf("checkpoint container %q terminated with exit code %d", failed.Name, term.ExitCode)
	if term.Reason != "" {
		message = fmt.Sprintf("%s: %s", message, term.Reason)
	}
	logger := logr.FromContextOrDiscard(ctx).WithValues("container", failed.Name)
	logger.Info("Checkpoint container failed", "exit_code", term.ExitCode, "reason", term.Reason)
	emitPodEvent(ctx, w.clientset, logger, pod, "snapshot", corev1.EventTypeWarning, "CheckpointFailed", message)
	w.killRunningContainers(ctx, logger, pod, fmt.Sprintf("checkpoint container %s failed", failed.Name))
	w.setSnapshotContentFailed(ctx, content, "CheckpointContainerFailed", errors.New(message))
	return true
}

// killRunningContainers SIGKILLs every still-running container in the pod, resolving each
// container's host PID through the node runtime. Best-effort: resolution and signal errors are
// logged and skipped so one stuck container does not block terminating the rest.
func (w *NodeController) killRunningContainers(ctx context.Context, logger logr.Logger, pod *corev1.Pod, reason string) {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.State.Running == nil || cs.ContainerID == "" {
			continue
		}
		containerID := snapshotruntime.StripCRIScheme(cs.ContainerID)
		resolveCtx, cancel := context.WithTimeout(ctx, containerResolveAttemptTimeout)
		pid, _, err := w.runtime.ResolveContainer(resolveCtx, containerID)
		cancel()
		if err != nil {
			logger.Error(err, "Failed to resolve running checkpoint container", "container", cs.Name)
			continue
		}
		if err := snapshotruntime.SendSignalToPID(logger, pid, syscall.SIGKILL, reason); err != nil {
			logger.Error(err, "Failed to signal running checkpoint container", "container", cs.Name)
		}
	}
}

// podLabelPatchBase returns a minimal Pod carrying only the identity + a clone of the source pod's
// labels, suitable as the MergeFrom base for a label-only patch — so the informer-cached pod is not
// mutated and the whole object is not deep-copied.
func podLabelPatchBase(pod *corev1.Pod) *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace: pod.Namespace,
		Name:      pod.Name,
		Labels:    maps.Clone(pod.Labels),
	}}
}

// labelCaptureEligible promotes a gate-validated source pod by adding CaptureEligibleLabel, which the
// source-pod informer keys on. Idempotent.
func (w *NodeController) labelCaptureEligible(ctx context.Context, pod *corev1.Pod) error {
	if pod.Labels[snapshotprotocol.CaptureEligibleLabel] == "true" {
		return nil
	}
	base := podLabelPatchBase(pod)
	updated := base.DeepCopy()
	if updated.Labels == nil {
		updated.Labels = map[string]string{}
	}
	updated.Labels[snapshotprotocol.CaptureEligibleLabel] = "true"
	return w.client.Patch(ctx, updated, client.MergeFrom(base))
}

// removeCaptureEligibleLabel drops CaptureEligibleLabel so the source-pod informer stops driving the
// pod after a terminal cancellation. Best-effort: a failure is logged, not surfaced.
func (w *NodeController) removeCaptureEligibleLabel(ctx context.Context, pod *corev1.Pod) {
	if _, ok := pod.Labels[snapshotprotocol.CaptureEligibleLabel]; !ok {
		return
	}
	base := podLabelPatchBase(pod)
	updated := base.DeepCopy()
	delete(updated.Labels, snapshotprotocol.CaptureEligibleLabel)
	if err := w.client.Patch(ctx, updated, client.MergeFrom(base)); err != nil {
		logr.FromContextOrDiscard(ctx).Error(err, "Failed to remove capture-eligible label", "pod", pod.Name)
	}
}

// setSnapshotContentSucceeded patches status with the Ready condition.
func (w *NodeController) setSnapshotContentSucceeded(ctx context.Context, content *nvidiacomv1alpha1.PodSnapshotContent) {
	patch := client.MergeFrom(content.DeepCopy())
	meta.SetStatusCondition(&content.Status.Conditions, metav1.Condition{
		Type:    nvidiacomv1alpha1.PodSnapshotConditionReady,
		Status:  metav1.ConditionTrue,
		Reason:  "Captured",
		Message: "Checkpoint captured and verified",
	})
	if err := w.client.Status().Patch(ctx, content, patch); err != nil {
		logr.FromContextOrDiscard(ctx).Error(err, "Failed to write PodSnapshotContent ready status", "content", content.Name)
	}
}

// setSnapshotContentFailed patches status with the Failed condition.
func (w *NodeController) setSnapshotContentFailed(ctx context.Context, content *nvidiacomv1alpha1.PodSnapshotContent, reason string, cause error) {
	patch := client.MergeFrom(content.DeepCopy())
	meta.SetStatusCondition(&content.Status.Conditions, metav1.Condition{
		Type:    nvidiacomv1alpha1.PodSnapshotConditionFailed,
		Status:  metav1.ConditionTrue,
		Reason:  reason,
		Message: cause.Error(),
	})
	if err := w.client.Status().Patch(ctx, content, patch); err != nil {
		logr.FromContextOrDiscard(ctx).Error(err, "Failed to write PodSnapshotContent failed status", "content", content.Name, "reason", reason)
	}
}

// executorCheckpoint is the production checkpointFn. The reconciler has already resolved the
// container ID and host PID. It runs executor.Checkpoint to the destination, verifies the
// artifact directory, and writes the snapshot-complete sentinel. On dump or verification
// failure it SIGKILLs the CUDA-locked process before returning the error.
func (w *NodeController) executorCheckpoint(ctx context.Context, params CheckpointParams) error {
	log := logr.FromContextOrDiscard(ctx)

	req := executor.CheckpointRequest{
		ContainerID:        params.ContainerID,
		ContainerName:      params.ContainerName,
		CheckpointID:       params.CheckpointID,
		CheckpointLocation: params.HostPath,
		StartedAt:          params.StartedAt,
		NodeName:           w.config.NodeName,
		PodName:            params.Pod.Name,
		PodNamespace:       params.Pod.Namespace,
		PodIP:              params.Pod.Status.PodIP,
		Clientset:          w.clientset,
	}
	if err := executor.Checkpoint(ctx, w.runtime, log, req, w.config); err != nil {
		w.killCheckpointProcess(log, params.ContainerPID, "checkpoint failed")
		return fmt.Errorf("checkpoint: %w", err)
	}

	info, statErr := os.Stat(params.HostPath)
	if statErr != nil || !info.IsDir() {
		w.killCheckpointProcess(log, params.ContainerPID, "checkpoint verification failed")
		if statErr != nil {
			return fmt.Errorf("verify checkpoint path %s: %w", params.HostPath, statErr)
		}
		return fmt.Errorf("verify checkpoint path %s: not a directory", params.HostPath)
	}

	if err := snapshotruntime.WriteControlSentinel(params.ContainerPID, snapshotprotocol.SnapshotCompleteFile); err != nil {
		w.killCheckpointProcess(log, params.ContainerPID, "checkpoint sentinel failed")
		return fmt.Errorf("write snapshot-complete sentinel: %w", err)
	}
	return nil
}

// killCheckpointProcess signals the CUDA-locked process so it does not hang after a failed dump.
func (w *NodeController) killCheckpointProcess(log logr.Logger, pid int, reason string) {
	if err := snapshotruntime.SendSignalToPID(log, pid, syscall.SIGKILL, reason); err != nil {
		log.Error(err, "Failed to signal checkpoint process", "reason", reason)
	}
}

// containerIDForName returns the running container's CRI-stripped ID, or "" if absent.
func containerIDForName(pod *corev1.Pod, containerName string) string {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName {
			return snapshotruntime.StripCRIScheme(cs.ContainerID)
		}
	}
	return ""
}

// isContentTerminal reports whether the work order already has a terminal condition.
func isContentTerminal(content *nvidiacomv1alpha1.PodSnapshotContent) bool {
	for _, t := range []string{nvidiacomv1alpha1.PodSnapshotConditionReady, nvidiacomv1alpha1.PodSnapshotConditionFailed} {
		if cond := meta.FindStatusCondition(content.Status.Conditions, t); cond != nil && cond.Status == metav1.ConditionTrue {
			return true
		}
	}
	return false
}

// artifactPresent reports whether a completed checkpoint directory already exists on disk.
func artifactPresent(destination string) bool {
	info, err := os.Stat(destination)
	return err == nil && info.IsDir()
}

// contentNameFromInformerObj extracts the object name from a dynamic informer object,
// handling the DeletedFinalStateUnknown tombstone.
func contentNameFromInformerObj(obj interface{}) (string, bool) {
	if accessor, err := meta.Accessor(obj); err == nil {
		return accessor.GetName(), true
	}
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		return "", false
	}
	accessor, err := meta.Accessor(tombstone.Obj)
	if err != nil {
		return "", false
	}
	return accessor.GetName(), true
}
