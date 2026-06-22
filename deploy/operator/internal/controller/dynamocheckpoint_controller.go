/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

var errCheckpointCleanupPending = errors.New("checkpoint cleanup pending")

// CheckpointReconciler reconciles a DynamoCheckpoint object
type CheckpointReconciler struct {
	client.Client
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig
	Recorder      record.EventRecorder
}

// GetRecorder returns the event recorder (implements controller_common.Reconciler interface)
func (r *CheckpointReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=podsnapshots,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;watch
// +kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch

//nolint:gocyclo
func (r *CheckpointReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Fetch the DynamoCheckpoint instance
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
	if err := r.Get(ctx, req.NamespacedName, ckpt); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	logger.Info("Reconciling DynamoCheckpoint", "name", ckpt.Name, "phase", ckpt.Status.Phase)

	if ckpt.GetDeletionTimestamp().IsZero() {
		if ckpt.Annotations != nil &&
			ckpt.Annotations[consts.CheckpointAutoAnnotation] == consts.KubeLabelValueTrue &&
			!commonController.ContainsFinalizer(ckpt) {
			commonController.AddFinalizer(ckpt)
			if err := r.Update(ctx, ckpt); err != nil {
				logger.Error(err, "Failed to add finalizer")
				return ctrl.Result{}, err
			}
		}
	} else {
		if commonController.ContainsFinalizer(ckpt) {
			if err := r.FinalizeResource(ctx, ckpt); err != nil {
				if errors.Is(err, errCheckpointCleanupPending) {
					logger.Info("Checkpoint cleanup pending", "reason", err.Error())
					return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
				}
				logger.Error(err, "Failed to call finalizer")
				return ctrl.Result{}, err
			}
			commonController.RemoveFinalizer(ckpt)
			if err := r.Update(ctx, ckpt); err != nil {
				logger.Error(err, "Failed to remove finalizer")
				return ctrl.Result{}, err
			}
		}
		return ctrl.Result{}, nil
	}

	checkpointID, err := checkpoint.CheckpointID(ckpt)
	if err != nil {
		logger.Error(err, "Failed to resolve checkpoint ID")
		return ctrl.Result{}, fmt.Errorf("failed to resolve checkpoint ID: %w", err)
	}

	if ckpt.Labels == nil {
		ckpt.Labels = map[string]string{}
	}
	if ckpt.Labels[snapshotprotocol.CheckpointIDLabel] != checkpointID {
		ckpt.Labels[snapshotprotocol.CheckpointIDLabel] = checkpointID
		if err := r.Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.Get(ctx, req.NamespacedName, ckpt); err != nil {
			return ctrl.Result{}, err
		}
	}

	needsStatusUpdate := false
	phaseWasEmpty := ckpt.Status.Phase == ""
	if ckpt.Status.CheckpointID != checkpointID {
		ckpt.Status.CheckpointID = checkpointID
		needsStatusUpdate = true
	}
	if ckpt.Status.IdentityHash != checkpointID {
		ckpt.Status.IdentityHash = checkpointID
		needsStatusUpdate = true
	}
	existing, err := checkpoint.FindCheckpointByCheckpointID(ctx, r.Client, ckpt.Namespace, checkpointID, ckpt.Name)
	if err != nil {
		return ctrl.Result{}, err
	}
	if existing != nil {
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
		ckpt.Status.JobName = ""
		ckpt.Status.CreatedAt = nil
		ckpt.Status.Message = fmt.Sprintf("checkpoint ID %s is already owned by %s", checkpointID, existing.Name)
		if err := r.Status().Update(ctx, ckpt); err != nil {
			logger.Error(err, "Failed to mark duplicate DynamoCheckpoint as failed")
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}
	desiredJobName := snapshotprotocol.GetCheckpointJobName(
		checkpointID,
		ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation],
	)
	switch ckpt.Status.Phase {
	case "", nvidiacomv1alpha1.DynamoCheckpointPhasePending, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, nvidiacomv1alpha1.DynamoCheckpointPhaseReady, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed:
	default:
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
		ckpt.Status.Message = ""
		needsStatusUpdate = true
	}
	if ckpt.Status.Phase == "" {
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
		ckpt.Status.Message = ""
		needsStatusUpdate = true
	}
	if ckpt.Status.Phase != nvidiacomv1alpha1.DynamoCheckpointPhaseCreating &&
		ckpt.Status.JobName != "" &&
		ckpt.Status.JobName != desiredJobName {
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
		ckpt.Status.JobName = ""
		ckpt.Status.CreatedAt = nil
		ckpt.Status.Message = ""
		needsStatusUpdate = true
	}
	if needsStatusUpdate {
		if err := r.Status().Update(ctx, ckpt); err != nil {
			logger.Error(err, "Failed to initialize DynamoCheckpoint status")
			return ctrl.Result{}, err
		}
		if phaseWasEmpty {
			return ctrl.Result{}, nil
		}
	}

	// Handle based on current phase
	switch ckpt.Status.Phase {
	case nvidiacomv1alpha1.DynamoCheckpointPhasePending:
		return r.handlePending(ctx, ckpt)
	case nvidiacomv1alpha1.DynamoCheckpointPhaseCreating:
		return r.handleCreating(ctx, ckpt)
	case nvidiacomv1alpha1.DynamoCheckpointPhaseReady:
		// Nothing to do, checkpoint is ready
		return ctrl.Result{}, nil
	case nvidiacomv1alpha1.DynamoCheckpointPhaseFailed:
		return ctrl.Result{}, nil
	default:
		// Unknown phase, reset to Pending
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}
}

func (r *CheckpointReconciler) handlePending(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if err := checkpoint.ValidateGMSSnapshotGate("spec.gpuMemoryService", true, ckpt.Spec.GPUMemoryService); err != nil {
		return r.failPendingCheckpoint(ctx, ckpt, "GMSSnapshotDisabled", err)
	}
	if err := checkpoint.ValidatePreparedGPUMemoryServicePodTemplate(ckpt); err != nil {
		return r.failPendingCheckpoint(ctx, ckpt, "GMSPodTemplateNotPrepared", err)
	}

	hash := ckpt.Status.CheckpointID
	if hash == "" {
		hash = ckpt.Status.IdentityHash
	}
	if hash == "" {
		var err error
		hash, err = checkpoint.CheckpointID(ckpt)
		if err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to resolve checkpoint ID: %w", err)
		}
	}

	jobName := snapshotprotocol.GetCheckpointJobName(
		hash,
		ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation],
	)

	// Use SyncResource to create/update the checkpoint Job
	modified, _, err := commonController.SyncResource(ctx, r, ckpt, func(ctx context.Context) (*batchv1.Job, bool, error) {
		job, err := buildCheckpointJob(ctx, r.Client, r.Config, ckpt, jobName)
		return job, false, err
	})
	if err != nil {
		logger.Error(err, "Failed to sync checkpoint Job")
		return ctrl.Result{}, err
	}

	if modified {
		logger.Info("Created/updated checkpoint Job", "job", jobName)
	}

	// Update status to Creating phase
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseCreating
	ckpt.Status.JobName = jobName
	ckpt.Status.CreatedAt = nil
	ckpt.Status.Message = ""
	meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
		Type:    string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated),
		Status:  metav1.ConditionTrue,
		Reason:  "JobCreated",
		Message: fmt.Sprintf("Checkpoint job %s created", jobName),
	})

	if err := r.Status().Update(ctx, ckpt); err != nil {
		return ctrl.Result{}, err
	}

	// Status update will trigger next reconcile via watch
	return ctrl.Result{}, nil
}

func (r *CheckpointReconciler) failPendingCheckpoint(
	ctx context.Context,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	reason string,
	err error,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
	ckpt.Status.JobName = ""
	ckpt.Status.CreatedAt = nil
	ckpt.Status.Message = err.Error()
	meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
		Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated),
		Status:             metav1.ConditionFalse,
		Reason:             reason,
		Message:            err.Error(),
		LastTransitionTime: metav1.Now(),
	})
	if updateErr := r.Status().Update(ctx, ckpt); updateErr != nil {
		logger.Error(updateErr, "Failed to mark DynamoCheckpoint as failed")
		return ctrl.Result{}, updateErr
	}
	return ctrl.Result{}, nil
}

func (r *CheckpointReconciler) handleCreating(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error) {
	if ckpt.Status.JobName == "" {
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
		ckpt.Status.Message = "checkpoint job is missing from status"
		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	// Check Job status
	job := &batchv1.Job{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: ckpt.Namespace, Name: ckpt.Status.JobName}, job); err != nil {
		if apierrors.IsNotFound(err) {
			ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
			ckpt.Status.Message = "checkpoint job was deleted"
			meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
				Type:    string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated),
				Status:  metav1.ConditionFalse,
				Reason:  "JobDeleted",
				Message: "Checkpoint job was deleted",
			})
			if err := r.Status().Update(ctx, ckpt); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Required step: create the PodSnapshot once the source pod exists. The checkpoint cannot
	// reach Ready without it, so creation failure fails or requeues the capture.
	checkpointID, err := checkpoint.CheckpointID(ckpt)
	if err != nil {
		return ctrl.Result{}, err
	}
	pod, err := r.findSourcePod(ctx, job)
	if err != nil {
		if client.IgnoreNotFound(err) == nil {
			return ctrl.Result{RequeueAfter: time.Second}, nil
		}
		return ctrl.Result{}, err
	}
	if err := r.ensurePodSnapshot(ctx, ckpt, checkpointID, pod.Name); err != nil {
		if commonController.IgnoreIntermediateError(err) != nil {
			r.updateFailedStatus(ctx, ckpt, err)
		}
		return ctrl.Result{}, err
	}

	return r.observePodSnapshot(ctx, ckpt, job, checkpointID)
}

// observePodSnapshot maps the bound PodSnapshot's status (and the owned Job's failure / deadline
// hang guards) onto the DynamoCheckpoint phase. Completion cascades up from PodSnapshotContent
// → PodSnapshot → DynamoCheckpoint, so this never reads the Job's terminal annotation.
func (r *CheckpointReconciler) observePodSnapshot(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, job *batchv1.Job, checkpointID string) (ctrl.Result, error) {
	snap := &nvidiacomv1alpha1.PodSnapshot{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: ckpt.Namespace, Name: podSnapshotName(checkpointID)}, snap); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{RequeueAfter: time.Second}, nil
		}
		return ctrl.Result{}, err
	}

	// A PodSnapshot can fail before it is bound (e.g. the PodSnapshotReconciler rejects the
	// source pod), so always observe Failed. Ready is only meaningful once bound.
	if nvidiacomv1alpha1.IsPodSnapshotFailed(snap) {
		return r.failCreating(ctx, ckpt, "PodSnapshotFailed", podSnapshotConditionMessage(snap, nvidiacomv1alpha1.PodSnapshotConditionFailed))
	}
	if snap.Status.BoundPodSnapshotContentName != nil && nvidiacomv1alpha1.IsPodSnapshotSucceeded(snap) {
		return r.markCheckpointReady(ctx, ckpt, checkpointID, podSnapshotConditionMessage(snap, nvidiacomv1alpha1.PodSnapshotConditionReady))
	}

	// Hang guard 1: the owned Job failed while the PodSnapshot is still non-terminal.
	if jobFailed, message := checkpointJobFailed(job); jobFailed {
		return r.failCreating(ctx, ckpt, "JobFailed", message)
	}

	// Hang guard 2: the Job ran past its deadline without a terminal PodSnapshot.
	if job.Spec.ActiveDeadlineSeconds != nil {
		deadline := job.CreationTimestamp.Add(time.Duration(*job.Spec.ActiveDeadlineSeconds) * time.Second)
		if time.Now().After(deadline) {
			return r.failCreating(ctx, ckpt, "CheckpointDeadlineExceeded",
				fmt.Sprintf("checkpoint did not complete before the Job deadline (%s)", deadline.Format(time.RFC3339)))
		}
	}

	return ctrl.Result{}, nil
}

// failCreating marks the DynamoCheckpoint Failed with a completion-condition reason.
func (r *CheckpointReconciler) failCreating(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, reason, message string) (ctrl.Result, error) {
	log.FromContext(ctx).Info("Checkpoint failed", "reason", reason, "message", message)
	r.Recorder.Event(ckpt, corev1.EventTypeWarning, "CheckpointFailed", message)
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
	ckpt.Status.Message = message
	meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
		Type:    string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
		Status:  metav1.ConditionFalse,
		Reason:  reason,
		Message: message,
	})
	return ctrl.Result{}, r.Status().Update(ctx, ckpt)
}

// markCheckpointReady marks the DynamoCheckpoint Ready after its bound PodSnapshot succeeded.
func (r *CheckpointReconciler) markCheckpointReady(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint, checkpointID, message string) (ctrl.Result, error) {
	log.FromContext(ctx).Info("Checkpoint ready", "checkpointID", checkpointID)
	r.Recorder.Event(ckpt, corev1.EventTypeNormal, "CheckpointReady", message)
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseReady
	ckpt.Status.CheckpointID = checkpointID
	ckpt.Status.CreatedAt = ptr.To(metav1.Now())
	ckpt.Status.Message = ""
	meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
		Type:    string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
		Status:  metav1.ConditionTrue,
		Reason:  "PodSnapshotReady",
		Message: message,
	})
	return ctrl.Result{}, r.Status().Update(ctx, ckpt)
}

// podSnapshotConditionMessage returns the message of the named PodSnapshot condition, or "".
func podSnapshotConditionMessage(snap *nvidiacomv1alpha1.PodSnapshot, condType string) string {
	if cond := meta.FindStatusCondition(snap.Status.Conditions, condType); cond != nil {
		return cond.Message
	}
	return ""
}

// checkpointJobFailed reports whether the Job has a True JobFailed condition.
func checkpointJobFailed(job *batchv1.Job) (bool, string) {
	for _, condition := range job.Status.Conditions {
		if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
			message := "checkpoint job failed"
			if condition.Message != "" {
				message = fmt.Sprintf("%s: %s", message, condition.Message)
			}
			return true, message
		}
	}
	return false, ""
}

//nolint:gocyclo
func (r *CheckpointReconciler) FinalizeResource(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) error {
	logger := log.FromContext(ctx)
	if ckpt == nil || ckpt.Annotations == nil || ckpt.Annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue {
		return nil
	}
	if r.Config == nil {
		logger.Info("Automatic checkpoint artifact cleanup skipped because operator configuration is not available")
		return nil
	}

	checkpointID, err := checkpoint.CheckpointID(ckpt)
	if err != nil {
		return err
	}

	storage, ok, err := checkpoint.StorageFromConfig(r.Config.Checkpoint.Storage)
	if err != nil {
		return err
	}
	if !ok {
		daemonSets := &appsv1.DaemonSetList{}
		if err := r.List(
			ctx,
			daemonSets,
			client.InNamespace(ckpt.Namespace),
			client.MatchingLabels{snapshotprotocol.SnapshotAgentLabelKey: snapshotprotocol.SnapshotAgentLabelValue},
		); err != nil {
			return fmt.Errorf("list snapshot-agent daemonsets in %s: %w", ckpt.Namespace, err)
		}
		storage, err = snapshotprotocol.DiscoverStorageFromDaemonSets(ckpt.Namespace, daemonSets.Items)
		if err != nil {
			return fmt.Errorf("discover snapshot-agent storage for automatic checkpoint cleanup: %w", err)
		}
	}

	job, err := buildCheckpointCleanupJob(r.Config, ckpt, checkpointID, storage)
	if err != nil {
		return err
	}
	current := &batchv1.Job{}
	jobKey := client.ObjectKey{Namespace: job.Namespace, Name: job.Name}
	if err := r.Get(ctx, jobKey, current); err != nil {
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("get checkpoint cleanup job %s/%s: %w", job.Namespace, job.Name, err)
		}
		if err := r.Create(ctx, job.DeepCopy()); err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("create checkpoint cleanup job %s/%s: %w", job.Namespace, job.Name, err)
		}
		return fmt.Errorf("%w: job %s/%s created", errCheckpointCleanupPending, job.Namespace, job.Name)
	}
	if current.Labels[snapshotprotocol.CheckpointIDLabel] != checkpointID {
		return fmt.Errorf("checkpoint cleanup job %s/%s already exists for checkpoint ID %q", job.Namespace, job.Name, current.Labels[snapshotprotocol.CheckpointIDLabel])
	}

	for _, condition := range current.Status.Conditions {
		switch {
		case condition.Type == batchv1.JobComplete && condition.Status == corev1.ConditionTrue:
			if err := r.Delete(ctx, current); err != nil && !apierrors.IsNotFound(err) {
				return fmt.Errorf("delete completed checkpoint cleanup job %s/%s: %w", current.Namespace, current.Name, err)
			}
			return nil
		case condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue:
			if err := r.Delete(ctx, current); err != nil && !apierrors.IsNotFound(err) {
				return fmt.Errorf("delete failed checkpoint cleanup job %s/%s: %w", current.Namespace, current.Name, err)
			}
			return fmt.Errorf("%w: job %s/%s failed and was deleted for retry: %s", errCheckpointCleanupPending, current.Namespace, current.Name, condition.Message)
		}
	}
	return fmt.Errorf("%w: job %s/%s is still running", errCheckpointCleanupPending, job.Namespace, job.Name)
}

// SetupWithManager sets up the controller with the Manager.
func (r *CheckpointReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoCheckpoint{}).
		Owns(&batchv1.Job{}, builder.WithPredicates(predicate.Funcs{
			// Ignore creation - we don't need to reconcile when we just created the Job
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		Owns(&nvidiacomv1alpha1.PodSnapshot{}, builder.WithPredicates(predicate.Funcs{
			// Ignore create (we just created it). Watch update (status mirror) and
			// delete (re-enqueue to recreate / unblock). Delete is safe: reconcile
			// exits at the deletion-timestamp guard before reaching observePodSnapshot.
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return false },
		})).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}
