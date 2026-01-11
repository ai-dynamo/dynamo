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
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
)

// CheckpointReconciler reconciles a DynamoCheckpoint object
type CheckpointReconciler struct {
	client.Client
	Config   commonController.Config
	Recorder record.EventRecorder
}

// GetRecorder returns the event recorder (implements controller_common.Reconciler interface)
func (r *CheckpointReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}

// getCheckpointPVCName returns the configured PVC name, or the default if not set
func (r *CheckpointReconciler) getCheckpointPVCName() string {
	if r.Config.Checkpoint.Enabled && r.Config.Checkpoint.Storage.PVC.PVCName != "" {
		return r.Config.Checkpoint.Storage.PVC.PVCName
	}
	return checkpoint.DefaultCheckpointPVCName
}

// getSignalHostPath returns the configured signal host path, or the default if not set
func (r *CheckpointReconciler) getSignalHostPath() string {
	if r.Config.Checkpoint.Enabled && r.Config.Checkpoint.Storage.SignalHostPath != "" {
		return r.Config.Checkpoint.Storage.SignalHostPath
	}
	return consts.CheckpointSignalHostPath
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete

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

	// Compute identity hash if not already set
	if ckpt.Status.IdentityHash == "" {
		hash, err := checkpoint.ComputeIdentityHash(ckpt.Spec.Identity)
		if err != nil {
			logger.Error(err, "Failed to compute identity hash")
			return ctrl.Result{}, fmt.Errorf("failed to compute identity hash: %w", err)
		}

		ckpt.Status.IdentityHash = hash
		ckpt.Status.TarPath = checkpoint.GetTarPath(checkpoint.GetCheckpointBasePath(&r.Config.Checkpoint), hash)
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending

		if err := r.Status().Update(ctx, ckpt); err != nil {
			logger.Error(err, "Failed to update DynamoCheckpoint status with hash")
			return ctrl.Result{}, err
		}
		// Status update will trigger a new reconcile via the watch
		return ctrl.Result{}, nil
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
		// Could implement retry logic here
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

	jobName := fmt.Sprintf("checkpoint-%s", ckpt.Name)

	// Use SyncResource to create/update the checkpoint Job
	modified, _, err := commonController.SyncResource(ctx, r, ckpt, func(ctx context.Context) (*batchv1.Job, bool, error) {
		job := r.buildCheckpointJob(ckpt, jobName)
		return job, false, nil
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
	meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
		Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated),
		Status:             metav1.ConditionTrue,
		Reason:             "JobCreated",
		Message:            fmt.Sprintf("Checkpoint job %s created", jobName),
		LastTransitionTime: metav1.Now(),
	})

	if err := r.Status().Update(ctx, ckpt); err != nil {
		return ctrl.Result{}, err
	}

	// Status update will trigger next reconcile via watch
	return ctrl.Result{}, nil
}

func (r *CheckpointReconciler) handleCreating(ctx context.Context, ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Check Job status
	job := &batchv1.Job{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: ckpt.Namespace, Name: ckpt.Status.JobName}, job); err != nil {
		if apierrors.IsNotFound(err) {
			// Job was deleted, go back to Pending
			ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
			ckpt.Status.JobName = ""
			meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
				Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated),
				Status:             metav1.ConditionFalse,
				Reason:             "JobDeleted",
				Message:            "Checkpoint job was deleted",
				LastTransitionTime: metav1.Now(),
			})
			if err := r.Status().Update(ctx, ckpt); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Check if job succeeded
	if job.Status.Succeeded > 0 {
		logger.Info("Checkpoint Job succeeded", "job", job.Name)
		r.Recorder.Event(ckpt, corev1.EventTypeNormal, "CheckpointReady", "Checkpoint creation completed successfully")

		now := metav1.Now()
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseReady
		ckpt.Status.CreatedAt = &now
		meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
			Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
			Status:             metav1.ConditionTrue,
			Reason:             "JobSucceeded",
			Message:            "Checkpoint job completed successfully",
			LastTransitionTime: metav1.Now(),
		})
		meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
			Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionTarAvailable),
			Status:             metav1.ConditionTrue,
			Reason:             "TarCreated",
			Message:            fmt.Sprintf("Checkpoint tar available at %s", ckpt.Status.TarPath),
			LastTransitionTime: metav1.Now(),
		})

		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	// Check if job failed
	if job.Status.Failed > 0 {
		logger.Info("Checkpoint Job failed", "job", job.Name)
		r.Recorder.Event(ckpt, corev1.EventTypeWarning, "CheckpointFailed", "Checkpoint creation failed")

		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
		ckpt.Status.Message = "Checkpoint job failed"
		meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
			Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
			Status:             metav1.ConditionFalse,
			Reason:             "JobFailed",
			Message:            "Checkpoint job failed",
			LastTransitionTime: metav1.Now(),
		})

		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	// Job is still running - we'll be notified via Update event when status changes
	return ctrl.Result{}, nil
}

func (r *CheckpointReconciler) buildCheckpointJob(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, jobName string) *batchv1.Job {
	// Use the pod template from the spec
	podTemplate := ckpt.Spec.Job.PodTemplateSpec.DeepCopy()

	// Add checkpoint-related labels
	if podTemplate.Labels == nil {
		podTemplate.Labels = make(map[string]string)
	}
	podTemplate.Labels[consts.KubeLabelCheckpointName] = ckpt.Name
	podTemplate.Labels[consts.KubeLabelCheckpointHash] = ckpt.Status.IdentityHash
	podTemplate.Labels[consts.KubeLabelCheckpointSource] = "true"

	// Add checkpoint PVC volume
	podTemplate.Spec.Volumes = append(podTemplate.Spec.Volumes, corev1.Volume{
		Name: consts.CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: r.getCheckpointPVCName(),
			},
		},
	})

	// Add signal volume (hostPath for communication with DaemonSet)
	// Both the checkpoint pod and DaemonSet mount the same hostPath directory
	hostPathType := corev1.HostPathDirectoryOrCreate
	podTemplate.Spec.Volumes = append(podTemplate.Spec.Volumes, corev1.Volume{
		Name: consts.CheckpointSignalVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: r.getSignalHostPath(),
				Type: &hostPathType,
			},
		},
	})

	// Add checkpoint env vars and volume mounts to main container
	if len(podTemplate.Spec.Containers) > 0 {
		mainContainer := &podTemplate.Spec.Containers[0]

		// Add environment variables
		// Only CHECKPOINT_PATH is required - its presence indicates checkpoint mode
		// Compute the signal file path - unique per checkpoint hash
		// The DaemonSet writes this file after checkpoint is complete
		// The pod waits for this file, then exits successfully
		signalFilePath := consts.CheckpointSignalMountPath + "/" + ckpt.Status.IdentityHash + ".done"

		mainContainer.Env = append(mainContainer.Env,
			corev1.EnvVar{
				Name:  consts.EnvCheckpointPath,
				Value: ckpt.Status.TarPath,
			},
			corev1.EnvVar{
				Name:  consts.EnvCheckpointHash,
				Value: ckpt.Status.IdentityHash,
			},
			corev1.EnvVar{
				Name:  consts.EnvCheckpointSignalFile,
				Value: signalFilePath,
			},
		)

		// Add volume mounts
		mainContainer.VolumeMounts = append(mainContainer.VolumeMounts,
			corev1.VolumeMount{
				Name:      consts.CheckpointVolumeName,
				MountPath: checkpoint.GetCheckpointBasePath(&r.Config.Checkpoint),
			},
			corev1.VolumeMount{
				Name:      consts.CheckpointSignalVolumeName,
				MountPath: consts.CheckpointSignalMountPath,
			},
		)
	}

	// Set restart policy to Never for Jobs
	podTemplate.Spec.RestartPolicy = corev1.RestartPolicyNever

	// Build the Job
	activeDeadlineSeconds := ckpt.Spec.Job.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		defaultDeadline := int64(3600)
		activeDeadlineSeconds = &defaultDeadline
	}

	backoffLimit := ckpt.Spec.Job.BackoffLimit
	if backoffLimit == nil {
		defaultBackoff := int32(3)
		backoffLimit = &defaultBackoff
	}

	ttlSeconds := ckpt.Spec.Job.TTLSecondsAfterFinished
	if ttlSeconds == nil {
		defaultTTL := int32(300)
		ttlSeconds = &defaultTTL
	}

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: ckpt.Namespace,
			Labels: map[string]string{
				consts.KubeLabelCheckpointName: ckpt.Name,
				consts.KubeLabelCheckpointHash: ckpt.Status.IdentityHash,
			},
		},
		Spec: batchv1.JobSpec{
			ActiveDeadlineSeconds:   activeDeadlineSeconds,
			BackoffLimit:            backoffLimit,
			TTLSecondsAfterFinished: ttlSeconds,
			Template:                *podTemplate,
		},
	}

	return job
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
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config)).
		Complete(r)
}
