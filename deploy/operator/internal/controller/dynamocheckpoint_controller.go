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
	"k8s.io/utils/ptr"
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

// Helper function to compute checkpoint location from operator config
func (r *CheckpointReconciler) getCheckpointLocation(identityHash string) string {
	basePath := checkpoint.GetPVCBasePath(&r.Config.Checkpoint)
	return fmt.Sprintf("%s/%s", basePath, identityHash)
}

// Helper function to get checkpoint storage type from operator config
func (r *CheckpointReconciler) getCheckpointStorageType() nvidiacomv1alpha1.DynamoCheckpointStorageType {
	return nvidiacomv1alpha1.DynamoCheckpointStorageType(commonController.CheckpointStorageTypePVC)
}

// GetRecorder returns the event recorder (implements controller_common.Reconciler interface)
func (r *CheckpointReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
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

		// Set checkpoint location and storage type using helper functions
		ckpt.Status.Location = r.getCheckpointLocation(ckpt.Status.IdentityHash)
		ckpt.Status.StorageType = r.getCheckpointStorageType()

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
			Message:            fmt.Sprintf("Checkpoint available at %s", ckpt.Status.Location),
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

	// Add signal volume (hostPath for communication with DaemonSet)
	// The DaemonSet writes a signal file after checkpoint is complete
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

	// Compute the signal file path - unique per checkpoint hash
	signalFilePath := consts.CheckpointSignalMountPath + "/" + ckpt.Status.IdentityHash + ".done"

	// Add initContainer to clean up any leftover signal file from previous runs
	// This ensures a fresh start for each checkpoint job without affecting the checkpoint itself
	// InitContainers complete before the main container starts, so they don't appear in the checkpoint
	initContainerImage := r.Config.Checkpoint.InitContainerImage

	podTemplate.Spec.InitContainers = append(podTemplate.Spec.InitContainers, corev1.Container{
		Name:  "cleanup-signal-file",
		Image: initContainerImage,
		Command: []string{
			"sh",
			"-c",
			fmt.Sprintf("rm -f %s || true; echo 'Signal file cleanup complete'", signalFilePath),
		},
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      consts.CheckpointSignalVolumeName,
				MountPath: consts.CheckpointSignalMountPath,
			},
		},
	})

	// Add checkpoint env vars and volume mounts to main container
	if len(podTemplate.Spec.Containers) > 0 {
		mainContainer := &podTemplate.Spec.Containers[0]

		// Compute checkpoint location and storage type using helper functions
		checkpointLocation := r.getCheckpointLocation(ckpt.Status.IdentityHash)
		storageType := string(r.getCheckpointStorageType())

		// Add checkpoint-related env vars
		mainContainer.Env = append(mainContainer.Env,
			// Signal file: DaemonSet writes this after checkpoint completes
			corev1.EnvVar{
				Name:  consts.EnvCheckpointSignalFile,
				Value: signalFilePath,
			},
			// Ready file: Worker creates this when model is loaded
			corev1.EnvVar{
				Name:  consts.EnvCheckpointReadyFile,
				Value: consts.CheckpointReadyFilePath,
			},
			// Checkpoint hash: For idempotency check
			corev1.EnvVar{
				Name:  consts.EnvCheckpointHash,
				Value: ckpt.Status.IdentityHash,
			},
			// Checkpoint location: For idempotency check
			corev1.EnvVar{
				Name:  consts.EnvCheckpointLocation,
				Value: checkpointLocation,
			},
			// Storage type: For idempotency check (pvc, s3, oci)
			corev1.EnvVar{
				Name:  consts.EnvCheckpointStorageType,
				Value: storageType,
			},
		)

		// Add signal volume mount (required for DaemonSet communication)
		mainContainer.VolumeMounts = append(mainContainer.VolumeMounts,
			corev1.VolumeMount{
				Name:      consts.CheckpointSignalVolumeName,
				MountPath: consts.CheckpointSignalMountPath,
			},
		)

		// Add checkpoint PVC volume and mount for mount namespace consistency with restore pods
		// CRIU requires the exact same mount layout between checkpoint and restore
		if r.Config.Checkpoint.Storage.PVC.PVCName != "" {
			pvcName := r.Config.Checkpoint.Storage.PVC.PVCName
			basePath := r.Config.Checkpoint.Storage.PVC.BasePath
			if basePath == "" {
				basePath = consts.CheckpointBasePath
			}
			checkpoint.InjectCheckpointVolume(&podTemplate.Spec, pvcName)
			checkpoint.InjectCheckpointVolumeMount(mainContainer, basePath)
		}

		// Add Downward API volume for pod identity (mount namespace consistency with restore pods)
		checkpoint.InjectPodInfoVolume(&podTemplate.Spec)
		checkpoint.InjectPodInfoVolumeMount(mainContainer)

		// Override probes for checkpoint mode
		// Checkpoint jobs need different probe behavior than regular worker pods:
		// - Readiness: Wait for model to load before checkpoint
		// - Liveness/Startup: Remove to prevent restarts during slow model loading
		mainContainer.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{"cat", consts.CheckpointReadyFilePath},
				},
			},
			InitialDelaySeconds: 15,
			PeriodSeconds:       2,
		}
		// Remove liveness probe - we don't want restarts during model loading
		mainContainer.LivenessProbe = nil
		// Remove startup probe - not needed for checkpoint jobs
		mainContainer.StartupProbe = nil
	}

	// Set restart policy to Never for Jobs
	podTemplate.Spec.RestartPolicy = corev1.RestartPolicyNever

	// Apply seccomp profile to block io_uring syscalls
	// CRIU doesn't support io_uring memory mappings, so we must block these syscalls
	podTemplate.Spec.SecurityContext = &corev1.PodSecurityContext{
		SeccompProfile: &corev1.SeccompProfile{
			Type:             corev1.SeccompProfileTypeLocalhost,
			LocalhostProfile: ptr.To("profiles/block-iouring.json"),
		},
	}

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
