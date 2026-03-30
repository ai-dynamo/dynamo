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
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
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
)

const (
	checkpointStatusAnnotation = "nvidia.com/snapshot-checkpoint-status"
	checkpointStatusCompleted  = "completed"
	checkpointStatusFailed     = "failed"
)

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

func checkpointLeaseExpired(lease *coordinationv1.Lease, now time.Time) bool {
	if lease.Spec.LeaseDurationSeconds == nil {
		return true
	}
	leaseTime := lease.Spec.RenewTime
	if leaseTime == nil {
		leaseTime = lease.Spec.AcquireTime
	}
	if leaseTime == nil {
		return true
	}
	return now.After(leaseTime.Time.Add(time.Duration(*lease.Spec.LeaseDurationSeconds) * time.Second))
}

func desiredArtifactVersion(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) string {
	version := consts.DefaultCheckpointArtifactVersion
	if ckpt.Annotations == nil {
		return version
	}

	annotatedVersion := strings.TrimSpace(ckpt.Annotations[consts.KubeAnnotationCheckpointArtifactVersion])
	if annotatedVersion != "" {
		version = annotatedVersion
	}
	return version
}

func desiredCheckpointJobName(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, identityHash string) string {
	return "checkpoint-job-" + identityHash + "-" + desiredArtifactVersion(ckpt)
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch

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

	identityHash, err := checkpoint.ComputeIdentityHash(ckpt.Spec.Identity)
	if err != nil {
		logger.Error(err, "Failed to compute checkpoint identity hash")
		return ctrl.Result{}, fmt.Errorf("failed to compute checkpoint identity hash: %w", err)
	}

	if ckpt.Labels == nil {
		ckpt.Labels = map[string]string{}
	}
	if ckpt.Labels[consts.KubeLabelCheckpointHash] != identityHash {
		ckpt.Labels[consts.KubeLabelCheckpointHash] = identityHash
		if err := r.Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.Get(ctx, req.NamespacedName, ckpt); err != nil {
			return ctrl.Result{}, err
		}
	}

	needsStatusUpdate := false
	phaseWasEmpty := ckpt.Status.Phase == ""
	if ckpt.Status.IdentityHash != identityHash {
		ckpt.Status.IdentityHash = identityHash
		needsStatusUpdate = true
	}
	existing, err := checkpoint.FindCheckpointByIdentityHash(ctx, r.Client, ckpt.Namespace, identityHash, ckpt.Name)
	if err != nil {
		return ctrl.Result{}, err
	}
	if existing != nil {
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
		ckpt.Status.JobName = ""
		ckpt.Status.CreatedAt = nil
		ckpt.Status.Message = fmt.Sprintf("checkpoint identity hash %s is already owned by %s", identityHash, existing.Name)
		if err := r.Status().Update(ctx, ckpt); err != nil {
			logger.Error(err, "Failed to mark duplicate DynamoCheckpoint as failed")
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}
	desiredJobName := desiredCheckpointJobName(ckpt, identityHash)
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

	hash := ckpt.Status.IdentityHash
	if hash == "" {
		var err error
		hash, err = checkpoint.ComputeIdentityHash(ckpt.Spec.Identity)
		if err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to compute checkpoint identity hash: %w", err)
		}
	}
	version := desiredArtifactVersion(ckpt)
	jobName := desiredCheckpointJobName(ckpt, hash)
	location, storageType, err := checkpoint.ResolveCheckpointStorage(hash, version, &r.Config.Checkpoint)
	if err != nil {
		return ctrl.Result{}, err
	}

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
	ckpt.Status.Location = location
	ckpt.Status.StorageType = storageType
	ckpt.Status.CreatedAt = nil
	ckpt.Status.Message = ""
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

	jobComplete := false
	jobFailed := false
	for _, condition := range job.Status.Conditions {
		if condition.Status != corev1.ConditionTrue {
			continue
		}
		if condition.Type == batchv1.JobComplete {
			jobComplete = true
			continue
		}
		if condition.Type == batchv1.JobFailed {
			jobFailed = true
		}
	}

	status := job.Annotations[checkpointStatusAnnotation]
	if status == checkpointStatusFailed {
		reason := "JobFailed"
		message := "Checkpoint job failed"
		if jobComplete {
			reason = "CheckpointVerificationFailed"
			message = "Checkpoint job completed but snapshot-agent reported checkpoint failure"
		}

		logger.Info("Checkpoint Job failed", "job", job.Name, "checkpoint_status", status)
		r.Recorder.Event(ckpt, corev1.EventTypeWarning, "CheckpointFailed", message)

		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
		ckpt.Status.Message = message
		meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
			Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
			Status:             metav1.ConditionFalse,
			Reason:             reason,
			Message:            message,
			LastTransitionTime: metav1.Now(),
		})

		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	if jobComplete {
		if status != checkpointStatusCompleted {
			lease := &coordinationv1.Lease{}
			leaseKey := client.ObjectKey{Namespace: job.Namespace, Name: job.Name}
			if err := r.Get(ctx, leaseKey, lease); err != nil {
				if !apierrors.IsNotFound(err) {
					return ctrl.Result{}, err
				}
			} else if !checkpointLeaseExpired(lease, time.Now()) {
				logger.V(1).Info("Checkpoint job is complete but checkpoint lease is still active; waiting for terminal watcher status", "job", job.Name)
				return ctrl.Result{RequeueAfter: time.Second}, nil
			}

			reason := "CheckpointVerificationFailed"
			message := "Checkpoint job completed without snapshot-agent completion confirmation"
			if status == checkpointStatusFailed {
				message = "Checkpoint job completed but snapshot-agent reported checkpoint failure"
			}

			logger.Info("Checkpoint Job completed without usable artifact", "job", job.Name, "checkpoint_status", status)
			r.Recorder.Event(ckpt, corev1.EventTypeWarning, "CheckpointFailed", message)

			ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
			ckpt.Status.Message = message
			meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
				Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
				Status:             metav1.ConditionFalse,
				Reason:             reason,
				Message:            message,
				LastTransitionTime: metav1.Now(),
			})

			if err := r.Status().Update(ctx, ckpt); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil
		}

		logger.Info("Checkpoint Job succeeded", "job", job.Name)
		r.Recorder.Event(ckpt, corev1.EventTypeNormal, "CheckpointReady", "Checkpoint creation completed successfully")

		if ckpt.Status.Location == "" || ckpt.Status.StorageType == "" {
			version := desiredArtifactVersion(ckpt)
			location, storageType, err := checkpoint.ResolveCheckpointStorage(
				ckpt.Status.IdentityHash,
				version,
				&r.Config.Checkpoint,
			)
			if err != nil {
				return ctrl.Result{}, err
			}
			ckpt.Status.Location = location
			ckpt.Status.StorageType = storageType
		}

		now := metav1.Now()
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseReady
		ckpt.Status.CreatedAt = &now
		ckpt.Status.Message = ""
		meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
			Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
			Status:             metav1.ConditionTrue,
			Reason:             "JobSucceeded",
			Message:            fmt.Sprintf("Checkpoint job completed, available at %s", ckpt.Status.Location),
			LastTransitionTime: metav1.Now(),
		})

		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	if jobFailed {
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

func (r *CheckpointReconciler) buildCheckpointWorkerDefaultEnv(
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	podTemplate *corev1.PodTemplateSpec,
) []corev1.EnvVar {
	componentType := consts.ComponentTypeWorker
	dynamoNamespace := consts.GlobalDynamoNamespace
	parentGraphDeploymentName := podTemplate.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	workerHashSuffix := podTemplate.Labels[consts.KubeLabelDynamoWorkerHash]
	discoveryBackend := configv1alpha1.DiscoveryBackendKubernetes

	if podTemplate.Labels[consts.KubeLabelDynamoNamespace] != "" {
		dynamoNamespace = podTemplate.Labels[consts.KubeLabelDynamoNamespace]
	}
	if podTemplate.Labels[consts.KubeLabelDynamoComponentType] != "" &&
		dynamo.IsWorkerComponent(podTemplate.Labels[consts.KubeLabelDynamoComponentType]) {
		componentType = podTemplate.Labels[consts.KubeLabelDynamoComponentType]
	}

	defaultContainer, _ := dynamo.NewWorkerDefaults().GetBaseContainer(dynamo.ComponentContext{
		ComponentType:                  componentType,
		DynamoNamespace:                dynamoNamespace,
		ParentGraphDeploymentName:      parentGraphDeploymentName,
		ParentGraphDeploymentNamespace: ckpt.Namespace,
		DiscoveryBackend:               discoveryBackend,
		WorkerHashSuffix:               workerHashSuffix,
	})
	return defaultContainer.Env
}

func (r *CheckpointReconciler) buildCheckpointJob(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, jobName string) *batchv1.Job {
	// Use the pod template from the spec
	podTemplate := ckpt.Spec.Job.PodTemplateSpec.DeepCopy()
	hash := ckpt.Status.IdentityHash
	if hash == "" {
		hash, _ = checkpoint.ComputeIdentityHash(ckpt.Spec.Identity)
	}
	version := desiredArtifactVersion(ckpt)

	// Add checkpoint-related labels
	if podTemplate.Labels == nil {
		podTemplate.Labels = make(map[string]string)
	}
	if podTemplate.Annotations == nil {
		podTemplate.Annotations = make(map[string]string)
	}
	location, storageType, err := checkpoint.ResolveCheckpointStorage(
		hash,
		version,
		&r.Config.Checkpoint,
	)
	if err != nil {
		location = ""
		storageType = ""
	}
	checkpoint.ApplyCheckpointSourcePodMetadata(podTemplate.Labels, podTemplate.Annotations, hash, location, storageType)

	hasPodInfoVolume := false
	for _, volume := range podTemplate.Spec.Volumes {
		if volume.Name == consts.PodInfoVolumeName {
			hasPodInfoVolume = true
			break
		}
	}
	if !hasPodInfoVolume {
		podTemplate.Spec.Volumes = append(podTemplate.Spec.Volumes, corev1.Volume{
			Name: consts.PodInfoVolumeName,
			VolumeSource: corev1.VolumeSource{
				DownwardAPI: &corev1.DownwardAPIVolumeSource{
					Items: []corev1.DownwardAPIVolumeFile{
						{
							Path: consts.PodInfoFileDynNamespace,
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels['" + consts.KubeLabelDynamoNamespace + "']",
							},
						},
						{
							Path: consts.PodInfoFileDynNamespaceWorkerSuffix,
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels['" + consts.KubeLabelDynamoWorkerHash + "']",
							},
						},
						{
							Path: consts.PodInfoFileDynComponent,
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels['" + consts.KubeLabelDynamoComponentType + "']",
							},
						},
						{
							Path: consts.PodInfoFileDynParentDGDName,
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels['" + consts.KubeLabelDynamoGraphDeploymentName + "']",
							},
						},
						{
							Path: consts.PodInfoFileDynParentDGDNamespace,
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.namespace",
							},
						},
						{
							Path: "pod_name",
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  consts.PodInfoFieldPodName,
							},
						},
						{
							Path: "pod_uid",
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  consts.PodInfoFieldPodUID,
							},
						},
						{
							Path: "pod_namespace",
							FieldRef: &corev1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  consts.PodInfoFieldPodNamespace,
							},
						},
					},
				},
			},
		})
	}

	// Configure the main container for checkpoint mode.
	if len(podTemplate.Spec.Containers) > 0 {
		mainContainer := &podTemplate.Spec.Containers[0]

		// Manual checkpoints start from a raw pod template, so re-apply the worker
		// runtime env defaults before layering checkpoint-specific env on top.
		mainContainer.Env = dynamo.MergeEnvs(
			r.buildCheckpointWorkerDefaultEnv(ckpt, podTemplate),
			mainContainer.Env,
		)
		dynamo.AddStandardEnvVars(mainContainer, r.Config)

		// Add the ready-for-checkpoint signal path.
		mainContainer.Env = append(mainContainer.Env,
			corev1.EnvVar{
				Name:  consts.EnvReadyForCheckpointFile,
				Value: r.Config.Checkpoint.ReadyForCheckpointFilePath,
			},
		)
		if gpus, ok := mainContainer.Resources.Limits[corev1.ResourceName(consts.KubeResourceGPUNvidia)]; ok && gpus.Cmp(*resource.NewQuantity(1, resource.DecimalSI)) > 0 {
			mainContainer.Command = append([]string{"cuda-checkpoint", "--launch-job"}, mainContainer.Command...)
		}

		// Override probes for checkpoint mode
		// Checkpoint jobs need different probe behavior than regular worker pods:
		// - Readiness: Wait for model to load before checkpoint
		// - Liveness/Startup: Remove to prevent restarts during slow model loading
		mainContainer.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{"cat", r.Config.Checkpoint.ReadyForCheckpointFilePath},
				},
			},
			InitialDelaySeconds: 15,
			PeriodSeconds:       2,
		}
		// Remove liveness probe - we don't want restarts during model loading
		mainContainer.LivenessProbe = nil
		// Remove startup probe - not needed for checkpoint jobs
		mainContainer.StartupProbe = nil

		hasPodInfoMount := false
		for _, mount := range mainContainer.VolumeMounts {
			if mount.Name == consts.PodInfoVolumeName {
				hasPodInfoMount = true
				break
			}
		}
		if !hasPodInfoMount {
			mainContainer.VolumeMounts = append(mainContainer.VolumeMounts, corev1.VolumeMount{
				Name:      consts.PodInfoVolumeName,
				MountPath: consts.PodInfoMountPath,
				ReadOnly:  true,
			})
		}

		dynamo.ApplySharedMemoryVolumeAndMount(&podTemplate.Spec, mainContainer, ckpt.Spec.Job.SharedMemory)
	}

	// Set restart policy to Never for Jobs
	podTemplate.Spec.RestartPolicy = corev1.RestartPolicyNever

	// Apply seccomp profile to block io_uring syscalls
	// CRIU doesn't support io_uring memory mappings, so we must block these syscalls
	if podTemplate.Spec.SecurityContext == nil {
		podTemplate.Spec.SecurityContext = &corev1.PodSecurityContext{}
	}
	podTemplate.Spec.SecurityContext.SeccompProfile = &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: ptr.To(consts.SeccompProfilePath),
	}

	// Build the Job
	activeDeadlineSeconds := ckpt.Spec.Job.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		defaultDeadline := int64(3600) // 1 hour
		activeDeadlineSeconds = &defaultDeadline
	}

	ttlSeconds := ckpt.Spec.Job.TTLSecondsAfterFinished
	if ttlSeconds == nil {
		defaultTTL := int32(300) // 5 minutes
		ttlSeconds = &defaultTTL
	}

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: ckpt.Namespace,
			Labels: map[string]string{
				consts.KubeLabelCheckpointHash: hash,
			},
		},
		Spec: batchv1.JobSpec{
			ActiveDeadlineSeconds: activeDeadlineSeconds,
			// Checkpoint jobs are single-attempt to keep snapshot-agent status terminal.
			BackoffLimit:            ptr.To[int32](0),
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
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}
