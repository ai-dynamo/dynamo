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

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
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
	snapshotv1alpha1 "github.com/ai-dynamo/dynamo/deploy/snapshot/api/v1alpha1"
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

func desiredArtifactVersion(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) string {
	return checkpoint.ArtifactVersionForCheckpoint(ckpt)
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints/finalizers,verbs=update
// +kubebuilder:rbac:groups=snapshot.nvidia.com,resources=snapshotrequests,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=snapshot.nvidia.com,resources=snapshotrequests/status,verbs=get;update;patch

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
	desiredVersion := desiredArtifactVersion(ckpt)
	desiredLocation, desiredStorageType, err := checkpoint.ResolveCheckpointStorage(identityHash, desiredVersion, &r.Config.Checkpoint)
	if err != nil {
		return ctrl.Result{}, err
	}
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
		((ckpt.Status.Location != "" && ckpt.Status.Location != desiredLocation) ||
			(ckpt.Status.StorageType != "" && ckpt.Status.StorageType != desiredStorageType)) {
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
	requestName := checkpoint.CheckpointRequestName(hash, version)
	location, storageType, err := checkpoint.ResolveCheckpointStorage(hash, version, &r.Config.Checkpoint)
	if err != nil {
		return ctrl.Result{}, err
	}

	modified, _, err := commonController.SyncResource(ctx, r, ckpt, func(ctx context.Context) (*snapshotv1alpha1.SnapshotRequest, bool, error) {
		request, err := r.buildCheckpointRequest(ckpt, requestName)
		return request, false, err
	})
	if err != nil {
		logger.Error(err, "Failed to sync checkpoint SnapshotRequest")
		return ctrl.Result{}, err
	}

	if modified {
		logger.Info("Created/updated checkpoint SnapshotRequest", "request", requestName)
	}

	// Update status to Creating phase
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseCreating
	ckpt.Status.JobName = requestName
	ckpt.Status.Location = location
	ckpt.Status.StorageType = storageType
	ckpt.Status.CreatedAt = nil
	ckpt.Status.Message = ""
	meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
		Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated),
		Status:             metav1.ConditionTrue,
		Reason:             "SnapshotRequestCreated",
		Message:            fmt.Sprintf("SnapshotRequest %s created", requestName),
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
		ckpt.Status.Message = "checkpoint SnapshotRequest is missing from status"
		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	request := &snapshotv1alpha1.SnapshotRequest{}
	requestName := strings.TrimSpace(ckpt.Status.JobName)
	if strings.HasSuffix(requestName, "-checkpoint") {
		requestName = strings.TrimSuffix(requestName, "-checkpoint")
	}
	if requestName == "" {
		requestName = checkpoint.CheckpointRequestName(ckpt.Status.IdentityHash, desiredArtifactVersion(ckpt))
	}
	if err := r.Get(ctx, client.ObjectKey{Namespace: ckpt.Namespace, Name: requestName}, request); err != nil {
		if apierrors.IsNotFound(err) {
			ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhasePending
			ckpt.Status.CreatedAt = nil
			ckpt.Status.Message = "checkpoint SnapshotRequest not found, retrying"
			meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
				Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated),
				Status:             metav1.ConditionUnknown,
				Reason:             "SnapshotRequestMissing",
				Message:            "Checkpoint SnapshotRequest not found, retrying",
				LastTransitionTime: metav1.Now(),
			})
			if err := r.Status().Update(ctx, ckpt); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{}, err
	}

	switch request.Status.State {
	case snapshotv1alpha1.SnapshotRequestStateFailed:
		ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseFailed
		ckpt.Status.Message = request.Status.Message
		meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
			Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
			Status:             metav1.ConditionFalse,
			Reason:             "SnapshotRequestFailed",
			Message:            request.Status.Message,
			LastTransitionTime: metav1.Now(),
		})
		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		r.Recorder.Event(ckpt, corev1.EventTypeWarning, "CheckpointFailed", request.Status.Message)
		return ctrl.Result{}, nil
	case snapshotv1alpha1.SnapshotRequestStateSucceeded:
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
		ckpt.Status.JobName = firstNonEmpty(request.Status.JobName, ckpt.Status.JobName)
		ckpt.Status.CreatedAt = &now
		ckpt.Status.Message = ""
		meta.SetStatusCondition(&ckpt.Status.Conditions, metav1.Condition{
			Type:               string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCompleted),
			Status:             metav1.ConditionTrue,
			Reason:             "SnapshotRequestSucceeded",
			Message:            fmt.Sprintf("Checkpoint job completed, available at %s", ckpt.Status.Location),
			LastTransitionTime: metav1.Now(),
		})
		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		logger.Info("Checkpoint SnapshotRequest succeeded", "request", request.Name, "job", request.Status.JobName)
		r.Recorder.Event(ckpt, corev1.EventTypeNormal, "CheckpointReady", "Checkpoint creation completed successfully")
		return ctrl.Result{}, nil
	default:
		ckpt.Status.JobName = firstNonEmpty(request.Status.JobName, ckpt.Status.JobName)
		if ckpt.Status.Location == "" {
			ckpt.Status.Location = request.Status.Location
		}
		if ckpt.Status.StorageType == "" {
			ckpt.Status.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(request.Status.StorageType)
		}
		ckpt.Status.Message = request.Status.Message
		if err := r.Status().Update(ctx, ckpt); err != nil {
			return ctrl.Result{}, err
		}
		logger.V(1).Info("Checkpoint SnapshotRequest still running", "request", request.Name, "state", request.Status.State)
		return ctrl.Result{}, nil
	}
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

func (r *CheckpointReconciler) buildCheckpointPodTemplate(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) *corev1.PodTemplateSpec {
	podTemplate := ckpt.Spec.Job.PodTemplateSpec.DeepCopy()

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

	podTemplate.Spec.RestartPolicy = corev1.RestartPolicyNever
	return podTemplate
}

func (r *CheckpointReconciler) buildCheckpointRequest(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, requestName string) (*snapshotv1alpha1.SnapshotRequest, error) {
	hash := ckpt.Status.IdentityHash
	if hash == "" {
		var err error
		hash, err = checkpoint.ComputeIdentityHash(ckpt.Spec.Identity)
		if err != nil {
			return nil, err
		}
	}

	activeDeadlineSeconds := ckpt.Spec.Job.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		activeDeadlineSeconds = ptr.To[int64](3600)
	}
	ttlSecondsAfterFinished := ckpt.Spec.Job.TTLSecondsAfterFinished
	if ttlSecondsAfterFinished == nil {
		ttlSecondsAfterFinished = ptr.To[int32](300)
	}

	return &snapshotv1alpha1.SnapshotRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      requestName,
			Namespace: ckpt.Namespace,
			Labels: map[string]string{
				consts.KubeLabelCheckpointHash: hash,
			},
		},
		Spec: snapshotv1alpha1.SnapshotRequestSpec{
			Phase:                   snapshotv1alpha1.SnapshotRequestPhaseCheckpoint,
			SnapshotID:              hash,
			ArtifactVersion:         desiredArtifactVersion(ckpt),
			PodTemplate:             r.buildCheckpointPodTemplate(ckpt),
			DisableCudaCheckpointJobFile: true,
			ActiveDeadlineSeconds:   activeDeadlineSeconds,
			TTLSecondsAfterFinished: ttlSecondsAfterFinished,
		},
	}, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

// SetupWithManager sets up the controller with the Manager.
func (r *CheckpointReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoCheckpoint{}).
		Owns(&snapshotv1alpha1.SnapshotRequest{}, builder.WithPredicates(predicate.Funcs{
			// Ignore creation - we don't need to reconcile when we just created the Job
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}
