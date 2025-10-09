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

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
)

const (
	dynamoModelFinalizerName = "nvidia.com/dynamomodel-finalizer"
)

// DynamoModelReconciler reconciles a DynamoModel object
type DynamoModelReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomodels,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomodels/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamomodels/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *DynamoModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var err error
	reason := Reason("undefined")
	message := Message("")
	state := PendingState

	// Retrieve the DynamoModel CRD
	dynamoModel := &nvidiacomv1alpha1.DynamoModel{}
	if err = r.Get(ctx, req.NamespacedName, dynamoModel); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	defer func() {
		if err != nil {
			state = FailedState
			message = Message(err.Error())
			logger.Error(err, "Reconciliation failed")
		}
		dynamoModel.SetState(string(state))

		readyStatus := metav1.ConditionFalse
		if state == ReadyState {
			readyStatus = metav1.ConditionTrue
		}

		// Update Ready condition
		dynamoModel.AddStatusCondition(metav1.Condition{
			Type:               "Ready",
			Status:             readyStatus,
			Reason:             string(reason),
			Message:            string(message),
			LastTransitionTime: metav1.Now(),
		})

		err = r.Status().Update(ctx, dynamoModel)
		if err != nil {
			logger.Error(err, "Unable to update the CRD status", "crd", req.NamespacedName, "state", state, "reason", reason, "message", message)
		}
		logger.Info("Reconciliation done")
	}()

	// Handle finalizer
	deleted, err := commonController.HandleFinalizer(ctx, dynamoModel, r.Client, r)
	if err != nil {
		logger.Error(err, "failed to handle the finalizer")
		reason = "failed_to_handle_the_finalizer"
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}

	// Reconcile resources
	state, reason, message, err = r.reconcileResources(ctx, dynamoModel)
	if err != nil {
		logger.Error(err, "failed to reconcile the resources")
		reason = "failed_to_reconcile_the_resources"
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// reconcileResources handles the main reconciliation logic for DynamoModel
func (r *DynamoModelReconciler) reconcileResources(ctx context.Context, dynamoModel *nvidiacomv1alpha1.DynamoModel) (State, Reason, Message, error) {
	logger := log.FromContext(ctx)

	// Step 1: Reconcile PVC
	pvc, err := r.reconcilePVC(ctx, dynamoModel)
	if err != nil {
		return FailedState, "PVCReconciliationFailed", Message(err.Error()), err
	}

	// Update status with PVC name
	if dynamoModel.Status.PVCName != pvc.Name {
		dynamoModel.Status.PVCName = pvc.Name
	}

	// Step 2: Reconcile download Job
	job, err := r.reconcileDownloadJob(ctx, dynamoModel, pvc)
	if err != nil {
		return FailedState, "JobReconciliationFailed", Message(err.Error()), err
	}

	// Update status with Job name
	if dynamoModel.Status.DownloadJobName != job.Name {
		dynamoModel.Status.DownloadJobName = job.Name
	}

	// Step 3: Check Job status
	if job.Status.Succeeded > 0 {
		// Job completed successfully
		logger.Info("Model download completed successfully", "model", dynamoModel.Name)
		if dynamoModel.Status.LastDownloadTime == nil {
			now := metav1.Now()
			dynamoModel.Status.LastDownloadTime = &now
		}
		return ReadyState, "ModelReady", "Model downloaded and ready", nil
	} else if job.Status.Failed > 0 {
		// Job failed
		return FailedState, "DownloadFailed", "Model download job failed", fmt.Errorf("download job failed")
	} else if job.Status.Active > 0 {
		// Job is still running
		return PendingState, "Downloading", "Model download in progress", nil
	}

	// Job is pending
	return PendingState, "JobPending", "Model download job is pending", nil
}

// reconcilePVC creates or retrieves the PVC for the model
func (r *DynamoModelReconciler) reconcilePVC(ctx context.Context, dynamoModel *nvidiacomv1alpha1.DynamoModel) (*corev1.PersistentVolumeClaim, error) {
	logger := log.FromContext(ctx)

	pvcName := dynamoModel.GetPVCName()
	pvc := &corev1.PersistentVolumeClaim{}
	pvcNamespacedName := types.NamespacedName{Name: pvcName, Namespace: dynamoModel.Namespace}

	err := r.Get(ctx, pvcNamespacedName, pvc)
	if err != nil && !errors.IsNotFound(err) {
		logger.Error(err, "Unable to retrieve PVC", "pvcName", pvcName)
		return nil, err
	}

	// If PVC does not exist, create it
	if errors.IsNotFound(err) {
		create := true
		if dynamoModel.Spec.PVC.Create != nil {
			create = *dynamoModel.Spec.PVC.Create
		}

		if !create {
			logger.Error(err, "PVC does not exist and create is not enabled", "pvcName", pvcName)
			return nil, fmt.Errorf("PVC %s does not exist and create is disabled", pvcName)
		}

		pvc = r.constructPVC(dynamoModel)
		if err := controllerutil.SetControllerReference(dynamoModel, pvc, r.Scheme); err != nil {
			logger.Error(err, "Failed to set controller reference for PVC", "pvcName", pvcName)
			return nil, err
		}

		err = r.Create(ctx, pvc)
		if err != nil {
			logger.Error(err, "Failed to create PVC", "pvcName", pvcName)
			return nil, err
		}
		logger.Info("PVC created", "pvcName", pvcName, "namespace", dynamoModel.Namespace)
	}

	return pvc, nil
}

// constructPVC creates a PVC object from the DynamoModel spec
func (r *DynamoModelReconciler) constructPVC(dynamoModel *nvidiacomv1alpha1.DynamoModel) *corev1.PersistentVolumeClaim {
	pvcName := dynamoModel.GetPVCName()
	storageClassName := dynamoModel.Spec.PVC.StorageClass
	accessMode := dynamoModel.Spec.PVC.VolumeAccessMode
	if accessMode == "" {
		accessMode = corev1.ReadWriteMany
	}

	return &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pvcName,
			Namespace: dynamoModel.Namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "model-storage",
				"dynamo.nvidia.com/model":      dynamoModel.Name,
			},
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{accessMode},
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: dynamoModel.Spec.PVC.Size,
				},
			},
			StorageClassName: &storageClassName,
		},
	}
}

// reconcileDownloadJob creates or retrieves the download Job for the model
func (r *DynamoModelReconciler) reconcileDownloadJob(ctx context.Context, dynamoModel *nvidiacomv1alpha1.DynamoModel, pvc *corev1.PersistentVolumeClaim) (*batchv1.Job, error) {
	logger := log.FromContext(ctx)

	jobName := fmt.Sprintf("%s-download", dynamoModel.Name)
	job := &batchv1.Job{}
	jobNamespacedName := types.NamespacedName{Name: jobName, Namespace: dynamoModel.Namespace}

	err := r.Get(ctx, jobNamespacedName, job)
	if err != nil && !errors.IsNotFound(err) {
		logger.Error(err, "Unable to retrieve Job", "jobName", jobName)
		return nil, err
	}

	// If Job does not exist, create it
	if errors.IsNotFound(err) {
		job = r.constructDownloadJob(dynamoModel, pvc)
		if err := controllerutil.SetControllerReference(dynamoModel, job, r.Scheme); err != nil {
			logger.Error(err, "Failed to set controller reference for Job", "jobName", jobName)
			return nil, err
		}

		err = r.Create(ctx, job)
		if err != nil {
			logger.Error(err, "Failed to create Job", "jobName", jobName)
			return nil, err
		}
		logger.Info("Download Job created", "jobName", jobName, "namespace", dynamoModel.Namespace)
	}

	return job, nil
}

// constructDownloadJob creates a Job object for downloading the model
func (r *DynamoModelReconciler) constructDownloadJob(dynamoModel *nvidiacomv1alpha1.DynamoModel, pvc *corev1.PersistentVolumeClaim) *batchv1.Job {
	jobName := fmt.Sprintf("%s-download", dynamoModel.Name)
	backoffLimit := int32(3)
	completions := int32(1)
	parallelism := int32(1)

	// Parse source URL to determine download strategy
	downloadScript := r.generateDownloadScript(dynamoModel)

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: dynamoModel.Namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "model-downloader",
				"dynamo.nvidia.com/model":      dynamoModel.Name,
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Completions:  &completions,
			Parallelism:  &parallelism,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app":                     "model-download",
						"dynamo.nvidia.com/model": dynamoModel.Name,
					},
				},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:    "model-download",
							Image:   "python:3.10-slim",
							Command: []string{"sh", "-c"},
							Args:    []string{downloadScript},
							Env:     r.generateEnvVars(dynamoModel),
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "model-cache",
									MountPath: "/model-cache",
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "model-cache",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: pvc.Name,
								},
							},
						},
					},
				},
			},
		},
	}

	// Add secret reference if specified
	if dynamoModel.Spec.SecretRef != "" {
		job.Spec.Template.Spec.Containers[0].EnvFrom = []corev1.EnvFromSource{
			{
				SecretRef: &corev1.SecretEnvSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: dynamoModel.Spec.SecretRef,
					},
				},
			},
		}
	}

	return job
}

// generateDownloadScript generates the download script based on the source URL
func (r *DynamoModelReconciler) generateDownloadScript(dynamoModel *nvidiacomv1alpha1.DynamoModel) string {
	sourceURL := dynamoModel.Spec.SourceURL

	// Determine the download method based on the source URL prefix
	if strings.HasPrefix(sourceURL, "hf://") {
		// HuggingFace download using modern 'hf download' command
		// Build revision flag if version is specified
		revisionFlag := ""
		if dynamoModel.Spec.Version != "" {
			revisionFlag = fmt.Sprintf("--revision $MODEL_REVISION")
		}

		script := fmt.Sprintf(`
set -eux
pip install --no-cache-dir huggingface_hub hf_transfer
hf download $MODEL_NAME %s --exclude "original/*" --exclude "metal/*"
`, revisionFlag)
		return script
	} else if strings.HasPrefix(sourceURL, "s3://") {
		// S3 download
		script := `
set -eux
pip install --no-cache-dir awscli
aws s3 sync $SOURCE_URL /model-cache --no-progress
`
		return script
	} else if strings.HasPrefix(sourceURL, "ngc://") {
		// NGC download
		script := `
set -eux
pip install --no-cache-dir ngc-cli
ngc registry model download-version $MODEL_NAME --dest /model-cache
`
		return script
	}

	// Default: generic download (assumes HTTP/HTTPS URL)
	script := `
set -eux
pip install --no-cache-dir wget
wget -P /model-cache $SOURCE_URL
`
	return script
}

// generateEnvVars generates environment variables for the download job
func (r *DynamoModelReconciler) generateEnvVars(dynamoModel *nvidiacomv1alpha1.DynamoModel) []corev1.EnvVar {
	envVars := []corev1.EnvVar{
		{
			Name:  "SOURCE_URL",
			Value: dynamoModel.Spec.SourceURL,
		},
		{
			Name:  "HF_HOME",
			Value: "/model-cache",
		},
		{
			Name:  "HF_HUB_ENABLE_HF_TRANSFER",
			Value: "1",
		},
	}

	// Add model name for HuggingFace downloads
	if strings.HasPrefix(dynamoModel.Spec.SourceURL, "hf://") {
		modelName := strings.TrimPrefix(dynamoModel.Spec.SourceURL, "hf://")
		envVars = append(envVars, corev1.EnvVar{
			Name:  "MODEL_NAME",
			Value: modelName,
		})
	}

	// Add version/revision if specified
	if dynamoModel.Spec.Version != "" {
		envVars = append(envVars, corev1.EnvVar{
			Name:  "MODEL_REVISION",
			Value: dynamoModel.Spec.Version,
		})
	}

	return envVars
}

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoModel{}).
		Owns(&corev1.PersistentVolumeClaim{}).
		Owns(&batchv1.Job{}).
		Complete(r)
}

// Cleanup implements the Cleanup interface for finalizer handling
func (r *DynamoModelReconciler) Cleanup(ctx context.Context, obj client.Object) error {
	logger := log.FromContext(ctx)
	dynamoModel := obj.(*nvidiacomv1alpha1.DynamoModel)

	logger.Info("Cleaning up DynamoModel resources", "name", dynamoModel.Name, "namespace", dynamoModel.Namespace)

	// The PVC and Job will be automatically deleted due to owner references
	// Additional cleanup logic can be added here if needed

	return nil
}

// GetFinalizerName returns the finalizer name for this controller
func (r *DynamoModelReconciler) GetFinalizerName() string {
	return dynamoModelFinalizerName
}
