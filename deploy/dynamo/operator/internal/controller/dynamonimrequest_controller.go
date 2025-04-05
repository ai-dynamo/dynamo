/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
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
 * Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
 */

package controller

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"text/template"
	"time"

	"emperror.dev/errors"
	commonconfig "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/config"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/controller_common"
	"github.com/apparentlymart/go-shquot/shquot"
	"github.com/ettle/strcase"
	"github.com/huandu/xstrings"
	"github.com/mitchellh/hashstructure/v2"
	"github.com/prune998/docker-registry-client/registry"
	"github.com/rs/xid"
	"github.com/sergeymakinen/go-quote/unix"
	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v2"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	dynamoCommon "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/schemas"
	yataiclient "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/yatai-client"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
)

const (
	KubeAnnotationDynamoNimRequestHash            = "yatai.ai/bento-request-hash"
	KubeAnnotationDynamoNimRequestImageBuiderHash = "yatai.ai/bento-request-image-builder-hash"
	KubeAnnotationDynamoNimRequestModelSeederHash = "yatai.ai/bento-request-model-seeder-hash"
	KubeLabelYataiImageBuilderSeparateModels      = "yatai.ai/yatai-image-builder-separate-models"
	KubeAnnotationDynamoNimStorageNS              = "yatai.ai/bento-storage-namespace"
	KubeAnnotationModelStorageNS                  = "yatai.ai/model-storage-namespace"
	StoreSchemaAWS                                = "aws"
	StoreSchemaGCP                                = "gcp"
)

// DynamoNimRequestReconciler reconciles a DynamoNimRequest object
type DynamoNimRequestReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Config   controller_common.Config
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamonimrequests,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamonimrequests/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamonimrequests/finalizers,verbs=update
//+kubebuilder:rbac:groups=nvidia.com,resources=dynamonims,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=nvidia.com,resources=dynamonims/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=events,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=serviceaccounts,verbs=get;list;watch
//+kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the DynamoNimRequest object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.18.2/pkg/reconcile
//
//nolint:gocyclo,nakedret
func (r *DynamoNimRequestReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logs := log.FromContext(ctx)

	dynamoNimRequest := &nvidiacomv1alpha1.DynamoNimRequest{}

	err = r.Get(ctx, req.NamespacedName, dynamoNimRequest)

	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			logs.Info("dynamoNimRequest resource not found. Ignoring since object must be deleted")
			err = nil
			return
		}
		// Error reading the object - requeue the request.
		logs.Error(err, "Failed to get dynamoNimRequest")
		return
	}

	for _, condition := range dynamoNimRequest.Status.Conditions {
		if condition.Type == nvidiacomv1alpha1.DynamoDeploymentConditionTypeAvailable && condition.Status == metav1.ConditionTrue {
			logs.Info("Skip available dynamoNimRequest")
			return
		}
	}

	if len(dynamoNimRequest.Status.Conditions) == 0 {
		dynamoNimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeModelsSeeding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile dynamoNimRequest",
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile dynamoNimRequest",
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExists,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile dynamoNimRequest",
			},
		)
		if err != nil {
			return
		}
	}

	logs = logs.WithValues("dynamoNimRequest", dynamoNimRequest.Name, "dynamoNimRequestNamespace", dynamoNimRequest.Namespace)

	defer func() {
		if err == nil {
			logs.Info("Reconcile success")
			return
		}
		logs.Error(err, "Failed to reconcile dynamoNimRequest.")
		r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeWarning, "ReconcileError", "Failed to reconcile dynamoNimRequest: %v", err)
		_, err_ := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeDynamoNimAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: err.Error(),
			},
		)
		if err_ != nil {
			logs.Error(err_, "Failed to update dynamoNimRequest status")
			return
		}
	}()

	dynamoNimAvailableCondition := meta.FindStatusCondition(dynamoNimRequest.Status.Conditions, nvidiacomv1alpha1.DynamoNimRequestConditionTypeDynamoNimAvailable)
	if dynamoNimAvailableCondition == nil || dynamoNimAvailableCondition.Status != metav1.ConditionUnknown {
		dynamoNimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeDynamoNimAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Reconciling",
			},
		)
		if err != nil {
			return
		}
	}

	if isSeparateModels(dynamoNimRequest) {
		err = errors.New("separate models, unsupported feature")
		logs.Error(err, "unsupported feature")
		return
	}

	dynamoNimRequest, imageInfo, imageExists, imageExistsResult, err := r.ensureImageExists(ctx, ensureImageExistsOption{
		dynamoNimRequest: dynamoNimRequest,
		req:              req,
	})

	if err != nil {
		err = errors.Wrapf(err, "ensure image exists")
		return
	}

	if !imageExists {
		result = imageExistsResult
		dynamoNimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeDynamoNimAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "DynamoNim image is building",
			},
		)
		if err != nil {
			return
		}
		return
	}

	dynamoNimCR := &nvidiacomv1alpha1.DynamoNim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dynamoNimRequest.Name,
			Namespace: dynamoNimRequest.Namespace,
		},
		Spec: nvidiacomv1alpha1.DynamoNimSpec{
			Tag:         dynamoNimRequest.Spec.BentoTag,
			Image:       imageInfo.ImageName,
			ServiceName: dynamoNimRequest.Spec.ServiceName,
			Context:     dynamoNimRequest.Spec.Context,
			Models:      dynamoNimRequest.Spec.Models,
		},
	}

	err = ctrl.SetControllerReference(dynamoNimRequest, dynamoNimCR, r.Scheme)
	if err != nil {
		err = errors.Wrap(err, "set controller reference")
		return
	}

	if imageInfo.DockerConfigJSONSecretName != "" {
		dynamoNimCR.Spec.ImagePullSecrets = []corev1.LocalObjectReference{
			{
				Name: imageInfo.DockerConfigJSONSecretName,
			},
		}
	}

	if dynamoNimRequest.Spec.DownloadURL == "" {
		var dynamoNim *schemas.DynamoNIM
		dynamoNim, err = r.getDynamoNim(ctx, dynamoNimRequest)
		if err != nil {
			err = errors.Wrap(err, "get dynamoNim")
			return
		}
		dynamoNimCR.Spec.Context = &nvidiacomv1alpha1.BentoContext{
			BentomlVersion: dynamoNim.Manifest.BentomlVersion,
		}
	}

	r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "DynamoNimImageBuilder", "Creating DynamoNim CR %s in namespace %s", dynamoNimCR.Name, dynamoNimCR.Namespace)
	err = r.Create(ctx, dynamoNimCR)
	isAlreadyExists := k8serrors.IsAlreadyExists(err)
	if err != nil && !isAlreadyExists {
		err = errors.Wrap(err, "create DynamoNim resource")
		return
	}
	if isAlreadyExists {
		oldDynamoNimCR := &nvidiacomv1alpha1.DynamoNim{}
		r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "DynamoNimImageBuilder", "Updating DynamoNim CR %s in namespace %s", dynamoNimCR.Name, dynamoNimCR.Namespace)
		err = r.Get(ctx, types.NamespacedName{Name: dynamoNimCR.Name, Namespace: dynamoNimCR.Namespace}, oldDynamoNimCR)
		if err != nil {
			err = errors.Wrap(err, "get DynamoNim resource")
			return
		}
		if !reflect.DeepEqual(oldDynamoNimCR.Spec, dynamoNimCR.Spec) {
			oldDynamoNimCR.OwnerReferences = dynamoNimCR.OwnerReferences
			oldDynamoNimCR.Spec = dynamoNimCR.Spec
			err = r.Update(ctx, oldDynamoNimCR)
			if err != nil {
				err = errors.Wrap(err, "update DynamoNim resource")
				return
			}
		}
	}

	dynamoNimRequest, err = r.setStatusConditions(ctx, req,
		metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeDynamoNimAvailable,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: "DynamoNim is generated",
		},
	)
	if err != nil {
		return
	}

	return
}

func isEstargzEnabled() bool {
	return os.Getenv("ESTARGZ_ENABLED") == commonconsts.KubeLabelValueTrue
}

type ensureImageExistsOption struct {
	dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest
	req              ctrl.Request
}

//nolint:gocyclo,nakedret
func (r *DynamoNimRequestReconciler) ensureImageExists(ctx context.Context, opt ensureImageExistsOption) (dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest, imageInfo ImageInfo, imageExists bool, result ctrl.Result, err error) { // nolint: unparam
	logs := log.FromContext(ctx)

	dynamoNimRequest = opt.dynamoNimRequest
	req := opt.req

	imageInfo, err = r.getImageInfo(ctx, GetImageInfoOption{
		DynamoNimRequest: dynamoNimRequest,
	})
	if err != nil {
		err = errors.Wrap(err, "get image info")
		return
	}

	imageExistsCheckedCondition := meta.FindStatusCondition(dynamoNimRequest.Status.Conditions, nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExistsChecked)
	imageExistsCondition := meta.FindStatusCondition(dynamoNimRequest.Status.Conditions, nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExists)
	if imageExistsCheckedCondition == nil || imageExistsCheckedCondition.Status != metav1.ConditionTrue || imageExistsCheckedCondition.Message != imageInfo.ImageName {
		imageExistsCheckedCondition = &metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExistsChecked,
			Status:  metav1.ConditionUnknown,
			Reason:  "Reconciling",
			Message: imageInfo.ImageName,
		}
		dynamoNimAvailableCondition := &metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeDynamoNimAvailable,
			Status:  metav1.ConditionUnknown,
			Reason:  "Reconciling",
			Message: "Checking image exists",
		}
		dynamoNimRequest, err = r.setStatusConditions(ctx, req, *imageExistsCheckedCondition, *dynamoNimAvailableCondition)
		if err != nil {
			return
		}
		r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "CheckingImage", "Checking image exists: %s", imageInfo.ImageName)
		imageExists, err = checkImageExists(dynamoNimRequest, imageInfo.DockerRegistry, imageInfo.InClusterImageName)
		if err != nil {
			err = errors.Wrapf(err, "check image %s exists", imageInfo.ImageName)
			return
		}

		err = r.Get(ctx, req.NamespacedName, dynamoNimRequest)
		if err != nil {
			logs.Error(err, "Failed to re-fetch dynamoNimRequest")
			return
		}

		if imageExists {
			r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "CheckingImage", "Image exists: %s", imageInfo.ImageName)
			imageExistsCheckedCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExistsChecked,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: imageInfo.ImageName,
			}
			imageExistsCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExists,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: imageInfo.ImageName,
			}
			dynamoNimRequest, err = r.setStatusConditions(ctx, req, *imageExistsCondition, *imageExistsCheckedCondition)
			if err != nil {
				return
			}
		} else {
			r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "CheckingImage", "Image not exists: %s", imageInfo.ImageName)
			imageExistsCheckedCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExistsChecked,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image not exists: %s", imageInfo.ImageName),
			}
			imageExistsCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExists,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image %s is not exists", imageInfo.ImageName),
			}
			dynamoNimRequest, err = r.setStatusConditions(ctx, req, *imageExistsCondition, *imageExistsCheckedCondition)
			if err != nil {
				return
			}
		}
	}

	var dynamoNimRequestHashStr string
	dynamoNimRequestHashStr, err = r.getHashStr(dynamoNimRequest)
	if err != nil {
		err = errors.Wrapf(err, "get dynamoNimRequest %s/%s hash string", dynamoNimRequest.Namespace, dynamoNimRequest.Name)
		return
	}

	imageExists = imageExistsCondition != nil && imageExistsCondition.Status == metav1.ConditionTrue && imageExistsCondition.Message == imageInfo.ImageName
	if imageExists {
		return
	}

	jobLabels := map[string]string{
		commonconsts.KubeLabelBentoRequest:        dynamoNimRequest.Name,
		commonconsts.KubeLabelIsBentoImageBuilder: commonconsts.KubeLabelValueTrue,
	}

	if isSeparateModels(opt.dynamoNimRequest) {
		jobLabels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueTrue
	} else {
		jobLabels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueFalse
	}

	jobs := &batchv1.JobList{}
	err = r.List(ctx, jobs, client.InNamespace(req.Namespace), client.MatchingLabels(jobLabels))
	if err != nil {
		err = errors.Wrap(err, "list jobs")
		return
	}

	reservedJobs := make([]*batchv1.Job, 0)
	for _, job_ := range jobs.Items {
		job_ := job_

		oldHash := job_.Annotations[KubeAnnotationDynamoNimRequestHash]
		if oldHash != dynamoNimRequestHashStr {
			logs.Info("Because hash changed, delete old job", "job", job_.Name, "oldHash", oldHash, "newHash", dynamoNimRequestHashStr)
			// --cascade=foreground
			err = r.Delete(ctx, &job_, &client.DeleteOptions{
				PropagationPolicy: &[]metav1.DeletionPropagation{metav1.DeletePropagationForeground}[0],
			})
			if err != nil {
				err = errors.Wrapf(err, "delete job %s", job_.Name)
				return
			}
			return
		} else {
			reservedJobs = append(reservedJobs, &job_)
		}
	}

	var job *batchv1.Job
	if len(reservedJobs) > 0 {
		job = reservedJobs[0]
	}

	if len(reservedJobs) > 1 {
		for _, job_ := range reservedJobs[1:] {
			logs.Info("Because has more than one job, delete old job", "job", job_.Name)
			// --cascade=foreground
			err = r.Delete(ctx, job_, &client.DeleteOptions{
				PropagationPolicy: &[]metav1.DeletionPropagation{metav1.DeletePropagationForeground}[0],
			})
			if err != nil {
				err = errors.Wrapf(err, "delete job %s", job_.Name)
				return
			}
		}
	}

	if job == nil {
		job, err = r.generateImageBuilderJob(ctx, GenerateImageBuilderJobOption{
			ImageInfo:        imageInfo,
			DynamoNimRequest: dynamoNimRequest,
		})
		if err != nil {
			err = errors.Wrap(err, "generate image builder job")
			return
		}
		r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderJob", "Creating image builder job: %s", job.Name)
		err = r.Create(ctx, job)
		if err != nil {
			err = errors.Wrapf(err, "create image builder job %s", job.Name)
			return
		}
		r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderJob", "Created image builder job: %s", job.Name)
		return
	}

	r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "CheckingImageBuilderJob", "Found image builder job: %s", job.Name)

	err = r.Get(ctx, req.NamespacedName, dynamoNimRequest)
	if err != nil {
		logs.Error(err, "Failed to re-fetch dynamoNimRequest")
		return
	}
	imageBuildingCondition := meta.FindStatusCondition(dynamoNimRequest.Status.Conditions, nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageBuilding)

	isJobFailed := false
	isJobRunning := true

	if job.Spec.Completions != nil {
		if job.Status.Succeeded != *job.Spec.Completions {
			if job.Status.Failed > 0 {
				for _, condition := range job.Status.Conditions {
					if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
						isJobFailed = true
						break
					}
				}
			}
			isJobRunning = !isJobFailed
		} else {
			isJobRunning = false
		}
	}

	if isJobRunning {
		conditions := make([]metav1.Condition, 0)
		if job.Status.Active > 0 {
			conditions = append(conditions, metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is running", job.Name),
			})
		} else {
			conditions = append(conditions, metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is waiting", job.Name),
			})
		}
		if dynamoNimRequest.Spec.ImageBuildTimeout != nil {
			if imageBuildingCondition != nil && imageBuildingCondition.LastTransitionTime.Add(time.Duration(*dynamoNimRequest.Spec.ImageBuildTimeout)).Before(time.Now()) {
				conditions = append(conditions, metav1.Condition{
					Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageBuilding,
					Status:  metav1.ConditionFalse,
					Reason:  "Timeout",
					Message: fmt.Sprintf("Image building job %s is timeout", job.Name),
				})
				if _, err = r.setStatusConditions(ctx, req, conditions...); err != nil {
					return
				}
				err = errors.New("image build timeout")
				return
			}
		}

		if dynamoNimRequest, err = r.setStatusConditions(ctx, req, conditions...); err != nil {
			return
		}

		if imageBuildingCondition != nil && imageBuildingCondition.Status != metav1.ConditionTrue && isJobRunning {
			r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "DynamoNimImageBuilder", "Image is building now")
		}

		return
	}

	if isJobFailed {
		dynamoNimRequest, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageBuilding,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is failed.", job.Name),
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeDynamoNimAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is failed.", job.Name),
			},
		)
		if err != nil {
			return
		}
		return
	}

	dynamoNimRequest, err = r.setStatusConditions(ctx, req,
		metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageBuilding,
			Status:  metav1.ConditionFalse,
			Reason:  "Reconciling",
			Message: fmt.Sprintf("Image building job %s is succeeded.", job.Name),
		},
		metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoNimRequestConditionTypeImageExists,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: imageInfo.ImageName,
		},
	)
	if err != nil {
		return
	}

	r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "DynamoNimImageBuilder", "Image has been built successfully")

	imageExists = true

	return
}

func (r *DynamoNimRequestReconciler) setStatusConditions(ctx context.Context, req ctrl.Request, conditions ...metav1.Condition) (dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest, err error) {
	dynamoNimRequest = &nvidiacomv1alpha1.DynamoNimRequest{}
	/*
		Please don't blame me when you see this kind of code,
		this is to avoid "the object has been modified; please apply your changes to the latest version and try again" when updating CR status,
		don't doubt that almost all CRD operators (e.g. cert-manager) can't avoid this stupid error and can only try to avoid this by this stupid way.
	*/
	for i := 0; i < 3; i++ {
		if err = r.Get(ctx, req.NamespacedName, dynamoNimRequest); err != nil {
			err = errors.Wrap(err, "Failed to re-fetch dynamoNimRequest")
			return
		}
		for _, condition := range conditions {
			meta.SetStatusCondition(&dynamoNimRequest.Status.Conditions, condition)
		}
		if err = r.Status().Update(ctx, dynamoNimRequest); err != nil {
			time.Sleep(100 * time.Millisecond)
		} else {
			break
		}
	}
	if err != nil {
		err = errors.Wrap(err, "Failed to update dynamoNimRequest status")
		return
	}
	if err = r.Get(ctx, req.NamespacedName, dynamoNimRequest); err != nil {
		err = errors.Wrap(err, "Failed to re-fetch dynamoNimRequest")
		return
	}
	return
}

type DynamoNimImageBuildEngine string

const (
	DynamoNimImageBuildEngineKaniko           DynamoNimImageBuildEngine = "kaniko"
	DynamoNimImageBuildEngineBuildkit         DynamoNimImageBuildEngine = "buildkit"
	DynamoNimImageBuildEngineBuildkitRootless DynamoNimImageBuildEngine = "buildkit-rootless"
)

const (
	EnvDynamoNimImageBuildEngine = "BENTO_IMAGE_BUILD_ENGINE"
)

func getDynamoNimImageBuildEngine() DynamoNimImageBuildEngine {
	engine := os.Getenv(EnvDynamoNimImageBuildEngine)
	if engine == "" {
		return DynamoNimImageBuildEngineKaniko
	}
	return DynamoNimImageBuildEngine(engine)
}

//nolint:nakedret
func (r *DynamoNimRequestReconciler) makeSureDockerConfigJSONSecret(ctx context.Context, namespace string, dockerRegistryConf *commonconfig.DockerRegistryConfig) (dockerConfigJSONSecret *corev1.Secret, err error) {
	if dockerRegistryConf.Username == "" {
		return
	}

	// nolint: gosec
	dockerConfigSecretName := commonconsts.KubeSecretNameRegcred
	dockerConfigObj := struct {
		Auths map[string]struct {
			Auth string `json:"auth"`
		} `json:"auths"`
	}{
		Auths: map[string]struct {
			Auth string `json:"auth"`
		}{
			dockerRegistryConf.Server: {
				Auth: base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", dockerRegistryConf.Username, dockerRegistryConf.Password))),
			},
		},
	}

	dockerConfigContent, err := json.Marshal(dockerConfigObj)
	if err != nil {
		err = errors.Wrap(err, "marshal docker config")
		return nil, err
	}

	dockerConfigJSONSecret = &corev1.Secret{}

	err = r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: dockerConfigSecretName}, dockerConfigJSONSecret)
	dockerConfigIsNotFound := k8serrors.IsNotFound(err)
	// nolint: gocritic
	if err != nil && !dockerConfigIsNotFound {
		err = errors.Wrap(err, "get docker config secret")
		return nil, err
	}
	err = nil
	if dockerConfigIsNotFound {
		dockerConfigJSONSecret = &corev1.Secret{
			Type: corev1.SecretTypeDockerConfigJson,
			ObjectMeta: metav1.ObjectMeta{
				Name:      dockerConfigSecretName,
				Namespace: namespace,
			},
			Data: map[string][]byte{
				".dockerconfigjson": dockerConfigContent,
			},
		}
		err_ := r.Create(ctx, dockerConfigJSONSecret)
		if err_ != nil {
			dockerConfigJSONSecret = &corev1.Secret{}
			err = r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: dockerConfigSecretName}, dockerConfigJSONSecret)
			dockerConfigIsNotFound = k8serrors.IsNotFound(err)
			if err != nil && !dockerConfigIsNotFound {
				err = errors.Wrap(err, "get docker config secret")
				return nil, err
			}
			if dockerConfigIsNotFound {
				err_ = errors.Wrap(err_, "create docker config secret")
				return nil, err_
			}
			if err != nil {
				err = nil
			}
		}
	} else {
		dockerConfigJSONSecret.Data[".dockerconfigjson"] = dockerConfigContent
		err = r.Update(ctx, dockerConfigJSONSecret)
		if err != nil {
			err = errors.Wrap(err, "update docker config secret")
			return nil, err
		}
	}

	return
}

//nolint:nakedret
func (r *DynamoNimRequestReconciler) getYataiClient(ctx context.Context) (yataiClient **yataiclient.YataiClient, yataiConf **commonconfig.YataiConfig, err error) {
	yataiConf_, err := commonconfig.GetYataiConfig(ctx)
	isNotFound := k8serrors.IsNotFound(err)
	if err != nil && !isNotFound {
		err = errors.Wrap(err, "get yatai config")
		return
	}

	if isNotFound {
		return
	}

	if yataiConf_.Endpoint == "" {
		return
	}

	if yataiConf_.ClusterName == "" {
		yataiConf_.ClusterName = "default"
	}

	yataiClient_ := yataiclient.NewYataiClient(yataiConf_.Endpoint, fmt.Sprintf("%s:%s:%s", commonconsts.YataiImageBuilderComponentName, yataiConf_.ClusterName, yataiConf_.ApiToken))

	yataiClient = &yataiClient_
	yataiConf = &yataiConf_
	return
}

func (r *DynamoNimRequestReconciler) getYataiClientWithAuth(ctx context.Context, dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) (**yataiclient.YataiClient, **commonconfig.YataiConfig, error) {
	orgId, ok := dynamoNimRequest.Labels[commonconsts.NgcOrganizationHeaderName]
	if !ok {
		orgId = commonconsts.DefaultOrgId
	}

	userId, ok := dynamoNimRequest.Labels[commonconsts.NgcUserHeaderName]
	if !ok {
		userId = commonconsts.DefaultUserId
	}

	auth := yataiclient.DynamoAuthHeaders{
		OrgId:  orgId,
		UserId: userId,
	}

	client, yataiConf, err := r.getYataiClient(ctx)
	if err != nil {
		return nil, nil, err
	}

	(*client).SetAuth(auth)
	return client, yataiConf, err
}

//nolint:nakedret
func (r *DynamoNimRequestReconciler) getDockerRegistry(ctx context.Context, dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) (dockerRegistry schemas.DockerRegistrySchema, err error) {
	if dynamoNimRequest != nil && dynamoNimRequest.Spec.DockerConfigJSONSecretName != "" {
		secret := &corev1.Secret{}
		err = r.Get(ctx, types.NamespacedName{
			Namespace: dynamoNimRequest.Namespace,
			Name:      dynamoNimRequest.Spec.DockerConfigJSONSecretName,
		}, secret)
		if err != nil {
			err = errors.Wrapf(err, "get docker config json secret %s", dynamoNimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		configJSON, ok := secret.Data[".dockerconfigjson"]
		if !ok {
			err = errors.Errorf("docker config json secret %s does not have .dockerconfigjson key", dynamoNimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		var configObj struct {
			Auths map[string]struct {
				Auth string `json:"auth"`
			} `json:"auths"`
		}
		err = json.Unmarshal(configJSON, &configObj)
		if err != nil {
			err = errors.Wrapf(err, "unmarshal docker config json secret %s", dynamoNimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		imageRegistryURI, _, _ := xstrings.Partition(dynamoNimRequest.Spec.Image, "/")
		var server string
		var auth string
		if imageRegistryURI != "" {
			for k, v := range configObj.Auths {
				if k == imageRegistryURI {
					server = k
					auth = v.Auth
					break
				}
			}
			if server == "" {
				for k, v := range configObj.Auths {
					if strings.Contains(k, imageRegistryURI) {
						server = k
						auth = v.Auth
						break
					}
				}
			}
		}
		if server == "" {
			for k, v := range configObj.Auths {
				server = k
				auth = v.Auth
				break
			}
		}
		if server == "" {
			err = errors.Errorf("no auth in docker config json secret %s", dynamoNimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		dockerRegistry.Server = server
		var credentials []byte
		credentials, err = base64.StdEncoding.DecodeString(auth)
		if err != nil {
			err = errors.Wrapf(err, "cannot base64 decode auth in docker config json secret %s", dynamoNimRequest.Spec.DockerConfigJSONSecretName)
			return
		}
		dockerRegistry.Username, _, dockerRegistry.Password = xstrings.Partition(string(credentials), ":")
		if dynamoNimRequest.Spec.OCIRegistryInsecure != nil {
			dockerRegistry.Secure = !*dynamoNimRequest.Spec.OCIRegistryInsecure
		}
		return
	}

	dockerRegistryConfig, err := commonconfig.GetDockerRegistryConfig()
	if err != nil {
		err = errors.Wrap(err, "get docker registry")
		return
	}

	dynamoNimRepositoryName := "yatai-bentos"
	modelRepositoryName := "yatai-models"
	if dockerRegistryConfig.BentoRepositoryName != "" {
		dynamoNimRepositoryName = dockerRegistryConfig.BentoRepositoryName
	}
	if dockerRegistryConfig.ModelRepositoryName != "" {
		modelRepositoryName = dockerRegistryConfig.ModelRepositoryName
	}
	dynamoNimRepositoryURI := fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.Server, "/"), dynamoNimRepositoryName)
	modelRepositoryURI := fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.Server, "/"), modelRepositoryName)
	if strings.Contains(dockerRegistryConfig.Server, "docker.io") {
		dynamoNimRepositoryURI = fmt.Sprintf("docker.io/%s", dynamoNimRepositoryName)
		modelRepositoryURI = fmt.Sprintf("docker.io/%s", modelRepositoryName)
	}
	dynamoNimRepositoryInClusterURI := dynamoNimRepositoryURI
	modelRepositoryInClusterURI := modelRepositoryURI
	if dockerRegistryConfig.InClusterServer != "" {
		dynamoNimRepositoryInClusterURI = fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.InClusterServer, "/"), dynamoNimRepositoryName)
		modelRepositoryInClusterURI = fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.InClusterServer, "/"), modelRepositoryName)
		if strings.Contains(dockerRegistryConfig.InClusterServer, "docker.io") {
			dynamoNimRepositoryInClusterURI = fmt.Sprintf("docker.io/%s", dynamoNimRepositoryName)
			modelRepositoryInClusterURI = fmt.Sprintf("docker.io/%s", modelRepositoryName)
		}
	}
	dockerRegistry = schemas.DockerRegistrySchema{
		Server:                       dockerRegistryConfig.Server,
		Username:                     dockerRegistryConfig.Username,
		Password:                     dockerRegistryConfig.Password,
		Secure:                       dockerRegistryConfig.Secure,
		BentosRepositoryURI:          dynamoNimRepositoryURI,
		BentosRepositoryURIInCluster: dynamoNimRepositoryInClusterURI,
		ModelsRepositoryURI:          modelRepositoryURI,
		ModelsRepositoryURIInCluster: modelRepositoryInClusterURI,
	}

	return
}

func isAddNamespacePrefix() bool {
	return os.Getenv("ADD_NAMESPACE_PREFIX_TO_IMAGE_NAME") == trueStr
}

func getDynamoNimImagePrefix(dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) string {
	if dynamoNimRequest == nil {
		return ""
	}
	prefix, exist := dynamoNimRequest.Annotations[KubeAnnotationDynamoNimStorageNS]
	if exist && prefix != "" {
		return fmt.Sprintf("%s.", prefix)
	}
	if isAddNamespacePrefix() {
		return fmt.Sprintf("%s.", dynamoNimRequest.Namespace)
	}
	return ""
}

func getDynamoNimImageName(dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest, dockerRegistry schemas.DockerRegistrySchema, dynamoNimRepositoryName, dynamoNimVersion string, inCluster bool) string {
	if dynamoNimRequest != nil && dynamoNimRequest.Spec.Image != "" {
		return dynamoNimRequest.Spec.Image
	}
	var uri, tag string
	if inCluster {
		uri = dockerRegistry.BentosRepositoryURIInCluster
	} else {
		uri = dockerRegistry.BentosRepositoryURI
	}
	tail := fmt.Sprintf("%s.%s", dynamoNimRepositoryName, dynamoNimVersion)
	separateModels := isSeparateModels(dynamoNimRequest)
	if separateModels {
		tail += ".nomodels"
	}
	if isEstargzEnabled() {
		tail += ".esgz"
	}

	tag = fmt.Sprintf("yatai.%s%s", getDynamoNimImagePrefix(dynamoNimRequest), tail)

	if len(tag) > 128 {
		hashStr := hash(tail)
		tag = fmt.Sprintf("yatai.%s%s", getDynamoNimImagePrefix(dynamoNimRequest), hashStr)
		if len(tag) > 128 {
			tag = fmt.Sprintf("yatai.%s", hash(fmt.Sprintf("%s%s", getDynamoNimImagePrefix(dynamoNimRequest), tail)))[:128]
		}
	}
	return fmt.Sprintf("%s:%s", uri, tag)
}

func isSeparateModels(dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) (separateModels bool) {
	return dynamoNimRequest.Annotations[commonconsts.KubeAnnotationYataiImageBuilderSeparateModels] == commonconsts.KubeLabelValueTrue
}

func checkImageExists(dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest, dockerRegistry schemas.DockerRegistrySchema, imageName string) (bool, error) {
	if dynamoNimRequest.Annotations["yatai.ai/force-build-image"] == commonconsts.KubeLabelValueTrue {
		return false, nil
	}

	server, _, imageName := xstrings.Partition(imageName, "/")
	if strings.Contains(server, "docker.io") {
		server = "index.docker.io"
	}
	if dockerRegistry.Secure {
		server = fmt.Sprintf("https://%s", server)
	} else {
		server = fmt.Sprintf("http://%s", server)
	}
	hub, err := registry.New(server, dockerRegistry.Username, dockerRegistry.Password, logrus.Debugf)
	if err != nil {
		err = errors.Wrapf(err, "create docker registry client for %s", server)
		return false, err
	}
	imageName, _, tag := xstrings.LastPartition(imageName, ":")
	tags, err := hub.Tags(imageName)
	isNotFound := err != nil && strings.Contains(err.Error(), "404")
	if isNotFound {
		return false, nil
	}
	if err != nil {
		err = errors.Wrapf(err, "get tags for docker image %s", imageName)
		return false, err
	}
	for _, tag_ := range tags {
		if tag_ == tag {
			return true, nil
		}
	}
	return false, nil
}

type ImageInfo struct {
	DockerRegistry             schemas.DockerRegistrySchema
	DockerConfigJSONSecretName string
	ImageName                  string
	InClusterImageName         string
	DockerRegistryInsecure     bool
}

type GetImageInfoOption struct {
	DynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest
}

//nolint:nakedret
func (r *DynamoNimRequestReconciler) getImageInfo(ctx context.Context, opt GetImageInfoOption) (imageInfo ImageInfo, err error) {
	dynamoNimRepositoryName, _, dynamoNimVersion := xstrings.Partition(opt.DynamoNimRequest.Spec.BentoTag, ":")
	dockerRegistry, err := r.getDockerRegistry(ctx, opt.DynamoNimRequest)
	if err != nil {
		err = errors.Wrap(err, "get docker registry")
		return
	}
	imageInfo.DockerRegistry = dockerRegistry
	imageInfo.ImageName = getDynamoNimImageName(opt.DynamoNimRequest, dockerRegistry, dynamoNimRepositoryName, dynamoNimVersion, false)
	imageInfo.InClusterImageName = getDynamoNimImageName(opt.DynamoNimRequest, dockerRegistry, dynamoNimRepositoryName, dynamoNimVersion, true)

	imageInfo.DockerConfigJSONSecretName = opt.DynamoNimRequest.Spec.DockerConfigJSONSecretName

	imageInfo.DockerRegistryInsecure = opt.DynamoNimRequest.Annotations[commonconsts.KubeAnnotationDockerRegistryInsecure] == "true"
	if opt.DynamoNimRequest.Spec.OCIRegistryInsecure != nil {
		imageInfo.DockerRegistryInsecure = *opt.DynamoNimRequest.Spec.OCIRegistryInsecure
	}

	if imageInfo.DockerConfigJSONSecretName == "" {
		var dockerRegistryConf *commonconfig.DockerRegistryConfig
		dockerRegistryConf, err = commonconfig.GetDockerRegistryConfig()
		if err != nil {
			err = errors.Wrap(err, "get docker registry")
			return
		}
		imageInfo.DockerRegistryInsecure = !dockerRegistryConf.Secure
		var dockerConfigSecret *corev1.Secret
		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Making sure docker config secret %s in namespace %s", commonconsts.KubeSecretNameRegcred, opt.DynamoNimRequest.Namespace)
		dockerConfigSecret, err = r.makeSureDockerConfigJSONSecret(ctx, opt.DynamoNimRequest.Namespace, dockerRegistryConf)
		if err != nil {
			err = errors.Wrap(err, "make sure docker config secret")
			return
		}
		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Docker config secret %s in namespace %s is ready", commonconsts.KubeSecretNameRegcred, opt.DynamoNimRequest.Namespace)
		if dockerConfigSecret != nil {
			imageInfo.DockerConfigJSONSecretName = dockerConfigSecret.Name
		}
	}
	return
}

func (r *DynamoNimRequestReconciler) getDynamoNim(ctx context.Context, dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) (dynamoNim *schemas.DynamoNIM, err error) {
	dynamoNimRepositoryName, _, dynamoNimVersion := xstrings.Partition(dynamoNimRequest.Spec.BentoTag, ":")

	yataiClient_, _, err := r.getYataiClient(ctx)
	if err != nil {
		err = errors.Wrap(err, "get yatai client")
		return
	}

	if yataiClient_ == nil {
		err = errors.New("can't get yatai client, please check yatai configuration")
		return
	}

	yataiClient := *yataiClient_

	r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "FetchDynamoNim", "Getting dynamoNim %s from yatai service", dynamoNimRequest.Spec.BentoTag)
	dynamoNim, err = yataiClient.GetBento(ctx, dynamoNimRepositoryName, dynamoNimVersion)
	if err != nil {
		err = errors.Wrap(err, "get dynamoNim")
		return
	}
	r.Recorder.Eventf(dynamoNimRequest, corev1.EventTypeNormal, "FetchDynamoNim", "Got dynamoNim %s from yatai service", dynamoNimRequest.Spec.BentoTag)
	return
}

func (r *DynamoNimRequestReconciler) getImageBuilderJobName() string {
	guid := xid.New()
	return fmt.Sprintf("yatai-dynamonim-image-builder-%s", guid.String())
}

func (r *DynamoNimRequestReconciler) getImageBuilderJobLabels(dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) map[string]string {
	dynamoNimRepositoryName, _, dynamoNimVersion := xstrings.Partition(dynamoNimRequest.Spec.BentoTag, ":")
	labels := map[string]string{
		commonconsts.KubeLabelBentoRequest:         dynamoNimRequest.Name,
		commonconsts.KubeLabelIsBentoImageBuilder:  "true",
		commonconsts.KubeLabelYataiBentoRepository: dynamoNimRepositoryName,
		commonconsts.KubeLabelYataiBento:           dynamoNimVersion,
	}

	if isSeparateModels(dynamoNimRequest) {
		labels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueTrue
	} else {
		labels[KubeLabelYataiImageBuilderSeparateModels] = commonconsts.KubeLabelValueFalse
	}
	return labels
}

func (r *DynamoNimRequestReconciler) getImageBuilderPodLabels(dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) map[string]string {
	dynamoNimRepositoryName, _, dynamoNimVersion := xstrings.Partition(dynamoNimRequest.Spec.BentoTag, ":")
	return map[string]string{
		commonconsts.KubeLabelBentoRequest:         dynamoNimRequest.Name,
		commonconsts.KubeLabelIsBentoImageBuilder:  "true",
		commonconsts.KubeLabelYataiBentoRepository: dynamoNimRepositoryName,
		commonconsts.KubeLabelYataiBento:           dynamoNimVersion,
	}
}

func hash(text string) string {
	// nolint: gosec
	hasher := md5.New()
	hasher.Write([]byte(text))
	return hex.EncodeToString(hasher.Sum(nil))
}

type GenerateImageBuilderJobOption struct {
	ImageInfo        ImageInfo
	DynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest
}

//nolint:nakedret
func (r *DynamoNimRequestReconciler) generateImageBuilderJob(ctx context.Context, opt GenerateImageBuilderJobOption) (job *batchv1.Job, err error) {
	// nolint: gosimple
	podTemplateSpec, err := r.generateImageBuilderPodTemplateSpec(ctx, GenerateImageBuilderPodTemplateSpecOption(opt))
	if err != nil {
		err = errors.Wrap(err, "generate image builder pod template spec")
		return
	}
	kubeAnnotations := make(map[string]string)
	hashStr, err := r.getHashStr(opt.DynamoNimRequest)
	if err != nil {
		err = errors.Wrap(err, "failed to get hash string")
		return
	}
	kubeAnnotations[KubeAnnotationDynamoNimRequestHash] = hashStr
	job = &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        r.getImageBuilderJobName(),
			Namespace:   opt.DynamoNimRequest.Namespace,
			Labels:      r.getImageBuilderJobLabels(opt.DynamoNimRequest),
			Annotations: kubeAnnotations,
		},
		Spec: batchv1.JobSpec{
			TTLSecondsAfterFinished: ptr.To(int32(60 * 60 * 24)),
			Completions:             ptr.To(int32(1)),
			Parallelism:             ptr.To(int32(1)),
			PodFailurePolicy: &batchv1.PodFailurePolicy{
				Rules: []batchv1.PodFailurePolicyRule{
					{
						Action: batchv1.PodFailurePolicyActionFailJob,
						OnExitCodes: &batchv1.PodFailurePolicyOnExitCodesRequirement{
							ContainerName: ptr.To(BuilderContainerName),
							Operator:      batchv1.PodFailurePolicyOnExitCodesOpIn,
							Values:        []int32{BuilderJobFailedExitCode},
						},
					},
				},
			},
			Template: *podTemplateSpec,
		},
	}
	err = ctrl.SetControllerReference(opt.DynamoNimRequest, job, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set controller reference for job %s", job.Name)
		return
	}
	return
}

func injectPodAffinity(podSpec *corev1.PodSpec, dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) {
	if podSpec.Affinity == nil {
		podSpec.Affinity = &corev1.Affinity{}
	}

	if podSpec.Affinity.PodAffinity == nil {
		podSpec.Affinity.PodAffinity = &corev1.PodAffinity{}
	}

	podSpec.Affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution = append(podSpec.Affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution, corev1.WeightedPodAffinityTerm{
		Weight: 100,
		PodAffinityTerm: corev1.PodAffinityTerm{
			LabelSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					commonconsts.KubeLabelBentoRequest: dynamoNimRequest.Name,
				},
			},
			TopologyKey: corev1.LabelHostname,
		},
	})
}

const BuilderContainerName = "builder"
const BuilderJobFailedExitCode = 42
const ModelSeederContainerName = "seeder"
const ModelSeederJobFailedExitCode = 42

type GenerateImageBuilderPodTemplateSpecOption struct {
	ImageInfo        ImageInfo
	DynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest
}

//nolint:gocyclo,nakedret
func (r *DynamoNimRequestReconciler) generateImageBuilderPodTemplateSpec(ctx context.Context, opt GenerateImageBuilderPodTemplateSpecOption) (pod *corev1.PodTemplateSpec, err error) {
	dynamoNimRepositoryName, _, dynamoNimVersion := xstrings.Partition(opt.DynamoNimRequest.Spec.BentoTag, ":")
	kubeLabels := r.getImageBuilderPodLabels(opt.DynamoNimRequest)

	inClusterImageName := opt.ImageInfo.InClusterImageName

	dockerConfigJSONSecretName := opt.ImageInfo.DockerConfigJSONSecretName

	dockerRegistryInsecure := opt.ImageInfo.DockerRegistryInsecure

	volumes := []corev1.Volume{
		{
			Name: "yatai",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
		{
			Name: "workspace",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
	}

	volumeMounts := []corev1.VolumeMount{
		{
			Name:      "yatai",
			MountPath: "/yatai",
		},
		{
			Name:      "workspace",
			MountPath: "/workspace",
		},
	}

	if dockerConfigJSONSecretName != "" {
		volumes = append(volumes, corev1.Volume{
			Name: dockerConfigJSONSecretName,
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{
					SecretName: dockerConfigJSONSecretName,
					Items: []corev1.KeyToPath{
						{
							Key:  ".dockerconfigjson",
							Path: "config.json",
						},
					},
				},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      dockerConfigJSONSecretName,
			MountPath: "/kaniko/.docker/",
		})
	}

	var dynamoNim *schemas.DynamoNIM
	yataiAPITokenSecretName := ""
	dynamoNimDownloadURL := opt.DynamoNimRequest.Spec.DownloadURL
	dynamoNimDownloadHeader := ""

	if dynamoNimDownloadURL == "" {
		var yataiClient_ **yataiclient.YataiClient
		var yataiConf_ **commonconfig.YataiConfig

		yataiClient_, yataiConf_, err = r.getYataiClientWithAuth(ctx, opt.DynamoNimRequest)
		if err != nil {
			err = errors.Wrap(err, "get yatai client")
			return
		}

		if yataiClient_ == nil || yataiConf_ == nil {
			err = errors.New("can't get yatai client, please check yatai configuration")
			return
		}

		yataiClient := *yataiClient_
		yataiConf := *yataiConf_

		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting dynamoNim %s from yatai service", opt.DynamoNimRequest.Spec.BentoTag)
		dynamoNim, err = yataiClient.GetBento(ctx, dynamoNimRepositoryName, dynamoNimVersion)
		if err != nil {
			err = errors.Wrap(err, "get dynamoNim")
			return
		}
		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got dynamoNim %s from yatai service", opt.DynamoNimRequest.Spec.BentoTag)

		if dynamoNim.TransmissionStrategy != nil && *dynamoNim.TransmissionStrategy == schemas.TransmissionStrategyPresignedURL {
			var dynamoNim_ *schemas.DynamoNIM
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting presigned url for dynamoNim %s from yatai service", opt.DynamoNimRequest.Spec.BentoTag)
			dynamoNim_, err = yataiClient.PresignBentoDownloadURL(ctx, dynamoNimRepositoryName, dynamoNimVersion)
			if err != nil {
				err = errors.Wrap(err, "presign dynamoNim download url")
				return
			}
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got presigned url for dynamoNim %s from yatai service", opt.DynamoNimRequest.Spec.BentoTag)
			dynamoNimDownloadURL = dynamoNim_.PresignedDownloadUrl
		} else {
			dynamoNimDownloadURL = fmt.Sprintf("%s/api/v1/dynamo_nims/%s/versions/%s/download", yataiConf.Endpoint, dynamoNimRepositoryName, dynamoNimVersion)
			dynamoNimDownloadHeader = fmt.Sprintf("%s: %s:%s:$%s", commonconsts.YataiApiTokenHeaderName, commonconsts.YataiImageBuilderComponentName, yataiConf.ClusterName, commonconsts.EnvYataiApiToken)
		}

		// nolint: gosec
		yataiAPITokenSecretName = "yatai-api-token"

		yataiAPITokenSecret := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      yataiAPITokenSecretName,
				Namespace: opt.DynamoNimRequest.Namespace,
			},
			StringData: map[string]string{
				commonconsts.EnvYataiApiToken: yataiConf.ApiToken,
			},
		}

		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting secret %s in namespace %s", yataiAPITokenSecretName, opt.DynamoNimRequest.Namespace)
		_yataiAPITokenSecret := &corev1.Secret{}
		err = r.Get(ctx, types.NamespacedName{Namespace: opt.DynamoNimRequest.Namespace, Name: yataiAPITokenSecretName}, _yataiAPITokenSecret)
		isNotFound := k8serrors.IsNotFound(err)
		if err != nil && !isNotFound {
			err = errors.Wrapf(err, "failed to get secret %s", yataiAPITokenSecretName)
			return
		}

		if isNotFound {
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is not found, so creating it in namespace %s", yataiAPITokenSecretName, opt.DynamoNimRequest.Namespace)
			err = r.Create(ctx, yataiAPITokenSecret)
			isExists := k8serrors.IsAlreadyExists(err)
			if err != nil && !isExists {
				err = errors.Wrapf(err, "failed to create secret %s", yataiAPITokenSecretName)
				return
			}
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is created in namespace %s", yataiAPITokenSecretName, opt.DynamoNimRequest.Namespace)
		} else {
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is found in namespace %s, so updating it", yataiAPITokenSecretName, opt.DynamoNimRequest.Namespace)
			err = r.Update(ctx, yataiAPITokenSecret)
			if err != nil {
				err = errors.Wrapf(err, "failed to update secret %s", yataiAPITokenSecretName)
				return
			}
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is updated in namespace %s", yataiAPITokenSecretName, opt.DynamoNimRequest.Namespace)
		}
	}
	internalImages := commonconfig.GetInternalImages()
	logrus.Infof("Image builder is using the images %v", *internalImages)

	buildEngine := getDynamoNimImageBuildEngine()

	privileged := buildEngine != DynamoNimImageBuildEngineBuildkitRootless

	dynamoNimDownloadCommandTemplate, err := template.New("downloadCommand").Parse(`
set -e

mkdir -p /workspace/buildcontext
url="{{.DynamoNimDownloadURL}}"
echo "Downloading dynamoNim {{.DynamoNimRepositoryName}}:{{.DynamoNimVersion}} to /tmp/downloaded.tar..."
if [[ ${url} == s3://* ]]; then
	echo "Downloading from s3..."
	aws s3 cp ${url} /tmp/downloaded.tar
elif [[ ${url} == gs://* ]]; then
	echo "Downloading from GCS..."
	gsutil cp ${url} /tmp/downloaded.tar
else
	curl --fail -L -H "{{.DynamoNimDownloadHeader}}" ${url} --output /tmp/downloaded.tar --progress-bar
fi
cd /workspace/buildcontext
echo "Extracting dynamoNim tar file..."
tar -xvf /tmp/downloaded.tar
echo "Removing dynamoNim tar file..."
rm /tmp/downloaded.tar
{{if not .Privileged}}
echo "Changing directory permission..."
chown -R 1000:1000 /workspace
{{end}}
echo "Done"
	`)

	if err != nil {
		err = errors.Wrap(err, "failed to parse download command template")
		return
	}

	var dynamoNimDownloadCommandBuffer bytes.Buffer

	err = dynamoNimDownloadCommandTemplate.Execute(&dynamoNimDownloadCommandBuffer, map[string]interface{}{
		"DynamoNimDownloadURL":    dynamoNimDownloadURL,
		"DynamoNimDownloadHeader": dynamoNimDownloadHeader,
		"DynamoNimRepositoryName": dynamoNimRepositoryName,
		"DynamoNimVersion":        dynamoNimVersion,
		"Privileged":              privileged,
	})
	if err != nil {
		err = errors.Wrap(err, "failed to execute download command template")
		return
	}

	dynamoNimDownloadCommand := dynamoNimDownloadCommandBuffer.String()

	downloaderContainerResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1000m"),
			corev1.ResourceMemory: resource.MustParse("3000Mi"),
		},
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("100m"),
			corev1.ResourceMemory: resource.MustParse("1000Mi"),
		},
	}

	downloaderContainerEnvFrom := opt.DynamoNimRequest.Spec.DownloaderContainerEnvFrom

	if yataiAPITokenSecretName != "" {
		downloaderContainerEnvFrom = append(downloaderContainerEnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: yataiAPITokenSecretName,
				},
			},
		})
	}

	initContainers := []corev1.Container{
		{
			Name:  "dynamonim-downloader",
			Image: internalImages.BentoDownloader,
			Command: []string{
				"bash",
				"-c",
				dynamoNimDownloadCommand,
			},
			VolumeMounts: volumeMounts,
			Resources:    downloaderContainerResources,
			EnvFrom:      downloaderContainerEnvFrom,
			Env: []corev1.EnvVar{
				{
					Name:  "AWS_EC2_METADATA_DISABLED",
					Value: "true",
				},
			},
		},
	}

	containers := make([]corev1.Container, 0)

	models := opt.DynamoNimRequest.Spec.Models
	modelsSeen := map[string]struct{}{}
	for _, model := range models {
		modelsSeen[model.Tag] = struct{}{}
	}

	if dynamoNim != nil {
		for _, modelTag := range dynamoNim.Manifest.Models {
			if _, ok := modelsSeen[modelTag]; !ok {
				models = append(models, nvidiacomv1alpha1.BentoModel{
					Tag: modelTag,
				})
			}
		}
	}

	var globalExtraPodMetadata *dynamoCommon.ExtraPodMetadata
	var globalExtraPodSpec *dynamoCommon.ExtraPodSpec
	var globalExtraContainerEnv []corev1.EnvVar
	var globalDefaultImageBuilderContainerResources *corev1.ResourceRequirements
	var buildArgs []string
	var builderArgs []string

	configNamespace, err := commonconfig.GetYataiImageBuilderNamespace(ctx)
	if err != nil {
		err = errors.Wrap(err, "failed to get Yatai image builder namespace")
		return
	}

	configCmName := "yatai-image-builder-config"
	r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting configmap %s from namespace %s", configCmName, configNamespace)
	configCm := &corev1.ConfigMap{}
	err = r.Get(ctx, types.NamespacedName{Name: configCmName, Namespace: configNamespace}, configCm)
	configCmIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !configCmIsNotFound {
		err = errors.Wrap(err, "failed to get configmap")
		return
	}
	err = nil // nolint: ineffassign

	if !configCmIsNotFound {
		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Configmap %s is got from namespace %s", configCmName, configNamespace)

		globalExtraPodMetadata = &dynamoCommon.ExtraPodMetadata{}

		if val, ok := configCm.Data["extra_pod_metadata"]; ok {
			err = yaml.Unmarshal([]byte(val), globalExtraPodMetadata)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal extra_pod_metadata, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		globalExtraPodSpec = &dynamoCommon.ExtraPodSpec{}

		if val, ok := configCm.Data["extra_pod_spec"]; ok {
			err = yaml.Unmarshal([]byte(val), globalExtraPodSpec)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal extra_pod_spec, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		globalExtraContainerEnv = []corev1.EnvVar{}

		if val, ok := configCm.Data["extra_container_env"]; ok {
			err = yaml.Unmarshal([]byte(val), &globalExtraContainerEnv)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal extra_container_env, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		if val, ok := configCm.Data["default_image_builder_container_resources"]; ok {
			globalDefaultImageBuilderContainerResources = &corev1.ResourceRequirements{}
			err = yaml.Unmarshal([]byte(val), globalDefaultImageBuilderContainerResources)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal default_image_builder_container_resources, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		buildArgs = []string{}

		if val, ok := configCm.Data["build_args"]; ok {
			err = yaml.Unmarshal([]byte(val), &buildArgs)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal build_args, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}

		builderArgs = []string{}
		if val, ok := configCm.Data["builder_args"]; ok {
			err = yaml.Unmarshal([]byte(val), &builderArgs)
			if err != nil {
				err = errors.Wrapf(err, "failed to yaml unmarshal builder_args, please check the configmap %s in namespace %s", configCmName, configNamespace)
				return
			}
		}
		logrus.Info("passed in builder args: ", builderArgs)
	} else {
		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Configmap %s is not found in namespace %s", configCmName, configNamespace)
	}

	if buildArgs == nil {
		buildArgs = make([]string, 0)
	}

	if opt.DynamoNimRequest.Spec.BuildArgs != nil {
		buildArgs = append(buildArgs, opt.DynamoNimRequest.Spec.BuildArgs...)
	}

	dockerFilePath := "/workspace/buildcontext/env/docker/Dockerfile"

	builderContainerEnvFrom := make([]corev1.EnvFromSource, 0)
	builderContainerEnvs := []corev1.EnvVar{
		{
			Name:  "DOCKER_CONFIG",
			Value: "/kaniko/.docker/",
		},
		{
			Name:  "IFS",
			Value: "''",
		},
	}

	kanikoCacheRepo := os.Getenv("KANIKO_CACHE_REPO")
	if kanikoCacheRepo == "" {
		kanikoCacheRepo = opt.ImageInfo.DockerRegistry.BentosRepositoryURIInCluster
	}

	kubeAnnotations := make(map[string]string)
	kubeAnnotations[KubeAnnotationDynamoNimRequestImageBuiderHash] = opt.DynamoNimRequest.Annotations[KubeAnnotationDynamoNimRequestImageBuiderHash]

	command := []string{
		"/kaniko/executor",
	}
	args := []string{
		"--context=/workspace/buildcontext",
		"--verbosity=info",
		"--image-fs-extract-retry=3",
		"--cache=false",
		fmt.Sprintf("--cache-repo=%s", kanikoCacheRepo),
		"--compressed-caching=false",
		"--compression=zstd",
		"--compression-level=-7",
		fmt.Sprintf("--dockerfile=%s", dockerFilePath),
		fmt.Sprintf("--insecure=%v", dockerRegistryInsecure),
		fmt.Sprintf("--destination=%s", inClusterImageName),
	}

	kanikoSnapshotMode := os.Getenv("KANIKO_SNAPSHOT_MODE")
	if kanikoSnapshotMode != "" {
		args = append(args, fmt.Sprintf("--snapshot-mode=%s", kanikoSnapshotMode))
	}

	var builderImage string
	switch buildEngine {
	case DynamoNimImageBuildEngineKaniko:
		builderImage = internalImages.Kaniko
		if isEstargzEnabled() {
			builderContainerEnvs = append(builderContainerEnvs, corev1.EnvVar{
				Name:  "GGCR_EXPERIMENT_ESTARGZ",
				Value: "1",
			})
		}
	case DynamoNimImageBuildEngineBuildkit:
		builderImage = internalImages.Buildkit
	case DynamoNimImageBuildEngineBuildkitRootless:
		builderImage = internalImages.BuildkitRootless
	default:
		err = errors.Errorf("unknown dynamoNim image build engine %s", buildEngine)
		return
	}

	isBuildkit := buildEngine == DynamoNimImageBuildEngineBuildkit || buildEngine == DynamoNimImageBuildEngineBuildkitRootless

	if isBuildkit {
		output := fmt.Sprintf("type=image,name=%s,push=true,registry.insecure=%v", inClusterImageName, dockerRegistryInsecure)
		buildkitdFlags := []string{}
		if !privileged {
			buildkitdFlags = append(buildkitdFlags, "--oci-worker-no-process-sandbox")
		}
		if isEstargzEnabled() {
			buildkitdFlags = append(buildkitdFlags, "--oci-worker-snapshotter=stargz")
			output += ",oci-mediatypes=true,compression=estargz,force-compression=true"
		}
		if len(buildkitdFlags) > 0 {
			builderContainerEnvs = append(builderContainerEnvs, corev1.EnvVar{
				Name:  "BUILDKITD_FLAGS",
				Value: strings.Join(buildkitdFlags, " "),
			})
		}
		buildkitURL := os.Getenv("BUILDKIT_URL")
		if buildkitURL == "" {
			err = errors.New("BUILDKIT_URL is not set")
			return
		}
		command = []string{
			"buildctl",
		}
		args = []string{
			"--addr",
			buildkitURL,
			"build",
			"--frontend",
			"dockerfile.v0",
			"--local",
			"context=/workspace/buildcontext",
			"--local",
			fmt.Sprintf("dockerfile=%s", filepath.Dir(dockerFilePath)),
			"--output",
			output,
		}
		cacheRepo := os.Getenv("BUILDKIT_CACHE_REPO")
		if cacheRepo != "" {
			args = append(args, "--export-cache", fmt.Sprintf("type=registry,ref=%s:buildcache,mode=max,compression=zstd,ignore-error=true", cacheRepo))
			args = append(args, "--import-cache", fmt.Sprintf("type=registry,ref=%s:buildcache", cacheRepo))
		}
	}

	var builderContainerSecurityContext *corev1.SecurityContext

	if buildEngine == DynamoNimImageBuildEngineBuildkit {
		builderContainerSecurityContext = &corev1.SecurityContext{
			Privileged: ptr.To(true),
		}
	} else if buildEngine == DynamoNimImageBuildEngineBuildkitRootless {
		kubeAnnotations["container.apparmor.security.beta.kubernetes.io/builder"] = "unconfined"
		builderContainerSecurityContext = &corev1.SecurityContext{
			SeccompProfile: &corev1.SeccompProfile{
				Type: corev1.SeccompProfileTypeUnconfined,
			},
			RunAsUser:  ptr.To(int64(1000)),
			RunAsGroup: ptr.To(int64(1000)),
		}
	}

	// add build args to pass via --build-arg
	for _, buildArg := range buildArgs {
		quotedBuildArg := unix.SingleQuote.Quote(buildArg)
		if isBuildkit {
			args = append(args, "--opt", fmt.Sprintf("build-arg:%s", quotedBuildArg))
		} else {
			args = append(args, fmt.Sprintf("--build-arg=%s", quotedBuildArg))
		}
	}
	// add other arguments to builder
	args = append(args, builderArgs...)
	logrus.Info("yatai-image-builder args: ", args)

	// nolint: gosec
	buildArgsSecretName := "yatai-image-builder-build-args"
	r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting secret %s from namespace %s", buildArgsSecretName, configNamespace)
	buildArgsSecret := &corev1.Secret{}
	err = r.Get(ctx, types.NamespacedName{Name: buildArgsSecretName, Namespace: configNamespace}, buildArgsSecret)
	buildArgsSecretIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !buildArgsSecretIsNotFound {
		err = errors.Wrap(err, "failed to get secret")
		return
	}

	if !buildArgsSecretIsNotFound {
		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is got from namespace %s", buildArgsSecretName, configNamespace)
		if configNamespace != opt.DynamoNimRequest.Namespace {
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is in namespace %s, but DynamoNimRequest is in namespace %s, so we need to copy the secret to DynamoNimRequest namespace", buildArgsSecretName, configNamespace, opt.DynamoNimRequest.Namespace)
			r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting secret %s in namespace %s", buildArgsSecretName, opt.DynamoNimRequest.Namespace)
			_buildArgsSecret := &corev1.Secret{}
			err = r.Get(ctx, types.NamespacedName{Namespace: opt.DynamoNimRequest.Namespace, Name: buildArgsSecretName}, _buildArgsSecret)
			localBuildArgsSecretIsNotFound := k8serrors.IsNotFound(err)
			if err != nil && !localBuildArgsSecretIsNotFound {
				err = errors.Wrapf(err, "failed to get secret %s from namespace %s", buildArgsSecretName, opt.DynamoNimRequest.Namespace)
				return
			}
			if localBuildArgsSecretIsNotFound {
				r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Copying secret %s from namespace %s to namespace %s", buildArgsSecretName, configNamespace, opt.DynamoNimRequest.Namespace)
				err = r.Create(ctx, &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      buildArgsSecretName,
						Namespace: opt.DynamoNimRequest.Namespace,
					},
					Data: buildArgsSecret.Data,
				})
				if err != nil {
					err = errors.Wrapf(err, "failed to create secret %s in namespace %s", buildArgsSecretName, opt.DynamoNimRequest.Namespace)
					return
				}
			} else {
				r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is already in namespace %s", buildArgsSecretName, opt.DynamoNimRequest.Namespace)
				r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Updating secret %s in namespace %s", buildArgsSecretName, opt.DynamoNimRequest.Namespace)
				err = r.Update(ctx, &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      buildArgsSecretName,
						Namespace: opt.DynamoNimRequest.Namespace,
					},
					Data: buildArgsSecret.Data,
				})
				if err != nil {
					err = errors.Wrapf(err, "failed to update secret %s in namespace %s", buildArgsSecretName, opt.DynamoNimRequest.Namespace)
					return
				}
			}
		}

		for key := range buildArgsSecret.Data {
			envName := fmt.Sprintf("BENTOML_BUILD_ARG_%s", strings.ReplaceAll(strings.ToUpper(strcase.ToKebab(key)), "-", "_"))
			builderContainerEnvs = append(builderContainerEnvs, corev1.EnvVar{
				Name: envName,
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{
							Name: buildArgsSecretName,
						},
						Key: key,
					},
				},
			})

			if isBuildkit {
				args = append(args, "--opt", fmt.Sprintf("build-arg:%s=$(%s)", key, envName))
			} else {
				args = append(args, fmt.Sprintf("--build-arg=%s=$(%s)", key, envName))
			}
		}
	} else {
		r.Recorder.Eventf(opt.DynamoNimRequest, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Secret %s is not found in namespace %s", buildArgsSecretName, configNamespace)
	}

	builderContainerArgs := []string{
		"-c",
		fmt.Sprintf("sleep 15; %s && exit 0 || exit %d", shquot.POSIXShell(append(command, args...)), BuilderJobFailedExitCode), // TODO: remove once functionality exists to wait for istio sidecar.
	}

	container := corev1.Container{
		Name:            BuilderContainerName,
		Image:           builderImage,
		ImagePullPolicy: corev1.PullAlways,
		Command:         []string{"sh"},
		Args:            builderContainerArgs,
		VolumeMounts:    volumeMounts,
		Env:             builderContainerEnvs,
		EnvFrom:         builderContainerEnvFrom,
		TTY:             true,
		Stdin:           true,
		SecurityContext: builderContainerSecurityContext,
	}

	if globalDefaultImageBuilderContainerResources != nil {
		container.Resources = *globalDefaultImageBuilderContainerResources
	}

	if opt.DynamoNimRequest.Spec.ImageBuilderContainerResources != nil {
		container.Resources = *opt.DynamoNimRequest.Spec.ImageBuilderContainerResources
	}

	containers = append(containers, container)

	pod = &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      kubeLabels,
			Annotations: kubeAnnotations,
		},
		Spec: corev1.PodSpec{
			RestartPolicy:  corev1.RestartPolicyNever,
			Volumes:        volumes,
			InitContainers: initContainers,
			Containers:     containers,
		},
	}

	if globalExtraPodMetadata != nil {
		for k, v := range globalExtraPodMetadata.Annotations {
			pod.Annotations[k] = v
		}

		for k, v := range globalExtraPodMetadata.Labels {
			pod.Labels[k] = v
		}
	}

	if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodMetadata != nil {
		for k, v := range opt.DynamoNimRequest.Spec.ImageBuilderExtraPodMetadata.Annotations {
			pod.Annotations[k] = v
		}

		for k, v := range opt.DynamoNimRequest.Spec.ImageBuilderExtraPodMetadata.Labels {
			pod.Labels[k] = v
		}
	}

	if globalExtraPodSpec != nil {
		pod.Spec.PriorityClassName = globalExtraPodSpec.PriorityClassName
		pod.Spec.SchedulerName = globalExtraPodSpec.SchedulerName
		pod.Spec.NodeSelector = globalExtraPodSpec.NodeSelector
		pod.Spec.Affinity = globalExtraPodSpec.Affinity
		pod.Spec.Tolerations = globalExtraPodSpec.Tolerations
		pod.Spec.TopologySpreadConstraints = globalExtraPodSpec.TopologySpreadConstraints
		pod.Spec.ServiceAccountName = globalExtraPodSpec.ServiceAccountName
	}

	if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec != nil {
		if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.PriorityClassName != "" {
			pod.Spec.PriorityClassName = opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.PriorityClassName
		}

		if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.SchedulerName != "" {
			pod.Spec.SchedulerName = opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.SchedulerName
		}

		if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.NodeSelector != nil {
			pod.Spec.NodeSelector = opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.NodeSelector
		}

		if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.Affinity != nil {
			pod.Spec.Affinity = opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.Affinity
		}

		if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.Tolerations != nil {
			pod.Spec.Tolerations = opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.Tolerations
		}

		if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.TopologySpreadConstraints != nil {
			pod.Spec.TopologySpreadConstraints = opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.TopologySpreadConstraints
		}

		if opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.ServiceAccountName != "" {
			pod.Spec.ServiceAccountName = opt.DynamoNimRequest.Spec.ImageBuilderExtraPodSpec.ServiceAccountName
		}
	}

	injectPodAffinity(&pod.Spec, opt.DynamoNimRequest)

	if pod.Spec.ServiceAccountName == "" {
		serviceAccounts := &corev1.ServiceAccountList{}
		err = r.List(ctx, serviceAccounts, client.InNamespace(opt.DynamoNimRequest.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelYataiImageBuilderPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			err = errors.Wrapf(err, "failed to list service accounts in namespace %s", opt.DynamoNimRequest.Namespace)
			return
		}
		if len(serviceAccounts.Items) > 0 {
			pod.Spec.ServiceAccountName = serviceAccounts.Items[0].Name
		} else {
			pod.Spec.ServiceAccountName = "default"
		}
	}

	for i, c := range pod.Spec.InitContainers {
		env := c.Env
		if globalExtraContainerEnv != nil {
			env = append(env, globalExtraContainerEnv...)
		}
		env = append(env, opt.DynamoNimRequest.Spec.ImageBuilderExtraContainerEnv...)
		pod.Spec.InitContainers[i].Env = env
	}
	for i, c := range pod.Spec.Containers {
		env := c.Env
		if globalExtraContainerEnv != nil {
			env = append(env, globalExtraContainerEnv...)
		}
		env = append(env, opt.DynamoNimRequest.Spec.ImageBuilderExtraContainerEnv...)
		pod.Spec.Containers[i].Env = env
	}

	return
}

func (r *DynamoNimRequestReconciler) getHashStr(dynamoNimRequest *nvidiacomv1alpha1.DynamoNimRequest) (string, error) {
	var hash uint64
	hash, err := hashstructure.Hash(struct {
		Spec        nvidiacomv1alpha1.DynamoNimRequestSpec
		Labels      map[string]string
		Annotations map[string]string
	}{
		Spec:        dynamoNimRequest.Spec,
		Labels:      dynamoNimRequest.Labels,
		Annotations: dynamoNimRequest.Annotations,
	}, hashstructure.FormatV2, nil)
	if err != nil {
		err = errors.Wrap(err, "get dynamoNimRequest CR spec hash")
		return "", err
	}
	hashStr := strconv.FormatUint(hash, 10)
	return hashStr, nil
}

const (
	trueStr = "true"
)

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoNimRequestReconciler) SetupWithManager(mgr ctrl.Manager) error {

	err := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoNimRequest{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&nvidiacomv1alpha1.DynamoNim{}).
		Owns(&batchv1.Job{}).
		WithEventFilter(controller_common.EphemeralDeploymentEventFilter(r.Config)).
		Complete(r)
	return errors.Wrap(err, "failed to setup DynamoNimRequest controller")
}
