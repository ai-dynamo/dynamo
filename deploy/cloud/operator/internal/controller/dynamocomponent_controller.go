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
	"strconv"
	"strings"
	"text/template"
	"time"

	"emperror.dev/errors"
	commonconfig "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/apparentlymart/go-shquot/shquot"
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

	apiStoreClient "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/api_store_client"
	dynamoCommon "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/schemas"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
)

const (
	dockerConfigSecretKey = ".dockerconfigjson"
)

// DynamoComponentReconciler reconciles a DynamoComponent object
type DynamoComponentReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Config   controller_common.Config
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponents,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponents/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponents/finalizers,verbs=update
//+kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponents,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponents/status,verbs=get;update;patch
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
// the DynamoComponent object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.18.2/pkg/reconcile
//
//nolint:gocyclo,nakedret
func (r *DynamoComponentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logs := log.FromContext(ctx)

	DynamoComponent := &nvidiacomv1alpha1.DynamoComponent{}

	err = r.Get(ctx, req.NamespacedName, DynamoComponent)

	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			logs.Info("DynamoComponent resource not found. Ignoring since object must be deleted")
			err = nil
			return
		}
		// Error reading the object - requeue the request.
		logs.Error(err, "Failed to get DynamoComponent")
		return
	}

	if DynamoComponent.IsReady() {
		logs.Info("Skip available DynamoComponent")
		return
	}

	if len(DynamoComponent.Status.Conditions) == 0 {
		DynamoComponent, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageBuilding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile DynamoComponent",
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageExists,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile DynamoComponent",
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeDynamoComponentAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Reconciling",
			},
		)
		if err != nil {
			return
		}
	}

	logs = logs.WithValues("DynamoComponent", DynamoComponent.Name, "DynamoComponentNamespace", DynamoComponent.Namespace)

	defer func() {
		if err == nil {
			logs.Info("Reconcile success")
			return
		}
		logs.Error(err, "Failed to reconcile DynamoComponent.")
		r.Recorder.Eventf(DynamoComponent, corev1.EventTypeWarning, "ReconcileError", "Failed to reconcile DynamoComponent: %v", err)
		_, err_ := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeDynamoComponentAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: err.Error(),
			},
		)
		if err_ != nil {
			logs.Error(err_, "Failed to update DynamoComponent status")
			return
		}
	}()

	DynamoComponent, _, imageExists, imageExistsResult, err := r.ensureImageExists(ctx, ensureImageExistsOption{
		DynamoComponent: DynamoComponent,
		req:             req,
	})

	if err != nil {
		err = errors.Wrapf(err, "ensure image exists")
		return
	}

	if !imageExists {
		result = imageExistsResult
		DynamoComponent, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeDynamoComponentAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "DynamoComponent image is building",
			},
		)
		if err != nil {
			return
		}
		return
	}

	DynamoComponent, err = r.setStatusConditions(ctx, req,
		metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeDynamoComponentAvailable,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: "DynamoComponent image is generated",
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
	DynamoComponent *nvidiacomv1alpha1.DynamoComponent
	req             ctrl.Request
}

//nolint:gocyclo,nakedret
func (r *DynamoComponentReconciler) ensureImageExists(ctx context.Context, opt ensureImageExistsOption) (DynamoComponent *nvidiacomv1alpha1.DynamoComponent, imageInfo ImageInfo, imageExists bool, result ctrl.Result, err error) { // nolint: unparam
	logs := log.FromContext(ctx)

	DynamoComponent = opt.DynamoComponent
	req := opt.req

	imageInfo, err = r.getImageInfo(ctx, GetImageInfoOption{
		DynamoComponent: DynamoComponent,
	})
	if err != nil {
		err = errors.Wrap(err, "get image info")
		return
	}

	imageExistsCheckedCondition := meta.FindStatusCondition(DynamoComponent.Status.Conditions, nvidiacomv1alpha1.DynamoComponentConditionTypeImageExistsChecked)
	imageExistsCondition := meta.FindStatusCondition(DynamoComponent.Status.Conditions, nvidiacomv1alpha1.DynamoComponentConditionTypeImageExists)
	if imageExistsCheckedCondition == nil || imageExistsCheckedCondition.Status != metav1.ConditionTrue || imageExistsCheckedCondition.Message != imageInfo.ImageName {
		imageExistsCheckedCondition = &metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageExistsChecked,
			Status:  metav1.ConditionUnknown,
			Reason:  "Reconciling",
			Message: imageInfo.ImageName,
		}
		dynamoComponentAvailableCondition := &metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeDynamoComponentAvailable,
			Status:  metav1.ConditionUnknown,
			Reason:  "Reconciling",
			Message: "Checking image exists",
		}
		DynamoComponent, err = r.setStatusConditions(ctx, req, *imageExistsCheckedCondition, *dynamoComponentAvailableCondition)
		if err != nil {
			return
		}
		r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "CheckingImage", "Checking image exists: %s", imageInfo.ImageName)
		imageExists, err = checkImageExists(DynamoComponent, imageInfo.DockerRegistry, imageInfo.ImageName)
		if err != nil {
			err = errors.Wrapf(err, "check image %s exists", imageInfo.ImageName)
			return
		}

		err = r.Get(ctx, req.NamespacedName, DynamoComponent)
		if err != nil {
			logs.Error(err, "Failed to re-fetch DynamoComponent")
			return
		}

		if imageExists {
			r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "CheckingImage", "Image exists: %s", imageInfo.ImageName)
			imageExistsCheckedCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageExistsChecked,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: imageInfo.ImageName,
			}
			imageExistsCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageExists,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: imageInfo.ImageName,
			}
			DynamoComponent, err = r.setStatusConditions(ctx, req, *imageExistsCondition, *imageExistsCheckedCondition)
			if err != nil {
				return
			}
		} else {
			r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "CheckingImage", "Image not exists: %s", imageInfo.ImageName)
			imageExistsCheckedCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageExistsChecked,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image not exists: %s", imageInfo.ImageName),
			}
			imageExistsCondition = &metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageExists,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image %s is not exists", imageInfo.ImageName),
			}
			DynamoComponent, err = r.setStatusConditions(ctx, req, *imageExistsCondition, *imageExistsCheckedCondition)
			if err != nil {
				return
			}
		}
	}

	var DynamoComponentHashStr string
	DynamoComponentHashStr, err = r.getHashStr(DynamoComponent)
	if err != nil {
		err = errors.Wrapf(err, "get DynamoComponent %s/%s hash string", DynamoComponent.Namespace, DynamoComponent.Name)
		return
	}

	imageExists = imageExistsCondition != nil && imageExistsCondition.Status == metav1.ConditionTrue && imageExistsCondition.Message == imageInfo.ImageName
	if imageExists {
		return
	}

	jobLabels := map[string]string{
		commonconsts.KubeLabelDynamoComponent:      DynamoComponent.Name,
		commonconsts.KubeLabelIsDynamoImageBuilder: commonconsts.KubeLabelValueTrue,
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

		oldHash := job_.Annotations[consts.KubeAnnotationDynamoComponentHash]
		if oldHash != DynamoComponentHashStr {
			logs.Info("Because hash changed, delete old job", "job", job_.Name, "oldHash", oldHash, "newHash", DynamoComponentHashStr)
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
			ImageInfo:       imageInfo,
			DynamoComponent: DynamoComponent,
		})
		if err != nil {
			err = errors.Wrap(err, "generate image builder job")
			return
		}
		r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderJob", "Creating image builder job: %s", job.Name)
		err = r.Create(ctx, job)
		if err != nil {
			err = errors.Wrapf(err, "create image builder job %s", job.Name)
			return
		}
		r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderJob", "Created image builder job: %s", job.Name)
		return
	}

	r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "CheckingImageBuilderJob", "Found image builder job: %s", job.Name)

	err = r.Get(ctx, req.NamespacedName, DynamoComponent)
	if err != nil {
		logs.Error(err, "Failed to re-fetch DynamoComponent")
		return
	}
	imageBuildingCondition := meta.FindStatusCondition(DynamoComponent.Status.Conditions, nvidiacomv1alpha1.DynamoComponentConditionTypeImageBuilding)

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
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageBuilding,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is running", job.Name),
			})
		} else {
			conditions = append(conditions, metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageBuilding,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is waiting", job.Name),
			})
		}
		if DynamoComponent.Spec.ImageBuildTimeout != nil {
			if imageBuildingCondition != nil && imageBuildingCondition.LastTransitionTime.Add(time.Duration(*DynamoComponent.Spec.ImageBuildTimeout)).Before(time.Now()) {
				conditions = append(conditions, metav1.Condition{
					Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageBuilding,
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

		if DynamoComponent, err = r.setStatusConditions(ctx, req, conditions...); err != nil {
			return
		}

		if imageBuildingCondition != nil && imageBuildingCondition.Status != metav1.ConditionTrue && isJobRunning {
			r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "DynamoComponentImageBuilder", "Image is building now")
		}

		return
	}

	if isJobFailed {
		DynamoComponent, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageBuilding,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Image building job %s is failed.", job.Name),
			},
			metav1.Condition{
				Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeDynamoComponentAvailable,
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

	DynamoComponent, err = r.setStatusConditions(ctx, req,
		metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageBuilding,
			Status:  metav1.ConditionFalse,
			Reason:  "Reconciling",
			Message: fmt.Sprintf("Image building job %s is succeeded.", job.Name),
		},
		metav1.Condition{
			Type:    nvidiacomv1alpha1.DynamoComponentConditionTypeImageExists,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: imageInfo.ImageName,
		},
	)
	if err != nil {
		return
	}

	r.Recorder.Eventf(DynamoComponent, corev1.EventTypeNormal, "DynamoComponentImageBuilder", "Image has been built successfully")

	imageExists = true

	return
}

func (r *DynamoComponentReconciler) setStatusConditions(ctx context.Context, req ctrl.Request, conditions ...metav1.Condition) (DynamoComponent *nvidiacomv1alpha1.DynamoComponent, err error) {
	DynamoComponent = &nvidiacomv1alpha1.DynamoComponent{}
	/*
		Please don't blame me when you see this kind of code,
		this is to avoid "the object has been modified; please apply your changes to the latest version and try again" when updating CR status,
		don't doubt that almost all CRD operators (e.g. cert-manager) can't avoid this stupid error and can only try to avoid this by this stupid way.
	*/
	for i := 0; i < 3; i++ {
		if err = r.Get(ctx, req.NamespacedName, DynamoComponent); err != nil {
			err = errors.Wrap(err, "Failed to re-fetch DynamoComponent")
			return
		}
		for _, condition := range conditions {
			meta.SetStatusCondition(&DynamoComponent.Status.Conditions, condition)
		}
		if err = r.Status().Update(ctx, DynamoComponent); err != nil {
			time.Sleep(100 * time.Millisecond)
		} else {
			break
		}
	}
	if err != nil {
		err = errors.Wrap(err, "Failed to update DynamoComponent status")
		return
	}
	if err = r.Get(ctx, req.NamespacedName, DynamoComponent); err != nil {
		err = errors.Wrap(err, "Failed to re-fetch DynamoComponent")
		return
	}
	return
}

type DynamoComponentImageBuildEngine string

const (
	DynamoComponentImageBuildEngineKaniko           DynamoComponentImageBuildEngine = "kaniko"
	DynamoComponentImageBuildEngineBuildkit         DynamoComponentImageBuildEngine = "buildkit"
	DynamoComponentImageBuildEngineBuildkitRootless DynamoComponentImageBuildEngine = "buildkit-rootless"
)

const (
	EnvDynamoImageBuildEngine = "DYNAMO_IMAGE_BUILD_ENGINE"
)

func getDynamoComponentImageBuildEngine() DynamoComponentImageBuildEngine {
	engine := os.Getenv(EnvDynamoImageBuildEngine)
	if engine == "" {
		return DynamoComponentImageBuildEngineKaniko
	}
	return DynamoComponentImageBuildEngine(engine)
}

//nolint:nakedret
func (r *DynamoComponentReconciler) getApiStoreClient(ctx context.Context) (*apiStoreClient.ApiStoreClient, *commonconfig.ApiStoreConfig, error) {
	apiStoreConf, err := commonconfig.GetApiStoreConfig(ctx)
	isNotFound := k8serrors.IsNotFound(err)
	if err != nil && !isNotFound {
		err = errors.Wrap(err, "get api store config")
		return nil, nil, err
	}

	if isNotFound {
		return nil, nil, err
	}

	if apiStoreConf.Endpoint == "" {
		return nil, nil, err
	}

	if apiStoreConf.ClusterName == "" {
		apiStoreConf.ClusterName = "default"
	}

	apiStoreClient := apiStoreClient.NewApiStoreClient(apiStoreConf.Endpoint)

	return apiStoreClient, apiStoreConf, nil
}

func (r *DynamoComponentReconciler) RetrieveDockerRegistrySecret(ctx context.Context, secretName string, namespace string, dockerRegistry *schemas.DockerRegistrySchema) error {
	secret := &corev1.Secret{}
	err := r.Get(ctx, types.NamespacedName{
		Namespace: namespace,
		Name:      secretName,
	}, secret)
	if err != nil {
		err = errors.Wrapf(err, "get docker config json secret %s", secretName)
		return err
	}
	configJSON, ok := secret.Data[dockerConfigSecretKey]
	if !ok {
		err = errors.Errorf("docker config json secret %s does not have %s key", secretName, dockerConfigSecretKey)
		return err
	}
	var configObj struct {
		Auths map[string]struct {
			Auth string `json:"auth"`
		} `json:"auths"`
	}
	err = json.Unmarshal(configJSON, &configObj)
	if err != nil {
		err = errors.Wrapf(err, "unmarshal docker config json secret %s", secretName)
		return err
	}
	var server string
	var auth string
	if dockerRegistry.Server != "" {
		for k, v := range configObj.Auths {
			if k == dockerRegistry.Server {
				server = k
				auth = v.Auth
				break
			}
		}
		if server == "" {
			for k, v := range configObj.Auths {
				if strings.Contains(k, dockerRegistry.Server) {
					server = k
					auth = v.Auth
					break
				}
			}
		}
	}
	if server == "" {
		err = errors.Errorf("no auth in docker config json secret %s for server %s", secretName, dockerRegistry.Server)
		return err
	}
	var credentials []byte
	credentials, err = base64.StdEncoding.DecodeString(auth)
	if err != nil {
		err = errors.Wrapf(err, "cannot base64 decode auth in docker config json secret %s", secretName)
		return err
	}
	dockerRegistry.Username, _, dockerRegistry.Password = xstrings.Partition(string(credentials), ":")
	return nil
}

//nolint:nakedret
func (r *DynamoComponentReconciler) getDockerRegistry(ctx context.Context, DynamoComponent *nvidiacomv1alpha1.DynamoComponent) (dockerRegistry *schemas.DockerRegistrySchema, err error) {

	dockerRegistryConfig := commonconfig.GetDockerRegistryConfig()

	dynamoRepositoryName := "dynamo-components"
	if dockerRegistryConfig.DynamoComponentsRepositoryName != "" {
		dynamoRepositoryName = dockerRegistryConfig.DynamoComponentsRepositoryName
	}
	dynamoRepositoryURI := fmt.Sprintf("%s/%s", strings.TrimRight(dockerRegistryConfig.Server, "/"), dynamoRepositoryName)
	if strings.Contains(dockerRegistryConfig.Server, "docker.io") {
		dynamoRepositoryURI = fmt.Sprintf("docker.io/%s", dynamoRepositoryName)
	}

	if DynamoComponent != nil && DynamoComponent.Spec.DockerConfigJSONSecretName != "" {
		dockerRegistryConfig.SecretName = DynamoComponent.Spec.DockerConfigJSONSecretName
	}

	dockerRegistry = &schemas.DockerRegistrySchema{
		Server:              dockerRegistryConfig.Server,
		Secure:              dockerRegistryConfig.Secure,
		DynamoRepositoryURI: dynamoRepositoryURI,
		SecretName:          dockerRegistryConfig.SecretName,
	}

	err = r.RetrieveDockerRegistrySecret(ctx, dockerRegistryConfig.SecretName, DynamoComponent.Namespace, dockerRegistry)
	if err != nil {
		err = errors.Wrap(err, "retrieve docker registry secret")
		return
	}

	return
}

func isAddNamespacePrefix() bool {
	return os.Getenv("ADD_NAMESPACE_PREFIX_TO_IMAGE_NAME") == trueStr
}

func getDynamoComponentImagePrefix(DynamoComponent *nvidiacomv1alpha1.DynamoComponent) string {
	if DynamoComponent == nil {
		return ""
	}
	prefix, exist := DynamoComponent.Annotations[consts.KubeAnnotationDynamoComponentStorageNS]
	if exist && prefix != "" {
		return fmt.Sprintf("%s.", prefix)
	}
	if isAddNamespacePrefix() {
		return fmt.Sprintf("%s.", DynamoComponent.Namespace)
	}
	return ""
}

func getDynamoComponentImageName(DynamoComponent *nvidiacomv1alpha1.DynamoComponent, dockerRegistry schemas.DockerRegistrySchema, dynamoComponentRepositoryName, dynamoComponentVersion string) string {
	if DynamoComponent != nil && DynamoComponent.Spec.Image != "" {
		return DynamoComponent.Spec.Image
	}
	var uri, tag string
	uri = dockerRegistry.DynamoRepositoryURI
	tail := fmt.Sprintf("%s.%s", dynamoComponentRepositoryName, dynamoComponentVersion)
	if isEstargzEnabled() {
		tail += ".esgz"
	}

	tag = fmt.Sprintf("dynamo.%s%s", getDynamoComponentImagePrefix(DynamoComponent), tail)

	if len(tag) > 128 {
		hashStr := hash(tail)
		tag = fmt.Sprintf("dynamo.%s%s", getDynamoComponentImagePrefix(DynamoComponent), hashStr)
		if len(tag) > 128 {
			tag = fmt.Sprintf("dynamo.%s", hash(fmt.Sprintf("%s%s", getDynamoComponentImagePrefix(DynamoComponent), tail)))[:128]
		}
	}
	return fmt.Sprintf("%s:%s", uri, tag)
}

func checkImageExists(DynamoComponent *nvidiacomv1alpha1.DynamoComponent, dockerRegistry schemas.DockerRegistrySchema, imageName string) (bool, error) {
	if DynamoComponent.Annotations["nvidia.com/force-build-image"] == commonconsts.KubeLabelValueTrue {
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
	DockerRegistryInsecure     bool
}

type GetImageInfoOption struct {
	DynamoComponent *nvidiacomv1alpha1.DynamoComponent
}

//nolint:nakedret
func (r *DynamoComponentReconciler) getImageInfo(ctx context.Context, opt GetImageInfoOption) (imageInfo ImageInfo, err error) {
	dynamoComponentRepositoryName, _, dynamoComponentVersion := xstrings.Partition(opt.DynamoComponent.Spec.DynamoComponent, ":")
	dockerRegistry, err := r.getDockerRegistry(ctx, opt.DynamoComponent)
	if err != nil {
		err = errors.Wrap(err, "get docker registry")
		return
	}
	imageInfo.DockerRegistry = *dockerRegistry
	imageInfo.ImageName = getDynamoComponentImageName(opt.DynamoComponent, *dockerRegistry, dynamoComponentRepositoryName, dynamoComponentVersion)

	imageInfo.DockerConfigJSONSecretName = dockerRegistry.SecretName

	imageInfo.DockerRegistryInsecure = opt.DynamoComponent.Annotations[commonconsts.KubeAnnotationDynamoDockerRegistryInsecure] == "true"
	return
}

func (r *DynamoComponentReconciler) getImageBuilderJobName() string {
	guid := xid.New()
	return fmt.Sprintf("dynamo-image-builder-%s", guid.String())
}

func (r *DynamoComponentReconciler) getImageBuilderJobLabels(DynamoComponent *nvidiacomv1alpha1.DynamoComponent) map[string]string {
	return map[string]string{
		commonconsts.KubeLabelDynamoComponent:      DynamoComponent.Name,
		commonconsts.KubeLabelIsDynamoImageBuilder: "true",
	}
}

func (r *DynamoComponentReconciler) getImageBuilderPodLabels(DynamoComponent *nvidiacomv1alpha1.DynamoComponent) map[string]string {
	return map[string]string{
		commonconsts.KubeLabelDynamoComponent:      DynamoComponent.Name,
		commonconsts.KubeLabelIsDynamoImageBuilder: "true",
	}
}

func hash(text string) string {
	// nolint: gosec
	hasher := md5.New()
	hasher.Write([]byte(text))
	return hex.EncodeToString(hasher.Sum(nil))
}

type GenerateImageBuilderJobOption struct {
	ImageInfo       ImageInfo
	DynamoComponent *nvidiacomv1alpha1.DynamoComponent
}

//nolint:nakedret
func (r *DynamoComponentReconciler) generateImageBuilderJob(ctx context.Context, opt GenerateImageBuilderJobOption) (job *batchv1.Job, err error) {
	// nolint: gosimple
	podTemplateSpec, err := r.generateImageBuilderPodTemplateSpec(ctx, GenerateImageBuilderPodTemplateSpecOption(opt))
	if err != nil {
		err = errors.Wrap(err, "generate image builder pod template spec")
		return
	}
	kubeAnnotations := make(map[string]string)
	hashStr, err := r.getHashStr(opt.DynamoComponent)
	if err != nil {
		err = errors.Wrap(err, "failed to get hash string")
		return
	}
	kubeAnnotations[consts.KubeAnnotationDynamoComponentHash] = hashStr
	job = &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        r.getImageBuilderJobName(),
			Namespace:   opt.DynamoComponent.Namespace,
			Labels:      r.getImageBuilderJobLabels(opt.DynamoComponent),
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
	err = ctrl.SetControllerReference(opt.DynamoComponent, job, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set controller reference for job %s", job.Name)
		return
	}
	return
}

func injectPodAffinity(podSpec *corev1.PodSpec, DynamoComponent *nvidiacomv1alpha1.DynamoComponent) {
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
					commonconsts.KubeLabelDynamoComponent: DynamoComponent.Name,
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
	ImageInfo       ImageInfo
	DynamoComponent *nvidiacomv1alpha1.DynamoComponent
}

//nolint:gocyclo,nakedret
func (r *DynamoComponentReconciler) generateImageBuilderPodTemplateSpec(ctx context.Context, opt GenerateImageBuilderPodTemplateSpecOption) (pod *corev1.PodTemplateSpec, err error) {
	dynamoComponentRepositoryName, _, dynamoComponentVersion := xstrings.Partition(opt.DynamoComponent.Spec.DynamoComponent, ":")
	kubeLabels := r.getImageBuilderPodLabels(opt.DynamoComponent)

	imageName := opt.ImageInfo.ImageName

	dockerConfigJSONSecretName := opt.ImageInfo.DockerConfigJSONSecretName

	dockerRegistryInsecure := opt.ImageInfo.DockerRegistryInsecure

	volumes := []corev1.Volume{
		{
			Name: "dynamo",
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
			Name:      "dynamo",
			MountPath: "/dynamo",
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

	var dynamoComponent *schemas.DynamoComponent
	dynamoComponentDownloadURL := opt.DynamoComponent.Spec.DownloadURL

	if dynamoComponentDownloadURL == "" {
		var apiStoreClient *apiStoreClient.ApiStoreClient
		var apiStoreConf *commonconfig.ApiStoreConfig

		apiStoreClient, apiStoreConf, err = r.getApiStoreClient(ctx)
		if err != nil {
			err = errors.Wrap(err, "get api store client")
			return
		}

		if apiStoreClient == nil || apiStoreConf == nil {
			err = errors.New("can't get api store client, please check api store configuration")
			return
		}

		r.Recorder.Eventf(opt.DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting dynamoComponent %s from api store service", opt.DynamoComponent.Spec.DynamoComponent)
		dynamoComponent, err = apiStoreClient.GetDynamoComponent(ctx, dynamoComponentRepositoryName, dynamoComponentVersion)
		if err != nil {
			err = errors.Wrap(err, "get dynamoComponent")
			return
		}
		r.Recorder.Eventf(opt.DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got dynamoComponent %s from api store service", opt.DynamoComponent.Spec.DynamoComponent)

		if dynamoComponent.TransmissionStrategy != nil && *dynamoComponent.TransmissionStrategy == schemas.TransmissionStrategyPresignedURL {
			var dynamoComponent_ *schemas.DynamoComponent
			r.Recorder.Eventf(opt.DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting presigned url for dynamoComponent %s from api store service", opt.DynamoComponent.Spec.DynamoComponent)
			dynamoComponent_, err = apiStoreClient.PresignDynamoComponentDownloadURL(ctx, dynamoComponentRepositoryName, dynamoComponentVersion)
			if err != nil {
				err = errors.Wrap(err, "presign dynamoComponent download url")
				return
			}
			r.Recorder.Eventf(opt.DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Got presigned url for dynamoComponent %s from api store service", opt.DynamoComponent.Spec.DynamoComponent)
			dynamoComponentDownloadURL = dynamoComponent_.PresignedDownloadUrl
		} else {
			dynamoComponentDownloadURL = fmt.Sprintf("%s/api/v1/dynamo_nims/%s/versions/%s/download", apiStoreConf.Endpoint, dynamoComponentRepositoryName, dynamoComponentVersion)
		}

	}
	internalImages := commonconfig.GetInternalImages()
	logrus.Infof("Image builder is using the images %v", *internalImages)

	buildEngine := getDynamoComponentImageBuildEngine()

	privileged := buildEngine != DynamoComponentImageBuildEngineBuildkitRootless

	dynamoComponentDownloadCommandTemplate, err := template.New("downloadCommand").Parse(`
set -e

mkdir -p /workspace/buildcontext
url="{{.DynamoComponentDownloadURL}}"
echo "Downloading dynamoComponent {{.DynamoComponentRepositoryName}}:{{.DynamoComponentVersion}} to /tmp/downloaded.tar..."
if [[ ${url} == s3://* ]]; then
	echo "Downloading from s3..."
	aws s3 cp ${url} /tmp/downloaded.tar
elif [[ ${url} == gs://* ]]; then
	echo "Downloading from GCS..."
	gsutil cp ${url} /tmp/downloaded.tar
else
	curl --fail -L ${url} --output /tmp/downloaded.tar --progress-bar
fi
cd /workspace/buildcontext
echo "Extracting dynamoComponent tar file..."
tar -xvf /tmp/downloaded.tar
echo "Removing dynamoComponent tar file..."
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

	var dynamoComponentDownloadCommandBuffer bytes.Buffer

	err = dynamoComponentDownloadCommandTemplate.Execute(&dynamoComponentDownloadCommandBuffer, map[string]interface{}{
		"DynamoComponentDownloadURL":    dynamoComponentDownloadURL,
		"DynamoComponentRepositoryName": dynamoComponentRepositoryName,
		"DynamoComponentVersion":        dynamoComponentVersion,
		"Privileged":                    privileged,
	})
	if err != nil {
		err = errors.Wrap(err, "failed to execute download command template")
		return
	}

	dynamoComponentDownloadCommand := dynamoComponentDownloadCommandBuffer.String()

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

	downloaderContainerEnvFrom := opt.DynamoComponent.Spec.DownloaderContainerEnvFrom

	initContainers := []corev1.Container{
		{
			Name:  "dynamocomponent-downloader",
			Image: internalImages.DynamoComponentsDownloader,
			Command: []string{
				"sh",
				"-c",
				dynamoComponentDownloadCommand,
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

	var globalExtraPodMetadata *dynamoCommon.ExtraPodMetadata
	var globalExtraPodSpec *dynamoCommon.ExtraPodSpec
	var globalExtraContainerEnv []corev1.EnvVar
	var globalDefaultImageBuilderContainerResources *corev1.ResourceRequirements
	var buildArgs []string
	var builderArgs []string

	configNamespace, err := commonconfig.GetDynamoImageBuilderNamespace(ctx)
	if err != nil {
		err = errors.Wrap(err, "failed to get dynamo image builder namespace")
		return
	}

	configCmName := "dynamo-image-builder-config"
	r.Recorder.Eventf(opt.DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Getting configmap %s from namespace %s", configCmName, configNamespace)
	configCm := &corev1.ConfigMap{}
	err = r.Get(ctx, types.NamespacedName{Name: configCmName, Namespace: configNamespace}, configCm)
	configCmIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !configCmIsNotFound {
		err = errors.Wrap(err, "failed to get configmap")
		return
	}
	err = nil // nolint: ineffassign

	if !configCmIsNotFound {
		r.Recorder.Eventf(opt.DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Configmap %s is got from namespace %s", configCmName, configNamespace)

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
		r.Recorder.Eventf(opt.DynamoComponent, corev1.EventTypeNormal, "GenerateImageBuilderPod", "Configmap %s is not found in namespace %s", configCmName, configNamespace)
	}

	if buildArgs == nil {
		buildArgs = make([]string, 0)
	}

	if opt.DynamoComponent.Spec.BuildArgs != nil {
		buildArgs = append(buildArgs, opt.DynamoComponent.Spec.BuildArgs...)
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
		kanikoCacheRepo = opt.ImageInfo.DockerRegistry.DynamoRepositoryURI
	}

	kubeAnnotations := make(map[string]string)
	kubeAnnotations[consts.KubeAnnotationDynamoComponentImageBuiderHash] = opt.DynamoComponent.Annotations[consts.KubeAnnotationDynamoComponentImageBuiderHash]

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
		fmt.Sprintf("--destination=%s", imageName),
	}

	kanikoSnapshotMode := os.Getenv("KANIKO_SNAPSHOT_MODE")
	if kanikoSnapshotMode != "" {
		args = append(args, fmt.Sprintf("--snapshot-mode=%s", kanikoSnapshotMode))
	}

	var builderImage string
	switch buildEngine {
	case DynamoComponentImageBuildEngineKaniko:
		builderImage = internalImages.Kaniko
		if isEstargzEnabled() {
			builderContainerEnvs = append(builderContainerEnvs, corev1.EnvVar{
				Name:  "GGCR_EXPERIMENT_ESTARGZ",
				Value: "1",
			})
		}
	case DynamoComponentImageBuildEngineBuildkit:
		builderImage = internalImages.Buildkit
	case DynamoComponentImageBuildEngineBuildkitRootless:
		builderImage = internalImages.BuildkitRootless
	default:
		err = errors.Errorf("unknown dynamoComponent image build engine %s", buildEngine)
		return
	}

	isBuildkit := buildEngine == DynamoComponentImageBuildEngineBuildkit || buildEngine == DynamoComponentImageBuildEngineBuildkitRootless

	if isBuildkit {
		output := fmt.Sprintf("type=image,name=%s,push=true,registry.insecure=%v", imageName, dockerRegistryInsecure)
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

	if buildEngine == DynamoComponentImageBuildEngineBuildkit {
		builderContainerSecurityContext = &corev1.SecurityContext{
			Privileged: ptr.To(true),
		}
	} else if buildEngine == DynamoComponentImageBuildEngineBuildkitRootless {
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
	logrus.Info("dynamo-image-builder args: ", args)

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

	if opt.DynamoComponent.Spec.ImageBuilderContainerResources != nil {
		container.Resources = *opt.DynamoComponent.Spec.ImageBuilderContainerResources
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

	if opt.DynamoComponent.Spec.ImageBuilderExtraPodMetadata != nil {
		for k, v := range opt.DynamoComponent.Spec.ImageBuilderExtraPodMetadata.Annotations {
			pod.Annotations[k] = v
		}

		for k, v := range opt.DynamoComponent.Spec.ImageBuilderExtraPodMetadata.Labels {
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

	if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec != nil {
		if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.PriorityClassName != "" {
			pod.Spec.PriorityClassName = opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.PriorityClassName
		}

		if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.SchedulerName != "" {
			pod.Spec.SchedulerName = opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.SchedulerName
		}

		if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.NodeSelector != nil {
			pod.Spec.NodeSelector = opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.NodeSelector
		}

		if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.Affinity != nil {
			pod.Spec.Affinity = opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.Affinity
		}

		if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.Tolerations != nil {
			pod.Spec.Tolerations = opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.Tolerations
		}

		if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.TopologySpreadConstraints != nil {
			pod.Spec.TopologySpreadConstraints = opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.TopologySpreadConstraints
		}

		if opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.ServiceAccountName != "" {
			pod.Spec.ServiceAccountName = opt.DynamoComponent.Spec.ImageBuilderExtraPodSpec.ServiceAccountName
		}
	}

	injectPodAffinity(&pod.Spec, opt.DynamoComponent)

	if pod.Spec.ServiceAccountName == "" {
		serviceAccounts := &corev1.ServiceAccountList{}
		err = r.List(ctx, serviceAccounts, client.InNamespace(opt.DynamoComponent.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelDynamoImageBuilderPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			err = errors.Wrapf(err, "failed to list service accounts in namespace %s", opt.DynamoComponent.Namespace)
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
		env = append(env, opt.DynamoComponent.Spec.ImageBuilderExtraContainerEnv...)
		pod.Spec.InitContainers[i].Env = env
	}
	for i, c := range pod.Spec.Containers {
		env := c.Env
		if globalExtraContainerEnv != nil {
			env = append(env, globalExtraContainerEnv...)
		}
		env = append(env, opt.DynamoComponent.Spec.ImageBuilderExtraContainerEnv...)
		pod.Spec.Containers[i].Env = env
	}

	return
}

func (r *DynamoComponentReconciler) getHashStr(DynamoComponent *nvidiacomv1alpha1.DynamoComponent) (string, error) {
	var hash uint64
	hash, err := hashstructure.Hash(struct {
		Spec        nvidiacomv1alpha1.DynamoComponentSpec
		Labels      map[string]string
		Annotations map[string]string
	}{
		Spec:        DynamoComponent.Spec,
		Labels:      DynamoComponent.Labels,
		Annotations: DynamoComponent.Annotations,
	}, hashstructure.FormatV2, nil)
	if err != nil {
		err = errors.Wrap(err, "get DynamoComponent CR spec hash")
		return "", err
	}
	hashStr := strconv.FormatUint(hash, 10)
	return hashStr, nil
}

const (
	trueStr = "true"
)

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoComponentReconciler) SetupWithManager(mgr ctrl.Manager) error {

	err := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoComponent{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&nvidiacomv1alpha1.DynamoComponent{}).
		Owns(&batchv1.Job{}).
		WithEventFilter(controller_common.EphemeralDeploymentEventFilter(r.Config)).
		Complete(r)
	return errors.Wrap(err, "failed to setup DynamoComponent controller")
}
