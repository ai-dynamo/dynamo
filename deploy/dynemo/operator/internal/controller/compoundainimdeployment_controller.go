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
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"emperror.dev/errors"
	"github.com/cisco-open/k8s-objectmatcher/patch"
	compoundaiCommon "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/common"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/schemasv1"
	yataiclient "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/yatai-client"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/v1alpha1"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/internal/controller_common"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/internal/envoy"
	commonconfig "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/pkg/compoundai/config"
	commonconsts "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/pkg/compoundai/consts"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/pkg/compoundai/system"
	"github.com/huandu/xstrings"
	"github.com/jinzhu/copier"
	"github.com/prometheus/common/version"
	istioNetworking "istio.io/api/networking/v1beta1"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	compounadaiConversion "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/conversion"
)

const (
	DefaultClusterName                                        = "default"
	DefaultServiceAccountName                                 = "default"
	KubeValueNameSharedMemory                                 = "shared-memory"
	KubeAnnotationDeploymentStrategy                          = "yatai.ai/deployment-strategy"
	KubeAnnotationYataiEnableStealingTrafficDebugMode         = "yatai.ai/enable-stealing-traffic-debug-mode"
	KubeAnnotationYataiEnableDebugMode                        = "yatai.ai/enable-debug-mode"
	KubeAnnotationYataiEnableDebugPodReceiveProductionTraffic = "yatai.ai/enable-debug-pod-receive-production-traffic"
	KubeAnnotationYataiProxySidecarResourcesLimitsCPU         = "yatai.ai/proxy-sidecar-resources-limits-cpu"
	KubeAnnotationYataiProxySidecarResourcesLimitsMemory      = "yatai.ai/proxy-sidecar-resources-limits-memory"
	KubeAnnotationYataiProxySidecarResourcesRequestsCPU       = "yatai.ai/proxy-sidecar-resources-requests-cpu"
	KubeAnnotationYataiProxySidecarResourcesRequestsMemory    = "yatai.ai/proxy-sidecar-resources-requests-memory"
	DeploymentTargetTypeProduction                            = "production"
	DeploymentTargetTypeDebug                                 = "debug"
	ContainerPortNameHTTPProxy                                = "http-proxy"
	ServicePortNameHTTPNonProxy                               = "http-non-proxy"
	HeaderNameDebug                                           = "X-Yatai-Debug"
)

var ServicePortHTTPNonProxy = commonconsts.BentoServicePort + 1

// CompoundAINimDeploymentReconciler reconciles a CompoundAINimDeployment object
type CompoundAINimDeploymentReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Config   controller_common.Config
	NatsAddr string
	EtcdAddr string
}

// +kubebuilder:rbac:groups=nvidia.com,resources=compoundainimdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=compoundainimdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=compoundainimdeployments/finalizers,verbs=update

//+kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
//+kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=events,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=networking.k8s.io,resources=ingressclasses,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=networking.k8s.io,resources=ingresses,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=networking.istio.io,resources=virtualservices,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;create;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the CompoundAINimDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.18.2/pkg/reconcile
//
//nolint:gocyclo,nakedret
func (r *CompoundAINimDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logs := log.FromContext(ctx)

	compoundAINimDeployment := &v1alpha1.CompoundAINimDeployment{}
	err = r.Get(ctx, req.NamespacedName, compoundAINimDeployment)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			logs.Info("CompoundAINimDeployment resource not found. Ignoring since object must be deleted.")
			err = nil
			return
		}
		// Error reading the object - requeue the request.
		logs.Error(err, "Failed to get CompoundAINimDeployment.")
		return
	}

	logs = logs.WithValues("compoundAINimDeployment", compoundAINimDeployment.Name, "namespace", compoundAINimDeployment.Namespace)

	if len(compoundAINimDeployment.Status.Conditions) == 0 {
		logs.Info("Starting to reconcile CompoundAINimDeployment")
		logs.Info("Initializing CompoundAINimDeployment status")
		r.Recorder.Event(compoundAINimDeployment, corev1.EventTypeNormal, "Reconciling", "Starting to reconcile CompoundAINimDeployment")
		compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.CompoundAIDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile CompoundAINimDeployment",
			},
			metav1.Condition{
				Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimFound,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile CompoundAINimDeployment",
			},
		)
		if err != nil {
			return
		}
	}

	defer func() {
		if err == nil {
			return
		}
		logs.Error(err, "Failed to reconcile CompoundAINimDeployment.")
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "ReconcileError", "Failed to reconcile CompoundAINimDeployment: %v", err)
		_, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.CompoundAIDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Failed to reconcile CompoundAINimDeployment: %v", err),
			},
		)
		if err != nil {
			return
		}
	}()

	yataiClient, clusterName, err := r.getYataiClientWithAuth(ctx, compoundAINimDeployment)
	if err != nil {
		err = errors.Wrap(err, "get yatai client with auth")
		return
	}

	compoundAINimFoundCondition := meta.FindStatusCondition(compoundAINimDeployment.Status.Conditions, v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimFound)
	if compoundAINimFoundCondition != nil && compoundAINimFoundCondition.Status == metav1.ConditionUnknown {
		logs.Info(fmt.Sprintf("Getting Compound AI NIM %s", compoundAINimDeployment.Spec.CompoundAINim))
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetCompoundAINim", "Getting Compound AI NIM %s", compoundAINimDeployment.Spec.CompoundAINim)
	}
	compoundAINimRequest := &v1alpha1.CompoundAINimRequest{}
	compoundAINimCR := &v1alpha1.CompoundAINim{}
	err = r.Get(ctx, types.NamespacedName{
		Namespace: compoundAINimDeployment.Namespace,
		Name:      compoundAINimDeployment.Spec.CompoundAINim,
	}, compoundAINimCR)
	compoundAINimIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !compoundAINimIsNotFound {
		err = errors.Wrapf(err, "get CompoundAINim %s/%s", compoundAINimDeployment.Namespace, compoundAINimDeployment.Spec.CompoundAINim)
		return
	}
	if compoundAINimIsNotFound {
		if compoundAINimFoundCondition != nil && compoundAINimFoundCondition.Status == metav1.ConditionUnknown {
			logs.Info(fmt.Sprintf("CompoundAINim %s not found", compoundAINimDeployment.Spec.CompoundAINim))
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetCompoundAINim", "CompoundAINim %s not found", compoundAINimDeployment.Spec.CompoundAINim)
		}
		compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimFound,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: "CompoundAINim not found",
			},
		)
		if err != nil {
			return
		}
		compoundAINimRequestFoundCondition := meta.FindStatusCondition(compoundAINimDeployment.Status.Conditions, v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimRequestFound)
		if compoundAINimRequestFoundCondition == nil || compoundAINimRequestFoundCondition.Status != metav1.ConditionUnknown {
			compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
				metav1.Condition{
					Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimRequestFound,
					Status:  metav1.ConditionUnknown,
					Reason:  "Reconciling",
					Message: "CompoundAINim not found",
				},
			)
			if err != nil {
				return
			}
		}
		if compoundAINimRequestFoundCondition != nil && compoundAINimRequestFoundCondition.Status == metav1.ConditionUnknown {
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetCompoundAINimRequest", "Getting CompoundAINimRequest %s", compoundAINimDeployment.Spec.CompoundAINim)
		}
		err = r.Get(ctx, types.NamespacedName{
			Namespace: compoundAINimDeployment.Namespace,
			Name:      compoundAINimDeployment.Spec.CompoundAINim,
		}, compoundAINimRequest)
		if err != nil {
			err = errors.Wrapf(err, "get CompoundAINimRequest %s/%s", compoundAINimDeployment.Namespace, compoundAINimDeployment.Spec.CompoundAINim)
			compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
				metav1.Condition{
					Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimFound,
					Status:  metav1.ConditionFalse,
					Reason:  "Reconciling",
					Message: err.Error(),
				},
				metav1.Condition{
					Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimRequestFound,
					Status:  metav1.ConditionFalse,
					Reason:  "Reconciling",
					Message: err.Error(),
				},
			)
			if err != nil {
				return
			}
		}
		if compoundAINimRequestFoundCondition != nil && compoundAINimRequestFoundCondition.Status == metav1.ConditionUnknown {
			logs.Info(fmt.Sprintf("CompoundAINimRequest %s found", compoundAINimDeployment.Spec.CompoundAINim))
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetCompoundAINimRequest", "CompoundAINimRequest %s is found and waiting for its compoundAINim to be provided", compoundAINimDeployment.Spec.CompoundAINim)
		}
		compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimRequestFound,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: "CompoundAINim not found",
			},
		)
		if err != nil {
			return
		}
		compoundAINimRequestAvailableCondition := meta.FindStatusCondition(compoundAINimRequest.Status.Conditions, v1alpha1.CompoundAIDeploymentConditionTypeAvailable)
		if compoundAINimRequestAvailableCondition != nil && compoundAINimRequestAvailableCondition.Status == metav1.ConditionFalse {
			err = errors.Errorf("CompoundAINimRequest %s/%s is not available: %s", compoundAINimRequest.Namespace, compoundAINimRequest.Name, compoundAINimRequestAvailableCondition.Message)
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "GetCompoundAINimRequest", err.Error())
			_, err_ := r.setStatusConditions(ctx, req,
				metav1.Condition{
					Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimFound,
					Status:  metav1.ConditionFalse,
					Reason:  "Reconciling",
					Message: err.Error(),
				},
				metav1.Condition{
					Type:    v1alpha1.CompoundAIDeploymentConditionTypeAvailable,
					Status:  metav1.ConditionFalse,
					Reason:  "Reconciling",
					Message: err.Error(),
				},
			)
			if err_ != nil {
				err = err_
				return
			}
			return
		}
		return
	} else {
		if compoundAINimFoundCondition != nil && compoundAINimFoundCondition.Status != metav1.ConditionTrue {
			logs.Info(fmt.Sprintf("CompoundAINim %s found", compoundAINimDeployment.Spec.CompoundAINim))
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetCompoundAINim", "CompoundAINim %s is found", compoundAINimDeployment.Spec.CompoundAINim)
		}
		compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.CompoundAIDeploymentConditionTypeCompoundAINimFound,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: "CompoundAINim found",
			},
		)
		if err != nil {
			return
		}
	}

	modified := false

	// Reconcile PVC
	_, err = r.reconcilePVC(ctx, compoundAINimDeployment)
	if err != nil {
		logs.Error(err, "Unable to create PVC", "crd", req.NamespacedName)
		return ctrl.Result{}, err
	}

	// create or update api-server deployment
	modified_, err := r.createOrUpdateOrDeleteDeployments(ctx, createOrUpdateOrDeleteDeploymentsOption{
		yataiClient:             yataiClient,
		compoundAINimDeployment: compoundAINimDeployment,
		compoundAINim:           compoundAINimCR,
		clusterName:             clusterName,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server hpa
	modified_, err = r.createOrUpdateHPA(ctx, compoundAINimDeployment, compoundAINimCR)
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server service
	modified_, err = r.createOrUpdateOrDeleteServices(ctx, createOrUpdateOrDeleteServicesOption{
		compoundAINimDeployment: compoundAINimDeployment,
		compoundAINim:           compoundAINimCR,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server ingresses
	modified_, err = r.createOrUpdateIngresses(ctx, createOrUpdateIngressOption{
		yataiClient:             yataiClient,
		compoundAINimDeployment: compoundAINimDeployment,
		compoundAINim:           compoundAINimCR,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	if yataiClient != nil && clusterName != nil {
		yataiClient_ := *yataiClient
		clusterName_ := *clusterName
		compoundAINimRepositoryName, compoundAINimVersion := getCompoundAINimRepositoryNameAndCompoundAINimVersion(compoundAINimCR)
		_, err = yataiClient_.GetBento(ctx, compoundAINimRepositoryName, compoundAINimVersion)

		compoundAINimIsNotFound := isNotFoundError(err)
		if err != nil && !compoundAINimIsNotFound {
			return
		}
		if compoundAINimIsNotFound {
			compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
				metav1.Condition{
					Type:    v1alpha1.CompoundAIDeploymentConditionTypeAvailable,
					Status:  metav1.ConditionTrue,
					Reason:  "Reconciling",
					Message: "Remote compoundAINim from Yatai is not found",
				},
			)
			return
		}
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetYataiDeployment", "Fetching yatai deployment %s", compoundAINimDeployment.Name)
		var oldYataiDeployment *schemasv1.DeploymentSchema
		oldYataiDeployment, err = yataiClient_.GetDeployment(ctx, clusterName_, compoundAINimDeployment.Namespace, compoundAINimDeployment.Name)
		isNotFound := isNotFoundError(err)
		if err != nil && !isNotFound {
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "GetYataiDeployment", "Failed to fetch yatai deployment %s: %s", compoundAINimDeployment.Name, err)
			return
		}
		err = nil

		envs := make([]*modelschemas.LabelItemSchema, 0)

		specEnvs := compoundAINimDeployment.Spec.Envs

		for _, env := range specEnvs {
			envs = append(envs, &modelschemas.LabelItemSchema{
				Key:   env.Name,
				Value: env.Value,
			})
		}

		var hpaConf *modelschemas.DeploymentTargetHPAConf
		hpaConf, err = TransformToOldHPA(compoundAINimDeployment.Spec.Autoscaling)
		if err != nil {
			return
		}
		deploymentTargets := make([]*schemasv1.CreateDeploymentTargetSchema, 0, 1)
		deploymentTarget := &schemasv1.CreateDeploymentTargetSchema{
			DeploymentTargetTypeSchema: schemasv1.DeploymentTargetTypeSchema{
				Type: modelschemas.DeploymentTargetTypeStable,
			},
			BentoRepository: compoundAINimRepositoryName,
			Bento:           compoundAINimVersion,
			Config: &modelschemas.DeploymentTargetConfig{
				KubeResourceUid:                        string(compoundAINimDeployment.UID),
				KubeResourceVersion:                    compoundAINimDeployment.ResourceVersion,
				Resources:                              compounadaiConversion.ConvertToDeploymentTargetResources(compoundAINimDeployment.Spec.Resources),
				HPAConf:                                hpaConf,
				Envs:                                   &envs,
				EnableIngress:                          &compoundAINimDeployment.Spec.Ingress.Enabled,
				EnableStealingTrafficDebugMode:         &[]bool{checkIfIsStealingTrafficDebugModeEnabled(compoundAINimDeployment.Spec.Annotations)}[0],
				EnableDebugMode:                        &[]bool{checkIfIsDebugModeEnabled(compoundAINimDeployment.Spec.Annotations)}[0],
				EnableDebugPodReceiveProductionTraffic: &[]bool{checkIfIsDebugPodReceiveProductionTrafficEnabled(compoundAINimDeployment.Spec.Annotations)}[0],
				BentoDeploymentOverrides: &modelschemas.ApiServerBentoDeploymentOverrides{
					MonitorExporter:  compoundAINimDeployment.Spec.MonitorExporter,
					ExtraPodMetadata: compoundAINimDeployment.Spec.ExtraPodMetadata,
					ExtraPodSpec:     compoundAINimDeployment.Spec.ExtraPodSpec,
				},
				BentoRequestOverrides: &modelschemas.BentoRequestOverrides{
					ImageBuildTimeout:              compoundAINimRequest.Spec.ImageBuildTimeout,
					ImageBuilderExtraPodSpec:       compoundAINimRequest.Spec.ImageBuilderExtraPodSpec,
					ImageBuilderExtraPodMetadata:   compoundAINimRequest.Spec.ImageBuilderExtraPodMetadata,
					ImageBuilderExtraContainerEnv:  compoundAINimRequest.Spec.ImageBuilderExtraContainerEnv,
					ImageBuilderContainerResources: compoundAINimRequest.Spec.ImageBuilderContainerResources,
					DockerConfigJSONSecretName:     compoundAINimRequest.Spec.DockerConfigJSONSecretName,
					DownloaderContainerEnvFrom:     compoundAINimRequest.Spec.DownloaderContainerEnvFrom,
				},
			},
		}
		deploymentTargets = append(deploymentTargets, deploymentTarget)
		updateSchema := &schemasv1.UpdateDeploymentSchema{
			Targets:     deploymentTargets,
			DoNotDeploy: true,
		}
		if isNotFound {
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "CreateYataiDeployment", "Creating yatai deployment %s", compoundAINimDeployment.Name)
			_, err = yataiClient_.CreateDeployment(ctx, clusterName_, &schemasv1.CreateDeploymentSchema{
				Name:                   compoundAINimDeployment.Name,
				KubeNamespace:          compoundAINimDeployment.Namespace,
				UpdateDeploymentSchema: *updateSchema,
			})
			if err != nil {
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "CreateYataiDeployment", "Failed to create yatai deployment %s: %s", compoundAINimDeployment.Name, err)
				return
			}
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "CreateYataiDeployment", "Created yatai deployment %s", compoundAINimDeployment.Name)
		} else {
			noChange := false
			if oldYataiDeployment != nil && oldYataiDeployment.LatestRevision != nil && len(oldYataiDeployment.LatestRevision.Targets) > 0 {
				oldYataiDeployment.LatestRevision.Targets[0].Config.KubeResourceUid = updateSchema.Targets[0].Config.KubeResourceUid
				oldYataiDeployment.LatestRevision.Targets[0].Config.KubeResourceVersion = updateSchema.Targets[0].Config.KubeResourceVersion
				noChange = reflect.DeepEqual(oldYataiDeployment.LatestRevision.Targets[0].Config, updateSchema.Targets[0].Config)
			}
			if noChange {
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateYataiDeployment", "No change in yatai deployment %s, skipping", compoundAINimDeployment.Name)
			} else {
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateYataiDeployment", "Updating yatai deployment %s", compoundAINimDeployment.Name)
				_, err = yataiClient_.UpdateDeployment(ctx, clusterName_, compoundAINimDeployment.Namespace, compoundAINimDeployment.Name, updateSchema)
				if err != nil {
					r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "UpdateYataiDeployment", "Failed to update yatai deployment %s: %s", compoundAINimDeployment.Name, err)
					return
				}
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateYataiDeployment", "Updated yatai deployment %s", compoundAINimDeployment.Name)
			}
		}
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "SyncYataiDeploymentStatus", "Syncing yatai deployment %s status", compoundAINimDeployment.Name)
		_, err = yataiClient_.SyncDeploymentStatus(ctx, clusterName_, compoundAINimDeployment.Namespace, compoundAINimDeployment.Name)
		if err != nil {
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "SyncYataiDeploymentStatus", "Failed to sync yatai deployment %s status: %s", compoundAINimDeployment.Name, err)
			return
		}
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "SyncYataiDeploymentStatus", "Synced yatai deployment %s status", compoundAINimDeployment.Name)
	}

	if !modified {
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateYataiDeployment", "No changes to yatai deployment %s", compoundAINimDeployment.Name)
	}

	logs.Info("Finished reconciling.")
	r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "Update", "All resources updated!")
	compoundAINimDeployment, err = r.setStatusConditions(ctx, req,
		metav1.Condition{
			Type:    v1alpha1.CompoundAIDeploymentConditionTypeAvailable,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: "Reconciling",
		},
	)
	return
}

func isNotFoundError(err error) bool {
	if err == nil {
		return false
	}
	errMsg := strings.ToLower(err.Error())
	return strings.Contains(errMsg, "not found") || strings.Contains(errMsg, "could not find") || strings.Contains(errMsg, "404")
}

func (r *CompoundAINimDeploymentReconciler) reconcilePVC(ctx context.Context, crd *v1alpha1.CompoundAINimDeployment) (*corev1.PersistentVolumeClaim, error) {
	logger := log.FromContext(ctx)
	if crd.Spec.PVC == nil {
		return nil, nil
	}
	pvcConfig := *crd.Spec.PVC
	pvc := &corev1.PersistentVolumeClaim{}
	pvcName := types.NamespacedName{Name: getPvcName(crd, pvcConfig.Name), Namespace: crd.GetNamespace()}
	err := r.Get(ctx, pvcName, pvc)
	if err != nil && client.IgnoreNotFound(err) != nil {
		logger.Error(err, "Unable to retrieve PVC", "crd", crd.GetName())
		return nil, err
	}

	// If PVC does not exist, create a new one
	if err != nil {
		if pvcConfig.Create == nil || !*pvcConfig.Create {
			logger.Error(err, "Unknown PVC", "pvc", pvc.Name)
			return nil, err
		}
		pvc = constructPVC(crd, pvcConfig)
		if err := controllerutil.SetControllerReference(crd, pvc, r.Scheme); err != nil {
			logger.Error(err, "Failed to set controller reference", "pvc", pvc.Name)
			return nil, err
		}
		err = r.Create(ctx, pvc)
		if err != nil {
			logger.Error(err, "Failed to create pvc", "pvc", pvc.Name)
			return nil, err
		}
		logger.Info("PVC created", "pvc", pvcName)
	}
	return pvc, nil
}

func (r *CompoundAINimDeploymentReconciler) setStatusConditions(ctx context.Context, req ctrl.Request, conditions ...metav1.Condition) (compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, err error) {
	compoundAINimDeployment = &v1alpha1.CompoundAINimDeployment{}
	for i := 0; i < 3; i++ {
		if err = r.Get(ctx, req.NamespacedName, compoundAINimDeployment); err != nil {
			err = errors.Wrap(err, "Failed to re-fetch CompoundAINimDeployment")
			return
		}
		for _, condition := range conditions {
			meta.SetStatusCondition(&compoundAINimDeployment.Status.Conditions, condition)
		}
		if err = r.Status().Update(ctx, compoundAINimDeployment); err != nil {
			time.Sleep(100 * time.Millisecond)
		} else {
			break
		}
	}
	if err != nil {
		err = errors.Wrap(err, "Failed to update CompoundAINimDeployment status")
		return
	}
	if err = r.Get(ctx, req.NamespacedName, compoundAINimDeployment); err != nil {
		err = errors.Wrap(err, "Failed to re-fetch CompoundAINimDeployment")
		return
	}
	return
}

var cachedYataiConf *commonconfig.YataiConfig

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) getYataiClient(ctx context.Context) (yataiClient **yataiclient.YataiClient, clusterName *string, err error) {
	restConfig := config.GetConfigOrDie()
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		err = errors.Wrapf(err, "create kubernetes clientset")
		return
	}
	var yataiConf *commonconfig.YataiConfig

	if cachedYataiConf != nil {
		yataiConf = cachedYataiConf
	} else {
		yataiConf, err = commonconfig.GetYataiConfig(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
			secret, err := clientset.CoreV1().Secrets(namespace).Get(ctx, name, metav1.GetOptions{})
			return secret, errors.Wrap(err, "get secret")
		}, commonconsts.YataiDeploymentComponentName, false)
		isNotFound := k8serrors.IsNotFound(err)
		if err != nil && !isNotFound {
			err = errors.Wrap(err, "get yatai config")
			return
		}

		if isNotFound {
			return
		}
		cachedYataiConf = yataiConf
	}

	yataiEndpoint := yataiConf.Endpoint
	yataiAPIToken := yataiConf.ApiToken
	if yataiEndpoint == "" {
		return
	}

	clusterName_ := yataiConf.ClusterName
	if clusterName_ == "" {
		clusterName_ = DefaultClusterName
	}
	yataiClient_ := yataiclient.NewYataiClient(yataiEndpoint, fmt.Sprintf("%s:%s:%s", commonconsts.YataiDeploymentComponentName, clusterName_, yataiAPIToken))
	yataiClient = &yataiClient_
	clusterName = &clusterName_
	return
}

func (r *CompoundAINimDeploymentReconciler) getYataiClientWithAuth(ctx context.Context, compoundAINimDeployment *v1alpha1.CompoundAINimDeployment) (**yataiclient.YataiClient, *string, error) {
	orgId, ok := compoundAINimDeployment.Labels[commonconsts.NgcOrganizationHeaderName]
	if !ok {
		orgId = commonconsts.DefaultOrgId
	}

	userId, ok := compoundAINimDeployment.Labels[commonconsts.NgcUserHeaderName]
	if !ok {
		userId = commonconsts.DefaultUserId
	}

	auth := yataiclient.CompoundAIAuthHeaders{
		OrgId:  orgId,
		UserId: userId,
	}

	client, clusterName, err := r.getYataiClient(ctx)
	if err != nil {
		return nil, nil, err
	}

	(*client).SetAuth(auth)
	return client, clusterName, err
}

type createOrUpdateOrDeleteDeploymentsOption struct {
	yataiClient             **yataiclient.YataiClient
	compoundAINimDeployment *v1alpha1.CompoundAINimDeployment
	compoundAINim           *v1alpha1.CompoundAINim
	clusterName             *string
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) createOrUpdateOrDeleteDeployments(ctx context.Context, opt createOrUpdateOrDeleteDeploymentsOption) (modified bool, err error) {
	containsStealingTrafficDebugModeEnabled := checkIfContainsStealingTrafficDebugModeEnabled(opt.compoundAINimDeployment)
	modified, err = r.createOrUpdateDeployment(ctx, createOrUpdateDeploymentOption{
		createOrUpdateOrDeleteDeploymentsOption: opt,
		isStealingTrafficDebugModeEnabled:       false,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
	})
	if err != nil {
		err = errors.Wrap(err, "create or update deployment")
		return
	}
	if containsStealingTrafficDebugModeEnabled {
		modified, err = r.createOrUpdateDeployment(ctx, createOrUpdateDeploymentOption{
			createOrUpdateOrDeleteDeploymentsOption: opt,
			isStealingTrafficDebugModeEnabled:       true,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
		})
		if err != nil {
			err = errors.Wrap(err, "create or update deployment")
			return
		}
	} else {
		debugDeploymentName := r.getKubeName(opt.compoundAINimDeployment, opt.compoundAINim, true)
		debugDeployment := &appsv1.Deployment{}
		err = r.Get(ctx, types.NamespacedName{Name: debugDeploymentName, Namespace: opt.compoundAINimDeployment.Namespace}, debugDeployment)
		isNotFound := k8serrors.IsNotFound(err)
		if err != nil && !isNotFound {
			err = errors.Wrap(err, "get deployment")
			return
		}
		err = nil
		if !isNotFound {
			err = r.Delete(ctx, debugDeployment)
			if err != nil {
				err = errors.Wrap(err, "delete deployment")
				return
			}
			modified = true
		}
	}
	return
}

type createOrUpdateDeploymentOption struct {
	createOrUpdateOrDeleteDeploymentsOption
	isStealingTrafficDebugModeEnabled       bool
	containsStealingTrafficDebugModeEnabled bool
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) createOrUpdateDeployment(ctx context.Context, opt createOrUpdateDeploymentOption) (modified bool, err error) {
	logs := log.FromContext(ctx)

	deployment, err := r.generateDeployment(ctx, generateDeploymentOption{
		compoundAINimDeployment:                 opt.compoundAINimDeployment,
		compoundAINim:                           opt.compoundAINim,
		yataiClient:                             opt.yataiClient,
		clusterName:                             opt.clusterName,
		isStealingTrafficDebugModeEnabled:       opt.isStealingTrafficDebugModeEnabled,
		containsStealingTrafficDebugModeEnabled: opt.containsStealingTrafficDebugModeEnabled,
	})
	if err != nil {
		return
	}

	logs = logs.WithValues("namespace", deployment.Namespace, "deploymentName", deployment.Name)

	deploymentNamespacedName := fmt.Sprintf("%s/%s", deployment.Namespace, deployment.Name)

	r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "GetDeployment", "Getting Deployment %s", deploymentNamespacedName)

	oldDeployment := &appsv1.Deployment{}
	err = r.Get(ctx, types.NamespacedName{Name: deployment.Name, Namespace: deployment.Namespace}, oldDeployment)
	oldDeploymentIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !oldDeploymentIsNotFound {
		r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "GetDeployment", "Failed to get Deployment %s: %s", deploymentNamespacedName, err)
		logs.Error(err, "Failed to get Deployment.")
		return
	}

	if oldDeploymentIsNotFound {
		logs.Info("Deployment not found. Creating a new one.")

		err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(deployment), "set last applied annotation for deployment %s", deployment.Name)
		if err != nil {
			logs.Error(err, "Failed to set last applied annotation.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for Deployment %s: %s", deploymentNamespacedName, err)
			return
		}

		r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "CreateDeployment", "Creating a new Deployment %s", deploymentNamespacedName)
		err = r.Create(ctx, deployment)
		if err != nil {
			logs.Error(err, "Failed to create Deployment.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "CreateDeployment", "Failed to create Deployment %s: %s", deploymentNamespacedName, err)
			return
		}
		logs.Info("Deployment created.")
		r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "CreateDeployment", "Created Deployment %s", deploymentNamespacedName)
		modified = true
	} else {
		logs.Info("Deployment found.")

		var patchResult *patch.PatchResult
		patchResult, err = patch.DefaultPatchMaker.Calculate(oldDeployment, deployment)
		if err != nil {
			logs.Error(err, "Failed to calculate patch.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "CalculatePatch", "Failed to calculate patch for Deployment %s: %s", deploymentNamespacedName, err)
			return
		}

		if !patchResult.IsEmpty() {
			logs.Info("Deployment spec is different. Updating Deployment.")

			err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(deployment), "set last applied annotation for deployment %s", deployment.Name)
			if err != nil {
				logs.Error(err, "Failed to set last applied annotation.")
				r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for Deployment %s: %s", deploymentNamespacedName, err)
				return
			}

			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "UpdateDeployment", "Updating Deployment %s", deploymentNamespacedName)
			err = r.Update(ctx, deployment)
			if err != nil {
				logs.Error(err, "Failed to update Deployment.")
				r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "UpdateDeployment", "Failed to update Deployment %s: %s", deploymentNamespacedName, err)
				return
			}
			logs.Info("Deployment updated.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "UpdateDeployment", "Updated Deployment %s", deploymentNamespacedName)
			modified = true
		} else {
			logs.Info("Deployment spec is the same. Skipping update.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "UpdateDeployment", "Skipping update Deployment %s", deploymentNamespacedName)
		}
	}

	return
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) createOrUpdateHPA(ctx context.Context, compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, compoundAINim *v1alpha1.CompoundAINim) (modified bool, err error) {
	logs := log.FromContext(ctx)

	hpa, err := r.generateHPA(compoundAINimDeployment, compoundAINim)
	if err != nil {
		return
	}
	logs = logs.WithValues("namespace", hpa.Namespace, "hpaName", hpa.Name)
	hpaNamespacedName := fmt.Sprintf("%s/%s", hpa.Namespace, hpa.Name)

	r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetHPA", "Getting HPA %s", hpaNamespacedName)

	oldHPA, err := r.getHPA(ctx, hpa)
	oldHPAIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !oldHPAIsNotFound {
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "GetHPA", "Failed to get HPA %s: %s", hpaNamespacedName, err)
		logs.Error(err, "Failed to get HPA.")
		return
	}

	if oldHPAIsNotFound {
		logs.Info("HPA not found. Creating a new one.")

		err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(hpa), "set last applied annotation for hpa %s", hpa.Name)
		if err != nil {
			logs.Error(err, "Failed to set last applied annotation.")
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for HPA %s: %s", hpaNamespacedName, err)
			return
		}

		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "CreateHPA", "Creating a new HPA %s", hpaNamespacedName)
		err = r.Create(ctx, hpa)
		if err != nil {
			logs.Error(err, "Failed to create HPA.")
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "CreateHPA", "Failed to create HPA %s: %s", hpaNamespacedName, err)
			return
		}
		logs.Info("HPA created.")
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "CreateHPA", "Created HPA %s", hpaNamespacedName)
		modified = true
	} else {
		logs.Info("HPA found.")

		var patchResult *patch.PatchResult
		patchResult, err = patch.DefaultPatchMaker.Calculate(oldHPA, hpa)
		if err != nil {
			logs.Error(err, "Failed to calculate patch.")
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "CalculatePatch", "Failed to calculate patch for HPA %s: %s", hpaNamespacedName, err)
			return
		}

		if !patchResult.IsEmpty() {
			logs.Info(fmt.Sprintf("HPA spec is different. Updating HPA. The patch result is: %s", patchResult.String()))

			err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(hpa), "set last applied annotation for hpa %s", hpa.Name)
			if err != nil {
				logs.Error(err, "Failed to set last applied annotation.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for HPA %s: %s", hpaNamespacedName, err)
				return
			}

			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateHPA", "Updating HPA %s", hpaNamespacedName)
			err = r.Update(ctx, hpa)
			if err != nil {
				logs.Error(err, "Failed to update HPA.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "UpdateHPA", "Failed to update HPA %s: %s", hpaNamespacedName, err)
				return
			}
			logs.Info("HPA updated.")
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateHPA", "Updated HPA %s", hpaNamespacedName)
			modified = true
		} else {
			logs.Info("HPA spec is the same. Skipping update.")
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateHPA", "Skipping update HPA %s", hpaNamespacedName)
		}
	}

	return
}

func getResourceAnnotations(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment) map[string]string {
	resourceAnnotations := compoundAINimDeployment.Spec.Annotations
	if resourceAnnotations == nil {
		resourceAnnotations = map[string]string{}
	}

	return resourceAnnotations
}

func checkIfIsDebugModeEnabled(annotations map[string]string) bool {
	if annotations == nil {
		return false
	}

	return annotations[KubeAnnotationYataiEnableDebugMode] == commonconsts.KubeLabelValueTrue
}

func checkIfIsStealingTrafficDebugModeEnabled(annotations map[string]string) bool {
	if annotations == nil {
		return false
	}

	return annotations[KubeAnnotationYataiEnableStealingTrafficDebugMode] == commonconsts.KubeLabelValueTrue
}

func checkIfIsDebugPodReceiveProductionTrafficEnabled(annotations map[string]string) bool {
	if annotations == nil {
		return false
	}

	return annotations[KubeAnnotationYataiEnableDebugPodReceiveProductionTraffic] == commonconsts.KubeLabelValueTrue
}

func checkIfContainsStealingTrafficDebugModeEnabled(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment) bool {
	return checkIfIsStealingTrafficDebugModeEnabled(compoundAINimDeployment.Spec.Annotations)
}

type createOrUpdateOrDeleteServicesOption struct {
	compoundAINimDeployment *v1alpha1.CompoundAINimDeployment
	compoundAINim           *v1alpha1.CompoundAINim
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) createOrUpdateOrDeleteServices(ctx context.Context, opt createOrUpdateOrDeleteServicesOption) (modified bool, err error) {
	resourceAnnotations := getResourceAnnotations(opt.compoundAINimDeployment)
	isDebugPodReceiveProductionTrafficEnabled := checkIfIsDebugPodReceiveProductionTrafficEnabled(resourceAnnotations)
	containsStealingTrafficDebugModeEnabled := checkIfContainsStealingTrafficDebugModeEnabled(opt.compoundAINimDeployment)
	modified, err = r.createOrUpdateService(ctx, createOrUpdateServiceOption{
		compoundAINimDeployment:                 opt.compoundAINimDeployment,
		compoundAINim:                           opt.compoundAINim,
		isStealingTrafficDebugModeEnabled:       false,
		isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
		isGenericService:                        true,
	})
	if err != nil {
		return
	}
	if containsStealingTrafficDebugModeEnabled {
		var modified_ bool
		modified_, err = r.createOrUpdateService(ctx, createOrUpdateServiceOption{
			compoundAINimDeployment:                 opt.compoundAINimDeployment,
			compoundAINim:                           opt.compoundAINim,
			isStealingTrafficDebugModeEnabled:       false,
			isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
			isGenericService:                        false,
		})
		if err != nil {
			return
		}
		if modified_ {
			modified = true
		}
		modified_, err = r.createOrUpdateService(ctx, createOrUpdateServiceOption{
			compoundAINimDeployment:                 opt.compoundAINimDeployment,
			compoundAINim:                           opt.compoundAINim,
			isStealingTrafficDebugModeEnabled:       true,
			isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
			isGenericService:                        false,
		})
		if err != nil {
			return
		}
		if modified_ {
			modified = true
		}
	} else {
		productionServiceName := r.getServiceName(opt.compoundAINimDeployment, opt.compoundAINim, false)
		svc := &corev1.Service{}
		err = r.Get(ctx, types.NamespacedName{Name: productionServiceName, Namespace: opt.compoundAINimDeployment.Namespace}, svc)
		isNotFound := k8serrors.IsNotFound(err)
		if err != nil && !isNotFound {
			err = errors.Wrapf(err, "Failed to get service %s", productionServiceName)
			return
		}
		if !isNotFound {
			modified = true
			err = r.Delete(ctx, svc)
			if err != nil {
				err = errors.Wrapf(err, "Failed to delete service %s", productionServiceName)
				return
			}
		}
		debugServiceName := r.getServiceName(opt.compoundAINimDeployment, opt.compoundAINim, true)
		svc = &corev1.Service{}
		err = r.Get(ctx, types.NamespacedName{Name: debugServiceName, Namespace: opt.compoundAINimDeployment.Namespace}, svc)
		isNotFound = k8serrors.IsNotFound(err)
		if err != nil && !isNotFound {
			err = errors.Wrapf(err, "Failed to get service %s", debugServiceName)
			return
		}
		err = nil
		if !isNotFound {
			modified = true
			err = r.Delete(ctx, svc)
			if err != nil {
				err = errors.Wrapf(err, "Failed to delete service %s", debugServiceName)
				return
			}
		}
	}
	return
}

type createOrUpdateServiceOption struct {
	compoundAINimDeployment                 *v1alpha1.CompoundAINimDeployment
	compoundAINim                           *v1alpha1.CompoundAINim
	isStealingTrafficDebugModeEnabled       bool
	isDebugPodReceiveProductionTraffic      bool
	containsStealingTrafficDebugModeEnabled bool
	isGenericService                        bool
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) createOrUpdateService(ctx context.Context, opt createOrUpdateServiceOption) (modified bool, err error) {
	logs := log.FromContext(ctx)

	// nolint: gosimple
	service, err := r.generateService(generateServiceOption(opt))
	if err != nil {
		return
	}

	logs = logs.WithValues("namespace", service.Namespace, "serviceName", service.Name, "serviceSelector", service.Spec.Selector)

	serviceNamespacedName := fmt.Sprintf("%s/%s", service.Namespace, service.Name)

	r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "GetService", "Getting Service %s", serviceNamespacedName)

	oldService := &corev1.Service{}
	err = r.Get(ctx, types.NamespacedName{Name: service.Name, Namespace: service.Namespace}, oldService)
	oldServiceIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !oldServiceIsNotFound {
		r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "GetService", "Failed to get Service %s: %s", serviceNamespacedName, err)
		logs.Error(err, "Failed to get Service.")
		return
	}

	if oldServiceIsNotFound {
		logs.Info("Service not found. Creating a new one.")

		err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(service), "set last applied annotation for service %s", service.Name)
		if err != nil {
			logs.Error(err, "Failed to set last applied annotation.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for Service %s: %s", serviceNamespacedName, err)
			return
		}

		r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "CreateService", "Creating a new Service %s", serviceNamespacedName)
		err = r.Create(ctx, service)
		if err != nil {
			logs.Error(err, "Failed to create Service.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "CreateService", "Failed to create Service %s: %s", serviceNamespacedName, err)
			return
		}
		logs.Info("Service created.")
		r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "CreateService", "Created Service %s", serviceNamespacedName)
		modified = true
	} else {
		logs.Info("Service found.")

		var patchResult *patch.PatchResult
		patchResult, err = patch.DefaultPatchMaker.Calculate(oldService, service)
		if err != nil {
			logs.Error(err, "Failed to calculate patch.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "CalculatePatch", "Failed to calculate patch for Service %s: %s", serviceNamespacedName, err)
			return
		}

		if !patchResult.IsEmpty() {
			logs.Info("Service spec is different. Updating Service.")

			err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(service), "set last applied annotation for service %s", service.Name)
			if err != nil {
				logs.Error(err, "Failed to set last applied annotation.")
				r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for Service %s: %s", serviceNamespacedName, err)
				return
			}

			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "UpdateService", "Updating Service %s", serviceNamespacedName)
			oldService.Annotations = service.Annotations
			oldService.Labels = service.Labels
			oldService.Spec = service.Spec
			err = r.Update(ctx, oldService)
			if err != nil {
				logs.Error(err, "Failed to update Service.")
				r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeWarning, "UpdateService", "Failed to update Service %s: %s", serviceNamespacedName, err)
				return
			}
			logs.Info("Service updated.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "UpdateService", "Updated Service %s", serviceNamespacedName)
			modified = true
		} else {
			logs = logs.WithValues("oldServiceSelector", oldService.Spec.Selector)
			logs.Info("Service spec is the same. Skipping update.")
			r.Recorder.Eventf(opt.compoundAINimDeployment, corev1.EventTypeNormal, "UpdateService", "Skipping update Service %s", serviceNamespacedName)
		}
	}

	return
}

func (r *CompoundAINimDeploymentReconciler) createOrUpdateVirtualService(ctx context.Context, compoundAINimDeployment *v1alpha1.CompoundAINimDeployment) (bool, error) {
	log := log.FromContext(ctx)
	log.Info("Starting createOrUpdateVirtualService")
	vsName := compoundAINimDeployment.Name
	if compoundAINimDeployment.Spec.Ingress.HostPrefix != nil {
		vsName = *compoundAINimDeployment.Spec.Ingress.HostPrefix + vsName
	}
	vs := &networkingv1beta1.VirtualService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      compoundAINimDeployment.Name,
			Namespace: compoundAINimDeployment.Namespace,
		},
		Spec: istioNetworking.VirtualService{
			Hosts: []string{
				fmt.Sprintf("%s.dev.aire.nvidia.com", vsName),
			},
			Gateways: []string{"istio-system/ingress-alb"},
			Http: []*istioNetworking.HTTPRoute{
				{
					Match: []*istioNetworking.HTTPMatchRequest{
						{
							Uri: &istioNetworking.StringMatch{
								MatchType: &istioNetworking.StringMatch_Prefix{Prefix: "/"},
							},
						},
					},
					Route: []*istioNetworking.HTTPRouteDestination{
						{
							Destination: &istioNetworking.Destination{
								Host: fmt.Sprintf("%s.yatai.svc.cluster.local", compoundAINimDeployment.Name),
								Port: &istioNetworking.PortSelector{
									Number: 3000,
								},
							},
						},
					},
				},
			},
		},
	}

	log.Info("VirtualService object constructed", "VirtualService", vs)

	oldVS := &networkingv1beta1.VirtualService{}
	err := r.Get(ctx, types.NamespacedName{Name: vs.Name, Namespace: vs.Namespace}, oldVS)
	if client.IgnoreNotFound(err) != nil {
		log.Error(err, "Failed to get VirtualService")
		return false, err
	}

	vsEnabled := compoundAINimDeployment.Spec.Ingress.Enabled && compoundAINimDeployment.Spec.Ingress.UseVirtualService != nil && *compoundAINimDeployment.Spec.Ingress.UseVirtualService

	if err != nil {
		if vsEnabled {
			log.Info("VirtualService not found, creating new one")
			if err := r.Create(ctx, vs); err != nil {
				log.Error(err, "Failed to create VirtualService")
				return false, err
			}
			log.Info("VirtualService created successfully", "VirtualService", vs)
			return true, nil
		}
		return false, nil
	}

	if !vsEnabled {
		log.Info("VirtualService found, deleting", "OldVirtualService", oldVS)
		if err := r.Delete(ctx, oldVS); err != nil {
			log.Error(err, "Failed to delete VirtualService")
			return false, err
		}
		return true, err
	}

	log.Info("VirtualService found, updating", "OldVirtualService", oldVS)

	if err := r.Update(ctx, vs); err != nil {
		log.Error(err, "Failed to update VirtualService")
		return false, err
	}
	log.Info("VirtualService updated successfully", "VirtualService", oldVS)

	return true, nil
}

type createOrUpdateIngressOption struct {
	yataiClient             **yataiclient.YataiClient
	compoundAINimDeployment *v1alpha1.CompoundAINimDeployment
	compoundAINim           *v1alpha1.CompoundAINim
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) createOrUpdateIngresses(ctx context.Context, opt createOrUpdateIngressOption) (modified bool, err error) {
	logs := log.FromContext(ctx)

	compoundAINimDeployment := opt.compoundAINimDeployment
	compoundAINim := opt.compoundAINim

	modified, err = r.createOrUpdateVirtualService(ctx, compoundAINimDeployment)
	if err != nil {
		return false, err
	}

	// generateIngresses generates an ingress and actively waits for the ingress to come online ....
	// so disabling it for now unless explicitly enabled
	if !opt.compoundAINimDeployment.Spec.Ingress.Enabled || (opt.compoundAINimDeployment.Spec.Ingress.UseVirtualService != nil && *opt.compoundAINimDeployment.Spec.Ingress.UseVirtualService) {
		return false, nil
	}

	ingresses, err := r.generateIngresses(ctx, generateIngressesOption{
		yataiClient:             opt.yataiClient,
		compoundAINimDeployment: compoundAINimDeployment,
		compoundAINim:           compoundAINim,
	})
	if err != nil {
		return
	}

	for _, ingress := range ingresses {
		logs := logs.WithValues("namespace", ingress.Namespace, "ingressName", ingress.Name)
		ingressNamespacedName := fmt.Sprintf("%s/%s", ingress.Namespace, ingress.Name)

		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetIngress", "Getting Ingress %s", ingressNamespacedName)

		oldIngress := &networkingv1.Ingress{}
		err = r.Get(ctx, types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace}, oldIngress)
		oldIngressIsNotFound := k8serrors.IsNotFound(err)
		if err != nil && !oldIngressIsNotFound {
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "GetIngress", "Failed to get Ingress %s: %s", ingressNamespacedName, err)
			logs.Error(err, "Failed to get Ingress.")
			return
		}
		err = nil

		if oldIngressIsNotFound {
			if !compoundAINimDeployment.Spec.Ingress.Enabled {
				logs.Info("Ingress not enabled. Skipping.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GetIngress", "Skipping Ingress %s", ingressNamespacedName)
				continue
			}

			logs.Info("Ingress not found. Creating a new one.")

			err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(ingress), "set last applied annotation for ingress %s", ingress.Name)
			if err != nil {
				logs.Error(err, "Failed to set last applied annotation.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for Ingress %s: %s", ingressNamespacedName, err)
				return
			}

			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "CreateIngress", "Creating a new Ingress %s", ingressNamespacedName)
			err = r.Create(ctx, ingress)
			if err != nil {
				logs.Error(err, "Failed to create Ingress.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "CreateIngress", "Failed to create Ingress %s: %s", ingressNamespacedName, err)
				return
			}
			logs.Info("Ingress created.")
			r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "CreateIngress", "Created Ingress %s", ingressNamespacedName)
			modified = true
		} else {
			logs.Info("Ingress found.")

			if !compoundAINimDeployment.Spec.Ingress.Enabled {
				logs.Info("Ingress not enabled. Deleting.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "DeleteIngress", "Deleting Ingress %s", ingressNamespacedName)
				err = r.Delete(ctx, ingress)
				if err != nil {
					logs.Error(err, "Failed to delete Ingress.")
					r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "DeleteIngress", "Failed to delete Ingress %s: %s", ingressNamespacedName, err)
					return
				}
				logs.Info("Ingress deleted.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "DeleteIngress", "Deleted Ingress %s", ingressNamespacedName)
				modified = true
				continue
			}

			// Keep host unchanged
			ingress.Spec.Rules[0].Host = oldIngress.Spec.Rules[0].Host

			var patchResult *patch.PatchResult
			patchResult, err = patch.DefaultPatchMaker.Calculate(oldIngress, ingress)
			if err != nil {
				logs.Error(err, "Failed to calculate patch.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "CalculatePatch", "Failed to calculate patch for Ingress %s: %s", ingressNamespacedName, err)
				return
			}

			if !patchResult.IsEmpty() {
				logs.Info("Ingress spec is different. Updating Ingress.")

				err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(ingress), "set last applied annotation for ingress %s", ingress.Name)
				if err != nil {
					logs.Error(err, "Failed to set last applied annotation.")
					r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for Ingress %s: %s", ingressNamespacedName, err)
					return
				}

				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateIngress", "Updating Ingress %s", ingressNamespacedName)
				err = r.Update(ctx, ingress)
				if err != nil {
					logs.Error(err, "Failed to update Ingress.")
					r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "UpdateIngress", "Failed to update Ingress %s: %s", ingressNamespacedName, err)
					return
				}
				logs.Info("Ingress updated.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateIngress", "Updated Ingress %s", ingressNamespacedName)
				modified = true
			} else {
				logs.Info("Ingress spec is the same. Skipping update.")
				r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "UpdateIngress", "Skipping update Ingress %s", ingressNamespacedName)
			}
		}
	}

	return
}

func (r *CompoundAINimDeploymentReconciler) getKubeName(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, _ *v1alpha1.CompoundAINim, debug bool) string {
	if debug {
		return fmt.Sprintf("%s-d", compoundAINimDeployment.Name)
	}
	return compoundAINimDeployment.Name
}

func (r *CompoundAINimDeploymentReconciler) getServiceName(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, _ *v1alpha1.CompoundAINim, debug bool) string {
	var kubeName string
	if debug {
		kubeName = fmt.Sprintf("%s-d", compoundAINimDeployment.Name)
	} else {
		kubeName = fmt.Sprintf("%s-p", compoundAINimDeployment.Name)
	}
	return kubeName
}

func (r *CompoundAINimDeploymentReconciler) getGenericServiceName(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, compoundAINim *v1alpha1.CompoundAINim) string {
	return r.getKubeName(compoundAINimDeployment, compoundAINim, false)
}

func (r *CompoundAINimDeploymentReconciler) getKubeLabels(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, compoundAINim *v1alpha1.CompoundAINim) map[string]string {
	compoundAINimRepositoryName, _, compoundAINimVersion := xstrings.Partition(compoundAINim.Spec.Tag, ":")
	labels := map[string]string{
		commonconsts.KubeLabelYataiBentoDeployment:           compoundAINimDeployment.Name,
		commonconsts.KubeLabelBentoRepository:                compoundAINimRepositoryName,
		commonconsts.KubeLabelBentoVersion:                   compoundAINimVersion,
		commonconsts.KubeLabelYataiBentoDeploymentTargetType: DeploymentTargetTypeProduction,
		commonconsts.KubeLabelCreator:                        "yatai-deployment",
	}
	labels[commonconsts.KubeLabelYataiBentoDeploymentComponentType] = commonconsts.YataiBentoDeploymentComponentApiServer
	return labels
}

func (r *CompoundAINimDeploymentReconciler) getKubeAnnotations(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, compoundAINim *v1alpha1.CompoundAINim) map[string]string {
	compoundAINimRepositoryName, compoundAINimVersion := getCompoundAINimRepositoryNameAndCompoundAINimVersion(compoundAINim)
	annotations := map[string]string{
		commonconsts.KubeAnnotationBentoRepository: compoundAINimRepositoryName,
		commonconsts.KubeAnnotationBentoVersion:    compoundAINimVersion,
	}
	var extraAnnotations map[string]string
	if compoundAINimDeployment.Spec.ExtraPodMetadata != nil {
		extraAnnotations = compoundAINimDeployment.Spec.ExtraPodMetadata.Annotations
	} else {
		extraAnnotations = map[string]string{}
	}
	for k, v := range extraAnnotations {
		annotations[k] = v
	}
	return annotations
}

type generateDeploymentOption struct {
	compoundAINimDeployment                 *v1alpha1.CompoundAINimDeployment
	compoundAINim                           *v1alpha1.CompoundAINim
	yataiClient                             **yataiclient.YataiClient
	clusterName                             *string
	isStealingTrafficDebugModeEnabled       bool
	containsStealingTrafficDebugModeEnabled bool
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) generateDeployment(ctx context.Context, opt generateDeploymentOption) (kubeDeployment *appsv1.Deployment, err error) {
	kubeNs := opt.compoundAINimDeployment.Namespace

	// nolint: gosimple
	podTemplateSpec, err := r.generatePodTemplateSpec(ctx, generatePodTemplateSpecOption(opt))
	if err != nil {
		return
	}

	labels := r.getKubeLabels(opt.compoundAINimDeployment, opt.compoundAINim)

	annotations := r.getKubeAnnotations(opt.compoundAINimDeployment, opt.compoundAINim)

	kubeName := r.getKubeName(opt.compoundAINimDeployment, opt.compoundAINim, opt.isStealingTrafficDebugModeEnabled)

	defaultMaxSurge := intstr.FromString("25%")
	defaultMaxUnavailable := intstr.FromString("25%")

	strategy := appsv1.DeploymentStrategy{
		Type: appsv1.RollingUpdateDeploymentStrategyType,
		RollingUpdate: &appsv1.RollingUpdateDeployment{
			MaxSurge:       &defaultMaxSurge,
			MaxUnavailable: &defaultMaxUnavailable,
		},
	}

	resourceAnnotations := getResourceAnnotations(opt.compoundAINimDeployment)
	strategyStr := resourceAnnotations[KubeAnnotationDeploymentStrategy]
	if strategyStr != "" {
		strategyType := modelschemas.DeploymentStrategy(strategyStr)
		switch strategyType {
		case modelschemas.DeploymentStrategyRollingUpdate:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       &defaultMaxSurge,
					MaxUnavailable: &defaultMaxUnavailable,
				},
			}
		case modelschemas.DeploymentStrategyRecreate:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RecreateDeploymentStrategyType,
			}
		case modelschemas.DeploymentStrategyRampedSlowRollout:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       &[]intstr.IntOrString{intstr.FromInt(1)}[0],
					MaxUnavailable: &[]intstr.IntOrString{intstr.FromInt(0)}[0],
				},
			}
		case modelschemas.DeploymentStrategyBestEffortControlledRollout:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       &[]intstr.IntOrString{intstr.FromInt(0)}[0],
					MaxUnavailable: &[]intstr.IntOrString{intstr.FromString("20%")}[0],
				},
			}
		}
	}

	var replicas *int32
	if opt.isStealingTrafficDebugModeEnabled {
		replicas = &[]int32{int32(1)}[0]
	}

	kubeDeployment = &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        kubeName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					commonconsts.KubeLabelYataiSelector: kubeName,
				},
			},
			Template: *podTemplateSpec,
			Strategy: strategy,
		},
	}

	err = ctrl.SetControllerReference(opt.compoundAINimDeployment, kubeDeployment, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set deployment %s controller reference", kubeDeployment.Name)
	}

	return
}

func (r *CompoundAINimDeploymentReconciler) generateHPA(compoundAINimDeployment *v1alpha1.CompoundAINimDeployment, compoundAINim *v1alpha1.CompoundAINim) (*autoscalingv2.HorizontalPodAutoscaler, error) {
	labels := r.getKubeLabels(compoundAINimDeployment, compoundAINim)

	annotations := r.getKubeAnnotations(compoundAINimDeployment, compoundAINim)

	kubeName := r.getKubeName(compoundAINimDeployment, compoundAINim, false)

	kubeNs := compoundAINimDeployment.Namespace

	hpaConf := compoundAINimDeployment.Spec.Autoscaling

	if hpaConf == nil {
		hpaConf = &v1alpha1.Autoscaling{
			MinReplicas: 1,
			MaxReplicas: 1,
		}
	}

	minReplica := int32(hpaConf.MinReplicas)

	kubeHpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:        kubeName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			MinReplicas: &minReplica,
			MaxReplicas: int32(hpaConf.MaxReplicas),
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       kubeName,
			},
			Metrics: hpaConf.Metrics,
		},
	}

	if len(kubeHpa.Spec.Metrics) == 0 {
		averageUtilization := int32(commonconsts.HPACPUDefaultAverageUtilization)
		kubeHpa.Spec.Metrics = []autoscalingv2.MetricSpec{
			{
				Type: autoscalingv2.ResourceMetricSourceType,
				Resource: &autoscalingv2.ResourceMetricSource{
					Name: corev1.ResourceCPU,
					Target: autoscalingv2.MetricTarget{
						Type:               autoscalingv2.UtilizationMetricType,
						AverageUtilization: &averageUtilization,
					},
				},
			},
		}
	}

	err := ctrl.SetControllerReference(compoundAINimDeployment, kubeHpa, r.Scheme)
	if err != nil {
		return nil, errors.Wrapf(err, "set hpa %s controller reference", kubeName)
	}

	return kubeHpa, err
}

func (r *CompoundAINimDeploymentReconciler) getHPA(ctx context.Context, hpa *autoscalingv2.HorizontalPodAutoscaler) (client.Object, error) {
	name, ns := hpa.Name, hpa.Namespace
	obj := &autoscalingv2.HorizontalPodAutoscaler{}
	err := r.Get(ctx, types.NamespacedName{Name: name, Namespace: ns}, obj)
	if err == nil {
		legacyStatus := &autoscalingv2.HorizontalPodAutoscalerStatus{}
		if err := copier.Copy(legacyStatus, obj.Status); err != nil {
			return nil, err
		}
		obj.Status = *legacyStatus
	}
	return obj, err
}

func getCompoundAINimRepositoryNameAndCompoundAINimVersion(compoundAINim *v1alpha1.CompoundAINim) (repositoryName string, version string) {
	repositoryName, _, version = xstrings.Partition(compoundAINim.Spec.Tag, ":")

	return
}

type generatePodTemplateSpecOption struct {
	compoundAINimDeployment                 *v1alpha1.CompoundAINimDeployment
	compoundAINim                           *v1alpha1.CompoundAINim
	yataiClient                             **yataiclient.YataiClient
	clusterName                             *string
	isStealingTrafficDebugModeEnabled       bool
	containsStealingTrafficDebugModeEnabled bool
}

//nolint:gocyclo,nakedret
func (r *CompoundAINimDeploymentReconciler) generatePodTemplateSpec(ctx context.Context, opt generatePodTemplateSpecOption) (podTemplateSpec *corev1.PodTemplateSpec, err error) {
	compoundAINimRepositoryName, _ := getCompoundAINimRepositoryNameAndCompoundAINimVersion(opt.compoundAINim)
	podLabels := r.getKubeLabels(opt.compoundAINimDeployment, opt.compoundAINim)
	if opt.isStealingTrafficDebugModeEnabled {
		podLabels[commonconsts.KubeLabelYataiBentoDeploymentTargetType] = DeploymentTargetTypeDebug
	}

	podAnnotations := r.getKubeAnnotations(opt.compoundAINimDeployment, opt.compoundAINim)

	kubeName := r.getKubeName(opt.compoundAINimDeployment, opt.compoundAINim, opt.isStealingTrafficDebugModeEnabled)

	containerPort := commonconsts.BentoServicePort
	lastPort := containerPort + 1

	monitorExporter := opt.compoundAINimDeployment.Spec.MonitorExporter
	needMonitorContainer := monitorExporter != nil && monitorExporter.Enabled

	lastPort++
	monitorExporterPort := lastPort

	var envs []corev1.EnvVar
	envsSeen := make(map[string]struct{})

	resourceAnnotations := opt.compoundAINimDeployment.Spec.Annotations
	specEnvs := opt.compoundAINimDeployment.Spec.Envs

	if resourceAnnotations == nil {
		resourceAnnotations = make(map[string]string)
	}

	isDebugModeEnabled := checkIfIsDebugModeEnabled(resourceAnnotations)

	if specEnvs != nil {
		envs = make([]corev1.EnvVar, 0, len(specEnvs)+1)

		for _, env := range specEnvs {
			if _, ok := envsSeen[env.Name]; ok {
				continue
			}
			if env.Name == commonconsts.EnvBentoServicePort {
				// nolint: gosec
				containerPort, err = strconv.Atoi(env.Value)
				if err != nil {
					return nil, errors.Wrapf(err, "invalid port value %s", env.Value)
				}
			}
			envsSeen[env.Name] = struct{}{}
			envs = append(envs, corev1.EnvVar{
				Name:  env.Name,
				Value: env.Value,
			})
		}
	}

	defaultEnvs := []corev1.EnvVar{
		{
			Name:  commonconsts.EnvBentoServicePort,
			Value: fmt.Sprintf("%d", containerPort),
		},
		{
			Name:  commonconsts.EnvYataiDeploymentUID,
			Value: string(opt.compoundAINimDeployment.UID),
		},
		{
			Name:  commonconsts.EnvYataiBentoDeploymentName,
			Value: opt.compoundAINimDeployment.Name,
		},
		{
			Name:  commonconsts.EnvYataiBentoDeploymentNamespace,
			Value: opt.compoundAINimDeployment.Namespace,
		},
	}

	if r.NatsAddr != "" {
		defaultEnvs = append(defaultEnvs, corev1.EnvVar{
			Name:  "NATS_SERVER",
			Value: r.NatsAddr,
		})
	}

	if r.EtcdAddr != "" {
		defaultEnvs = append(defaultEnvs, corev1.EnvVar{
			Name:  "ETCD_ENDPOINTS",
			Value: r.EtcdAddr,
		})
	}

	if opt.yataiClient != nil {
		yataiClient := *opt.yataiClient

		var cluster *schemasv1.ClusterFullSchema
		clusterName := DefaultClusterName
		if opt.clusterName != nil {
			clusterName = *opt.clusterName
		}
		cluster, err = yataiClient.GetCluster(ctx, clusterName)
		if err != nil {
			return
		}

		var version *schemasv1.VersionSchema
		version, err = yataiClient.GetVersion(ctx)
		if err != nil {
			return
		}

		defaultEnvs = append(defaultEnvs, []corev1.EnvVar{
			{
				Name:  commonconsts.EnvYataiVersion,
				Value: fmt.Sprintf("%s-%s", version.Version, version.GitCommit),
			},
			{
				Name:  commonconsts.EnvYataiClusterUID,
				Value: cluster.Uid,
			},
		}...)
	}

	for _, env := range defaultEnvs {
		if _, ok := envsSeen[env.Name]; !ok {
			envs = append(envs, env)
		}
	}

	if needMonitorContainer {
		monitoringConfigTemplate := `monitoring.enabled=true
monitoring.type=otlp
monitoring.options.endpoint=http://127.0.0.1:%d
monitoring.options.insecure=true`
		var bentomlOptions string
		index := -1
		for i, env := range envs {
			if env.Name == "BENTOML_CONFIG_OPTIONS" {
				bentomlOptions = env.Value
				index = i
				break
			}
		}
		if index == -1 {
			// BENOML_CONFIG_OPTIONS not defined
			bentomlOptions = fmt.Sprintf(monitoringConfigTemplate, monitorExporterPort)
			envs = append(envs, corev1.EnvVar{
				Name:  "BENTOML_CONFIG_OPTIONS",
				Value: bentomlOptions,
			})
		} else if !strings.Contains(bentomlOptions, "monitoring") {
			// monitoring config not defined
			envs = append(envs[:index], envs[index+1:]...)
			bentomlOptions = strings.TrimSpace(bentomlOptions) // ' ' -> ''
			if bentomlOptions != "" {
				bentomlOptions += "\n"
			}
			bentomlOptions += fmt.Sprintf(monitoringConfigTemplate, monitorExporterPort)
			envs = append(envs, corev1.EnvVar{
				Name:  "BENTOML_CONFIG_OPTIONS",
				Value: bentomlOptions,
			})
		}
		// monitoring config already defined
		// do nothing
	}

	livenessProbe := &corev1.Probe{
		InitialDelaySeconds: 10,
		TimeoutSeconds:      20,
		FailureThreshold:    6,
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/livez",
				Port: intstr.FromString(commonconsts.BentoContainerPortName),
			},
		},
	}

	if opt.compoundAINimDeployment.Spec.LivenessProbe != nil {
		livenessProbe = opt.compoundAINimDeployment.Spec.LivenessProbe
	}

	readinessProbe := &corev1.Probe{
		InitialDelaySeconds: 5,
		TimeoutSeconds:      5,
		FailureThreshold:    12,
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/readyz",
				Port: intstr.FromString(commonconsts.BentoContainerPortName),
			},
		},
	}

	if opt.compoundAINimDeployment.Spec.ReadinessProbe != nil {
		readinessProbe = opt.compoundAINimDeployment.Spec.ReadinessProbe
	}

	volumes := make([]corev1.Volume, 0)
	volumeMounts := make([]corev1.VolumeMount, 0)

	args := make([]string, 0)

	args = append(args, "uv", "run", "compoundai", "start")

	if opt.compoundAINimDeployment.Spec.ServiceName != "" {
		args = append(args, []string{"--service-name", opt.compoundAINimDeployment.Spec.ServiceName}...)
	}

	if len(opt.compoundAINimDeployment.Spec.ExternalServices) > 0 {
		serviceSuffix := fmt.Sprintf("%s.svc.cluster.local:3000", opt.compoundAINimDeployment.Namespace)
		keys := make([]string, 0, len(opt.compoundAINimDeployment.Spec.ExternalServices))

		for key := range opt.compoundAINimDeployment.Spec.ExternalServices {
			keys = append(keys, key)
		}

		sort.Strings(keys)
		for _, key := range keys {
			service := opt.compoundAINimDeployment.Spec.ExternalServices[key]

			// Check if DeploymentSelectorKey is not "name"
			if service.DeploymentSelectorKey == "name" {
				dependsFlag := fmt.Sprintf("--depends \"%s=http://%s.%s\"", key, service.DeploymentSelectorValue, serviceSuffix)
				args = append(args, dependsFlag)
			} else if service.DeploymentSelectorKey == "nova" {
				dependsFlag := fmt.Sprintf("--depends \"%s=nova://%s\"", key, service.DeploymentSelectorValue)
				args = append(args, dependsFlag)
			} else {
				return nil, errors.Errorf("DeploymentSelectorKey '%s' not supported. Only 'name' and 'nova' are supported", service.DeploymentSelectorKey)
			}
		}
	}

	yataiResources := opt.compoundAINimDeployment.Spec.Resources

	resources, err := getResourcesConfig(yataiResources)
	if err != nil {
		err = errors.Wrap(err, "failed to get resources config")
		return nil, err
	}

	sharedMemorySizeLimit := resource.MustParse("64Mi")
	memoryLimit := resources.Limits[corev1.ResourceMemory]
	if !memoryLimit.IsZero() {
		sharedMemorySizeLimit.SetMilli(memoryLimit.MilliValue() / 2)
	}

	volumes = append(volumes, corev1.Volume{
		Name: KubeValueNameSharedMemory,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{
				Medium:    corev1.StorageMediumMemory,
				SizeLimit: &sharedMemorySizeLimit,
			},
		},
	})
	volumeMounts = append(volumeMounts, corev1.VolumeMount{
		Name:      KubeValueNameSharedMemory,
		MountPath: "/dev/shm",
	})
	if opt.compoundAINimDeployment.Spec.PVC != nil {
		volumes = append(volumes, corev1.Volume{
			Name: getPvcName(opt.compoundAINimDeployment, opt.compoundAINimDeployment.Spec.PVC.Name),
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: getPvcName(opt.compoundAINimDeployment, opt.compoundAINimDeployment.Spec.PVC.Name),
				},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      getPvcName(opt.compoundAINimDeployment, opt.compoundAINimDeployment.Spec.PVC.Name),
			MountPath: *opt.compoundAINimDeployment.Spec.PVC.MountPoint,
		})
	}

	imageName := opt.compoundAINim.Spec.Image

	var securityContext *corev1.SecurityContext
	var mainContainerSecurityContext *corev1.SecurityContext

	enableRestrictedSecurityContext := os.Getenv("ENABLE_RESTRICTED_SECURITY_CONTEXT") == "true"
	if enableRestrictedSecurityContext {
		securityContext = &corev1.SecurityContext{
			AllowPrivilegeEscalation: ptr.To(false),
			RunAsNonRoot:             ptr.To(true),
			RunAsUser:                ptr.To(int64(1000)),
			RunAsGroup:               ptr.To(int64(1000)),
			SeccompProfile: &corev1.SeccompProfile{
				Type: corev1.SeccompProfileTypeRuntimeDefault,
			},
			Capabilities: &corev1.Capabilities{
				Drop: []corev1.Capability{"ALL"},
			},
		}
		mainContainerSecurityContext = securityContext.DeepCopy()
		mainContainerSecurityContext.RunAsUser = ptr.To(int64(1034))
	}

	containers := make([]corev1.Container, 0, 2)

	// TODO: Temporarily disabling probes
	container := corev1.Container{
		Name:           "main",
		Image:          imageName,
		Command:        []string{"sh", "-c"},
		Args:           []string{strings.Join(args, " ")},
		LivenessProbe:  livenessProbe,
		ReadinessProbe: readinessProbe,
		Resources:      resources,
		Env:            envs,
		TTY:            true,
		Stdin:          true,
		VolumeMounts:   volumeMounts,
		Ports: []corev1.ContainerPort{
			{
				Protocol:      corev1.ProtocolTCP,
				Name:          commonconsts.BentoContainerPortName,
				ContainerPort: int32(containerPort), // nolint: gosec
			},
		},
		SecurityContext: mainContainerSecurityContext,
	}

	if opt.compoundAINimDeployment.Spec.EnvFromSecret != nil {
		container.EnvFrom = []corev1.EnvFromSource{
			{
				SecretRef: &corev1.SecretEnvSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: *opt.compoundAINimDeployment.Spec.EnvFromSecret,
					},
				},
			},
		}
	}

	if resourceAnnotations["yatai.ai/enable-container-privileged"] == commonconsts.KubeLabelValueTrue {
		if container.SecurityContext == nil {
			container.SecurityContext = &corev1.SecurityContext{}
		}
		container.SecurityContext.Privileged = &[]bool{true}[0]
	}

	if resourceAnnotations["yatai.ai/enable-container-ptrace"] == commonconsts.KubeLabelValueTrue {
		if container.SecurityContext == nil {
			container.SecurityContext = &corev1.SecurityContext{}
		}
		container.SecurityContext.Capabilities = &corev1.Capabilities{
			Add: []corev1.Capability{"SYS_PTRACE"},
		}
	}

	if resourceAnnotations["yatai.ai/run-container-as-root"] == commonconsts.KubeLabelValueTrue {
		if container.SecurityContext == nil {
			container.SecurityContext = &corev1.SecurityContext{}
		}
		container.SecurityContext.RunAsUser = &[]int64{0}[0]
	}

	containers = append(containers, container)

	lastPort++
	metricsPort := lastPort

	containers = append(containers, corev1.Container{
		Name:  "metrics-transformer",
		Image: commonconfig.GetInternalImages().MetricsTransformer,
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("10m"),
				corev1.ResourceMemory: resource.MustParse("10Mi"),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("100Mi"),
			},
		},
		ReadinessProbe: &corev1.Probe{
			InitialDelaySeconds: 5,
			TimeoutSeconds:      5,
			FailureThreshold:    10,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/healthz",
					Port: intstr.FromString("metrics"),
				},
			},
		},
		LivenessProbe: &corev1.Probe{
			InitialDelaySeconds: 5,
			TimeoutSeconds:      5,
			FailureThreshold:    10,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/healthz",
					Port: intstr.FromString("metrics"),
				},
			},
		},
		Env: []corev1.EnvVar{
			{
				Name:  "BENTOML_SERVER_HOST",
				Value: "localhost",
			},
			{
				Name:  "BENTOML_SERVER_PORT",
				Value: fmt.Sprintf("%d", containerPort),
			},
			{
				Name:  "PORT",
				Value: fmt.Sprintf("%d", metricsPort),
			},
			{
				Name:  "OLD_METRICS_PREFIX",
				Value: fmt.Sprintf("BENTOML_%s_", strings.ReplaceAll(compoundAINimRepositoryName, "-", ":")),
			},
			{
				Name:  "NEW_METRICS_PREFIX",
				Value: "BENTOML_",
			},
		},
		Ports: []corev1.ContainerPort{
			{
				Protocol:      corev1.ProtocolTCP,
				Name:          "metrics",
				ContainerPort: int32(metricsPort),
			},
		},
		SecurityContext: securityContext,
	})

	lastPort++
	proxyPort := lastPort

	proxyResourcesRequestsCPUStr := resourceAnnotations[KubeAnnotationYataiProxySidecarResourcesRequestsCPU]
	if proxyResourcesRequestsCPUStr == "" {
		proxyResourcesRequestsCPUStr = "100m"
	}
	var proxyResourcesRequestsCPU resource.Quantity
	proxyResourcesRequestsCPU, err = resource.ParseQuantity(proxyResourcesRequestsCPUStr)
	if err != nil {
		err = errors.Wrapf(err, "failed to parse proxy sidecar resources requests cpu: %s", proxyResourcesRequestsCPUStr)
		return nil, err
	}
	proxyResourcesRequestsMemoryStr := resourceAnnotations[KubeAnnotationYataiProxySidecarResourcesRequestsMemory]
	if proxyResourcesRequestsMemoryStr == "" {
		proxyResourcesRequestsMemoryStr = "200Mi"
	}
	var proxyResourcesRequestsMemory resource.Quantity
	proxyResourcesRequestsMemory, err = resource.ParseQuantity(proxyResourcesRequestsMemoryStr)
	if err != nil {
		err = errors.Wrapf(err, "failed to parse proxy sidecar resources requests memory: %s", proxyResourcesRequestsMemoryStr)
		return nil, err
	}
	proxyResourcesLimitsCPUStr := resourceAnnotations[KubeAnnotationYataiProxySidecarResourcesLimitsCPU]
	if proxyResourcesLimitsCPUStr == "" {
		proxyResourcesLimitsCPUStr = "300m"
	}
	var proxyResourcesLimitsCPU resource.Quantity
	proxyResourcesLimitsCPU, err = resource.ParseQuantity(proxyResourcesLimitsCPUStr)
	if err != nil {
		err = errors.Wrapf(err, "failed to parse proxy sidecar resources limits cpu: %s", proxyResourcesLimitsCPUStr)
		return nil, err
	}
	proxyResourcesLimitsMemoryStr := resourceAnnotations[KubeAnnotationYataiProxySidecarResourcesLimitsMemory]
	if proxyResourcesLimitsMemoryStr == "" {
		proxyResourcesLimitsMemoryStr = "1000Mi"
	}
	var proxyResourcesLimitsMemory resource.Quantity
	proxyResourcesLimitsMemory, err = resource.ParseQuantity(proxyResourcesLimitsMemoryStr)
	if err != nil {
		err = errors.Wrapf(err, "failed to parse proxy sidecar resources limits memory: %s", proxyResourcesLimitsMemoryStr)
		return nil, err
	}
	var envoyConfigContent string
	if opt.isStealingTrafficDebugModeEnabled {
		productionServiceName := r.getServiceName(opt.compoundAINimDeployment, opt.compoundAINim, false)
		envoyConfigContent, err = envoy.GenerateEnvoyConfigurationContent(envoy.CreateEnvoyConfig{
			ListenPort:              proxyPort,
			DebugHeaderName:         HeaderNameDebug,
			DebugHeaderValue:        commonconsts.KubeLabelValueTrue,
			DebugServerAddress:      "localhost",
			DebugServerPort:         containerPort,
			ProductionServerAddress: fmt.Sprintf("%s.%s.svc.cluster.local", productionServiceName, opt.compoundAINimDeployment.Namespace),
			ProductionServerPort:    ServicePortHTTPNonProxy,
		})
	} else {
		debugServiceName := r.getServiceName(opt.compoundAINimDeployment, opt.compoundAINim, true)
		envoyConfigContent, err = envoy.GenerateEnvoyConfigurationContent(envoy.CreateEnvoyConfig{
			ListenPort:              proxyPort,
			DebugHeaderName:         HeaderNameDebug,
			DebugHeaderValue:        commonconsts.KubeLabelValueTrue,
			DebugServerAddress:      fmt.Sprintf("%s.%s.svc.cluster.local", debugServiceName, opt.compoundAINimDeployment.Namespace),
			DebugServerPort:         ServicePortHTTPNonProxy,
			ProductionServerAddress: "localhost",
			ProductionServerPort:    containerPort,
		})
	}
	if err != nil {
		err = errors.Wrapf(err, "failed to generate envoy configuration content")
		return nil, err
	}
	envoyConfigConfigMapName := fmt.Sprintf("%s-envoy-config", kubeName)
	envoyConfigConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      envoyConfigConfigMapName,
			Namespace: opt.compoundAINimDeployment.Namespace,
		},
		Data: map[string]string{
			"envoy.yaml": envoyConfigContent,
		},
	}
	err = ctrl.SetControllerReference(opt.compoundAINimDeployment, envoyConfigConfigMap, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "failed to set controller reference for envoy config config map")
		return nil, err
	}
	_, err = ctrl.CreateOrUpdate(ctx, r.Client, envoyConfigConfigMap, func() error {
		envoyConfigConfigMap.Data["envoy.yaml"] = envoyConfigContent
		return nil
	})
	if err != nil {
		err = errors.Wrapf(err, "failed to create or update envoy config configmap")
		return nil, err
	}
	volumes = append(volumes, corev1.Volume{
		Name: "envoy-config",
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: envoyConfigConfigMapName,
				},
			},
		},
	})
	proxyImage := "quay.io/bentoml/bentoml-proxy:0.0.1"
	proxyImage_ := os.Getenv("INTERNAL_IMAGES_PROXY")
	if proxyImage_ != "" {
		proxyImage = proxyImage_
	}
	containers = append(containers, corev1.Container{
		Name:  "proxy",
		Image: proxyImage,
		Command: []string{
			"envoy",
			"--config-path",
			"/etc/envoy/envoy.yaml",
		},
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      "envoy-config",
				MountPath: "/etc/envoy",
			},
		},
		Ports: []corev1.ContainerPort{
			{
				Name:          ContainerPortNameHTTPProxy,
				ContainerPort: int32(proxyPort),
				Protocol:      corev1.ProtocolTCP,
			},
		},
		ReadinessProbe: &corev1.Probe{
			InitialDelaySeconds: 5,
			TimeoutSeconds:      5,
			FailureThreshold:    10,
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{
						"sh",
						"-c",
						"curl -s localhost:9901/server_info | grep state | grep -q LIVE",
					},
				},
			},
		},
		LivenessProbe: &corev1.Probe{
			InitialDelaySeconds: 5,
			TimeoutSeconds:      5,
			FailureThreshold:    10,
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{
						"sh",
						"-c",
						"curl -s localhost:9901/server_info | grep state | grep -q LIVE",
					},
				},
			},
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    proxyResourcesRequestsCPU,
				corev1.ResourceMemory: proxyResourcesRequestsMemory,
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    proxyResourcesLimitsCPU,
				corev1.ResourceMemory: proxyResourcesLimitsMemory,
			},
		},
		SecurityContext: securityContext,
	})

	if needMonitorContainer {
		lastPort++
		monitorExporterProbePort := lastPort

		monitorExporterImage := "quay.io/bentoml/bentoml-monitor-exporter:0.0.3"
		monitorExporterImage_ := os.Getenv("INTERNAL_IMAGES_MONITOR_EXPORTER")
		if monitorExporterImage_ != "" {
			monitorExporterImage = monitorExporterImage_
		}

		monitorOptEnvs := make([]corev1.EnvVar, 0, len(monitorExporter.Options)+len(monitorExporter.StructureOptions))
		monitorOptEnvsSeen := make(map[string]struct{})

		for _, env := range monitorExporter.StructureOptions {
			monitorOptEnvsSeen[strings.ToLower(env.Name)] = struct{}{}
			monitorOptEnvs = append(monitorOptEnvs, corev1.EnvVar{
				Name:      "FLUENTBIT_OUTPUT_OPTION_" + strings.ToUpper(env.Name),
				Value:     env.Value,
				ValueFrom: env.ValueFrom,
			})
		}

		for k, v := range monitorExporter.Options {
			if _, exists := monitorOptEnvsSeen[strings.ToLower(k)]; exists {
				continue
			}
			monitorOptEnvs = append(monitorOptEnvs, corev1.EnvVar{
				Name:  "FLUENTBIT_OUTPUT_OPTION_" + strings.ToUpper(k),
				Value: v,
			})
		}

		monitorVolumeMounts := make([]corev1.VolumeMount, 0, len(monitorExporter.Mounts))
		for idx, mount := range monitorExporter.Mounts {
			volumeName := fmt.Sprintf("monitor-exporter-%d", idx)
			volumes = append(volumes, corev1.Volume{
				Name:         volumeName,
				VolumeSource: mount.VolumeSource,
			})
			monitorVolumeMounts = append(monitorVolumeMounts, corev1.VolumeMount{
				Name:      volumeName,
				MountPath: mount.Path,
				ReadOnly:  mount.ReadOnly,
			})
		}

		containers = append(containers, corev1.Container{
			Name:         "monitor-exporter",
			Image:        monitorExporterImage,
			VolumeMounts: monitorVolumeMounts,
			Env: append([]corev1.EnvVar{
				{
					Name:  "FLUENTBIT_OTLP_PORT",
					Value: fmt.Sprint(monitorExporterPort),
				},
				{
					Name:  "FLUENTBIT_HTTP_PORT",
					Value: fmt.Sprint(monitorExporterProbePort),
				},
				{
					Name:  "FLUENTBIT_OUTPUT",
					Value: monitorExporter.Output,
				},
			}, monitorOptEnvs...),
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("100m"),
					corev1.ResourceMemory: resource.MustParse("24Mi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("1000m"),
					corev1.ResourceMemory: resource.MustParse("72Mi"),
				},
			},
			ReadinessProbe: &corev1.Probe{
				InitialDelaySeconds: 5,
				TimeoutSeconds:      5,
				FailureThreshold:    10,
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/readyz",
						Port: intstr.FromInt(monitorExporterProbePort),
					},
				},
			},
			LivenessProbe: &corev1.Probe{
				InitialDelaySeconds: 5,
				TimeoutSeconds:      5,
				FailureThreshold:    10,
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/livez",
						Port: intstr.FromInt(monitorExporterProbePort),
					},
				},
			},
			SecurityContext: securityContext,
		})
	}

	debuggerImage := "quay.io/bentoml/bento-debugger:0.0.8"
	debuggerImage_ := os.Getenv("INTERNAL_IMAGES_DEBUGGER")
	if debuggerImage_ != "" {
		debuggerImage = debuggerImage_
	}

	if opt.isStealingTrafficDebugModeEnabled || isDebugModeEnabled {
		containers = append(containers, corev1.Container{
			Name:  "debugger",
			Image: debuggerImage,
			Command: []string{
				"sleep",
				"infinity",
			},
			SecurityContext: &corev1.SecurityContext{
				Capabilities: &corev1.Capabilities{
					Add: []corev1.Capability{"SYS_PTRACE"},
				},
			},
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("100m"),
					corev1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("1000m"),
					corev1.ResourceMemory: resource.MustParse("1000Mi"),
				},
			},
			Stdin: true,
			TTY:   true,
		})
	}

	podLabels[commonconsts.KubeLabelYataiSelector] = kubeName

	podSpec := corev1.PodSpec{
		Containers: containers,
		Volumes:    volumes,
	}

	podSpec.ImagePullSecrets = opt.compoundAINim.Spec.ImagePullSecrets

	extraPodMetadata := opt.compoundAINimDeployment.Spec.ExtraPodMetadata

	if extraPodMetadata != nil {
		for k, v := range extraPodMetadata.Annotations {
			podAnnotations[k] = v
		}

		for k, v := range extraPodMetadata.Labels {
			podLabels[k] = v
		}
	}

	extraPodSpec := opt.compoundAINimDeployment.Spec.ExtraPodSpec

	if extraPodSpec != nil {
		podSpec.SchedulerName = extraPodSpec.SchedulerName
		podSpec.NodeSelector = extraPodSpec.NodeSelector
		podSpec.Affinity = extraPodSpec.Affinity
		podSpec.Tolerations = extraPodSpec.Tolerations
		podSpec.TopologySpreadConstraints = extraPodSpec.TopologySpreadConstraints
		podSpec.Containers = append(podSpec.Containers, extraPodSpec.Containers...)
		podSpec.ServiceAccountName = extraPodSpec.ServiceAccountName
	}

	if podSpec.ServiceAccountName == "" {
		serviceAccounts := &corev1.ServiceAccountList{}
		err = r.List(ctx, serviceAccounts, client.InNamespace(opt.compoundAINimDeployment.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelBentoDeploymentPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			err = errors.Wrapf(err, "failed to list service accounts in namespace %s", opt.compoundAINimDeployment.Namespace)
			return
		}
		if len(serviceAccounts.Items) > 0 {
			podSpec.ServiceAccountName = serviceAccounts.Items[0].Name
		} else {
			podSpec.ServiceAccountName = DefaultServiceAccountName
		}
	}

	if resourceAnnotations["yatai.ai/enable-host-ipc"] == commonconsts.KubeLabelValueTrue {
		podSpec.HostIPC = true
	}

	if resourceAnnotations["yatai.ai/enable-host-network"] == commonconsts.KubeLabelValueTrue {
		podSpec.HostNetwork = true
	}

	if resourceAnnotations["yatai.ai/enable-host-pid"] == commonconsts.KubeLabelValueTrue {
		podSpec.HostPID = true
	}

	if opt.isStealingTrafficDebugModeEnabled || isDebugModeEnabled {
		podSpec.ShareProcessNamespace = &[]bool{true}[0]
	}

	podTemplateSpec = &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      podLabels,
			Annotations: podAnnotations,
		},
		Spec: podSpec,
	}

	return
}

func getResourcesConfig(resources *compoundaiCommon.Resources) (corev1.ResourceRequirements, error) {
	currentResources := corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("300m"),
			corev1.ResourceMemory: resource.MustParse("500Mi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("500m"),
			corev1.ResourceMemory: resource.MustParse("1Gi"),
		},
	}

	if resources == nil {
		return currentResources, nil
	}

	if resources.Limits != nil {
		if resources.Limits.CPU != "" {
			q, err := resource.ParseQuantity(resources.Limits.CPU)
			if err != nil {
				return currentResources, errors.Wrapf(err, "parse limits cpu quantity")
			}
			if currentResources.Limits == nil {
				currentResources.Limits = make(corev1.ResourceList)
			}
			currentResources.Limits[corev1.ResourceCPU] = q
		}
		if resources.Limits.Memory != "" {
			q, err := resource.ParseQuantity(resources.Limits.Memory)
			if err != nil {
				return currentResources, errors.Wrapf(err, "parse limits memory quantity")
			}
			if currentResources.Limits == nil {
				currentResources.Limits = make(corev1.ResourceList)
			}
			currentResources.Limits[corev1.ResourceMemory] = q
		}
		if resources.Limits.GPU != "" {
			q, err := resource.ParseQuantity(resources.Limits.GPU)
			if err != nil {
				return currentResources, errors.Wrapf(err, "parse limits gpu quantity")
			}
			if currentResources.Limits == nil {
				currentResources.Limits = make(corev1.ResourceList)
			}
			currentResources.Limits[commonconsts.KubeResourceGPUNvidia] = q
		}
		for k, v := range resources.Limits.Custom {
			q, err := resource.ParseQuantity(v)
			if err != nil {
				return currentResources, errors.Wrapf(err, "parse limits %s quantity", k)
			}
			if currentResources.Limits == nil {
				currentResources.Limits = make(corev1.ResourceList)
			}
			currentResources.Limits[corev1.ResourceName(k)] = q
		}
	}
	if resources.Requests != nil {
		if resources.Requests.CPU != "" {
			q, err := resource.ParseQuantity(resources.Requests.CPU)
			if err != nil {
				return currentResources, errors.Wrapf(err, "parse requests cpu quantity")
			}
			if currentResources.Requests == nil {
				currentResources.Requests = make(corev1.ResourceList)
			}
			currentResources.Requests[corev1.ResourceCPU] = q
		}
		if resources.Requests.Memory != "" {
			q, err := resource.ParseQuantity(resources.Requests.Memory)
			if err != nil {
				return currentResources, errors.Wrapf(err, "parse requests memory quantity")
			}
			if currentResources.Requests == nil {
				currentResources.Requests = make(corev1.ResourceList)
			}
			currentResources.Requests[corev1.ResourceMemory] = q
		}
		for k, v := range resources.Requests.Custom {
			q, err := resource.ParseQuantity(v)
			if err != nil {
				return currentResources, errors.Wrapf(err, "parse requests %s quantity", k)
			}
			if currentResources.Requests == nil {
				currentResources.Requests = make(corev1.ResourceList)
			}
			currentResources.Requests[corev1.ResourceName(k)] = q
		}
	}
	return currentResources, nil
}

type generateServiceOption struct {
	compoundAINimDeployment                 *v1alpha1.CompoundAINimDeployment
	compoundAINim                           *v1alpha1.CompoundAINim
	isStealingTrafficDebugModeEnabled       bool
	isDebugPodReceiveProductionTraffic      bool
	containsStealingTrafficDebugModeEnabled bool
	isGenericService                        bool
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) generateService(opt generateServiceOption) (kubeService *corev1.Service, err error) {
	var kubeName string
	if opt.isGenericService {
		kubeName = r.getGenericServiceName(opt.compoundAINimDeployment, opt.compoundAINim)
	} else {
		kubeName = r.getServiceName(opt.compoundAINimDeployment, opt.compoundAINim, opt.isStealingTrafficDebugModeEnabled)
	}

	labels := r.getKubeLabels(opt.compoundAINimDeployment, opt.compoundAINim)

	selector := make(map[string]string)

	for k, v := range labels {
		selector[k] = v
	}

	if opt.isStealingTrafficDebugModeEnabled {
		selector[commonconsts.KubeLabelYataiBentoDeploymentTargetType] = DeploymentTargetTypeDebug
	}

	targetPort := intstr.FromString(commonconsts.BentoContainerPortName)
	if opt.isGenericService {
		delete(selector, commonconsts.KubeLabelYataiBentoDeploymentTargetType)
		if opt.containsStealingTrafficDebugModeEnabled {
			targetPort = intstr.FromString(ContainerPortNameHTTPProxy)
		}
	}

	spec := corev1.ServiceSpec{
		Selector: selector,
		Ports: []corev1.ServicePort{
			{
				Name:       commonconsts.BentoServicePortName,
				Port:       commonconsts.BentoServicePort,
				TargetPort: targetPort,
				Protocol:   corev1.ProtocolTCP,
			},
			{
				Name:       ServicePortNameHTTPNonProxy,
				Port:       int32(ServicePortHTTPNonProxy),
				TargetPort: intstr.FromString(commonconsts.BentoContainerPortName),
				Protocol:   corev1.ProtocolTCP,
			},
		},
	}

	annotations := r.getKubeAnnotations(opt.compoundAINimDeployment, opt.compoundAINim)

	kubeNs := opt.compoundAINimDeployment.Namespace

	kubeService = &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        kubeName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: spec,
	}

	err = ctrl.SetControllerReference(opt.compoundAINimDeployment, kubeService, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set controller reference for service %s", kubeService.Name)
		return
	}

	return
}

func (r *CompoundAINimDeploymentReconciler) generateIngressHost(ctx context.Context, compoundAINimDeployment *v1alpha1.CompoundAINimDeployment) (string, error) {
	return r.generateDefaultHostname(ctx, compoundAINimDeployment)
}

var cachedDomainSuffix *string

func (r *CompoundAINimDeploymentReconciler) generateDefaultHostname(ctx context.Context, compoundAINimDeployment *v1alpha1.CompoundAINimDeployment) (string, error) {
	var domainSuffix string

	if cachedDomainSuffix != nil {
		domainSuffix = *cachedDomainSuffix
	} else {
		restConfig := config.GetConfigOrDie()
		clientset, err := kubernetes.NewForConfig(restConfig)
		if err != nil {
			return "", errors.Wrapf(err, "create kubernetes clientset")
		}

		domainSuffix, err = system.GetDomainSuffix(ctx, func(ctx context.Context, namespace, name string) (*corev1.ConfigMap, error) {
			configmap, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
			return configmap, errors.Wrap(err, "get configmap")
		}, clientset)
		if err != nil {
			return "", errors.Wrapf(err, "get domain suffix")
		}

		cachedDomainSuffix = &domainSuffix
	}
	return fmt.Sprintf("%s-%s.%s", compoundAINimDeployment.Name, compoundAINimDeployment.Namespace, domainSuffix), nil
}

type TLSModeOpt string

const (
	TLSModeNone   TLSModeOpt = "none"
	TLSModeAuto   TLSModeOpt = "auto"
	TLSModeStatic TLSModeOpt = "static"
)

type IngressConfig struct {
	ClassName           *string
	Annotations         map[string]string
	Path                string
	PathType            networkingv1.PathType
	TLSMode             TLSModeOpt
	StaticTLSSecretName string
}

var cachedIngressConfig *IngressConfig

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) GetIngressConfig(ctx context.Context) (ingressConfig *IngressConfig, err error) {
	if cachedIngressConfig != nil {
		ingressConfig = cachedIngressConfig
		return
	}

	restConfig := config.GetConfigOrDie()
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		err = errors.Wrapf(err, "create kubernetes clientset")
		return
	}

	configMap, err := system.GetNetworkConfigConfigMap(ctx, func(ctx context.Context, namespace, name string) (*corev1.ConfigMap, error) {
		configmap, err := clientset.CoreV1().ConfigMaps(namespace).Get(ctx, name, metav1.GetOptions{})
		return configmap, errors.Wrap(err, "get network config configmap")
	})
	if err != nil {
		err = errors.Wrapf(err, "failed to get configmap %s", commonconsts.KubeConfigMapNameNetworkConfig)
		return
	}

	var className *string

	className_ := strings.TrimSpace(configMap.Data[commonconsts.KubeConfigMapKeyNetworkConfigIngressClass])
	if className_ != "" {
		className = &className_
	}

	annotations := make(map[string]string)

	annotations_ := strings.TrimSpace(configMap.Data[commonconsts.KubeConfigMapKeyNetworkConfigIngressAnnotations])
	if annotations_ != "" {
		err = json.Unmarshal([]byte(annotations_), &annotations)
		if err != nil {
			err = errors.Wrapf(err, "failed to json unmarshal %s in configmap %s: %s", commonconsts.KubeConfigMapKeyNetworkConfigIngressAnnotations, commonconsts.KubeConfigMapNameNetworkConfig, annotations_)
			return
		}
	}

	path := strings.TrimSpace(configMap.Data["ingress-path"])
	if path == "" {
		path = "/"
	}

	pathType := networkingv1.PathTypeImplementationSpecific

	pathType_ := strings.TrimSpace(configMap.Data["ingress-path-type"])
	if pathType_ != "" {
		pathType = networkingv1.PathType(pathType_)
	}

	tlsMode := TLSModeNone
	tlsModeStr := strings.TrimSpace(configMap.Data["ingress-tls-mode"])
	if tlsModeStr != "" && tlsModeStr != "none" {
		if tlsModeStr == "auto" || tlsModeStr == "static" {
			tlsMode = TLSModeOpt(tlsModeStr)
		} else {
			fmt.Println("Invalid TLS mode:", tlsModeStr)
			err = errors.Wrapf(err, "Invalid TLS mode: %s", tlsModeStr)
			return
		}
	}

	staticTLSSecretName := strings.TrimSpace(configMap.Data["ingress-static-tls-secret-name"])
	if tlsMode == TLSModeStatic && staticTLSSecretName == "" {
		err = errors.Wrapf(err, "TLS mode is static but ingress-static-tls-secret isn't set")
		return
	}

	ingressConfig = &IngressConfig{
		ClassName:           className,
		Annotations:         annotations,
		Path:                path,
		PathType:            pathType,
		TLSMode:             tlsMode,
		StaticTLSSecretName: staticTLSSecretName,
	}

	cachedIngressConfig = ingressConfig

	return
}

type generateIngressesOption struct {
	yataiClient             **yataiclient.YataiClient
	compoundAINimDeployment *v1alpha1.CompoundAINimDeployment
	compoundAINim           *v1alpha1.CompoundAINim
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) generateIngresses(ctx context.Context, opt generateIngressesOption) (ingresses []*networkingv1.Ingress, err error) {
	compoundAINimRepositoryName, compoundAINimVersion := getCompoundAINimRepositoryNameAndCompoundAINimVersion(opt.compoundAINim)
	compoundAINimDeployment := opt.compoundAINimDeployment
	compoundAINim := opt.compoundAINim

	kubeName := r.getKubeName(compoundAINimDeployment, compoundAINim, false)

	r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GenerateIngressHost", "Generating hostname for ingress")
	internalHost, err := r.generateIngressHost(ctx, compoundAINimDeployment)
	if err != nil {
		r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeWarning, "GenerateIngressHost", "Failed to generate hostname for ingress: %v", err)
		return
	}
	r.Recorder.Eventf(compoundAINimDeployment, corev1.EventTypeNormal, "GenerateIngressHost", "Generated hostname for ingress: %s", internalHost)

	annotations := r.getKubeAnnotations(compoundAINimDeployment, compoundAINim)

	tag := fmt.Sprintf("%s:%s", compoundAINimRepositoryName, compoundAINimVersion)
	orgName := "unknown"

	annotations["nginx.ingress.kubernetes.io/configuration-snippet"] = fmt.Sprintf(`
more_set_headers "X-Powered-By: Yatai";
more_set_headers "X-Yatai-Org-Name: %s";
more_set_headers "X-Yatai-Bento: %s";
`, orgName, tag)

	annotations["nginx.ingress.kubernetes.io/ssl-redirect"] = "false"

	labels := r.getKubeLabels(compoundAINimDeployment, compoundAINim)

	kubeNs := compoundAINimDeployment.Namespace

	ingressConfig, err := r.GetIngressConfig(ctx)
	if err != nil {
		err = errors.Wrapf(err, "get ingress config")
		return
	}

	ingressClassName := ingressConfig.ClassName
	ingressAnnotations := ingressConfig.Annotations
	ingressPath := ingressConfig.Path
	ingressPathType := ingressConfig.PathType
	ingressTLSMode := ingressConfig.TLSMode
	ingressStaticTLSSecretName := ingressConfig.StaticTLSSecretName

	for k, v := range ingressAnnotations {
		annotations[k] = v
	}

	for k, v := range opt.compoundAINimDeployment.Spec.Ingress.Annotations {
		annotations[k] = v
	}

	for k, v := range opt.compoundAINimDeployment.Spec.Ingress.Labels {
		labels[k] = v
	}

	var tls []networkingv1.IngressTLS

	// set default tls from network configmap
	switch ingressTLSMode {
	case TLSModeNone:
	case TLSModeAuto:
		tls = make([]networkingv1.IngressTLS, 0, 1)
		tls = append(tls, networkingv1.IngressTLS{
			Hosts:      []string{internalHost},
			SecretName: kubeName,
		})

	case TLSModeStatic:
		tls = make([]networkingv1.IngressTLS, 0, 1)
		tls = append(tls, networkingv1.IngressTLS{
			Hosts:      []string{internalHost},
			SecretName: ingressStaticTLSSecretName,
		})
	default:
		err = errors.Wrapf(err, "TLS mode is invalid: %s", ingressTLSMode)
		return
	}

	// override default tls if CompoundAINimDeployment defines its own tls section
	if opt.compoundAINimDeployment.Spec.Ingress.TLS != nil && opt.compoundAINimDeployment.Spec.Ingress.TLS.SecretName != "" {
		tls = make([]networkingv1.IngressTLS, 0, 1)
		tls = append(tls, networkingv1.IngressTLS{
			Hosts:      []string{internalHost},
			SecretName: opt.compoundAINimDeployment.Spec.Ingress.TLS.SecretName,
		})
	}

	serviceName := r.getGenericServiceName(compoundAINimDeployment, compoundAINim)

	interIng := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:        kubeName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: networkingv1.IngressSpec{
			IngressClassName: ingressClassName,
			TLS:              tls,
			Rules: []networkingv1.IngressRule{
				{
					Host: internalHost,
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									Path:     ingressPath,
									PathType: &ingressPathType,
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: serviceName,
											Port: networkingv1.ServiceBackendPort{
												Name: commonconsts.BentoServicePortName,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	err = ctrl.SetControllerReference(compoundAINimDeployment, interIng, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set ingress %s controller reference", interIng.Name)
		return
	}

	ings := []*networkingv1.Ingress{interIng}

	return ings, err
}

var cachedCompoundAINimDeploymentNamespaces *[]string

func (r *CompoundAINimDeploymentReconciler) doCleanUpAbandonedRunnerServices() error {
	logs := log.Log.WithValues("func", "doCleanUpAbandonedRunnerServices")
	logs.Info("start cleaning up abandoned runner services")
	ctx, cancel := context.WithTimeout(context.TODO(), time.Minute*10)
	defer cancel()

	var compoundAINimDeploymentNamespaces []string

	if cachedCompoundAINimDeploymentNamespaces != nil {
		compoundAINimDeploymentNamespaces = *cachedCompoundAINimDeploymentNamespaces
	} else {
		restConfig := config.GetConfigOrDie()
		clientset, err := kubernetes.NewForConfig(restConfig)
		if err != nil {
			return errors.Wrapf(err, "create kubernetes clientset")
		}

		compoundAINimDeploymentNamespaces, err = commonconfig.GetBentoDeploymentNamespaces(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
			secret, err := clientset.CoreV1().Secrets(namespace).Get(ctx, name, metav1.GetOptions{})
			return secret, errors.Wrap(err, "get secret")
		})
		if err != nil {
			err = errors.Wrapf(err, "get compoundAINim deployment namespaces")
			return err
		}

		cachedCompoundAINimDeploymentNamespaces = &compoundAINimDeploymentNamespaces
	}

	for _, compoundAINimDeploymentNamespace := range compoundAINimDeploymentNamespaces {
		serviceList := &corev1.ServiceList{}
		serviceListOpts := []client.ListOption{
			client.HasLabels{commonconsts.KubeLabelYataiBentoDeploymentRunner},
			client.InNamespace(compoundAINimDeploymentNamespace),
		}
		err := r.List(ctx, serviceList, serviceListOpts...)
		if err != nil {
			return errors.Wrap(err, "list services")
		}
		for _, service := range serviceList.Items {
			service := service
			podList := &corev1.PodList{}
			podListOpts := []client.ListOption{
				client.InNamespace(service.Namespace),
				client.MatchingLabels(service.Spec.Selector),
			}
			err := r.List(ctx, podList, podListOpts...)
			if err != nil {
				return errors.Wrap(err, "list pods")
			}
			if len(podList.Items) > 0 {
				continue
			}
			createdAt := service.ObjectMeta.CreationTimestamp
			if time.Since(createdAt.Time) < time.Minute*3 {
				continue
			}
			logs.Info("deleting abandoned runner service", "name", service.Name, "namespace", service.Namespace)
			err = r.Delete(ctx, &service)
			if err != nil {
				return errors.Wrapf(err, "delete service %s", service.Name)
			}
		}
	}
	logs.Info("finished cleaning up abandoned runner services")
	return nil
}

func (r *CompoundAINimDeploymentReconciler) cleanUpAbandonedRunnerServices() {
	logs := log.Log.WithValues("func", "cleanUpAbandonedRunnerServices")
	err := r.doCleanUpAbandonedRunnerServices()
	if err != nil {
		logs.Error(err, "cleanUpAbandonedRunnerServices")
	}
	ticker := time.NewTicker(time.Second * 30)
	for range ticker.C {
		err := r.doCleanUpAbandonedRunnerServices()
		if err != nil {
			logs.Error(err, "cleanUpAbandonedRunnerServices")
		}
	}
}

//nolint:nakedret
func (r *CompoundAINimDeploymentReconciler) doRegisterCompoundComponent() (err error) {
	logs := log.Log.WithValues("func", "doRegisterYataiComponent")

	ctx, cancel := context.WithTimeout(context.TODO(), time.Minute*5)
	defer cancel()

	logs.Info("getting yatai client")
	yataiClient, clusterName, err := r.getYataiClient(ctx)
	if err != nil {
		err = errors.Wrap(err, "get yatai client")
		return
	}

	if yataiClient == nil {
		logs.Info("yatai client is nil")
		return
	}

	yataiClient_ := *yataiClient

	namespace, err := commonconfig.GetYataiDeploymentNamespace(ctx, func(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
		secret := &corev1.Secret{}
		err := r.Client.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, secret)
		return secret, errors.Wrap(err, "get secret")
	})
	if err != nil {
		err = errors.Wrap(err, "get yatai deployment namespace")
		return
	}

	_, err = yataiClient_.RegisterYataiComponent(ctx, *clusterName, &schemasv1.RegisterYataiComponentSchema{
		Name:          modelschemas.YataiComponentNameDeployment,
		KubeNamespace: namespace,
		Version:       version.Version,
		SelectorLabels: map[string]string{
			"app.kubernetes.io/name": "yatai-deployment",
		},
		Manifest: &modelschemas.YataiComponentManifestSchema{
			SelectorLabels: map[string]string{
				"app.kubernetes.io/name": "yatai-deployment",
			},
			LatestCRDVersion: "v2alpha1",
		},
	})

	return err
}

func (r *CompoundAINimDeploymentReconciler) registerCompoundComponent() {
	logs := log.Log.WithValues("func", "registerYataiComponent")
	err := r.doRegisterCompoundComponent()
	if err != nil {
		logs.Error(err, "registerYataiComponent")
	}
	ticker := time.NewTicker(time.Minute * 5)
	for range ticker.C {
		err := r.doRegisterCompoundComponent()
		if err != nil {
			logs.Error(err, "registerYataiComponent")
		}
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *CompoundAINimDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	logs := log.Log.WithValues("func", "SetupWithManager")

	if os.Getenv("DISABLE_CLEANUP_ABANDONED_RUNNER_SERVICES") != commonconsts.KubeLabelValueTrue {
		go r.cleanUpAbandonedRunnerServices()
	} else {
		logs.Info("cleanup abandoned runner services is disabled")
	}

	if os.Getenv("DISABLE_YATAI_COMPONENT_REGISTRATION") != commonconsts.KubeLabelValueTrue {
		go r.registerCompoundComponent()
	} else {
		logs.Info("yatai component registration is disabled")
	}

	m := ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.CompoundAINimDeployment{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.Deployment{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&corev1.Service{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&networkingv1beta1.VirtualService{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&networkingv1.Ingress{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&corev1.PersistentVolumeClaim{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Watches(&v1alpha1.CompoundAINimRequest{}, handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, compoundAINimRequest client.Object) []reconcile.Request {
			reqs := make([]reconcile.Request, 0)
			logs := log.Log.WithValues("func", "Watches", "kind", "CompoundAINimRequest", "name", compoundAINimRequest.GetName(), "namespace", compoundAINimRequest.GetNamespace())
			logs.Info("Triggering reconciliation for CompoundAINimRequest", "CompoundAINimRequestName", compoundAINimRequest.GetName(), "Namespace", compoundAINimRequest.GetNamespace())
			compoundAINim := &v1alpha1.CompoundAINim{}
			err := r.Get(context.Background(), types.NamespacedName{
				Name:      compoundAINimRequest.GetName(),
				Namespace: compoundAINimRequest.GetNamespace(),
			}, compoundAINim)
			compoundAINimIsNotFound := k8serrors.IsNotFound(err)
			if err != nil && !compoundAINimIsNotFound {
				logs.Info("Failed to get CompoundAINim", "name", compoundAINimRequest.GetName(), "namespace", compoundAINimRequest.GetNamespace(), "error", err)
				return reqs
			}
			if !compoundAINimIsNotFound {
				logs.Info("CompoundAINim found, skipping enqueue as it's already present", "CompoundAINimName", compoundAINimRequest.GetName())
				return reqs
			}
			compoundAINimDeployments := &v1alpha1.CompoundAINimDeploymentList{}
			err = r.List(context.Background(), compoundAINimDeployments, &client.ListOptions{
				Namespace: compoundAINimRequest.GetNamespace(),
			})
			if err != nil {
				logs.Info("Failed to list CompoundAINimDeployments", "Namespace", compoundAINimRequest.GetNamespace(), "error", err)
				return reqs
			}
			for _, compoundAINimDeployment := range compoundAINimDeployments.Items {
				compoundAINimDeployment := compoundAINimDeployment
				if compoundAINimDeployment.Spec.CompoundAINim == compoundAINimRequest.GetName() {
					reqs = append(reqs, reconcile.Request{
						NamespacedName: client.ObjectKeyFromObject(&compoundAINimDeployment),
					})
				}
			}
			// Log the list of CompoundAINimDeployments being enqueued for reconciliation
			logs.Info("Enqueuing CompoundAINimDeployments for reconciliation", "ReconcileRequests", reqs)
			return reqs
		})).WithEventFilter(controller_common.EphemeralDeploymentEventFilter(r.Config)).
		Watches(&v1alpha1.CompoundAINim{}, handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, compoundAINim client.Object) []reconcile.Request {
			logs := log.Log.WithValues("func", "Watches", "kind", "CompoundAINim", "name", compoundAINim.GetName(), "namespace", compoundAINim.GetNamespace())
			logs.Info("Triggering reconciliation for CompoundAINim", "CompoundAINimName", compoundAINim.GetName(), "Namespace", compoundAINim.GetNamespace())
			compoundAINimDeployments := &v1alpha1.CompoundAINimDeploymentList{}
			err := r.List(context.Background(), compoundAINimDeployments, &client.ListOptions{
				Namespace: compoundAINim.GetNamespace(),
			})
			if err != nil {
				logs.Info("Failed to list CompoundAINimDeployments", "Namespace", compoundAINim.GetNamespace(), "error", err)
				return []reconcile.Request{}
			}
			reqs := make([]reconcile.Request, 0)
			for _, compoundAINimDeployment := range compoundAINimDeployments.Items {
				compoundAINimDeployment := compoundAINimDeployment
				if compoundAINimDeployment.Spec.CompoundAINim == compoundAINim.GetName() {
					reqs = append(reqs, reconcile.Request{
						NamespacedName: client.ObjectKeyFromObject(&compoundAINimDeployment),
					})
				}
			}
			// Log the list of CompoundAINimDeployments being enqueued for reconciliation
			logs.Info("Enqueuing CompoundAINimDeployments for reconciliation", "ReconcileRequests", reqs)
			return reqs
		}))

	m.Owns(&autoscalingv2.HorizontalPodAutoscaler{})
	return m.Complete(r)
}

//nolint:nakedret
func TransformToOldHPA(hpa *v1alpha1.Autoscaling) (oldHpa *modelschemas.DeploymentTargetHPAConf, err error) {
	if hpa == nil {
		return
	}
	minReplicas := int32(hpa.MinReplicas)
	maxReplicas := int32(hpa.MaxReplicas)
	oldHpa = &modelschemas.DeploymentTargetHPAConf{
		MinReplicas: &minReplicas,
		MaxReplicas: &maxReplicas,
	}

	for _, metric := range hpa.Metrics {
		if metric.Type == autoscalingv2.PodsMetricSourceType {
			if metric.Pods == nil {
				continue
			}
			if metric.Pods.Metric.Name == commonconsts.KubeHPAQPSMetric {
				if metric.Pods.Target.Type != autoscalingv2.UtilizationMetricType {
					continue
				}
				if metric.Pods.Target.AverageValue == nil {
					continue
				}
				qps := metric.Pods.Target.AverageValue.Value()
				oldHpa.QPS = &qps
			}
		} else if metric.Type == autoscalingv2.ResourceMetricSourceType {
			if metric.Resource == nil {
				continue
			}
			if metric.Resource.Name == corev1.ResourceCPU {
				if metric.Resource.Target.Type != autoscalingv2.UtilizationMetricType {
					continue
				}
				if metric.Resource.Target.AverageUtilization == nil {
					continue
				}
				cpu := *metric.Resource.Target.AverageUtilization
				oldHpa.CPU = &cpu
			} else if metric.Resource.Name == corev1.ResourceMemory {
				if metric.Resource.Target.Type != autoscalingv2.UtilizationMetricType {
					continue
				}
				if metric.Resource.Target.AverageUtilization == nil {
					continue
				}
				memory := metric.Resource.Target.AverageValue.String()
				oldHpa.Memory = &memory
			}
		}
	}
	return
}
