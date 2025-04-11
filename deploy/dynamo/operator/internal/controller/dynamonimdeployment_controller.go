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
	"context"
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
	dynamoCommon "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/schemas"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	commonconfig "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/config"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/envoy"
	"github.com/cisco-open/k8s-objectmatcher/patch"
	"github.com/huandu/xstrings"
	istioNetworking "istio.io/api/networking/v1beta1"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
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
	kDefaultIngressSuffix                                     = "local"
)

var ServicePortHTTPNonProxy = commonconsts.BentoServicePort + 1

// DynamoNimDeploymentReconciler reconciles a DynamoNimDeployment object
type DynamoNimDeploymentReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
	Config   controller_common.Config
	NatsAddr string
	EtcdAddr string
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamonimdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamonimdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamonimdeployments/finalizers,verbs=update

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
// the DynamoNimDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.18.2/pkg/reconcile
//
//nolint:gocyclo,nakedret
func (r *DynamoNimDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logs := log.FromContext(ctx)

	dynamoNimDeployment := &v1alpha1.DynamoNimDeployment{}
	err = r.Get(ctx, req.NamespacedName, dynamoNimDeployment)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			logs.Info("DynamoNimDeployment resource not found. Ignoring since object must be deleted.")
			err = nil
			return
		}
		// Error reading the object - requeue the request.
		logs.Error(err, "Failed to get DynamoNimDeployment.")
		return
	}

	logs = logs.WithValues("dynamoNimDeployment", dynamoNimDeployment.Name, "namespace", dynamoNimDeployment.Namespace)

	if len(dynamoNimDeployment.Status.Conditions) == 0 {
		logs.Info("Starting to reconcile DynamoNimDeployment")
		logs.Info("Initializing DynamoNimDeployment status")
		r.Recorder.Event(dynamoNimDeployment, corev1.EventTypeNormal, "Reconciling", "Starting to reconcile DynamoNimDeployment")
		dynamoNimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile DynamoNimDeployment",
			},
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimFound,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile DynamoNimDeployment",
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
		logs.Error(err, "Failed to reconcile DynamoNimDeployment.")
		r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeWarning, "ReconcileError", "Failed to reconcile DynamoNimDeployment: %v", err)
		_, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Failed to reconcile DynamoNimDeployment: %v", err),
			},
		)
		if err != nil {
			return
		}
	}()

	dynamoNimFoundCondition := meta.FindStatusCondition(dynamoNimDeployment.Status.Conditions, v1alpha1.DynamoDeploymentConditionTypeDynamoNimFound)
	if dynamoNimFoundCondition != nil && dynamoNimFoundCondition.Status == metav1.ConditionUnknown {
		logs.Info(fmt.Sprintf("Getting Dynamo NIM %s", dynamoNimDeployment.Spec.DynamoNim))
		r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeNormal, "GetDynamoNim", "Getting Dynamo NIM %s", dynamoNimDeployment.Spec.DynamoNim)
	}
	dynamoNimRequest := &v1alpha1.DynamoNimRequest{}
	dynamoNimCR := &v1alpha1.DynamoNim{}
	err = r.Get(ctx, types.NamespacedName{
		Namespace: dynamoNimDeployment.Namespace,
		Name:      dynamoNimDeployment.Spec.DynamoNim,
	}, dynamoNimCR)
	dynamoNimIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !dynamoNimIsNotFound {
		err = errors.Wrapf(err, "get DynamoNim %s/%s", dynamoNimDeployment.Namespace, dynamoNimDeployment.Spec.DynamoNim)
		return
	}
	if dynamoNimIsNotFound {
		if dynamoNimFoundCondition != nil && dynamoNimFoundCondition.Status == metav1.ConditionUnknown {
			logs.Info(fmt.Sprintf("DynamoNim %s not found", dynamoNimDeployment.Spec.DynamoNim))
			r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeNormal, "GetDynamoNim", "DynamoNim %s not found", dynamoNimDeployment.Spec.DynamoNim)
		}
		dynamoNimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimFound,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: "DynamoNim not found",
			},
		)
		if err != nil {
			return
		}
		dynamoNimRequestFoundCondition := meta.FindStatusCondition(dynamoNimDeployment.Status.Conditions, v1alpha1.DynamoDeploymentConditionTypeDynamoNimRequestFound)
		if dynamoNimRequestFoundCondition == nil || dynamoNimRequestFoundCondition.Status != metav1.ConditionUnknown {
			dynamoNimDeployment, err = r.setStatusConditions(ctx, req,
				metav1.Condition{
					Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimRequestFound,
					Status:  metav1.ConditionUnknown,
					Reason:  "Reconciling",
					Message: "DynamoNim not found",
				},
			)
			if err != nil {
				return
			}
		}
		if dynamoNimRequestFoundCondition != nil && dynamoNimRequestFoundCondition.Status == metav1.ConditionUnknown {
			r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeNormal, "GetDynamoNimRequest", "Getting DynamoNimRequest %s", dynamoNimDeployment.Spec.DynamoNim)
		}
		err = r.Get(ctx, types.NamespacedName{
			Namespace: dynamoNimDeployment.Namespace,
			Name:      dynamoNimDeployment.Spec.DynamoNim,
		}, dynamoNimRequest)
		if err != nil {
			err = errors.Wrapf(err, "get DynamoNimRequest %s/%s", dynamoNimDeployment.Namespace, dynamoNimDeployment.Spec.DynamoNim)
			dynamoNimDeployment, err = r.setStatusConditions(ctx, req,
				metav1.Condition{
					Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimFound,
					Status:  metav1.ConditionFalse,
					Reason:  "Reconciling",
					Message: err.Error(),
				},
				metav1.Condition{
					Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimRequestFound,
					Status:  metav1.ConditionFalse,
					Reason:  "Reconciling",
					Message: err.Error(),
				},
			)
			if err != nil {
				return
			}
		}
		if dynamoNimRequestFoundCondition != nil && dynamoNimRequestFoundCondition.Status == metav1.ConditionUnknown {
			logs.Info(fmt.Sprintf("DynamoNimRequest %s found", dynamoNimDeployment.Spec.DynamoNim))
			r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeNormal, "GetDynamoNimRequest", "DynamoNimRequest %s is found and waiting for its dynamoNim to be provided", dynamoNimDeployment.Spec.DynamoNim)
		}
		dynamoNimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimRequestFound,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: "DynamoNim not found",
			},
		)
		if err != nil {
			return
		}
		dynamoNimRequestAvailableCondition := meta.FindStatusCondition(dynamoNimRequest.Status.Conditions, v1alpha1.DynamoDeploymentConditionTypeAvailable)
		if dynamoNimRequestAvailableCondition != nil && dynamoNimRequestAvailableCondition.Status == metav1.ConditionFalse {
			err = errors.Errorf("DynamoNimRequest %s/%s is not available: %s", dynamoNimRequest.Namespace, dynamoNimRequest.Name, dynamoNimRequestAvailableCondition.Message)
			r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeWarning, "GetDynamoNimRequest", err.Error())
			_, err_ := r.setStatusConditions(ctx, req,
				metav1.Condition{
					Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimFound,
					Status:  metav1.ConditionFalse,
					Reason:  "Reconciling",
					Message: err.Error(),
				},
				metav1.Condition{
					Type:    v1alpha1.DynamoDeploymentConditionTypeAvailable,
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
		if dynamoNimFoundCondition != nil && dynamoNimFoundCondition.Status != metav1.ConditionTrue {
			logs.Info(fmt.Sprintf("DynamoNim %s found", dynamoNimDeployment.Spec.DynamoNim))
			r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeNormal, "GetDynamoNim", "DynamoNim %s is found", dynamoNimDeployment.Spec.DynamoNim)
		}
		dynamoNimDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeDynamoNimFound,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: "DynamoNim found",
			},
		)
		if err != nil {
			return
		}
	}

	modified := false

	// Reconcile PVC
	_, err = r.reconcilePVC(ctx, dynamoNimDeployment)
	if err != nil {
		logs.Error(err, "Unable to create PVC", "crd", req.NamespacedName)
		return ctrl.Result{}, err
	}

	// create or update api-server deployment
	modified_, deployment, err := r.createOrUpdateOrDeleteDeployments(ctx, generateResourceOption{
		dynamoNimDeployment: dynamoNimDeployment,
		dynamoNim:           dynamoNimCR,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server hpa
	modified_, _, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoNimDeployment: dynamoNimDeployment,
		dynamoNim:           dynamoNimCR,
	}, r.generateHPA)
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server service
	modified_, err = r.createOrUpdateOrDeleteServices(ctx, generateResourceOption{
		dynamoNimDeployment: dynamoNimDeployment,
		dynamoNim:           dynamoNimCR,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server ingresses
	modified_, _, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoNimDeployment: dynamoNimDeployment,
		dynamoNim:           dynamoNimCR,
	}, r.generateVirtualService)
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	if !modified {
		r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeNormal, "UpdateYataiDeployment", "No changes to yatai deployment %s", dynamoNimDeployment.Name)
	}

	logs.Info("Finished reconciling.")
	r.Recorder.Eventf(dynamoNimDeployment, corev1.EventTypeNormal, "Update", "All resources updated!")
	err = r.computeAvailableStatusCondition(ctx, req, deployment)
	return
}

func (r *DynamoNimDeploymentReconciler) computeAvailableStatusCondition(ctx context.Context, req ctrl.Request, deployment *appsv1.Deployment) error {
	logs := log.FromContext(ctx)
	if IsDeploymentReady(deployment) {
		logs.Info("Deployment is ready. Setting available status condition to true.")
		_, err := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionTrue,
				Reason:  "DeploymentReady",
				Message: "Deployment is ready",
			},
		)
		return err
	} else {
		logs.Info("Deployment is not ready. Setting available status condition to false.")
		_, err := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "DeploymentNotReady",
				Message: "Deployment is not ready",
			},
		)
		return err
	}
}

// IsDeploymentReady determines if a Kubernetes Deployment is fully ready and available.
// It checks various status fields to ensure all replicas are available and the deployment
// configuration has been fully applied.
func IsDeploymentReady(deployment *appsv1.Deployment) bool {
	if deployment == nil {
		return false
	}
	// Paused deployments should not be considered ready
	if deployment.Spec.Paused {
		return false
	}
	// Default to 1 replica if not specified
	desiredReplicas := int32(1)
	if deployment.Spec.Replicas != nil {
		desiredReplicas = *deployment.Spec.Replicas
	}
	// Special case: if no replicas are desired, the deployment is considered ready
	if desiredReplicas == 0 {
		return true
	}
	status := deployment.Status
	// Check all basic status requirements:
	// 1. ObservedGeneration: Deployment controller has observed the latest configuration
	// 2. UpdatedReplicas: All replicas have been updated to the latest version
	// 3. AvailableReplicas: All desired replicas are available (schedulable and healthy)
	if status.ObservedGeneration < deployment.Generation ||
		status.UpdatedReplicas < desiredReplicas ||
		status.AvailableReplicas < desiredReplicas {
		return false
	}
	// Finally, check for the DeploymentAvailable condition
	// This is Kubernetes' own assessment that the deployment is available
	for _, cond := range deployment.Status.Conditions {
		if cond.Type == appsv1.DeploymentAvailable && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	// If we get here, the basic checks passed but the Available condition wasn't found
	return false
}

func (r *DynamoNimDeploymentReconciler) reconcilePVC(ctx context.Context, crd *v1alpha1.DynamoNimDeployment) (*corev1.PersistentVolumeClaim, error) {
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

func (r *DynamoNimDeploymentReconciler) setStatusConditions(ctx context.Context, req ctrl.Request, conditions ...metav1.Condition) (dynamoNimDeployment *v1alpha1.DynamoNimDeployment, err error) {
	dynamoNimDeployment = &v1alpha1.DynamoNimDeployment{}
	maxRetries := 3
	for range maxRetries - 1 {
		if err = r.Get(ctx, req.NamespacedName, dynamoNimDeployment); err != nil {
			err = errors.Wrap(err, "Failed to re-fetch DynamoNimDeployment")
			return
		}
		for _, condition := range conditions {
			meta.SetStatusCondition(&dynamoNimDeployment.Status.Conditions, condition)
		}
		if err = r.Status().Update(ctx, dynamoNimDeployment); err != nil {
			if k8serrors.IsConflict(err) {
				time.Sleep(100 * time.Millisecond)
				continue
			}
			break
		} else {
			break
		}
	}
	if err != nil {
		err = errors.Wrap(err, "Failed to update DynamoNimDeployment status")
		return
	}
	if err = r.Get(ctx, req.NamespacedName, dynamoNimDeployment); err != nil {
		err = errors.Wrap(err, "Failed to re-fetch DynamoNimDeployment")
		return
	}
	return
}

//nolint:nakedret
func (r *DynamoNimDeploymentReconciler) createOrUpdateOrDeleteDeployments(ctx context.Context, opt generateResourceOption) (modified bool, depl *appsv1.Deployment, err error) {
	containsStealingTrafficDebugModeEnabled := checkIfContainsStealingTrafficDebugModeEnabled(opt.dynamoNimDeployment)
	// create the main deployment
	modified, depl, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoNimDeployment:                     opt.dynamoNimDeployment,
		dynamoNim:                               opt.dynamoNim,
		isStealingTrafficDebugModeEnabled:       false,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
	}, r.generateDeployment)
	if err != nil {
		err = errors.Wrap(err, "create or update deployment")
		return
	}
	// create the debug deployment
	modified2, _, err := createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoNimDeployment:                     opt.dynamoNimDeployment,
		dynamoNim:                               opt.dynamoNim,
		isStealingTrafficDebugModeEnabled:       true,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
	}, r.generateDeployment)
	if err != nil {
		err = errors.Wrap(err, "create or update debug deployment")
	}
	modified = modified || modified2
	return
}

//nolint:nakedret
func createOrUpdateResource[T client.Object](ctx context.Context, r *DynamoNimDeploymentReconciler, opt generateResourceOption, generateResource func(ctx context.Context, opt generateResourceOption) (T, bool, error)) (modified bool, res T, err error) {
	logs := log.FromContext(ctx)

	resource, toDelete, err := generateResource(ctx, opt)
	if err != nil {
		return
	}
	resourceNamespace := resource.GetNamespace()
	resourceName := resource.GetName()
	resourceType := reflect.TypeOf(resource).Elem().Name()
	logs = logs.WithValues("namespace", resourceNamespace, "resourceName", resourceName, "resourceType", resourceType)

	// Retrieve the GroupVersionKind (GVK) of the desired object
	gvk, err := apiutil.GVKForObject(resource, r.Client.Scheme())
	if err != nil {
		logs.Error(err, "Failed to get GVK for object")
		return
	}

	// Create a new instance of the object
	obj, err := r.Client.Scheme().New(gvk)
	if err != nil {
		logs.Error(err, "Failed to create a new object for GVK")
		return
	}

	// Type assertion to ensure the object implements client.Object
	oldResource, ok := obj.(T)
	if !ok {
		return
	}

	err = r.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: resourceNamespace}, oldResource)
	oldResourceIsNotFound := k8serrors.IsNotFound(err)
	if err != nil && !oldResourceIsNotFound {
		r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeWarning, fmt.Sprintf("Get%s", resourceType), "Failed to get %s %s: %s", resourceType, resourceNamespace, err)
		logs.Error(err, "Failed to get HPA.")
		return
	}
	err = nil

	if oldResourceIsNotFound {
		if toDelete {
			logs.Info("Resource not found. Nothing to do.")
			return
		}
		logs.Info("Resource not found. Creating a new one.")

		err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(resource), "set last applied annotation for resource %s", resourceName)
		if err != nil {
			logs.Error(err, "Failed to set last applied annotation.")
			r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

		r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeNormal, fmt.Sprintf("Create%s", resourceType), "Creating a new %s %s", resourceType, resourceNamespace)
		err = r.Create(ctx, resource)
		if err != nil {
			logs.Error(err, "Failed to create Resource.")
			r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeWarning, fmt.Sprintf("Create%s", resourceType), "Failed to create %s %s: %s", resourceType, resourceNamespace, err)
			return
		}
		logs.Info(fmt.Sprintf("%s created.", resourceType))
		r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeNormal, fmt.Sprintf("Create%s", resourceType), "Created %s %s", resourceType, resourceNamespace)
		modified = true
		res = resource
	} else {
		logs.Info(fmt.Sprintf("%s found.", resourceType))
		if toDelete {
			logs.Info(fmt.Sprintf("%s not found. Deleting the existing one.", resourceType))
			err = r.Delete(ctx, oldResource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to delete %s.", resourceType))
				r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeWarning, fmt.Sprintf("Delete%s", resourceType), "Failed to delete %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
			logs.Info(fmt.Sprintf("%s deleted.", resourceType))
			r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeNormal, fmt.Sprintf("Delete%s", resourceType), "Deleted %s %s", resourceType, resourceNamespace)
			modified = true
			return
		}

		var patchResult *patch.PatchResult
		patchResult, err = patch.DefaultPatchMaker.Calculate(oldResource, resource)
		if err != nil {
			logs.Error(err, "Failed to calculate patch.")
			r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeWarning, fmt.Sprintf("CalculatePatch%s", resourceType), "Failed to calculate patch for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

		if !patchResult.IsEmpty() {
			logs.Info(fmt.Sprintf("%s spec is different. Updating %s. The patch result is: %s", resourceType, resourceType, patchResult.String()))

			err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(resource), "set last applied annotation for resource %s", resourceName)
			if err != nil {
				logs.Error(err, "Failed to set last applied annotation.")
				r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeWarning, fmt.Sprintf("SetLastAppliedAnnotation%s", resourceType), "Failed to set last applied annotation for %s %s: %s", resourceType, resourceNamespace, err)
				return
			}

			r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Updating %s %s", resourceType, resourceNamespace)
			resource.SetResourceVersion(oldResource.GetResourceVersion())
			err = r.Update(ctx, resource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to update %s.", resourceType))
				r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeWarning, fmt.Sprintf("Update%s", resourceType), "Failed to update %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
			logs.Info(fmt.Sprintf("%s updated.", resourceType))
			r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Updated %s %s", resourceType, resourceNamespace)
			modified = true
			res = resource
		} else {
			logs.Info(fmt.Sprintf("%s spec is the same. Skipping update.", resourceType))
			r.Recorder.Eventf(opt.dynamoNimDeployment, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Skipping update %s %s", resourceType, resourceNamespace)
			res = oldResource
		}
	}
	return
}

func getResourceAnnotations(dynamoNimDeployment *v1alpha1.DynamoNimDeployment) map[string]string {
	resourceAnnotations := dynamoNimDeployment.Spec.Annotations
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

func checkIfContainsStealingTrafficDebugModeEnabled(dynamoNimDeployment *v1alpha1.DynamoNimDeployment) bool {
	return checkIfIsStealingTrafficDebugModeEnabled(dynamoNimDeployment.Spec.Annotations)
}

//nolint:nakedret
func (r *DynamoNimDeploymentReconciler) createOrUpdateOrDeleteServices(ctx context.Context, opt generateResourceOption) (modified bool, err error) {
	resourceAnnotations := getResourceAnnotations(opt.dynamoNimDeployment)
	isDebugPodReceiveProductionTrafficEnabled := checkIfIsDebugPodReceiveProductionTrafficEnabled(resourceAnnotations)
	containsStealingTrafficDebugModeEnabled := checkIfContainsStealingTrafficDebugModeEnabled(opt.dynamoNimDeployment)
	// main generic service
	modified, _, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoNimDeployment:                     opt.dynamoNimDeployment,
		dynamoNim:                               opt.dynamoNim,
		isStealingTrafficDebugModeEnabled:       false,
		isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
		isGenericService:                        true,
	}, r.generateService)
	if err != nil {
		return
	}

	// debug production service (if enabled)
	modified_, _, err := createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoNimDeployment:                     opt.dynamoNimDeployment,
		dynamoNim:                               opt.dynamoNim,
		isStealingTrafficDebugModeEnabled:       false,
		isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
		isGenericService:                        false,
	}, r.generateService)
	if err != nil {
		return
	}
	modified = modified || modified_
	// debug service (if enabled)
	modified_, _, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoNimDeployment:                     opt.dynamoNimDeployment,
		dynamoNim:                               opt.dynamoNim,
		isStealingTrafficDebugModeEnabled:       true,
		isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
		isGenericService:                        false,
	}, r.generateService)
	if err != nil {
		return
	}
	modified = modified || modified_
	return
}

func (r *DynamoNimDeploymentReconciler) generateVirtualService(ctx context.Context, opt generateResourceOption) (*networkingv1beta1.VirtualService, bool, error) {
	log := log.FromContext(ctx)
	log.Info("Starting generateVirtualService")

	vsName := opt.dynamoNimDeployment.Name
	if opt.dynamoNimDeployment.Spec.Ingress.HostPrefix != nil {
		vsName = *opt.dynamoNimDeployment.Spec.Ingress.HostPrefix + vsName
	}
	ingressSuffix, found := os.LookupEnv("DYNAMO_INGRESS_SUFFIX")
	if !found || ingressSuffix == "" {
		ingressSuffix = kDefaultIngressSuffix
	}
	vs := &networkingv1beta1.VirtualService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opt.dynamoNimDeployment.Name,
			Namespace: opt.dynamoNimDeployment.Namespace,
		},
	}

	vsEnabled := opt.dynamoNimDeployment.Spec.Ingress.Enabled && opt.dynamoNimDeployment.Spec.Ingress.UseVirtualService != nil && *opt.dynamoNimDeployment.Spec.Ingress.UseVirtualService
	if !vsEnabled {
		log.Info("VirtualService is not enabled")
		return vs, true, nil
	}

	vs.Spec = istioNetworking.VirtualService{
		Hosts: []string{
			fmt.Sprintf("%s.%s", vsName, ingressSuffix),
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
							Host: opt.dynamoNimDeployment.Name,
							Port: &istioNetworking.PortSelector{
								Number: 3000,
							},
						},
					},
				},
			},
		},
	}
	return vs, false, nil
}

func (r *DynamoNimDeploymentReconciler) getKubeName(dynamoNimDeployment *v1alpha1.DynamoNimDeployment, _ *v1alpha1.DynamoNim, debug bool) string {
	if debug {
		return fmt.Sprintf("%s-d", dynamoNimDeployment.Name)
	}
	return dynamoNimDeployment.Name
}

func (r *DynamoNimDeploymentReconciler) getServiceName(dynamoNimDeployment *v1alpha1.DynamoNimDeployment, _ *v1alpha1.DynamoNim, debug bool) string {
	var kubeName string
	if debug {
		kubeName = fmt.Sprintf("%s-d", dynamoNimDeployment.Name)
	} else {
		kubeName = fmt.Sprintf("%s-p", dynamoNimDeployment.Name)
	}
	return kubeName
}

func (r *DynamoNimDeploymentReconciler) getGenericServiceName(dynamoNimDeployment *v1alpha1.DynamoNimDeployment, dynamoNim *v1alpha1.DynamoNim) string {
	return r.getKubeName(dynamoNimDeployment, dynamoNim, false)
}

func (r *DynamoNimDeploymentReconciler) getKubeLabels(dynamoNimDeployment *v1alpha1.DynamoNimDeployment, dynamoNim *v1alpha1.DynamoNim) map[string]string {
	dynamoNimRepositoryName, _, dynamoNimVersion := xstrings.Partition(dynamoNim.Spec.Tag, ":")
	labels := map[string]string{
		commonconsts.KubeLabelYataiBentoDeployment:           dynamoNimDeployment.Name,
		commonconsts.KubeLabelBentoRepository:                dynamoNimRepositoryName,
		commonconsts.KubeLabelBentoVersion:                   dynamoNimVersion,
		commonconsts.KubeLabelYataiBentoDeploymentTargetType: DeploymentTargetTypeProduction,
		commonconsts.KubeLabelCreator:                        "yatai-deployment",
	}
	labels[commonconsts.KubeLabelYataiBentoDeploymentComponentType] = commonconsts.YataiBentoDeploymentComponentApiServer
	return labels
}

func (r *DynamoNimDeploymentReconciler) getKubeAnnotations(dynamoNimDeployment *v1alpha1.DynamoNimDeployment, dynamoNim *v1alpha1.DynamoNim) map[string]string {
	dynamoNimRepositoryName, dynamoNimVersion := getDynamoNimRepositoryNameAndDynamoNimVersion(dynamoNim)
	annotations := map[string]string{
		commonconsts.KubeAnnotationBentoRepository: dynamoNimRepositoryName,
		commonconsts.KubeAnnotationBentoVersion:    dynamoNimVersion,
	}
	var extraAnnotations map[string]string
	if dynamoNimDeployment.Spec.ExtraPodMetadata != nil {
		extraAnnotations = dynamoNimDeployment.Spec.ExtraPodMetadata.Annotations
	} else {
		extraAnnotations = map[string]string{}
	}
	for k, v := range extraAnnotations {
		annotations[k] = v
	}
	return annotations
}

//nolint:nakedret
func (r *DynamoNimDeploymentReconciler) generateDeployment(ctx context.Context, opt generateResourceOption) (kubeDeployment *appsv1.Deployment, toDelete bool, err error) {
	kubeNs := opt.dynamoNimDeployment.Namespace

	labels := r.getKubeLabels(opt.dynamoNimDeployment, opt.dynamoNim)

	annotations := r.getKubeAnnotations(opt.dynamoNimDeployment, opt.dynamoNim)

	kubeName := r.getKubeName(opt.dynamoNimDeployment, opt.dynamoNim, opt.isStealingTrafficDebugModeEnabled)

	kubeDeployment = &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        kubeName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
	}

	if opt.isStealingTrafficDebugModeEnabled && !opt.containsStealingTrafficDebugModeEnabled {
		// if stealing traffic debug mode is enabked but disabled in the deployment, we need to delete the deployment
		return kubeDeployment, true, nil
	}

	// nolint: gosimple
	podTemplateSpec, err := r.generatePodTemplateSpec(ctx, opt)
	if err != nil {
		return
	}

	defaultMaxSurge := intstr.FromString("25%")
	defaultMaxUnavailable := intstr.FromString("25%")

	strategy := appsv1.DeploymentStrategy{
		Type: appsv1.RollingUpdateDeploymentStrategyType,
		RollingUpdate: &appsv1.RollingUpdateDeployment{
			MaxSurge:       &defaultMaxSurge,
			MaxUnavailable: &defaultMaxUnavailable,
		},
	}

	resourceAnnotations := getResourceAnnotations(opt.dynamoNimDeployment)
	strategyStr := resourceAnnotations[KubeAnnotationDeploymentStrategy]
	if strategyStr != "" {
		strategyType := schemas.DeploymentStrategy(strategyStr)
		switch strategyType {
		case schemas.DeploymentStrategyRollingUpdate:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       &defaultMaxSurge,
					MaxUnavailable: &defaultMaxUnavailable,
				},
			}
		case schemas.DeploymentStrategyRecreate:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RecreateDeploymentStrategyType,
			}
		case schemas.DeploymentStrategyRampedSlowRollout:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       &[]intstr.IntOrString{intstr.FromInt(1)}[0],
					MaxUnavailable: &[]intstr.IntOrString{intstr.FromInt(0)}[0],
				},
			}
		case schemas.DeploymentStrategyBestEffortControlledRollout:
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
	replicas = opt.dynamoNimDeployment.Spec.Replicas
	if opt.isStealingTrafficDebugModeEnabled {
		replicas = &[]int32{int32(1)}[0]
	}

	kubeDeployment.Spec = appsv1.DeploymentSpec{
		Replicas: replicas,
		Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				commonconsts.KubeLabelYataiSelector: kubeName,
			},
		},
		Template: *podTemplateSpec,
		Strategy: strategy,
	}

	err = ctrl.SetControllerReference(opt.dynamoNimDeployment, kubeDeployment, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set deployment %s controller reference", kubeDeployment.Name)
	}

	return
}

type generateResourceOption struct {
	dynamoNimDeployment                     *v1alpha1.DynamoNimDeployment
	dynamoNim                               *v1alpha1.DynamoNim
	isStealingTrafficDebugModeEnabled       bool
	containsStealingTrafficDebugModeEnabled bool
	isDebugPodReceiveProductionTraffic      bool
	isGenericService                        bool
}

func (r *DynamoNimDeploymentReconciler) generateHPA(ctx context.Context, opt generateResourceOption) (*autoscalingv2.HorizontalPodAutoscaler, bool, error) {
	labels := r.getKubeLabels(opt.dynamoNimDeployment, opt.dynamoNim)

	annotations := r.getKubeAnnotations(opt.dynamoNimDeployment, opt.dynamoNim)

	kubeName := r.getKubeName(opt.dynamoNimDeployment, opt.dynamoNim, false)

	kubeNs := opt.dynamoNimDeployment.Namespace

	hpaConf := opt.dynamoNimDeployment.Spec.Autoscaling

	kubeHpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:        kubeName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
	}

	if hpaConf == nil || !hpaConf.Enabled {
		// if hpa is not enabled, we need to delete the hpa
		return kubeHpa, true, nil
	}

	minReplica := int32(hpaConf.MinReplicas)

	kubeHpa.Spec = autoscalingv2.HorizontalPodAutoscalerSpec{
		MinReplicas: &minReplica,
		MaxReplicas: int32(hpaConf.MaxReplicas),
		ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       kubeName,
		},
		Metrics: hpaConf.Metrics,
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

	err := ctrl.SetControllerReference(opt.dynamoNimDeployment, kubeHpa, r.Scheme)
	if err != nil {
		return nil, false, errors.Wrapf(err, "set hpa %s controller reference", kubeName)
	}

	return kubeHpa, false, err
}

func getDynamoNimRepositoryNameAndDynamoNimVersion(dynamoNim *v1alpha1.DynamoNim) (repositoryName string, version string) {
	repositoryName, _, version = xstrings.Partition(dynamoNim.Spec.Tag, ":")

	return
}

//nolint:gocyclo,nakedret
func (r *DynamoNimDeploymentReconciler) generatePodTemplateSpec(ctx context.Context, opt generateResourceOption) (podTemplateSpec *corev1.PodTemplateSpec, err error) {
	dynamoNimRepositoryName, _ := getDynamoNimRepositoryNameAndDynamoNimVersion(opt.dynamoNim)
	podLabels := r.getKubeLabels(opt.dynamoNimDeployment, opt.dynamoNim)
	if opt.isStealingTrafficDebugModeEnabled {
		podLabels[commonconsts.KubeLabelYataiBentoDeploymentTargetType] = DeploymentTargetTypeDebug
	}

	podAnnotations := r.getKubeAnnotations(opt.dynamoNimDeployment, opt.dynamoNim)

	kubeName := r.getKubeName(opt.dynamoNimDeployment, opt.dynamoNim, opt.isStealingTrafficDebugModeEnabled)

	containerPort := commonconsts.BentoServicePort
	lastPort := containerPort + 1

	monitorExporter := opt.dynamoNimDeployment.Spec.MonitorExporter
	needMonitorContainer := monitorExporter != nil && monitorExporter.Enabled

	lastPort++
	monitorExporterPort := lastPort

	var envs []corev1.EnvVar
	envsSeen := make(map[string]struct{})

	resourceAnnotations := opt.dynamoNimDeployment.Spec.Annotations
	specEnvs := opt.dynamoNimDeployment.Spec.Envs

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
			Value: string(opt.dynamoNimDeployment.UID),
		},
		{
			Name:  commonconsts.EnvYataiBentoDeploymentName,
			Value: opt.dynamoNimDeployment.Name,
		},
		{
			Name:  commonconsts.EnvYataiBentoDeploymentNamespace,
			Value: opt.dynamoNimDeployment.Namespace,
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

	var livenessProbe *corev1.Probe
	if opt.dynamoNimDeployment.Spec.LivenessProbe != nil {
		livenessProbe = opt.dynamoNimDeployment.Spec.LivenessProbe
	}

	var readinessProbe *corev1.Probe
	if opt.dynamoNimDeployment.Spec.ReadinessProbe != nil {
		readinessProbe = opt.dynamoNimDeployment.Spec.ReadinessProbe
	}

	volumes := make([]corev1.Volume, 0)
	volumeMounts := make([]corev1.VolumeMount, 0)

	args := make([]string, 0)

	args = append(args, "cd", "src", "&&", "uv", "run", "dynamo", "serve")

	// todo : remove this line when https://github.com/ai-dynamo/dynamo/issues/345 is fixed
	enableDependsOption := false
	if len(opt.dynamoNimDeployment.Spec.ExternalServices) > 0 && enableDependsOption {
		serviceSuffix := fmt.Sprintf("%s.svc.cluster.local:3000", opt.dynamoNimDeployment.Namespace)
		keys := make([]string, 0, len(opt.dynamoNimDeployment.Spec.ExternalServices))

		for key := range opt.dynamoNimDeployment.Spec.ExternalServices {
			keys = append(keys, key)
		}

		sort.Strings(keys)
		for _, key := range keys {
			service := opt.dynamoNimDeployment.Spec.ExternalServices[key]

			// Check if DeploymentSelectorKey is not "name"
			if service.DeploymentSelectorKey == "name" {
				dependsFlag := fmt.Sprintf("--depends \"%s=http://%s.%s\"", key, service.DeploymentSelectorValue, serviceSuffix)
				args = append(args, dependsFlag)
			} else if service.DeploymentSelectorKey == "dynamo" {
				dependsFlag := fmt.Sprintf("--depends \"%s=dynamo://%s\"", key, service.DeploymentSelectorValue)
				args = append(args, dependsFlag)
			} else {
				return nil, errors.Errorf("DeploymentSelectorKey '%s' not supported. Only 'name' and 'dynamo' are supported", service.DeploymentSelectorKey)
			}
		}
	}

	if opt.dynamoNimDeployment.Spec.ServiceName != "" {
		args = append(args, []string{"--service-name", opt.dynamoNimDeployment.Spec.ServiceName}...)
		args = append(args, opt.dynamoNimDeployment.Spec.DynamoTag)
	}

	if len(opt.dynamoNimDeployment.Spec.Envs) > 0 {
		for _, env := range opt.dynamoNimDeployment.Spec.Envs {
			if env.Name == "DYNAMO_CONFIG_PATH" {
				args = append(args, "-f", env.Value)
			}
		}
	}

	yataiResources := opt.dynamoNimDeployment.Spec.Resources

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
	if opt.dynamoNimDeployment.Spec.PVC != nil {
		volumes = append(volumes, corev1.Volume{
			Name: getPvcName(opt.dynamoNimDeployment, opt.dynamoNimDeployment.Spec.PVC.Name),
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: getPvcName(opt.dynamoNimDeployment, opt.dynamoNimDeployment.Spec.PVC.Name),
				},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      getPvcName(opt.dynamoNimDeployment, opt.dynamoNimDeployment.Spec.PVC.Name),
			MountPath: *opt.dynamoNimDeployment.Spec.PVC.MountPoint,
		})
	}

	imageName := opt.dynamoNim.Spec.Image

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

	if opt.dynamoNimDeployment.Spec.EnvFromSecret != nil {
		container.EnvFrom = []corev1.EnvFromSource{
			{
				SecretRef: &corev1.SecretEnvSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: *opt.dynamoNimDeployment.Spec.EnvFromSecret,
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
				Value: fmt.Sprintf("BENTOML_%s_", strings.ReplaceAll(dynamoNimRepositoryName, "-", ":")),
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
		productionServiceName := r.getServiceName(opt.dynamoNimDeployment, opt.dynamoNim, false)
		envoyConfigContent, err = envoy.GenerateEnvoyConfigurationContent(envoy.CreateEnvoyConfig{
			ListenPort:              proxyPort,
			DebugHeaderName:         HeaderNameDebug,
			DebugHeaderValue:        commonconsts.KubeLabelValueTrue,
			DebugServerAddress:      "localhost",
			DebugServerPort:         containerPort,
			ProductionServerAddress: fmt.Sprintf("%s.%s.svc.cluster.local", productionServiceName, opt.dynamoNimDeployment.Namespace),
			ProductionServerPort:    ServicePortHTTPNonProxy,
		})
	} else {
		debugServiceName := r.getServiceName(opt.dynamoNimDeployment, opt.dynamoNim, true)
		envoyConfigContent, err = envoy.GenerateEnvoyConfigurationContent(envoy.CreateEnvoyConfig{
			ListenPort:              proxyPort,
			DebugHeaderName:         HeaderNameDebug,
			DebugHeaderValue:        commonconsts.KubeLabelValueTrue,
			DebugServerAddress:      fmt.Sprintf("%s.%s.svc.cluster.local", debugServiceName, opt.dynamoNimDeployment.Namespace),
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
			Namespace: opt.dynamoNimDeployment.Namespace,
		},
		Data: map[string]string{
			"envoy.yaml": envoyConfigContent,
		},
	}
	err = ctrl.SetControllerReference(opt.dynamoNimDeployment, envoyConfigConfigMap, r.Scheme)
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

	podSpec.ImagePullSecrets = opt.dynamoNim.Spec.ImagePullSecrets

	extraPodMetadata := opt.dynamoNimDeployment.Spec.ExtraPodMetadata

	if extraPodMetadata != nil {
		for k, v := range extraPodMetadata.Annotations {
			podAnnotations[k] = v
		}

		for k, v := range extraPodMetadata.Labels {
			podLabels[k] = v
		}
	}

	extraPodSpec := opt.dynamoNimDeployment.Spec.ExtraPodSpec

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
		err = r.List(ctx, serviceAccounts, client.InNamespace(opt.dynamoNimDeployment.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelBentoDeploymentPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			err = errors.Wrapf(err, "failed to list service accounts in namespace %s", opt.dynamoNimDeployment.Namespace)
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

func getResourcesConfig(resources *dynamoCommon.Resources) (corev1.ResourceRequirements, error) {
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

//nolint:nakedret
func (r *DynamoNimDeploymentReconciler) generateService(ctx context.Context, opt generateResourceOption) (kubeService *corev1.Service, toDelete bool, err error) {
	var kubeName string
	if opt.isGenericService {
		kubeName = r.getGenericServiceName(opt.dynamoNimDeployment, opt.dynamoNim)
	} else {
		kubeName = r.getServiceName(opt.dynamoNimDeployment, opt.dynamoNim, opt.isStealingTrafficDebugModeEnabled)
	}

	kubeNs := opt.dynamoNimDeployment.Namespace

	kubeService = &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeName,
			Namespace: kubeNs,
		},
	}

	if !opt.isGenericService && !opt.containsStealingTrafficDebugModeEnabled {
		// if it's not a generic service and not contains stealing traffic debug mode enabled, we don't need to create the service
		return kubeService, true, nil
	}

	labels := r.getKubeLabels(opt.dynamoNimDeployment, opt.dynamoNim)

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

	annotations := r.getKubeAnnotations(opt.dynamoNimDeployment, opt.dynamoNim)

	kubeService.ObjectMeta.Annotations = annotations
	kubeService.ObjectMeta.Labels = labels
	kubeService.Spec = spec

	err = ctrl.SetControllerReference(opt.dynamoNimDeployment, kubeService, r.Scheme)
	if err != nil {
		err = errors.Wrapf(err, "set controller reference for service %s", kubeService.Name)
		return
	}

	return
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

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoNimDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {

	m := ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.DynamoNimDeployment{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.Deployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		Owns(&corev1.Service{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&networkingv1beta1.VirtualService{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&networkingv1.Ingress{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&corev1.PersistentVolumeClaim{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Watches(&v1alpha1.DynamoNimRequest{}, handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, dynamoNimRequest client.Object) []reconcile.Request {
			reqs := make([]reconcile.Request, 0)
			logs := log.Log.WithValues("func", "Watches", "kind", "DynamoNimRequest", "name", dynamoNimRequest.GetName(), "namespace", dynamoNimRequest.GetNamespace())
			logs.Info("Triggering reconciliation for DynamoNimRequest", "DynamoNimRequestName", dynamoNimRequest.GetName(), "Namespace", dynamoNimRequest.GetNamespace())
			dynamoNim := &v1alpha1.DynamoNim{}
			err := r.Get(context.Background(), types.NamespacedName{
				Name:      dynamoNimRequest.GetName(),
				Namespace: dynamoNimRequest.GetNamespace(),
			}, dynamoNim)
			dynamoNimIsNotFound := k8serrors.IsNotFound(err)
			if err != nil && !dynamoNimIsNotFound {
				logs.Info("Failed to get DynamoNim", "name", dynamoNimRequest.GetName(), "namespace", dynamoNimRequest.GetNamespace(), "error", err)
				return reqs
			}
			if !dynamoNimIsNotFound {
				logs.Info("DynamoNim found, skipping enqueue as it's already present", "DynamoNimName", dynamoNimRequest.GetName())
				return reqs
			}
			dynamoNimDeployments := &v1alpha1.DynamoNimDeploymentList{}
			err = r.List(context.Background(), dynamoNimDeployments, &client.ListOptions{
				Namespace: dynamoNimRequest.GetNamespace(),
			})
			if err != nil {
				logs.Info("Failed to list DynamoNimDeployments", "Namespace", dynamoNimRequest.GetNamespace(), "error", err)
				return reqs
			}
			for _, dynamoNimDeployment := range dynamoNimDeployments.Items {
				dynamoNimDeployment := dynamoNimDeployment
				if dynamoNimDeployment.Spec.DynamoNim == dynamoNimRequest.GetName() {
					reqs = append(reqs, reconcile.Request{
						NamespacedName: client.ObjectKeyFromObject(&dynamoNimDeployment),
					})
				}
			}
			// Log the list of DynamoNimDeployments being enqueued for reconciliation
			logs.Info("Enqueuing DynamoNimDeployments for reconciliation", "ReconcileRequests", reqs)
			return reqs
		})).WithEventFilter(controller_common.EphemeralDeploymentEventFilter(r.Config)).
		Watches(&v1alpha1.DynamoNim{}, handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, dynamoNim client.Object) []reconcile.Request {
			logs := log.Log.WithValues("func", "Watches", "kind", "DynamoNim", "name", dynamoNim.GetName(), "namespace", dynamoNim.GetNamespace())
			logs.Info("Triggering reconciliation for DynamoNim", "DynamoNimName", dynamoNim.GetName(), "Namespace", dynamoNim.GetNamespace())
			dynamoNimDeployments := &v1alpha1.DynamoNimDeploymentList{}
			err := r.List(context.Background(), dynamoNimDeployments, &client.ListOptions{
				Namespace: dynamoNim.GetNamespace(),
			})
			if err != nil {
				logs.Info("Failed to list DynamoNimDeployments", "Namespace", dynamoNim.GetNamespace(), "error", err)
				return []reconcile.Request{}
			}
			reqs := make([]reconcile.Request, 0)
			for _, dynamoNimDeployment := range dynamoNimDeployments.Items {
				dynamoNimDeployment := dynamoNimDeployment
				if dynamoNimDeployment.Spec.DynamoNim == dynamoNim.GetName() {
					reqs = append(reqs, reconcile.Request{
						NamespacedName: client.ObjectKeyFromObject(&dynamoNimDeployment),
					})
				}
			}
			// Log the list of DynamoNimDeployments being enqueued for reconciliation
			logs.Info("Enqueuing DynamoNimDeployments for reconciliation", "ReconcileRequests", reqs)
			return reqs
		}))

	m.Owns(&autoscalingv2.HorizontalPodAutoscaler{})
	return m.Complete(r)
}
