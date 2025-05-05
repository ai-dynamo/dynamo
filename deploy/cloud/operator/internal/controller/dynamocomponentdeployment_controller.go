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
	dynamoCommon "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/schemas"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/config"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
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
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

const (
	DefaultClusterName                                   = "default"
	DefaultServiceAccountName                            = "default"
	KubeValueNameSharedMemory                            = "shared-memory"
	KubeAnnotationDeploymentStrategy                     = "nvidia.com/deployment-strategy"
	KubeAnnotationEnableStealingTrafficDebugMode         = "nvidia.com/enable-stealing-traffic-debug-mode"
	KubeAnnotationEnableDebugMode                        = "nvidia.com/enable-debug-mode"
	KubeAnnotationEnableDebugPodReceiveProductionTraffic = "nvidia.com/enable-debug-pod-receive-production-traffic"
	DeploymentTargetTypeProduction                       = "production"
	DeploymentTargetTypeDebug                            = "debug"
	HeaderNameDebug                                      = "X-Nvidia-Debug"
	DefaultIngressSuffix                                 = "local"
)

// DynamoComponentDeploymentReconciler reconciles a DynamoComponentDeployment object
type DynamoComponentDeploymentReconciler struct {
	client.Client
	Scheme            *runtime.Scheme
	Recorder          record.EventRecorder
	Config            controller_common.Config
	NatsAddr          string
	EtcdAddr          string
	EtcdStorage       etcdStorage
	UseVirtualService bool
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponentdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponentdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponentdeployments/finalizers,verbs=update

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
// the DynamoComponentDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.18.2/pkg/reconcile
//
//nolint:gocyclo,nakedret
func (r *DynamoComponentDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logs := log.FromContext(ctx)

	dynamoComponentDeployment := &v1alpha1.DynamoComponentDeployment{}
	err = r.Get(ctx, req.NamespacedName, dynamoComponentDeployment)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Object not found, return.  Created objects are automatically garbage collected.
			// For additional cleanup logic use finalizers.
			logs.Info("DynamoComponentDeployment resource not found. Ignoring since object must be deleted.")
			err = nil
			return
		}
		// Error reading the object - requeue the request.
		logs.Error(err, "Failed to get DynamoComponentDeployment.")
		return
	}

	logs = logs.WithValues("dynamoComponentDeployment", dynamoComponentDeployment.Name, "namespace", dynamoComponentDeployment.Namespace)

	deleted, err := commonController.HandleFinalizer(ctx, dynamoComponentDeployment, r.Client, r)
	if err != nil {
		logs.Error(err, "Failed to handle finalizer")
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}

	if len(dynamoComponentDeployment.Status.Conditions) == 0 {
		logs.Info("Starting to reconcile DynamoComponentDeployment")
		logs.Info("Initializing DynamoComponentDeployment status")
		r.Recorder.Event(dynamoComponentDeployment, corev1.EventTypeNormal, "Reconciling", "Starting to reconcile DynamoComponentDeployment")
		dynamoComponentDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile DynamoComponentDeployment",
			},
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeDynamoComponentReady,
				Status:  metav1.ConditionUnknown,
				Reason:  "Reconciling",
				Message: "Starting to reconcile DynamoComponentDeployment",
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
		logs.Error(err, "Failed to reconcile DynamoComponentDeployment.")
		r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeWarning, "ReconcileError", "Failed to reconcile DynamoComponentDeployment: %v", err)
		_, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Failed to reconcile DynamoComponentDeployment: %v", err),
			},
		)
		if err != nil {
			return
		}
	}()

	// retrieve the dynamo component
	dynamoComponentCR := &v1alpha1.DynamoComponent{}
	err = r.Get(ctx, types.NamespacedName{Name: getK8sName(dynamoComponentDeployment.Spec.DynamoComponent), Namespace: dynamoComponentDeployment.Namespace}, dynamoComponentCR)
	if err != nil {
		logs.Error(err, "Failed to get DynamoComponent")
		return
	}

	// check if the component is ready
	if dynamoComponentCR.IsReady() {
		logs.Info(fmt.Sprintf("DynamoComponent %s ready", dynamoComponentDeployment.Spec.DynamoComponent))
		r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeNormal, "GetDynamoComponent", "DynamoComponent %s is ready", dynamoComponentDeployment.Spec.DynamoComponent)
		dynamoComponentDeployment, err = r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeDynamoComponentReady,
				Status:  metav1.ConditionTrue,
				Reason:  "Reconciling",
				Message: "DynamoComponent is ready",
			},
		)
		if err != nil {
			return
		}
	} else {
		logs.Info(fmt.Sprintf("DynamoComponent %s not ready", dynamoComponentDeployment.Spec.DynamoComponent))
		r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeWarning, "GetDynamoComponent", "DynamoComponent %s is not ready", dynamoComponentDeployment.Spec.DynamoComponent)
		_, err_ := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeDynamoComponentReady,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: "DynamoComponent not ready",
			},
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: "DynamoComponent not ready",
			},
		)
		err = err_
		return
	}

	modified := false

	// Reconcile PVC
	_, err = r.reconcilePVC(ctx, dynamoComponentDeployment)
	if err != nil {
		logs.Error(err, "Unable to create PVC", "crd", req.NamespacedName)
		return ctrl.Result{}, err
	}

	// create or update api-server deployment
	modified_, deployment, err := r.createOrUpdateOrDeleteDeployments(ctx, generateResourceOption{
		dynamoComponentDeployment: dynamoComponentDeployment,
		dynamoComponent:           dynamoComponentCR,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server hpa
	modified_, _, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoComponentDeployment: dynamoComponentDeployment,
		dynamoComponent:           dynamoComponentCR,
	}, r.generateHPA)
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server service
	modified_, err = r.createOrUpdateOrDeleteServices(ctx, generateResourceOption{
		dynamoComponentDeployment: dynamoComponentDeployment,
		dynamoComponent:           dynamoComponentCR,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	// create or update api-server ingresses
	modified_, err = r.createOrUpdateOrDeleteIngress(ctx, generateResourceOption{
		dynamoComponentDeployment: dynamoComponentDeployment,
		dynamoComponent:           dynamoComponentCR,
	})
	if err != nil {
		return
	}

	if modified_ {
		modified = true
	}

	if !modified {
		r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeNormal, "UpdateDynamoGraphDeployment", "No changes to dynamo deployment %s", dynamoComponentDeployment.Name)
	}

	logs.Info("Finished reconciling.")
	r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeNormal, "Update", "All resources updated!")
	err = r.computeAvailableStatusCondition(ctx, req, deployment)
	return
}

func (r *DynamoComponentDeploymentReconciler) FinalizeResource(ctx context.Context, dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) error {
	logger := log.FromContext(ctx)
	logger.Info("Finalizing the DynamoComponentDeployment", "dynamoComponentDeployment", dynamoComponentDeployment)
	if dynamoComponentDeployment.Spec.ServiceName != "" && dynamoComponentDeployment.Spec.DynamoNamespace != nil && *dynamoComponentDeployment.Spec.DynamoNamespace != "" {
		logger.Info("Deleting the etcd keys for the service", "service", dynamoComponentDeployment.Spec.ServiceName, "dynamoNamespace", *dynamoComponentDeployment.Spec.DynamoNamespace)
		err := r.EtcdStorage.DeleteKeys(ctx, fmt.Sprintf("/%s/components/%s", *dynamoComponentDeployment.Spec.DynamoNamespace, dynamoComponentDeployment.Spec.ServiceName))
		if err != nil {
			logger.Error(err, "Failed to delete the etcd keys for the service", "service", dynamoComponentDeployment.Spec.ServiceName, "dynamoNamespace", *dynamoComponentDeployment.Spec.DynamoNamespace)
			return err
		}
	}
	return nil
}

func (r *DynamoComponentDeploymentReconciler) computeAvailableStatusCondition(ctx context.Context, req ctrl.Request, deployment *appsv1.Deployment) error {
	logs := log.FromContext(ctx)
	if IsDeploymentReady(deployment) {
		logs.Info("Deployment is ready. Setting available status condition to true.")
		_, err := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
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
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
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

func (r *DynamoComponentDeploymentReconciler) reconcilePVC(ctx context.Context, crd *v1alpha1.DynamoComponentDeployment) (*corev1.PersistentVolumeClaim, error) {
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

func (r *DynamoComponentDeploymentReconciler) setStatusConditions(ctx context.Context, req ctrl.Request, conditions ...metav1.Condition) (dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, err error) {
	dynamoComponentDeployment = &v1alpha1.DynamoComponentDeployment{}
	maxRetries := 3
	for range maxRetries - 1 {
		if err = r.Get(ctx, req.NamespacedName, dynamoComponentDeployment); err != nil {
			err = errors.Wrap(err, "Failed to re-fetch DynamoComponentDeployment")
			return
		}
		for _, condition := range conditions {
			meta.SetStatusCondition(&dynamoComponentDeployment.Status.Conditions, condition)
		}
		if err = r.Status().Update(ctx, dynamoComponentDeployment); err != nil {
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
		err = errors.Wrap(err, "Failed to update DynamoComponentDeployment status")
		return
	}
	if err = r.Get(ctx, req.NamespacedName, dynamoComponentDeployment); err != nil {
		err = errors.Wrap(err, "Failed to re-fetch DynamoComponentDeployment")
		return
	}
	return
}

//nolint:nakedret
func (r *DynamoComponentDeploymentReconciler) createOrUpdateOrDeleteDeployments(ctx context.Context, opt generateResourceOption) (modified bool, depl *appsv1.Deployment, err error) {
	containsStealingTrafficDebugModeEnabled := checkIfContainsStealingTrafficDebugModeEnabled(opt.dynamoComponentDeployment)
	// create the main deployment
	modified, depl, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoComponentDeployment:               opt.dynamoComponentDeployment,
		dynamoComponent:                         opt.dynamoComponent,
		isStealingTrafficDebugModeEnabled:       false,
		containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
	}, r.generateDeployment)
	if err != nil {
		err = errors.Wrap(err, "create or update deployment")
		return
	}
	// create the debug deployment
	modified2, _, err := createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoComponentDeployment:               opt.dynamoComponentDeployment,
		dynamoComponent:                         opt.dynamoComponent,
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
func createOrUpdateResource[T client.Object](ctx context.Context, r *DynamoComponentDeploymentReconciler, opt generateResourceOption, generateResource func(ctx context.Context, opt generateResourceOption) (T, bool, error)) (modified bool, res T, err error) {
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
		r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, fmt.Sprintf("Get%s", resourceType), "Failed to get %s %s: %s", resourceType, resourceNamespace, err)
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
			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

		err = ctrl.SetControllerReference(opt.dynamoComponentDeployment, resource, r.Scheme)
		if err != nil {
			logs.Error(err, "Failed to set controller reference.")
			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, "SetControllerReference", "Failed to set controller reference for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

		r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeNormal, fmt.Sprintf("Create%s", resourceType), "Creating a new %s %s", resourceType, resourceNamespace)
		err = r.Create(ctx, resource)
		if err != nil {
			logs.Error(err, "Failed to create Resource.")
			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, fmt.Sprintf("Create%s", resourceType), "Failed to create %s %s: %s", resourceType, resourceNamespace, err)
			return
		}
		logs.Info(fmt.Sprintf("%s created.", resourceType))
		r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeNormal, fmt.Sprintf("Create%s", resourceType), "Created %s %s", resourceType, resourceNamespace)
		modified = true
		res = resource
	} else {
		logs.Info(fmt.Sprintf("%s found.", resourceType))
		if toDelete {
			logs.Info(fmt.Sprintf("%s not found. Deleting the existing one.", resourceType))
			err = r.Delete(ctx, oldResource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to delete %s.", resourceType))
				r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, fmt.Sprintf("Delete%s", resourceType), "Failed to delete %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
			logs.Info(fmt.Sprintf("%s deleted.", resourceType))
			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeNormal, fmt.Sprintf("Delete%s", resourceType), "Deleted %s %s", resourceType, resourceNamespace)
			modified = true
			return
		}

		var patchResult *patch.PatchResult
		patchResult, err = patch.DefaultPatchMaker.Calculate(oldResource, resource)
		if err != nil {
			logs.Error(err, "Failed to calculate patch.")
			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, fmt.Sprintf("CalculatePatch%s", resourceType), "Failed to calculate patch for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

		if !patchResult.IsEmpty() {
			logs.Info(fmt.Sprintf("%s spec is different. Updating %s. The patch result is: %s", resourceType, resourceType, patchResult.String()))

			err = errors.Wrapf(patch.DefaultAnnotator.SetLastAppliedAnnotation(resource), "set last applied annotation for resource %s", resourceName)
			if err != nil {
				logs.Error(err, "Failed to set last applied annotation.")
				r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, fmt.Sprintf("SetLastAppliedAnnotation%s", resourceType), "Failed to set last applied annotation for %s %s: %s", resourceType, resourceNamespace, err)
				return
			}

			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Updating %s %s", resourceType, resourceNamespace)
			resource.SetResourceVersion(oldResource.GetResourceVersion())
			err = r.Update(ctx, resource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to update %s.", resourceType))
				r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeWarning, fmt.Sprintf("Update%s", resourceType), "Failed to update %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
			logs.Info(fmt.Sprintf("%s updated.", resourceType))
			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Updated %s %s", resourceType, resourceNamespace)
			modified = true
			res = resource
		} else {
			logs.Info(fmt.Sprintf("%s spec is the same. Skipping update.", resourceType))
			r.Recorder.Eventf(opt.dynamoComponentDeployment, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Skipping update %s %s", resourceType, resourceNamespace)
			res = oldResource
		}
	}
	return
}

func getResourceAnnotations(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) map[string]string {
	resourceAnnotations := dynamoComponentDeployment.Spec.Annotations
	if resourceAnnotations == nil {
		resourceAnnotations = map[string]string{}
	}

	return resourceAnnotations
}

func checkIfIsDebugModeEnabled(annotations map[string]string) bool {
	if annotations == nil {
		return false
	}

	return annotations[KubeAnnotationEnableDebugMode] == commonconsts.KubeLabelValueTrue
}

func checkIfIsStealingTrafficDebugModeEnabled(annotations map[string]string) bool {
	if annotations == nil {
		return false
	}

	return annotations[KubeAnnotationEnableStealingTrafficDebugMode] == commonconsts.KubeLabelValueTrue
}

func checkIfIsDebugPodReceiveProductionTrafficEnabled(annotations map[string]string) bool {
	if annotations == nil {
		return false
	}

	return annotations[KubeAnnotationEnableDebugPodReceiveProductionTraffic] == commonconsts.KubeLabelValueTrue
}

func checkIfContainsStealingTrafficDebugModeEnabled(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) bool {
	return checkIfIsStealingTrafficDebugModeEnabled(dynamoComponentDeployment.Spec.Annotations)
}

//nolint:nakedret
func (r *DynamoComponentDeploymentReconciler) createOrUpdateOrDeleteServices(ctx context.Context, opt generateResourceOption) (modified bool, err error) {
	resourceAnnotations := getResourceAnnotations(opt.dynamoComponentDeployment)
	isDebugPodReceiveProductionTrafficEnabled := checkIfIsDebugPodReceiveProductionTrafficEnabled(resourceAnnotations)
	containsStealingTrafficDebugModeEnabled := checkIfContainsStealingTrafficDebugModeEnabled(opt.dynamoComponentDeployment)
	// main generic service
	modified, _, err = createOrUpdateResource(ctx, r, generateResourceOption{
		dynamoComponentDeployment:               opt.dynamoComponentDeployment,
		dynamoComponent:                         opt.dynamoComponent,
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
		dynamoComponentDeployment:               opt.dynamoComponentDeployment,
		dynamoComponent:                         opt.dynamoComponent,
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
		dynamoComponentDeployment:               opt.dynamoComponentDeployment,
		dynamoComponent:                         opt.dynamoComponent,
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

func (r *DynamoComponentDeploymentReconciler) createOrUpdateOrDeleteIngress(ctx context.Context, opt generateResourceOption) (modified bool, err error) {
	modified, _, err = createOrUpdateResource(ctx, r, opt, r.generateIngress)
	if err != nil {
		return
	}
	modified_, _, err := createOrUpdateResource(ctx, r, opt, r.generateVirtualService)
	if err != nil {
		return
	}
	modified = modified || modified_
	return
}

func (r *DynamoComponentDeploymentReconciler) generateIngress(ctx context.Context, opt generateResourceOption) (*networkingv1.Ingress, bool, error) {
	log := log.FromContext(ctx)
	log.Info("Starting generateIngress")

	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opt.dynamoComponentDeployment.Name,
			Namespace: opt.dynamoComponentDeployment.Namespace,
		},
	}

	if !opt.dynamoComponentDeployment.Spec.Ingress.Enabled || opt.dynamoComponentDeployment.Spec.Ingress.IngressControllerClassName == nil {
		log.Info("Ingress is not enabled")
		return ingress, true, nil
	}
	host := getIngressHost(opt.dynamoComponentDeployment.Spec.Ingress)

	ingress.Spec = networkingv1.IngressSpec{
		IngressClassName: opt.dynamoComponentDeployment.Spec.Ingress.IngressControllerClassName,
		Rules: []networkingv1.IngressRule{
			{
				Host: host,
				IngressRuleValue: networkingv1.IngressRuleValue{
					HTTP: &networkingv1.HTTPIngressRuleValue{
						Paths: []networkingv1.HTTPIngressPath{
							{
								Path:     "/",
								PathType: &[]networkingv1.PathType{networkingv1.PathTypePrefix}[0],
								Backend: networkingv1.IngressBackend{
									Service: &networkingv1.IngressServiceBackend{
										Name: opt.dynamoComponentDeployment.Name,
										Port: networkingv1.ServiceBackendPort{
											Number: 3000,
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

	if opt.dynamoComponentDeployment.Spec.Ingress.TLS != nil {
		ingress.Spec.TLS = []networkingv1.IngressTLS{
			{
				Hosts:      []string{host},
				SecretName: opt.dynamoComponentDeployment.Spec.Ingress.TLS.SecretName,
			},
		}
	}

	return ingress, false, nil
}

func (r *DynamoComponentDeploymentReconciler) generateVirtualService(ctx context.Context, opt generateResourceOption) (*networkingv1beta1.VirtualService, bool, error) {
	log := log.FromContext(ctx)
	log.Info("Starting generateVirtualService")

	vs := &networkingv1beta1.VirtualService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opt.dynamoComponentDeployment.Name,
			Namespace: opt.dynamoComponentDeployment.Namespace,
		},
	}

	vsEnabled := opt.dynamoComponentDeployment.Spec.Ingress.Enabled && opt.dynamoComponentDeployment.Spec.Ingress.UseVirtualService && opt.dynamoComponentDeployment.Spec.Ingress.VirtualServiceGateway != nil
	if !vsEnabled {
		log.Info("VirtualService is not enabled")
		return vs, true, nil
	}

	vs.Spec = istioNetworking.VirtualService{
		Hosts: []string{
			getIngressHost(opt.dynamoComponentDeployment.Spec.Ingress),
		},
		Gateways: []string{*opt.dynamoComponentDeployment.Spec.Ingress.VirtualServiceGateway},
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
							Host: opt.dynamoComponentDeployment.Name,
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

func (r *DynamoComponentDeploymentReconciler) getKubeName(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, _ *v1alpha1.DynamoComponent, debug bool) string {
	if debug {
		return fmt.Sprintf("%s-d", dynamoComponentDeployment.Name)
	}
	return dynamoComponentDeployment.Name
}

func (r *DynamoComponentDeploymentReconciler) getServiceName(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, _ *v1alpha1.DynamoComponent, debug bool) string {
	var kubeName string
	if debug {
		kubeName = fmt.Sprintf("%s-d", dynamoComponentDeployment.Name)
	} else {
		kubeName = fmt.Sprintf("%s-p", dynamoComponentDeployment.Name)
	}
	return kubeName
}

func (r *DynamoComponentDeploymentReconciler) getGenericServiceName(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, dynamoComponent *v1alpha1.DynamoComponent) string {
	return r.getKubeName(dynamoComponentDeployment, dynamoComponent, false)
}

func (r *DynamoComponentDeploymentReconciler) getKubeLabels(_ *v1alpha1.DynamoComponentDeployment, dynamoComponent *v1alpha1.DynamoComponent) map[string]string {
	labels := map[string]string{
		commonconsts.KubeLabelDynamoComponent: dynamoComponent.Name,
	}
	labels[commonconsts.KubeLabelDynamoComponentType] = commonconsts.DynamoApiServerComponentName
	return labels
}

func (r *DynamoComponentDeploymentReconciler) getKubeAnnotations(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, dynamoComponent *v1alpha1.DynamoComponent) map[string]string {
	dynamoComponentRepositoryName, dynamoComponentVersion := getDynamoComponentRepositoryNameAndDynamoComponentVersion(dynamoComponent)
	annotations := map[string]string{
		commonconsts.KubeAnnotationDynamoRepository: dynamoComponentRepositoryName,
		commonconsts.KubeAnnotationDynamoVersion:    dynamoComponentVersion,
	}
	var extraAnnotations map[string]string
	if dynamoComponentDeployment.Spec.ExtraPodMetadata != nil {
		extraAnnotations = dynamoComponentDeployment.Spec.ExtraPodMetadata.Annotations
	} else {
		extraAnnotations = map[string]string{}
	}
	for k, v := range extraAnnotations {
		annotations[k] = v
	}
	return annotations
}

//nolint:nakedret
func (r *DynamoComponentDeploymentReconciler) generateDeployment(ctx context.Context, opt generateResourceOption) (kubeDeployment *appsv1.Deployment, toDelete bool, err error) {
	kubeNs := opt.dynamoComponentDeployment.Namespace

	labels := r.getKubeLabels(opt.dynamoComponentDeployment, opt.dynamoComponent)

	annotations := r.getKubeAnnotations(opt.dynamoComponentDeployment, opt.dynamoComponent)

	kubeName := r.getKubeName(opt.dynamoComponentDeployment, opt.dynamoComponent, opt.isStealingTrafficDebugModeEnabled)

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

	resourceAnnotations := getResourceAnnotations(opt.dynamoComponentDeployment)
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
	replicas = opt.dynamoComponentDeployment.Spec.Replicas
	if opt.isStealingTrafficDebugModeEnabled {
		replicas = &[]int32{int32(1)}[0]
	}

	kubeDeployment.Spec = appsv1.DeploymentSpec{
		Replicas: replicas,
		Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				commonconsts.KubeLabelDynamoSelector: kubeName,
			},
		},
		Template: *podTemplateSpec,
		Strategy: strategy,
	}

	return
}

type generateResourceOption struct {
	dynamoComponentDeployment               *v1alpha1.DynamoComponentDeployment
	dynamoComponent                         *v1alpha1.DynamoComponent
	isStealingTrafficDebugModeEnabled       bool
	containsStealingTrafficDebugModeEnabled bool
	isDebugPodReceiveProductionTraffic      bool
	isGenericService                        bool
}

func (r *DynamoComponentDeploymentReconciler) generateHPA(ctx context.Context, opt generateResourceOption) (*autoscalingv2.HorizontalPodAutoscaler, bool, error) {
	labels := r.getKubeLabels(opt.dynamoComponentDeployment, opt.dynamoComponent)

	annotations := r.getKubeAnnotations(opt.dynamoComponentDeployment, opt.dynamoComponent)

	kubeName := r.getKubeName(opt.dynamoComponentDeployment, opt.dynamoComponent, false)

	kubeNs := opt.dynamoComponentDeployment.Namespace

	hpaConf := opt.dynamoComponentDeployment.Spec.Autoscaling

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

	return kubeHpa, false, nil
}

func getDynamoComponentRepositoryNameAndDynamoComponentVersion(dynamoComponent *v1alpha1.DynamoComponent) (repositoryName string, version string) {
	repositoryName, _, version = xstrings.Partition(dynamoComponent.Spec.DynamoComponent, ":")

	return
}

//nolint:gocyclo,nakedret
func (r *DynamoComponentDeploymentReconciler) generatePodTemplateSpec(ctx context.Context, opt generateResourceOption) (podTemplateSpec *corev1.PodTemplateSpec, err error) {
	podLabels := r.getKubeLabels(opt.dynamoComponentDeployment, opt.dynamoComponent)
	if opt.isStealingTrafficDebugModeEnabled {
		podLabels[commonconsts.KubeLabelDynamoDeploymentTargetType] = DeploymentTargetTypeDebug
	}

	podAnnotations := r.getKubeAnnotations(opt.dynamoComponentDeployment, opt.dynamoComponent)

	kubeName := r.getKubeName(opt.dynamoComponentDeployment, opt.dynamoComponent, opt.isStealingTrafficDebugModeEnabled)

	containerPort := commonconsts.DynamoServicePort

	var envs []corev1.EnvVar
	envsSeen := make(map[string]struct{})

	resourceAnnotations := opt.dynamoComponentDeployment.Spec.Annotations
	specEnvs := opt.dynamoComponentDeployment.Spec.Envs

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
			if env.Name == commonconsts.EnvDynamoServicePort {
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
			Name:  commonconsts.EnvDynamoServicePort,
			Value: fmt.Sprintf("%d", containerPort),
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

	var livenessProbe *corev1.Probe
	if opt.dynamoComponentDeployment.Spec.LivenessProbe != nil {
		livenessProbe = opt.dynamoComponentDeployment.Spec.LivenessProbe
	}

	var readinessProbe *corev1.Probe
	if opt.dynamoComponentDeployment.Spec.ReadinessProbe != nil {
		readinessProbe = opt.dynamoComponentDeployment.Spec.ReadinessProbe
	}

	volumes := make([]corev1.Volume, 0)
	volumeMounts := make([]corev1.VolumeMount, 0)

	args := make([]string, 0)

	args = append(args, "cd", "src", "&&", "uv", "run", "dynamo", "serve")

	// todo : remove this line when https://github.com/ai-dynamo/dynamo/issues/345 is fixed
	enableDependsOption := false
	if len(opt.dynamoComponentDeployment.Spec.ExternalServices) > 0 && enableDependsOption {
		serviceSuffix := fmt.Sprintf("%s.svc.cluster.local:3000", opt.dynamoComponentDeployment.Namespace)
		keys := make([]string, 0, len(opt.dynamoComponentDeployment.Spec.ExternalServices))

		for key := range opt.dynamoComponentDeployment.Spec.ExternalServices {
			keys = append(keys, key)
		}

		sort.Strings(keys)
		for _, key := range keys {
			service := opt.dynamoComponentDeployment.Spec.ExternalServices[key]

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

	if opt.dynamoComponentDeployment.Spec.ServiceName != "" {
		args = append(args, []string{"--service-name", opt.dynamoComponentDeployment.Spec.ServiceName}...)
		args = append(args, opt.dynamoComponentDeployment.Spec.DynamoTag)
		if opt.dynamoComponentDeployment.Spec.DynamoNamespace != nil && *opt.dynamoComponentDeployment.Spec.DynamoNamespace != "" {
			args = append(args, fmt.Sprintf("--%s.ServiceArgs.dynamo.namespace=%s", opt.dynamoComponentDeployment.Spec.ServiceName, *opt.dynamoComponentDeployment.Spec.DynamoNamespace))
		}
	}

	if len(opt.dynamoComponentDeployment.Spec.Envs) > 0 {
		for _, env := range opt.dynamoComponentDeployment.Spec.Envs {
			if env.Name == "DYNAMO_CONFIG_PATH" {
				args = append(args, "-f", env.Value)
			}
		}
	}

	dynamoResources := opt.dynamoComponentDeployment.Spec.Resources

	resources, err := getResourcesConfig(dynamoResources)
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
	if opt.dynamoComponentDeployment.Spec.PVC != nil {
		volumes = append(volumes, corev1.Volume{
			Name: getPvcName(opt.dynamoComponentDeployment, opt.dynamoComponentDeployment.Spec.PVC.Name),
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: getPvcName(opt.dynamoComponentDeployment, opt.dynamoComponentDeployment.Spec.PVC.Name),
				},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      getPvcName(opt.dynamoComponentDeployment, opt.dynamoComponentDeployment.Spec.PVC.Name),
			MountPath: *opt.dynamoComponentDeployment.Spec.PVC.MountPoint,
		})
	}

	imageName := opt.dynamoComponent.GetImage()
	if imageName == "" {
		return nil, errors.Errorf("image is not ready for component %s", opt.dynamoComponent.Name)
	}

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
				Name:          commonconsts.DynamoContainerPortName,
				ContainerPort: int32(containerPort), // nolint: gosec
			},
		},
		SecurityContext: mainContainerSecurityContext,
	}

	if opt.dynamoComponentDeployment.Spec.EnvFromSecret != nil {
		container.EnvFrom = []corev1.EnvFromSource{
			{
				SecretRef: &corev1.SecretEnvSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: *opt.dynamoComponentDeployment.Spec.EnvFromSecret,
					},
				},
			},
		}
	}

	if resourceAnnotations["nvidia.com/enable-container-privileged"] == commonconsts.KubeLabelValueTrue {
		if container.SecurityContext == nil {
			container.SecurityContext = &corev1.SecurityContext{}
		}
		container.SecurityContext.Privileged = &[]bool{true}[0]
	}

	if resourceAnnotations["nvidia.com/enable-container-ptrace"] == commonconsts.KubeLabelValueTrue {
		if container.SecurityContext == nil {
			container.SecurityContext = &corev1.SecurityContext{}
		}
		container.SecurityContext.Capabilities = &corev1.Capabilities{
			Add: []corev1.Capability{"SYS_PTRACE"},
		}
	}

	if resourceAnnotations["nvidia.com/run-container-as-root"] == commonconsts.KubeLabelValueTrue {
		if container.SecurityContext == nil {
			container.SecurityContext = &corev1.SecurityContext{}
		}
		container.SecurityContext.RunAsUser = &[]int64{0}[0]
	}

	containers = append(containers, container)

	debuggerImage := "python:3.12-slim"
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

	podLabels[commonconsts.KubeLabelDynamoSelector] = kubeName

	podSpec := corev1.PodSpec{
		Containers: containers,
		Volumes:    volumes,
	}

	podSpec.ImagePullSecrets = []corev1.LocalObjectReference{
		{
			Name: config.GetDockerRegistryConfig().SecretName,
		},
	}
	if opt.dynamoComponent.Spec.DockerConfigJSONSecretName != "" {
		podSpec.ImagePullSecrets = append(podSpec.ImagePullSecrets, corev1.LocalObjectReference{
			Name: opt.dynamoComponent.Spec.DockerConfigJSONSecretName,
		})
	}
	podSpec.ImagePullSecrets = append(podSpec.ImagePullSecrets, opt.dynamoComponent.Spec.ImagePullSecrets...)

	extraPodMetadata := opt.dynamoComponentDeployment.Spec.ExtraPodMetadata

	if extraPodMetadata != nil {
		for k, v := range extraPodMetadata.Annotations {
			podAnnotations[k] = v
		}

		for k, v := range extraPodMetadata.Labels {
			podLabels[k] = v
		}
	}

	extraPodSpec := opt.dynamoComponentDeployment.Spec.ExtraPodSpec

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
		err = r.List(ctx, serviceAccounts, client.InNamespace(opt.dynamoComponentDeployment.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelDynamoDeploymentPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			err = errors.Wrapf(err, "failed to list service accounts in namespace %s", opt.dynamoComponentDeployment.Namespace)
			return
		}
		if len(serviceAccounts.Items) > 0 {
			podSpec.ServiceAccountName = serviceAccounts.Items[0].Name
		} else {
			podSpec.ServiceAccountName = DefaultServiceAccountName
		}
	}

	if resourceAnnotations["nvidia.com/enable-host-ipc"] == commonconsts.KubeLabelValueTrue {
		podSpec.HostIPC = true
	}

	if resourceAnnotations["nvidia.com/enable-host-network"] == commonconsts.KubeLabelValueTrue {
		podSpec.HostNetwork = true
	}

	if resourceAnnotations["nvidia.com/enable-host-pid"] == commonconsts.KubeLabelValueTrue {
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
func (r *DynamoComponentDeploymentReconciler) generateService(ctx context.Context, opt generateResourceOption) (kubeService *corev1.Service, toDelete bool, err error) {
	var kubeName string
	if opt.isGenericService {
		kubeName = r.getGenericServiceName(opt.dynamoComponentDeployment, opt.dynamoComponent)
	} else {
		kubeName = r.getServiceName(opt.dynamoComponentDeployment, opt.dynamoComponent, opt.isStealingTrafficDebugModeEnabled)
	}

	kubeNs := opt.dynamoComponentDeployment.Namespace

	kubeService = &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeName,
			Namespace: kubeNs,
		},
	}

	if !opt.dynamoComponentDeployment.IsMainComponent() || (!opt.isGenericService && !opt.containsStealingTrafficDebugModeEnabled) {
		// if it's not the main component or if it's not a generic service and not contains stealing traffic debug mode enabled, we don't need to create the service
		return kubeService, true, nil
	}

	labels := r.getKubeLabels(opt.dynamoComponentDeployment, opt.dynamoComponent)

	selector := make(map[string]string)

	for k, v := range labels {
		selector[k] = v
	}

	if opt.isStealingTrafficDebugModeEnabled {
		selector[commonconsts.KubeLabelDynamoDeploymentTargetType] = DeploymentTargetTypeDebug
	}

	targetPort := intstr.FromString(commonconsts.DynamoContainerPortName)

	spec := corev1.ServiceSpec{
		Selector: selector,
		Ports: []corev1.ServicePort{
			{
				Name:       commonconsts.DynamoServicePortName,
				Port:       commonconsts.DynamoServicePort,
				TargetPort: targetPort,
				Protocol:   corev1.ProtocolTCP,
			},
		},
	}

	annotations := r.getKubeAnnotations(opt.dynamoComponentDeployment, opt.dynamoComponent)

	kubeService.ObjectMeta.Annotations = annotations
	kubeService.ObjectMeta.Labels = labels
	kubeService.Spec = spec

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
func (r *DynamoComponentDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {

	m := ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.DynamoComponentDeployment{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.Deployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		Owns(&corev1.Service{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&networkingv1.Ingress{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&corev1.PersistentVolumeClaim{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		WithEventFilter(controller_common.EphemeralDeploymentEventFilter(r.Config))

	if r.UseVirtualService {
		m.Owns(&networkingv1beta1.VirtualService{}, builder.WithPredicates(predicate.GenerationChangedPredicate{}))
	}
	m.Owns(&autoscalingv2.HorizontalPodAutoscaler{})
	return m.Complete(r)
}
