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
	"github.com/huandu/xstrings"
	istioNetworking "istio.io/api/networking/v1beta1"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	volcanov1beta1 "volcano.sh/apis/pkg/apis/scheduling/v1beta1"
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
	KubernetesDeploymentStrategy                         = "kubernetes"

	KubeAnnotationDeploymentType = "nvidia.com/deployment-type"
	DeploymentTypeStandard       = "standard"
	DeploymentTypeLeaderWorker   = "leader-worker"
)

// DynamoComponentDeploymentReconciler reconciles a DynamoComponentDeployment object
type DynamoComponentDeploymentReconciler struct {
	client.Client
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

// +kubebuilder:rbac:groups=scheduling.volcano.sh,resources=podgroups,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=leaderworkerset.x-k8s.io,resources=leaderworkersets,verbs=get;list;watch;create;update;patch;delete

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

	// Determine deployment type
	deploymentType := GetDeploymentType(dynamoComponentDeployment)

	logs.Info("Using deployment type", "type", deploymentType)

	// Create the appropriate workload resource based on deployment type
	var leaderWorkerSets []*leaderworkersetv1.LeaderWorkerSet
	var deployment *appsv1.Deployment
	if r.Config.EnableLWS && deploymentType == DeploymentTypeLeaderWorker {
		desiredReplicas := int32(1)
		if dynamoComponentDeployment.Spec.Replicas != nil {
			desiredReplicas = *dynamoComponentDeployment.Spec.Replicas
		}

		anyModified := false

		for i := range int(desiredReplicas) {

			modified_, _, err := commonController.SyncResource(ctx, r, dynamoComponentDeployment, func(ctx context.Context) (*volcanov1beta1.PodGroup, bool, error) {
				return r.generateVolcanoPodGroup(ctx, generateResourceOption{
					dynamoComponentDeployment:               dynamoComponentDeployment,
					dynamoComponent:                         dynamoComponentCR,
					isStealingTrafficDebugModeEnabled:       false,
					containsStealingTrafficDebugModeEnabled: false,
					instanceID:                              i,
				})
			})

			if err != nil {
				return ctrl.Result{}, err
			}

			if modified_ {
				anyModified = true
			}

			modified_, lwsObj, err := commonController.SyncResource(ctx, r, dynamoComponentDeployment, func(ctx context.Context) (*leaderworkersetv1.LeaderWorkerSet, bool, error) {
				return r.generateLeaderWorkerSet(ctx, generateResourceOption{
					dynamoComponentDeployment:               dynamoComponentDeployment,
					dynamoComponent:                         dynamoComponentCR,
					isStealingTrafficDebugModeEnabled:       false,
					containsStealingTrafficDebugModeEnabled: false,
					instanceID:                              i,
				})
			})

			if err != nil {
				return ctrl.Result{}, err
			}

			if modified_ {
				anyModified = true
			}

			leaderWorkerSets = append(leaderWorkerSets, lwsObj)
		}

		// Clean up any excess LeaderWorkerSets (if replicas were decreased)
		baseKubeName := r.getKubeName(dynamoComponentDeployment, dynamoComponentCR, false)
		for i := int(desiredReplicas); ; i++ {
			// Try to find a LeaderWorkerSet with the next index
			nextLWSName := fmt.Sprintf("%s-%d", baseKubeName, i)
			lwsToDelete := &leaderworkersetv1.LeaderWorkerSet{}
			err := r.Get(ctx, types.NamespacedName{
				Name:      nextLWSName,
				Namespace: dynamoComponentDeployment.Namespace,
			}, lwsToDelete)

			if err != nil {
				if k8serrors.IsNotFound(err) {
					break
				}
				return ctrl.Result{}, err
			}

			err = r.Delete(ctx, lwsToDelete)
			if err != nil {
				return ctrl.Result{}, err
			}

			podGroupName := nextLWSName
			podGroupToDelete := &volcanov1beta1.PodGroup{}
			err = r.Get(ctx, types.NamespacedName{
				Name:      podGroupName,
				Namespace: dynamoComponentDeployment.Namespace,
			}, podGroupToDelete)

			if err != nil {
				if !k8serrors.IsNotFound(err) {
					logs.Error(err, "Failed to get PodGroup for deletion", "podGroupName", podGroupName)
				}
			} else {
				err = r.Delete(ctx, podGroupToDelete)
				if err != nil {
					logs.Error(err, "Failed to delete PodGroup", "podGroupName", podGroupName)
				}
			}

			anyModified = true
		}

		modified = anyModified

	} else {
		modified_, obj, err := r.createOrUpdateOrDeleteDeployments(ctx, generateResourceOption{
			dynamoComponentDeployment: dynamoComponentDeployment,
			dynamoComponent:           dynamoComponentCR,
		})

		if err != nil {
			return ctrl.Result{}, err
		}

		if modified_ {
			modified = true
		}

		deployment = obj

		// create or update api-server hpa
		modified_, _, err = commonController.SyncResource(ctx, r, dynamoComponentDeployment, func(ctx context.Context) (*autoscalingv2.HorizontalPodAutoscaler, bool, error) {
			return r.generateHPA(generateResourceOption{
				dynamoComponentDeployment: dynamoComponentDeployment,
				dynamoComponent:           dynamoComponentCR,
			})
		})
		if err != nil {
			return ctrl.Result{}, err
		}

		if modified_ {
			modified = true
		}

	}

	// create or update api-server service
	modified_, err := r.createOrUpdateOrDeleteServices(ctx, generateResourceOption{
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

	if deploymentType == DeploymentTypeLeaderWorker {
		err = r.computeAvailableStatusConditionForLeaderWorkerSets(ctx, req, leaderWorkerSets)
	} else {
		err = r.computeAvailableStatusCondition(ctx, req, deployment)
	}

	return
}

// computeAvailableStatusConditionForLeaderWorkerSet updates the status condition based on LeaderWorkerSet readiness
func (r *DynamoComponentDeploymentReconciler) computeAvailableStatusConditionForLeaderWorkerSets(ctx context.Context, req ctrl.Request, leaderWorkerSets []*leaderworkersetv1.LeaderWorkerSet) error {
	logs := log.FromContext(ctx)

	allReady := true
	for _, leaderWorkerSet := range leaderWorkerSets {
		if !IsLeaderWorkerSetReady(leaderWorkerSet) {
			allReady = false
			break
		}
	}

	if allReady {
		logs.Info("All LeaderWorkerSets are ready. Setting available status condition to true.")
		_, err := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionTrue,
				Reason:  "AllLeaderWorkerSetsReady",
				Message: "All LeaderWorkerSets are ready",
			},
		)
		return err
	} else {
		logs.Info("Not all LeaderWorkerSets are ready. Setting available status condition to false.")
		_, err := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "LeaderWorkerSetsNotReady",
				Message: "Not all LeaderWorkerSets are ready",
			},
		)
		return err
	}
}

// GetDeploymentType returns the deployment type from the annotations
// If not set, it returns the default DeploymentTypeStandard
func GetDeploymentType(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) string {
	resourceAnnotations := getResourceAnnotations(dynamoComponentDeployment)
	deploymentType := resourceAnnotations[KubeAnnotationDeploymentType]
	if deploymentType == "" {
		deploymentType = DeploymentTypeStandard
	}
	return deploymentType
}

// IsLeaderWorkerSetReady determines if a LeaderWorkerSet is fully ready and available
func IsLeaderWorkerSetReady(leaderWorkerSet *leaderworkersetv1.LeaderWorkerSet) bool {
	if leaderWorkerSet == nil {
		return false
	}

	desiredReplicas := int32(1)
	if leaderWorkerSet.Spec.Replicas != nil {
		desiredReplicas = *leaderWorkerSet.Spec.Replicas
	}

	// Special case: if no replicas are desired, the LeaderWorkerSet is considered ready
	if desiredReplicas == 0 {
		return true
	}

	status := leaderWorkerSet.Status

	if status.ReadyReplicas < desiredReplicas {
		return false
	}

	// Look for the Available condition specifically - this is defined in the CRD for LeaderWorkerSet
	for _, cond := range leaderWorkerSet.Status.Conditions {
		if cond.Type == string(leaderworkersetv1.LeaderWorkerSetAvailable) {
			return cond.Status == metav1.ConditionTrue
		}
	}

	return false
}

func (r *DynamoComponentDeploymentReconciler) generateVolcanoPodGroup(ctx context.Context, opt generateResourceOption) (*volcanov1beta1.PodGroup, bool, error) {
	logs := log.FromContext(ctx)
	logs.Info("Generating Volcano PodGroup")

	podGroupName := r.getKubeName(opt.dynamoComponentDeployment, opt.dynamoComponent, opt.isStealingTrafficDebugModeEnabled)
	instanceID := opt.instanceID
	podGroupName = fmt.Sprintf("%s-%d", podGroupName, instanceID)

	kubeNs := opt.dynamoComponentDeployment.Namespace
	labels := r.getKubeLabels(opt.dynamoComponentDeployment, opt.dynamoComponent)
	annotations := r.getKubeAnnotations(opt.dynamoComponentDeployment, opt.dynamoComponent)

	if labels == nil {
		labels = make(map[string]string)
	}
	labels["instance-id"] = fmt.Sprintf("%d", instanceID)

	// Set minMember to 2 (1 leader + 1 worker) as the minimum requirement for the set
	minMember := int32(2)

	podGroup := &volcanov1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podGroupName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: volcanov1beta1.PodGroupSpec{
			MinMember: minMember,
		},
	}

	return podGroup, false, nil
}

// generateLeaderWorkerSet creates a LeaderWorkerSet resource from the DynamoComponentDeployment
func (r *DynamoComponentDeploymentReconciler) generateLeaderWorkerSet(ctx context.Context, opt generateResourceOption) (*leaderworkersetv1.LeaderWorkerSet, bool, error) {
	logs := log.FromContext(ctx)
	logs.Info("Generating LeaderWorkerSet")

	// Extract the instance index from the name if it exists
	kubeName := r.getKubeName(opt.dynamoComponentDeployment, opt.dynamoComponent, opt.isStealingTrafficDebugModeEnabled)
	instanceID := opt.instanceID
	kubeName = fmt.Sprintf("%s-%d", kubeName, instanceID)

	kubeNs := opt.dynamoComponentDeployment.Namespace
	labels := r.getKubeLabels(opt.dynamoComponentDeployment, opt.dynamoComponent)
	annotations := r.getKubeAnnotations(opt.dynamoComponentDeployment, opt.dynamoComponent)

	// Add instance identifier to labels
	if labels == nil {
		labels = make(map[string]string)
	}
	labels["instance-id"] = fmt.Sprintf("%d", instanceID)

	leaderWorkerSet := &leaderworkersetv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:        kubeName,
			Namespace:   kubeNs,
			Labels:      labels,
			Annotations: annotations,
		},
	}

	leaderPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, opt)
	if err != nil {
		return nil, false, errors.Wrap(err, "failed to generate leader pod template")
	}

	if leaderPodTemplateSpec.ObjectMeta.Labels == nil {
		leaderPodTemplateSpec.ObjectMeta.Labels = make(map[string]string)
	}
	leaderPodTemplateSpec.ObjectMeta.Labels["role"] = "leader"
	leaderPodTemplateSpec.ObjectMeta.Labels["instance-id"] = fmt.Sprintf("%d", instanceID)
	delete(leaderPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	if leaderPodTemplateSpec.ObjectMeta.Annotations == nil {
		leaderPodTemplateSpec.ObjectMeta.Annotations = make(map[string]string)
	}
	leaderPodTemplateSpec.ObjectMeta.Annotations["scheduling.k8s.io/group-name"] = kubeName

	if leaderPodTemplateSpec.Spec.Containers[0].Command == nil {
		return nil, false, errors.New("container Command cannot be nil for Ray leader pod")
	}

	if len(leaderPodTemplateSpec.Spec.Containers[0].Args) == 0 {
		return nil, false, errors.New("container Args cannot be empty for Ray leader pod")
	}

	currentArgs := leaderPodTemplateSpec.Spec.Containers[0].Args[0]
	leaderPodTemplateSpec.Spec.Containers[0].Args[0] = "ray start --head --port=6379 && " + currentArgs

	workerPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, opt)
	if err != nil {
		return nil, false, errors.Wrap(err, "failed to generate worker pod template")
	}

	if workerPodTemplateSpec.ObjectMeta.Labels == nil {
		workerPodTemplateSpec.ObjectMeta.Labels = make(map[string]string)
	}
	workerPodTemplateSpec.ObjectMeta.Labels["role"] = "worker"
	workerPodTemplateSpec.ObjectMeta.Labels["instance-id"] = fmt.Sprintf("%d", instanceID)
	delete(workerPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	if workerPodTemplateSpec.ObjectMeta.Annotations == nil {
		workerPodTemplateSpec.ObjectMeta.Annotations = make(map[string]string)
	}
	workerPodTemplateSpec.ObjectMeta.Annotations["scheduling.k8s.io/group-name"] = kubeName

	if workerPodTemplateSpec.Spec.Containers[0].Command == nil {
		return nil, false, errors.New("container Command cannot be nil for Ray worker pod")
	}

	if len(workerPodTemplateSpec.Spec.Containers[0].Args) == 0 {
		return nil, false, errors.New("container Args cannot be empty for Ray worker pod")
	}

	currentArgs = workerPodTemplateSpec.Spec.Containers[0].Args[0]
	workerPodTemplateSpec.Spec.Containers[0].Args[0] = "ray start --address=$(LWS_LEADER_ADDRESS):6379 && " + currentArgs

	// Each individual LeaderWorkerSet always has exactly 1 replica
	singleReplica := int32(1)
	groupSize := int32(2)

	leaderWorkerSet.Spec = leaderworkersetv1.LeaderWorkerSetSpec{
		Replicas: &singleReplica,
		LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
			LeaderTemplate: leaderPodTemplateSpec,
			WorkerTemplate: *workerPodTemplateSpec,
			Size:           &groupSize,
		},
	}

	return leaderWorkerSet, false, nil
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
		if err := controllerutil.SetControllerReference(crd, pvc, r.Client.Scheme()); err != nil {
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
	modified, depl, err = commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*appsv1.Deployment, bool, error) {
		return r.generateDeployment(ctx, generateResourceOption{
			dynamoComponentDeployment:               opt.dynamoComponentDeployment,
			dynamoComponent:                         opt.dynamoComponent,
			isStealingTrafficDebugModeEnabled:       false,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
		})
	})
	if err != nil {
		err = errors.Wrap(err, "create or update deployment")
		return
	}
	// create the debug deployment
	modified2, _, err := commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*appsv1.Deployment, bool, error) {
		return r.generateDeployment(ctx, generateResourceOption{
			dynamoComponentDeployment:               opt.dynamoComponentDeployment,
			dynamoComponent:                         opt.dynamoComponent,
			isStealingTrafficDebugModeEnabled:       true,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
		})
	})
	if err != nil {
		err = errors.Wrap(err, "create or update debug deployment")
	}
	modified = modified || modified2
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
	modified, _, err = commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*corev1.Service, bool, error) {
		return r.generateService(generateResourceOption{
			dynamoComponentDeployment:               opt.dynamoComponentDeployment,
			dynamoComponent:                         opt.dynamoComponent,
			isStealingTrafficDebugModeEnabled:       false,
			isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
			isGenericService:                        true,
		})
	})
	if err != nil {
		return
	}

	// debug production service (if enabled)
	modified_, _, err := commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*corev1.Service, bool, error) {
		return r.generateService(generateResourceOption{
			dynamoComponentDeployment:               opt.dynamoComponentDeployment,
			dynamoComponent:                         opt.dynamoComponent,
			isStealingTrafficDebugModeEnabled:       false,
			isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
			isGenericService:                        false,
		})
	})
	if err != nil {
		return
	}
	modified = modified || modified_
	// debug service (if enabled)
	modified_, _, err = commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*corev1.Service, bool, error) {
		return r.generateService(generateResourceOption{
			dynamoComponentDeployment:               opt.dynamoComponentDeployment,
			dynamoComponent:                         opt.dynamoComponent,
			isStealingTrafficDebugModeEnabled:       true,
			isDebugPodReceiveProductionTraffic:      isDebugPodReceiveProductionTrafficEnabled,
			containsStealingTrafficDebugModeEnabled: containsStealingTrafficDebugModeEnabled,
			isGenericService:                        false,
		})
	})
	if err != nil {
		return
	}
	modified = modified || modified_
	return
}

func (r *DynamoComponentDeploymentReconciler) createOrUpdateOrDeleteIngress(ctx context.Context, opt generateResourceOption) (modified bool, err error) {
	modified, _, err = commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*networkingv1.Ingress, bool, error) {
		return r.generateIngress(ctx, opt)
	})
	if err != nil {
		return
	}
	modified_, _, err := commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*networkingv1beta1.VirtualService, bool, error) {
		return r.generateVirtualService(ctx, opt)
	})
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
											Number: commonconsts.DynamoServicePort,
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
								Number: commonconsts.DynamoServicePort,
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
	instanceID                              int
}

func (r *DynamoComponentDeploymentReconciler) generateHPA(opt generateResourceOption) (*autoscalingv2.HorizontalPodAutoscaler, bool, error) {
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
		serviceSuffix := fmt.Sprintf("%s.svc.cluster.local:%d", opt.dynamoComponentDeployment.Namespace, containerPort)
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
		args = append(args, fmt.Sprintf("--%s.environment=%s", opt.dynamoComponentDeployment.Spec.ServiceName, KubernetesDeploymentStrategy))
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

func (r *DynamoComponentDeploymentReconciler) generateService(opt generateResourceOption) (*corev1.Service, bool, error) {
	var kubeName string
	if opt.isGenericService {
		kubeName = r.getGenericServiceName(opt.dynamoComponentDeployment, opt.dynamoComponent)
	} else {
		kubeName = r.getServiceName(opt.dynamoComponentDeployment, opt.dynamoComponent, opt.isStealingTrafficDebugModeEnabled)
	}

	kubeNs := opt.dynamoComponentDeployment.Namespace

	kubeService := &corev1.Service{
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

	// Check if we're using LeaderWorkerSet
	deploymentType := GetDeploymentType(opt.dynamoComponentDeployment)

	// If using LeaderWorkerSet, modify selector to only target leaders
	if deploymentType == DeploymentTypeLeaderWorker {
		selector["role"] = "leader"
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

	return kubeService, false, nil
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

	if r.Config.EnableLWS {
		m.Owns(&leaderworkersetv1.LeaderWorkerSet{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the LeaderWorkerSet
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
			Owns(&volcanov1beta1.PodGroup{}, builder.WithPredicates(predicate.Funcs{
				// ignore creation cause we don't want to be called again after we create the LeaderWorkerSet
				CreateFunc:  func(ce event.CreateEvent) bool { return false },
				DeleteFunc:  func(de event.DeleteEvent) bool { return true },
				UpdateFunc:  func(de event.UpdateEvent) bool { return true },
				GenericFunc: func(ge event.GenericEvent) bool { return true },
			}))
	}

	if r.UseVirtualService {
		m.Owns(&networkingv1beta1.VirtualService{}, builder.WithPredicates(predicate.GenerationChangedPredicate{}))
	}
	m.Owns(&autoscalingv2.HorizontalPodAutoscaler{})
	return m.Complete(r)
}

func (r *DynamoComponentDeploymentReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}
