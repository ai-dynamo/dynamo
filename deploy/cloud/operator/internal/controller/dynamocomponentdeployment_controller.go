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
	"maps"
	"os"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"emperror.dev/errors"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/dynamo"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/webhook/validation"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	volcanov1beta1 "volcano.sh/apis/pkg/apis/scheduling/v1beta1"
)

const (
	DefaultClusterName                                   = "default"
	DefaultServiceAccountName                            = "default"
	KubeAnnotationDeploymentStrategy                     = "nvidia.com/deployment-strategy"
	KubeAnnotationDeploymentRollingUpdateMaxSurge        = "nvidia.com/deployment-rolling-update-max-surge"
	KubeAnnotationDeploymentRollingUpdateMaxUnavailable  = "nvidia.com/deployment-rolling-update-max-unavailable"
	KubeAnnotationEnableStealingTrafficDebugMode         = "nvidia.com/enable-stealing-traffic-debug-mode"
	KubeAnnotationEnableDebugMode                        = "nvidia.com/enable-debug-mode"
	KubeAnnotationEnableDebugPodReceiveProductionTraffic = "nvidia.com/enable-debug-pod-receive-production-traffic"
	DeploymentTargetTypeDebug                            = "debug"
)

// DynamoComponentDeploymentReconciler reconciles a DynamoComponentDeployment object
type DynamoComponentDeploymentReconciler struct {
	client.Client
	Recorder              record.EventRecorder
	Config                commonController.Config
	EtcdStorage           etcdStorage
	DockerSecretRetriever dockerSecretRetriever
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

	// Setup defer to handle errors and update status
	defer func() {
		if err == nil {
			return
		}
		reconcileErr := err
		logs.Error(reconcileErr, "Failed to reconcile DynamoComponentDeployment.")
		r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeWarning, "ReconcileError",
			"Failed to reconcile DynamoComponentDeployment: %v", reconcileErr)
		if _, statusErr := r.setStatusConditions(ctx, req,
			metav1.Condition{
				Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
				Status:  metav1.ConditionFalse,
				Reason:  "Reconciling",
				Message: fmt.Sprintf("Failed to reconcile DynamoComponentDeployment: %v", reconcileErr),
			},
		); statusErr != nil {
			logs.Error(statusErr, "Failed to update DynamoComponentDeployment status after reconcile error")
		}
	}()

	// Validate the DynamoComponentDeployment spec (defense in depth - only when webhooks are disabled)
	if !r.Config.WebhooksEnabled {
		validator := webhookvalidation.NewDynamoComponentDeploymentValidator(dynamoComponentDeployment)
		if _, validationErr := validator.Validate(); validationErr != nil {
			logs.Error(validationErr, "DynamoComponentDeployment validation failed, refusing to reconcile")

			// Set validation error condition
			meta.SetStatusCondition(&dynamoComponentDeployment.Status.Conditions, metav1.Condition{
				Type:               "Valid",
				Status:             metav1.ConditionFalse,
				ObservedGeneration: dynamoComponentDeployment.Generation,
				Reason:             "ValidationFailed",
				Message:            fmt.Sprintf("Validation failed: %v", validationErr),
			})

			// Update status and don't requeue (user must fix the spec)
			if statusErr := r.Status().Update(ctx, dynamoComponentDeployment); statusErr != nil {
				logs.Error(statusErr, "Failed to update DynamoComponentDeployment status with validation error")
				err = statusErr
				return ctrl.Result{}, err
			}

			// Record event for visibility
			r.Recorder.Event(dynamoComponentDeployment, corev1.EventTypeWarning, "ValidationFailed", validationErr.Error())

			// Don't requeue - user must fix the spec
			logs.Info("DynamoComponentDeployment is invalid, not reconciling until spec is fixed")
			err = nil
			return ctrl.Result{}, nil
		}

		// Set Valid condition to True and persist it
		meta.SetStatusCondition(&dynamoComponentDeployment.Status.Conditions, metav1.Condition{
			Type:               "Valid",
			Status:             metav1.ConditionTrue,
			ObservedGeneration: dynamoComponentDeployment.Generation,
			Reason:             "ValidationPassed",
			Message:            "DynamoComponentDeployment spec is valid",
		})
		if statusErr := r.Status().Update(ctx, dynamoComponentDeployment); statusErr != nil {
			logs.Error(statusErr, "Failed to update DynamoComponentDeployment status with validation success")
			err = statusErr
			return ctrl.Result{}, err
		}
	}

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

	// Create the appropriate workload resource based on deployment type
	var componentReconcileResult ComponentReconcileResult
	if r.Config.LWS.Enabled && dynamoComponentDeployment.IsMultinode() {
		componentReconcileResult, err = r.reconcileLeaderWorkerSetResources(ctx, dynamoComponentDeployment)
	} else {
		componentReconcileResult, err = r.reconcileDeploymentResources(ctx, dynamoComponentDeployment)
	}
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to reconcile the resources: %w", err)
	}
	modified := componentReconcileResult.modified

	// create or update api-server service
	serviceModified, err := r.createOrUpdateOrDeleteServices(ctx, generateResourceOption{
		dynamoComponentDeployment: dynamoComponentDeployment,
	})
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to create or update the service: %w", err)
	}

	// create or update headless service for model endpoint discovery
	componentMap := map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		dynamoComponentDeployment.Name: &dynamoComponentDeployment.Spec.DynamoComponentDeploymentSharedSpec,
	}
	if err := dynamo.ReconcileModelServicesForComponents(
		ctx,
		r,
		dynamoComponentDeployment,
		componentMap,
		dynamoComponentDeployment.Namespace,
	); err != nil {
		logs.Error(err, "Failed to reconcile model service")
		return ctrl.Result{}, err
	}

	// create or update api-server ingresses
	ingressModified, err := r.createOrUpdateOrDeleteIngress(ctx, generateResourceOption{
		dynamoComponentDeployment: dynamoComponentDeployment,
	})
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to create or update the ingress: %w", err)
	}

	if serviceModified || ingressModified {
		modified = true
	}

	if !modified {
		r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeNormal, "UpdateDynamoGraphDeployment", "No changes to dynamo deployment %s", dynamoComponentDeployment.Name)
	}

	logs.Info("Finished reconciling.")
	r.Recorder.Eventf(dynamoComponentDeployment, corev1.EventTypeNormal, "Update", "All resources updated!")

	err = r.setStatusConditionAndServiceReplicaStatus(ctx, dynamoComponentDeployment, componentReconcileResult)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to set status condition and service replica status: %w", err)
	}

	return
}

type ComponentReconcileResult struct {
	modified             bool
	status               metav1.ConditionStatus
	reason               string
	message              string
	serviceReplicaStatus *v1alpha1.ServiceReplicaStatus
}

func (r *DynamoComponentDeploymentReconciler) reconcileDeploymentResources(ctx context.Context, dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) (ComponentReconcileResult, error) {
	logger := log.FromContext(ctx)
	deploymentModified, deployment, err := r.createOrUpdateOrDeleteDeployments(ctx, generateResourceOption{
		dynamoComponentDeployment: dynamoComponentDeployment,
	})
	if err != nil {
		return ComponentReconcileResult{}, fmt.Errorf("failed to create or update the deployment: %w", err)
	}

	serviceReplicaStatus := &v1alpha1.ServiceReplicaStatus{
		ComponentKind:     v1alpha1.ComponentKindDeployment,
		ComponentName:     deployment.Name,
		Replicas:          deployment.Status.Replicas,
		UpdatedReplicas:   deployment.Status.UpdatedReplicas,
		ReadyReplicas:     &deployment.Status.ReadyReplicas,
		AvailableReplicas: &deployment.Status.AvailableReplicas,
	}

	if IsDeploymentReady(deployment) {
		logger.Info("Deployment is ready. Setting available status condition to true.")
		return ComponentReconcileResult{
			modified:             deploymentModified,
			status:               metav1.ConditionTrue,
			reason:               "DeploymentReady",
			message:              "Deployment is ready",
			serviceReplicaStatus: serviceReplicaStatus,
		}, nil
	}
	return ComponentReconcileResult{
		modified:             deploymentModified,
		status:               metav1.ConditionFalse,
		reason:               "DeploymentNotReady",
		message:              "Deployment is not ready",
		serviceReplicaStatus: serviceReplicaStatus,
	}, nil
}

func (r *DynamoComponentDeploymentReconciler) reconcileLeaderWorkerSetResources(ctx context.Context, dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) (ComponentReconcileResult, error) {
	logger := log.FromContext(ctx)

	desiredReplicas := int32(1)
	if dynamoComponentDeployment.Spec.Replicas != nil {
		desiredReplicas = *dynamoComponentDeployment.Spec.Replicas
	}

	anyModified := false
	leaderWorkerSets := make([]*leaderworkersetv1.LeaderWorkerSet, 0, desiredReplicas)
	for i := range int(desiredReplicas) {
		volcanoPodGroupModified, _, err := commonController.SyncResource(ctx, r, dynamoComponentDeployment, func(ctx context.Context) (*volcanov1beta1.PodGroup, bool, error) {
			return r.generateVolcanoPodGroup(ctx, generateResourceOption{
				dynamoComponentDeployment:               dynamoComponentDeployment,
				isStealingTrafficDebugModeEnabled:       false,
				containsStealingTrafficDebugModeEnabled: false,
				instanceID:                              &i,
			})
		})
		if err != nil {
			return ComponentReconcileResult{}, fmt.Errorf("failed to sync the PodGroup: %w", err)
		}

		leaderWorkerSetModified, lwsObj, err := commonController.SyncResource(ctx, r, dynamoComponentDeployment, func(ctx context.Context) (*leaderworkersetv1.LeaderWorkerSet, bool, error) {
			return r.generateLeaderWorkerSet(ctx, generateResourceOption{
				dynamoComponentDeployment:               dynamoComponentDeployment,
				isStealingTrafficDebugModeEnabled:       false,
				containsStealingTrafficDebugModeEnabled: false,
				instanceID:                              &i,
			})
		})
		if err != nil {
			return ComponentReconcileResult{}, fmt.Errorf("failed to sync the LeaderWorkerSet: %w", err)
		}

		if leaderWorkerSetModified || volcanoPodGroupModified {
			anyModified = true
		}
		leaderWorkerSets = append(leaderWorkerSets, lwsObj)
	}

	// Clean up any excess LeaderWorkerSets (if replicas were decreased)
	baseKubeName := r.getKubeName(dynamoComponentDeployment, false)
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
			return ComponentReconcileResult{}, fmt.Errorf("failed to get the LeaderWorkerSet for deletion: %w", err)
		}

		err = r.Delete(ctx, lwsToDelete)
		if err != nil {
			return ComponentReconcileResult{}, fmt.Errorf("failed to delete the LeaderWorkerSet: %w", err)
		}

		podGroupName := nextLWSName
		podGroupToDelete := &volcanov1beta1.PodGroup{}
		err = r.Get(ctx, types.NamespacedName{
			Name:      podGroupName,
			Namespace: dynamoComponentDeployment.Namespace,
		}, podGroupToDelete)

		if err != nil {
			if !k8serrors.IsNotFound(err) {
				logger.Error(err, "Failed to get PodGroup for deletion", "podGroupName", podGroupName)
			}
		} else {
			err = r.Delete(ctx, podGroupToDelete)
			if err != nil {
				logger.Error(err, "Failed to delete PodGroup", "podGroupName", podGroupName)
			}
		}

		anyModified = true
	}

	allReady := true
	lwsReplicaStatuses := []v1alpha1.ServiceReplicaStatus{}
	for _, leaderWorkerSet := range leaderWorkerSets {
		if !IsLeaderWorkerSetReady(leaderWorkerSet) {
			allReady = false
		}
		lwsReplicaStatuses = append(lwsReplicaStatuses, getLeaderWorkerSetReplicasStatus(leaderWorkerSet))
	}

	if allReady {
		return ComponentReconcileResult{
			modified:             anyModified,
			status:               metav1.ConditionTrue,
			reason:               "AllLeaderWorkerSetsReady",
			message:              "All LeaderWorkerSets are ready",
			serviceReplicaStatus: combineLWSReplicaStatuses(lwsReplicaStatuses),
		}, nil
	}
	return ComponentReconcileResult{
		modified:             anyModified,
		status:               metav1.ConditionFalse,
		reason:               "SomeLeaderWorkerSetsNotReady",
		message:              "Some LeaderWorkerSets are not ready",
		serviceReplicaStatus: combineLWSReplicaStatuses(lwsReplicaStatuses),
	}, nil

}

func (r *DynamoComponentDeploymentReconciler) setStatusConditionAndServiceReplicaStatus(ctx context.Context, dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, componentReconcileResult ComponentReconcileResult) error {
	condition := metav1.Condition{
		Type:    v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
		Status:  componentReconcileResult.status,
		Reason:  componentReconcileResult.reason,
		Message: componentReconcileResult.message,
	}

	meta.SetStatusCondition(&dynamoComponentDeployment.Status.Conditions, condition)
	dynamoComponentDeployment.Status.Service = componentReconcileResult.serviceReplicaStatus

	err := r.Status().Update(ctx, dynamoComponentDeployment)
	if err != nil {
		return fmt.Errorf("failed to update DynamoComponentDeployment status: %w", err)
	}
	return nil
}

func getLeaderWorkerSetReplicasStatus(leaderWorkerSet *leaderworkersetv1.LeaderWorkerSet) v1alpha1.ServiceReplicaStatus {
	return v1alpha1.ServiceReplicaStatus{
		ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
		ComponentName:   leaderWorkerSet.Name,
		Replicas:        leaderWorkerSet.Status.Replicas,
		UpdatedReplicas: leaderWorkerSet.Status.UpdatedReplicas,
		ReadyReplicas:   &leaderWorkerSet.Status.ReadyReplicas,
	}
}

func combineLWSReplicaStatuses(serviceReplicaStatuses []v1alpha1.ServiceReplicaStatus) *v1alpha1.ServiceReplicaStatus {
	if len(serviceReplicaStatuses) == 0 {
		return nil
	}

	firstServiceStatus := serviceReplicaStatuses[0]
	var readyReplicas int32 = 0
	if firstServiceStatus.ReadyReplicas != nil {
		readyReplicas = *firstServiceStatus.ReadyReplicas
	}
	for _, serviceReplicaStatus := range serviceReplicaStatuses[1:] {
		firstServiceStatus.Replicas += serviceReplicaStatus.Replicas
		firstServiceStatus.UpdatedReplicas += serviceReplicaStatus.UpdatedReplicas
		if serviceReplicaStatus.ReadyReplicas != nil {
			readyReplicas += *serviceReplicaStatus.ReadyReplicas
		}
	}

	firstServiceStatus.ReadyReplicas = &readyReplicas
	return &firstServiceStatus
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

	if opt.instanceID == nil {
		return nil, false, errors.New("generateVolcanoPodGroup: instanceID cannot be nil")
	}
	instanceID := *opt.instanceID

	if instanceID < 0 {
		return nil, false, fmt.Errorf("generateVolcanoPodGroup: instanceID cannot be negative, got %d", instanceID)
	}

	podGroupName := r.getKubeName(opt.dynamoComponentDeployment, opt.isStealingTrafficDebugModeEnabled)
	podGroupName = fmt.Sprintf("%s-%d", podGroupName, instanceID)

	kubeNs := opt.dynamoComponentDeployment.Namespace

	labels := make(map[string]string)
	labels["instance-id"] = fmt.Sprintf("%d", instanceID)

	minMember := opt.dynamoComponentDeployment.GetNumberOfNodes()

	podGroup := &volcanov1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podGroupName,
			Namespace: kubeNs,
			Labels:    labels,
		},
		Spec: volcanov1beta1.PodGroupSpec{
			MinMember: minMember,
		},
	}

	return podGroup, false, nil
}

func (r *DynamoComponentDeploymentReconciler) generateLeaderPodTemplateSpec(ctx context.Context, opt generateResourceOption, kubeName string, labels map[string]string, instanceID int) (*corev1.PodTemplateSpec, error) {
	leaderPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, opt, dynamo.RoleLeader)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate leader pod template")
	}

	maps.Copy(leaderPodTemplateSpec.ObjectMeta.Labels, labels)
	leaderPodTemplateSpec.ObjectMeta.Labels["role"] = "leader"
	leaderPodTemplateSpec.ObjectMeta.Labels["instance-id"] = fmt.Sprintf("%d", instanceID)
	delete(leaderPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	if leaderPodTemplateSpec.ObjectMeta.Annotations == nil {
		leaderPodTemplateSpec.ObjectMeta.Annotations = make(map[string]string)
	}
	leaderPodTemplateSpec.ObjectMeta.Annotations["scheduling.k8s.io/group-name"] = kubeName

	leaderPodTemplateSpec.Spec.SchedulerName = "volcano"

	err = checkMainContainer(&leaderPodTemplateSpec.Spec)

	if err != nil {
		return nil, errors.Wrap(err, "generateLeaderPodTemplateSpec: failed to check main container")
	}

	return leaderPodTemplateSpec, nil
}

func checkMainContainer(spec *corev1.PodSpec) error {

	if len(spec.Containers) == 0 {
		return errors.New("No containers found in pod spec")
	}

	mainContainerFound := false
	for _, container := range spec.Containers {
		if container.Name != commonconsts.MainContainerName {
			continue
		}

		if len(container.Command) == 0 {
			return errors.New("container Command cannot be nil for LWS pod")
		}

		if len(container.Args) == 0 {
			return errors.New("container Args cannot be empty for LWS pod")
		}

		mainContainerFound = true
		break
	}

	if !mainContainerFound {
		return errors.New("main container not found in pod spec")
	}

	return nil
}

func (r *DynamoComponentDeploymentReconciler) generateWorkerPodTemplateSpec(ctx context.Context, opt generateResourceOption, kubeName string, labels map[string]string, instanceID int) (*corev1.PodTemplateSpec, error) {
	workerPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, opt, dynamo.RoleWorker)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate worker pod template")
	}

	maps.Copy(workerPodTemplateSpec.ObjectMeta.Labels, labels)
	workerPodTemplateSpec.ObjectMeta.Labels["role"] = "worker"
	workerPodTemplateSpec.ObjectMeta.Labels["instance-id"] = fmt.Sprintf("%d", instanceID)
	delete(workerPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	workerPodTemplateSpec.Spec.SchedulerName = "volcano"

	if workerPodTemplateSpec.ObjectMeta.Annotations == nil {
		workerPodTemplateSpec.ObjectMeta.Annotations = make(map[string]string)
	}
	workerPodTemplateSpec.ObjectMeta.Annotations["scheduling.k8s.io/group-name"] = kubeName

	err = checkMainContainer(&workerPodTemplateSpec.Spec)

	if err != nil {
		return nil, errors.Wrap(err, "generateWorkerPodTemplateSpec: failed to check LWS worker main container")
	}

	if opt.dynamoComponentDeployment.Spec.Resources == nil || opt.dynamoComponentDeployment.Spec.Resources.Limits == nil || opt.dynamoComponentDeployment.Spec.Resources.Limits.GPU == "" {
		return nil, fmt.Errorf("generateWorkerPodTemplateSpec: GPU limit is not set for LWS worker pod")
	}

	return workerPodTemplateSpec, nil
}

// generateLeaderWorkerSet creates a LeaderWorkerSet resource from the DynamoComponentDeployment
func (r *DynamoComponentDeploymentReconciler) generateLeaderWorkerSet(ctx context.Context, opt generateResourceOption) (*leaderworkersetv1.LeaderWorkerSet, bool, error) {
	logs := log.FromContext(ctx)
	logs.Info("Generating LeaderWorkerSet")

	if opt.instanceID == nil {
		return nil, false, errors.New("generateLeaderWorkerSet: instanceID cannot be nil")
	}
	instanceID := *opt.instanceID

	if instanceID < 0 {
		return nil, false, fmt.Errorf("generateLeaderWorkerSet: instanceID cannot be negative, got %d", instanceID)
	}

	kubeName := r.getKubeName(opt.dynamoComponentDeployment, opt.isStealingTrafficDebugModeEnabled)
	kubeName = fmt.Sprintf("%s-%d", kubeName, instanceID)

	kubeNs := opt.dynamoComponentDeployment.Namespace
	labels := r.getKubeLabels(opt.dynamoComponentDeployment)

	if labels == nil {
		labels = make(map[string]string)
	}
	labels["instance-id"] = fmt.Sprintf("%d", instanceID)

	leaderWorkerSet := &leaderworkersetv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeName,
			Namespace: kubeNs,
			Labels:    labels,
		},
	}

	leaderPodLabels := make(map[string]string)
	for k, v := range labels {
		leaderPodLabels[k] = v
	}
	leaderPodTemplateSpec, err := r.generateLeaderPodTemplateSpec(ctx, opt, kubeName, leaderPodLabels, instanceID)
	if err != nil {
		return nil, false, errors.Wrap(err, "generateLeaderWorkerSet: failed to generate leader pod template")
	}

	workerPodLabels := make(map[string]string)
	for k, v := range labels {
		workerPodLabels[k] = v
	}
	workerPodTemplateSpec, err := r.generateWorkerPodTemplateSpec(ctx, opt, kubeName, workerPodLabels, instanceID)
	if err != nil {
		return nil, false, errors.Wrap(err, "generateLeaderWorkerSet: failed to generate worker pod template")
	}

	// Each individual LeaderWorkerSet always has exactly 1 replica
	singleReplica := int32(1)
	groupSize := opt.dynamoComponentDeployment.GetNumberOfNodes()

	leaderWorkerSet.Spec = leaderworkersetv1.LeaderWorkerSetSpec{
		Replicas:      &singleReplica,
		StartupPolicy: leaderworkersetv1.LeaderCreatedStartupPolicy,
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

func (r *DynamoComponentDeploymentReconciler) createOrUpdateOrDeleteIngress(ctx context.Context, opt generateResourceOption) (bool, error) {
	modified, _, err := commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*networkingv1.Ingress, bool, error) {
		return r.generateIngress(ctx, opt)
	})
	if err != nil {
		return false, err
	}
	if r.Config.IngressConfig.UseVirtualService() {
		modified_, _, err := commonController.SyncResource(ctx, r, opt.dynamoComponentDeployment, func(ctx context.Context) (*networkingv1beta1.VirtualService, bool, error) {
			return r.generateVirtualService(ctx, opt)
		})
		if err != nil {
			return false, err
		}
		return modified || modified_, nil
	}
	return modified, nil
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

	if opt.dynamoComponentDeployment.Spec.Ingress == nil || !opt.dynamoComponentDeployment.Spec.Ingress.Enabled || opt.dynamoComponentDeployment.Spec.Ingress.IngressControllerClassName == nil {
		log.Info("Ingress is not enabled")
		return ingress, true, nil
	}
	return dynamo.GenerateComponentIngress(ctx, opt.dynamoComponentDeployment.Name, opt.dynamoComponentDeployment.Namespace, *opt.dynamoComponentDeployment.Spec.Ingress), false, nil
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

	if !opt.dynamoComponentDeployment.Spec.Ingress.IsVirtualServiceEnabled() {
		log.Info("VirtualService is not enabled")
		return vs, true, nil
	}
	return dynamo.GenerateComponentVirtualService(ctx, opt.dynamoComponentDeployment.Name, opt.dynamoComponentDeployment.Namespace, *opt.dynamoComponentDeployment.Spec.Ingress), false, nil
}

func (r *DynamoComponentDeploymentReconciler) getKubeName(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, debug bool) string {
	if debug {
		return fmt.Sprintf("%s-d", dynamoComponentDeployment.Name)
	}
	return dynamoComponentDeployment.Name
}

func (r *DynamoComponentDeploymentReconciler) getServiceName(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, debug bool) string {
	var kubeName string
	if debug {
		kubeName = fmt.Sprintf("%s-d", dynamoComponentDeployment.Name)
	} else {
		kubeName = fmt.Sprintf("%s-p", dynamoComponentDeployment.Name)
	}
	return kubeName
}

func (r *DynamoComponentDeploymentReconciler) getGenericServiceName(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) string {
	return r.getKubeName(dynamoComponentDeployment, false)
}

func (r *DynamoComponentDeploymentReconciler) getKubeLabels(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) map[string]string {
	labels := map[string]string{}
	if dynamoComponentDeployment != nil {
		if dynamoComponentDeployment.Spec.Labels != nil {
			maps.Copy(labels, dynamoComponentDeployment.Spec.Labels)
		}
		if dynamoComponentDeployment.Labels != nil {
			maps.Copy(labels, dynamoComponentDeployment.Labels)
		}
		dynamo.AddBaseModelLabel(labels, dynamoComponentDeployment.Spec.ModelRef)
	}
	return labels
}

func (r *DynamoComponentDeploymentReconciler) getKubeAnnotations(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment) map[string]string {
	annotations := map[string]string{}
	if dynamoComponentDeployment != nil {
		if dynamoComponentDeployment.Spec.Annotations != nil {
			maps.Copy(annotations, dynamoComponentDeployment.Spec.Annotations)
		}
		if dynamoComponentDeployment.Spec.ExtraPodMetadata != nil && dynamoComponentDeployment.Spec.ExtraPodMetadata.Annotations != nil {
			maps.Copy(annotations, dynamoComponentDeployment.Spec.ExtraPodMetadata.Annotations)
		}
		dynamo.AddBaseModelAnnotation(annotations, dynamoComponentDeployment.Spec.ModelRef)
	}
	return annotations
}

//nolint:nakedret
func (r *DynamoComponentDeploymentReconciler) generateDeployment(ctx context.Context, opt generateResourceOption) (kubeDeployment *appsv1.Deployment, toDelete bool, err error) {
	kubeNs := opt.dynamoComponentDeployment.Namespace

	labels := r.getKubeLabels(opt.dynamoComponentDeployment)

	annotations := r.getKubeAnnotations(opt.dynamoComponentDeployment)

	kubeName := r.getKubeName(opt.dynamoComponentDeployment, opt.isStealingTrafficDebugModeEnabled)

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
	podTemplateSpec, err := r.generatePodTemplateSpec(ctx, opt, dynamo.RoleMain)
	if err != nil {
		return
	}

	maxSurge, maxUnavailable := getDeploymentRollingUpdateMaxSurgeAndMaxUnavailable(annotations)

	strategy := appsv1.DeploymentStrategy{
		Type: appsv1.RollingUpdateDeploymentStrategyType,
		RollingUpdate: &appsv1.RollingUpdateDeployment{
			MaxSurge:       &maxSurge,
			MaxUnavailable: &maxUnavailable,
		},
	}

	resourceAnnotations := getResourceAnnotations(opt.dynamoComponentDeployment)
	strategyStr := resourceAnnotations[KubeAnnotationDeploymentStrategy]
	if strategyStr != "" {
		strategyType := common.DeploymentStrategy(strategyStr)
		switch strategyType {
		case common.DeploymentStrategyRollingUpdate:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       &maxSurge,
					MaxUnavailable: &maxUnavailable,
				},
			}
		case common.DeploymentStrategyRecreate:
			strategy = appsv1.DeploymentStrategy{
				Type: appsv1.RecreateDeploymentStrategyType,
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

func getDeploymentRollingUpdateMaxSurgeAndMaxUnavailable(annotations map[string]string) (intstr.IntOrString, intstr.IntOrString) {
	maxSurge := intstr.FromString("25%")
	maxUnavailable := intstr.FromString("25%")

	if annotations[KubeAnnotationDeploymentRollingUpdateMaxSurge] != "" {
		maxSurge = intstr.Parse(annotations[KubeAnnotationDeploymentRollingUpdateMaxSurge])
	}
	if annotations[KubeAnnotationDeploymentRollingUpdateMaxUnavailable] != "" {
		maxUnavailable = intstr.Parse(annotations[KubeAnnotationDeploymentRollingUpdateMaxUnavailable])
	}

	return maxSurge, maxUnavailable
}

type generateResourceOption struct {
	dynamoComponentDeployment               *v1alpha1.DynamoComponentDeployment
	isStealingTrafficDebugModeEnabled       bool
	containsStealingTrafficDebugModeEnabled bool
	isDebugPodReceiveProductionTraffic      bool
	isGenericService                        bool
	instanceID                              *int
}

//nolint:gocyclo,nakedret
func (r *DynamoComponentDeploymentReconciler) generatePodTemplateSpec(ctx context.Context, opt generateResourceOption, role dynamo.Role) (podTemplateSpec *corev1.PodTemplateSpec, err error) {
	podLabels := r.getKubeLabels(opt.dynamoComponentDeployment)
	if opt.isStealingTrafficDebugModeEnabled {
		podLabels[commonconsts.KubeLabelDynamoDeploymentTargetType] = DeploymentTargetTypeDebug
	}

	// Convert user-provided metrics annotation into controller-managed label
	// By default (no annotation), metrics are enabled
	metricsAnnotationValue := ""
	if opt.dynamoComponentDeployment.Spec.Annotations != nil {
		metricsAnnotationValue = opt.dynamoComponentDeployment.Spec.Annotations[commonconsts.KubeAnnotationEnableMetrics]
	}
	switch metricsAnnotationValue {
	case commonconsts.KubeLabelValueFalse:
		// Explicitly disabled, don't add the label
	default:
		// Any other value (including empty) enables metrics
		podLabels[commonconsts.KubeLabelMetricsEnabled] = commonconsts.KubeLabelValueTrue
	}

	// Add label for the dynamo graph deployment on the pods themselves
	podLabels[commonconsts.KubeLabelDynamoGraphDeploymentName] = opt.dynamoComponentDeployment.Spec.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName]

	// Add component type label if specified
	if opt.dynamoComponentDeployment.Spec.ComponentType != "" {
		podLabels[commonconsts.KubeLabelDynamoComponentType] = opt.dynamoComponentDeployment.Spec.ComponentType
	}

	if opt.dynamoComponentDeployment.Spec.SubComponentType != "" {
		podLabels[commonconsts.KubeLabelDynamoSubComponentType] = opt.dynamoComponentDeployment.Spec.SubComponentType
	}

	podAnnotations := make(map[string]string)

	kubeName := r.getKubeName(opt.dynamoComponentDeployment, opt.isStealingTrafficDebugModeEnabled)

	resourceAnnotations := opt.dynamoComponentDeployment.Spec.Annotations

	if resourceAnnotations == nil {
		resourceAnnotations = make(map[string]string)
	}

	isDebugModeEnabled := checkIfIsDebugModeEnabled(resourceAnnotations)

	podSpec, err := dynamo.GenerateBasePodSpecForController(opt.dynamoComponentDeployment, r.DockerSecretRetriever, r.Config, role, commonconsts.MultinodeDeploymentTypeLWS)
	if err != nil {
		err = errors.Wrap(err, "failed to generate base pod spec")
		return nil, err
	}

	// Ensure we have at least one container (the main container should be there from GenerateBasePodSpec)
	if len(podSpec.Containers) == 0 {
		return nil, errors.New("no containers found in base pod spec")
	}

	debuggerImage := "python:3.12-slim"
	debuggerImage_ := os.Getenv("INTERNAL_IMAGES_DEBUGGER")
	if debuggerImage_ != "" {
		debuggerImage = debuggerImage_
	}

	if opt.isStealingTrafficDebugModeEnabled || isDebugModeEnabled {
		podSpec.Containers = append(podSpec.Containers, corev1.Container{
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

	extraPodMetadata := opt.dynamoComponentDeployment.Spec.ExtraPodMetadata

	if extraPodMetadata != nil {
		maps.Copy(podAnnotations, extraPodMetadata.Annotations)
		maps.Copy(podLabels, extraPodMetadata.Labels)
	}

	if podSpec.ServiceAccountName == "" {
		serviceAccounts := &corev1.ServiceAccountList{}
		err = r.List(ctx, serviceAccounts, client.InNamespace(opt.dynamoComponentDeployment.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelDynamoComponentPod: commonconsts.KubeLabelValueTrue,
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

	if opt.isStealingTrafficDebugModeEnabled || isDebugModeEnabled {
		podSpec.ShareProcessNamespace = &[]bool{true}[0]
	}

	podTemplateSpec = &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      podLabels,
			Annotations: podAnnotations,
		},
		Spec: *podSpec,
	}

	return
}

func (r *DynamoComponentDeploymentReconciler) generateService(opt generateResourceOption) (*corev1.Service, bool, error) {
	var kubeName string
	if opt.isGenericService {
		kubeName = r.getGenericServiceName(opt.dynamoComponentDeployment)
	} else {
		kubeName = r.getServiceName(opt.dynamoComponentDeployment, opt.isStealingTrafficDebugModeEnabled)
	}

	kubeNs := opt.dynamoComponentDeployment.Namespace

	kubeService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeName,
			Namespace: kubeNs,
		},
	}

	isK8sDiscovery := r.Config.IsK8sDiscoveryEnabled(opt.dynamoComponentDeployment.Spec.Annotations)

	// if discovery backend is k8s we want to create a service for each component
	// else, only create for the frontend component
	if !opt.isGenericService && !opt.containsStealingTrafficDebugModeEnabled && !(isK8sDiscovery || opt.dynamoComponentDeployment.IsFrontendComponent()) {
		// if it's not the main component or if it's not a generic service and not contains stealing traffic debug mode enabled, we don't need to create the service
		return kubeService, true, nil
	}

	labels := r.getKubeLabels(opt.dynamoComponentDeployment)

	if opt.dynamoComponentDeployment.Spec.DynamoNamespace == nil {
		return nil, false, fmt.Errorf("expected DynamoComponentDeployment %s to have a dynamoNamespace", opt.dynamoComponentDeployment.Name)
	}

	selector := map[string]string{
		commonconsts.KubeLabelDynamoComponentType: opt.dynamoComponentDeployment.Spec.ComponentType,
		commonconsts.KubeLabelDynamoNamespace:     *opt.dynamoComponentDeployment.Spec.DynamoNamespace,
	}
	// // If using LeaderWorkerSet, modify selector to only target leaders
	if opt.dynamoComponentDeployment.IsMultinode() {
		selector["role"] = "leader"
	}
	if opt.isStealingTrafficDebugModeEnabled {
		selector[commonconsts.KubeLabelDynamoDeploymentTargetType] = DeploymentTargetTypeDebug
	}
	if isK8sDiscovery {
		labels[commonconsts.KubeLabelDynamoDiscoveryBackend] = "kubernetes"
	}
	// Discovery is enabled for non frontend components
	if isK8sDiscovery && !opt.dynamoComponentDeployment.IsFrontendComponent() {
		labels[commonconsts.KubeLabelDynamoDiscoveryEnabled] = commonconsts.KubeLabelValueTrue
	}

	var servicePort corev1.ServicePort
	if opt.dynamoComponentDeployment.IsFrontendComponent() {
		servicePort = corev1.ServicePort{
			Name:       commonconsts.DynamoServicePortName,
			Port:       commonconsts.DynamoServicePort,
			TargetPort: intstr.FromString(commonconsts.DynamoContainerPortName),
			Protocol:   corev1.ProtocolTCP,
		}
	} else { // TODO: only for worker components
		servicePort = corev1.ServicePort{
			Name:       commonconsts.DynamoSystemPortName,
			Port:       commonconsts.DynamoSystemPort,
			TargetPort: intstr.FromString(commonconsts.DynamoSystemPortName),
			Protocol:   corev1.ProtocolTCP,
		}
	}

	spec := corev1.ServiceSpec{
		Selector: selector,
		Ports:    []corev1.ServicePort{servicePort},
	}

	annotations := r.getKubeAnnotations(opt.dynamoComponentDeployment)

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
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config))

	if r.Config.LWS.Enabled {
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

	if r.Config.IngressConfig.UseVirtualService() {
		m.Owns(&networkingv1beta1.VirtualService{}, builder.WithPredicates(predicate.GenerationChangedPredicate{}))
	}
	m.Owns(&autoscalingv2.HorizontalPodAutoscaler{})
	return m.Complete(r)
}

func (r *DynamoComponentDeploymentReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}
