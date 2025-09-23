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

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/secret"

	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/dynamo"
)

type State string
type Reason string
type Message string

const (
	FailedState  State = "failed"
	ReadyState   State = "successful"
	PendingState State = "pending"
)

type etcdStorage interface {
	DeleteKeys(ctx context.Context, prefix string) error
}

// DynamoGraphDeploymentReconciler reconciles a DynamoGraphDeployment object
type DynamoGraphDeploymentReconciler struct {
	client.Client
	Config                commonController.Config
	Recorder              record.EventRecorder
	DockerSecretRetriever dockerSecretRetriever
	ScaleClient           scale.ScalesGetter
	MPISecretReplicator   *secret.SecretReplicator
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=grove.io,resources=podcliquesets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=grove.io,resources=podcliques/scale,verbs=get;update;patch
// +kubebuilder:rbac:groups=grove.io,resources=podcliquescalinggroups/scale,verbs=get;update;patch
// +kubebuilder:rbac:groups=scheduling.run.ai,resources=queues,verbs=get;list

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the DynamoGraphDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *DynamoGraphDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var err error
	reason := Reason("undefined")
	message := Message("")
	state := PendingState
	// retrieve the CRD
	dynamoDeployment := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err = r.Get(ctx, req.NamespacedName, dynamoDeployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if err != nil {
		// not found, nothing to do
		return ctrl.Result{}, nil
	}

	defer func() {
		if err != nil {
			state = FailedState
			message = Message(err.Error())
			logger.Error(err, "Reconciliation failed")
		}
		dynamoDeployment.SetState(string(state))

		readyStatus := metav1.ConditionFalse
		if state == ReadyState {
			readyStatus = metav1.ConditionTrue
		}

		// Update Ready condition
		dynamoDeployment.AddStatusCondition(metav1.Condition{
			Type:               "Ready",
			Status:             readyStatus,
			Reason:             string(reason),
			Message:            string(message),
			LastTransitionTime: metav1.Now(),
		})

		err = r.Status().Update(ctx, dynamoDeployment)
		if err != nil {
			logger.Error(err, "Unable to update the CRD status", "crd", req.NamespacedName)
		}
		logger.Info("Reconciliation done")
	}()

	deleted, err := commonController.HandleFinalizer(ctx, dynamoDeployment, r.Client, r)
	if err != nil {
		logger.Error(err, "failed to handle the finalizer")
		reason = "failed_to_handle_the_finalizer"
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}
	state, reason, message, err = r.reconcileResources(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "failed to reconcile the resources")
		reason = "failed_to_reconcile_the_resources"
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

type Resource interface {
	IsReady() (ready bool, reason string)
	GetName() string
}

func (r *DynamoGraphDeploymentReconciler) reconcileResources(ctx context.Context, dynamoDeployment *nvidiacomv1alpha1.DynamoGraphDeployment) (State, Reason, Message, error) {
	logger := log.FromContext(ctx)
	// Orchestrator selection via single boolean annotation: nvidia.com/enable-grove
	// Unset or not "false": Grove if available; else component mode
	// "false": component mode (multinode -> LWS; single-node -> standard)
	enableGrove := true
	if dynamoDeployment.Annotations != nil && strings.ToLower(dynamoDeployment.Annotations[consts.KubeAnnotationEnableGrove]) == consts.KubeLabelValueFalse {
		enableGrove = false
	}

	// Determine if any service is multinode
	hasMultinode := dynamoDeployment.HasAnyMultinodeService()

	// Always ensure MPI SSH secret is available in this namespace
	if r.MPISecretReplicator != nil {
		err := r.MPISecretReplicator.Replicate(ctx, dynamoDeployment.Namespace)
		if err != nil {
			logger.Error(err, "Failed to replicate MPI secret", "namespace", dynamoDeployment.Namespace)
			return "", "", "", fmt.Errorf("failed to replicate MPI secret: %w", err)
		}
	}

	if enableGrove && r.Config.Grove.Enabled {
		logger.Info("Reconciling Grove resources", "enableGrove", enableGrove, "groveEnabled", r.Config.Grove.Enabled, "hasMultinode", hasMultinode, "lwsEnabled", r.Config.LWS.Enabled)
		return r.reconcileGroveResources(ctx, dynamoDeployment)
	}
	if hasMultinode && !r.Config.LWS.Enabled {
		err := fmt.Errorf("no multinode orchestrator available")
		logger.Error(err, err.Error(), "hasMultinode", hasMultinode, "lwsEnabled", r.Config.LWS.Enabled, "enableGrove", enableGrove, "groveEnabled", r.Config.Grove.Enabled)
		return "", "", "", err
	}
	logger.Info("Reconciling Dynamo components deployments", "hasMultinode", hasMultinode, "lwsEnabled", r.Config.LWS.Enabled, "enableGrove", enableGrove, "groveEnabled", r.Config.Grove.Enabled)
	return r.reconcileDynamoComponentsDeployments(ctx, dynamoDeployment)

}

// scaleGroveResource scales a Grove resource using the generic scaling function
func (r *DynamoGraphDeploymentReconciler) scaleGroveResource(ctx context.Context, resourceName, namespace string, newReplicas int32, resourceType string) error {
	logger := log.FromContext(ctx)
	// Determine the GroupVersionResource based on resource type
	var gvr schema.GroupVersionResource
	switch resourceType {
	case "PodClique":
		gvr = consts.PodCliqueGVR
	case "PodCliqueScalingGroup":
		gvr = consts.PodCliqueScalingGroupGVR
	default:
		return fmt.Errorf("unsupported Grove resource type: %s", resourceType)
	}

	// Use the generic scaling function
	err := commonController.ScaleResource(ctx, r.ScaleClient, gvr, namespace, resourceName, newReplicas)
	if err != nil {
		if errors.IsNotFound(err) {
			// Resource doesn't exist yet - this is normal during initial creation when Grove is still creating the resources asynchronously
			logger.V(1).Info("Grove resource not found yet, skipping scaling for now - will retry on next reconciliation", "gvr", gvr, "name", resourceName, "namespace", namespace)
			return nil
		}
	}
	return err
}

// reconcileGroveScaling handles scaling operations for Grove resources based on service replica changes
func (r *DynamoGraphDeploymentReconciler) reconcileGroveScaling(ctx context.Context, dynamoDeployment *nvidiacomv1alpha1.DynamoGraphDeployment) error {
	logger := log.FromContext(ctx)
	logger.V(1).Info("Reconciling Grove scaling operations")

	replicaIndex := 0
	for serviceName, component := range dynamoDeployment.Spec.Services {
		// Skip if replicas are not specified
		if component.Replicas == nil {
			continue
		}

		numberOfNodes := component.GetNumberOfNodes()
		isMultinode := numberOfNodes > 1

		if isMultinode {
			// Scale PodCliqueScalingGroup for multinode services
			// Grove naming pattern: {DGD.name}-{replicaIndex}-{serviceName}
			resourceName := fmt.Sprintf("%s-%d-%s", dynamoDeployment.Name, replicaIndex, strings.ToLower(serviceName))
			err := r.scaleGroveResource(ctx,
				resourceName,
				dynamoDeployment.Namespace,
				*component.Replicas,
				"PodCliqueScalingGroup")
			if err != nil {
				logger.Error(err, "Failed to scale PodCliqueScalingGroup", "serviceName", serviceName, "resourceName", resourceName, "replicas", *component.Replicas)
				return fmt.Errorf("failed to scale PodCliqueScalingGroup %s: %w", resourceName, err)
			}
		} else {
			// Scale individual PodClique for single-node services
			// Grove naming pattern: {DGD.name}-{replicaIndex}-{serviceName}
			resourceName := fmt.Sprintf("%s-%d-%s", dynamoDeployment.Name, replicaIndex, strings.ToLower(serviceName))
			err := r.scaleGroveResource(ctx,
				resourceName,
				dynamoDeployment.Namespace,
				*component.Replicas,
				"PodClique")
			if err != nil {
				logger.Error(err, "Failed to scale PodClique", "serviceName", serviceName, "resourceName", resourceName, "replicas", *component.Replicas)
				return fmt.Errorf("failed to scale PodClique %s: %w", resourceName, err)
			}
		}
	}

	logger.V(1).Info("Successfully reconciled Grove scaling operations")
	return nil
}

func (r *DynamoGraphDeploymentReconciler) reconcileGroveResources(ctx context.Context, dynamoDeployment *nvidiacomv1alpha1.DynamoGraphDeployment) (State, Reason, Message, error) {
	logger := log.FromContext(ctx)
	// generate the dynamoComponentsDeployments from the config
	groveGangSet, err := dynamo.GenerateGrovePodCliqueSet(ctx, dynamoDeployment, r.Config, r.DockerSecretRetriever)
	if err != nil {
		logger.Error(err, "failed to generate the Grove GangSet")
		return "", "", "", fmt.Errorf("failed to generate the Grove GangSet: %w", err)
	}
	_, syncedGroveGangSet, err := commonController.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*grovev1alpha1.PodCliqueSet, bool, error) {
		return groveGangSet, false, nil
	})
	if err != nil {
		logger.Error(err, "failed to sync the Grove GangSet")
		return "", "", "", fmt.Errorf("failed to sync the Grove GangSet: %w", err)
	}
	groveGangSetAsResource := commonController.WrapResource(
		syncedGroveGangSet,
		func() (bool, string) {
			// Grove readiness: all underlying PodCliques and PodCliqueScalingGroups have replicas == availableReplicas
			allComponentsReady, reason := dynamo.EvaluateAllComponentsReady(ctx, r.Client, dynamoDeployment)
			if !allComponentsReady {
				return false, reason
			}
			return true, ""
		},
	)

	// Handle Grove scaling operations after structural changes
	if err := r.reconcileGroveScaling(ctx, dynamoDeployment); err != nil {
		logger.Error(err, "failed to reconcile Grove scaling")
		return "", "", "", fmt.Errorf("failed to reconcile Grove scaling: %w", err)
	}

	resources := []Resource{groveGangSetAsResource}
	for componentName, component := range dynamoDeployment.Spec.Services {
		if component.ComponentType == consts.ComponentTypeFrontend {
			// generate the main component service
			mainComponentService, err := dynamo.GenerateComponentService(ctx, dynamo.GetDynamoComponentName(dynamoDeployment, componentName), dynamoDeployment.Namespace)
			if err != nil {
				logger.Error(err, "failed to generate the main component service")
				return "", "", "", fmt.Errorf("failed to generate the main component service: %w", err)
			}
			_, syncedMainComponentService, err := commonController.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*corev1.Service, bool, error) {
				return mainComponentService, false, nil
			})
			if err != nil {
				logger.Error(err, "failed to sync the main component service")
				return "", "", "", fmt.Errorf("failed to sync the main component service: %w", err)
			}
			mainComponentServiceAsResource := commonController.WrapResource(syncedMainComponentService,
				func() (bool, string) {
					return true, ""
				})
			resources = append(resources, mainComponentServiceAsResource)
			// generate the main component ingress
			ingressSpec := dynamo.GenerateDefaultIngressSpec(dynamoDeployment, r.Config.IngressConfig)
			if component.Ingress != nil {
				ingressSpec = *component.Ingress
			}
			mainComponentIngress := dynamo.GenerateComponentIngress(ctx, dynamo.GetDynamoComponentName(dynamoDeployment, componentName), dynamoDeployment.Namespace, ingressSpec)
			_, syncedMainComponentIngress, err := commonController.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*networkingv1.Ingress, bool, error) {
				if !ingressSpec.Enabled || ingressSpec.IngressControllerClassName == nil {
					logger.Info("Ingress is not enabled")
					return mainComponentIngress, true, nil
				}
				return mainComponentIngress, false, nil
			})
			if err != nil {
				logger.Error(err, "failed to sync the main component ingress")
				return "", "", "", fmt.Errorf("failed to sync the main component ingress: %w", err)
			}
			resources = append(resources, commonController.WrapResource(syncedMainComponentIngress,
				func() (bool, string) {
					return true, ""
				}))
			// generate the main component virtual service
			if r.Config.IngressConfig.UseVirtualService() {
				mainComponentVirtualService := dynamo.GenerateComponentVirtualService(ctx, dynamo.GetDynamoComponentName(dynamoDeployment, componentName), dynamoDeployment.Namespace, ingressSpec)
				_, syncedMainComponentVirtualService, err := commonController.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*networkingv1beta1.VirtualService, bool, error) {
					if !ingressSpec.IsVirtualServiceEnabled() {
						logger.Info("VirtualService is not enabled")
						return mainComponentVirtualService, true, nil
					}
					return mainComponentVirtualService, false, nil
				})
				if err != nil {
					logger.Error(err, "failed to sync the main component virtual service")
					return "", "", "", fmt.Errorf("failed to sync the main component virtual service: %w", err)
				}
				resources = append(resources, commonController.WrapResource(syncedMainComponentVirtualService,
					func() (bool, string) {
						return true, ""
					}))
			}
		}
	}
	return r.checkResourcesReadiness(resources)
}

func (r *DynamoGraphDeploymentReconciler) checkResourcesReadiness(resources []Resource) (State, Reason, Message, error) {
	var notReadyReasons []string
	notReadyResources := []string{}

	for _, resource := range resources {
		ready, reason := resource.IsReady()
		if !ready {
			notReadyResources = append(notReadyResources, resource.GetName())
			notReadyReasons = append(notReadyReasons, fmt.Sprintf("%s: %s", resource.GetName(), reason))
		}
	}

	if len(notReadyResources) == 0 {
		return ReadyState, "all_resources_are_ready", Message("All resources are ready"), nil
	}
	return PendingState, "some_resources_are_not_ready", Message(fmt.Sprintf("Resources not ready: %s", strings.Join(notReadyReasons, "; "))), nil
}

func (r *DynamoGraphDeploymentReconciler) reconcileDynamoComponentsDeployments(ctx context.Context, dynamoDeployment *nvidiacomv1alpha1.DynamoGraphDeployment) (State, Reason, Message, error) {
	resources := []Resource{}
	logger := log.FromContext(ctx)

	// generate the dynamoComponentsDeployments from the config
	defaultIngressSpec := dynamo.GenerateDefaultIngressSpec(dynamoDeployment, r.Config.IngressConfig)
	dynamoComponentsDeployments, err := dynamo.GenerateDynamoComponentsDeployments(ctx, dynamoDeployment, &defaultIngressSpec)
	if err != nil {
		logger.Error(err, "failed to generate the DynamoComponentsDeployments")
		return "", "", "", fmt.Errorf("failed to generate the DynamoComponentsDeployments: %w", err)
	}

	// reconcile the dynamoComponentsDeployments
	for serviceName, dynamoComponentDeployment := range dynamoComponentsDeployments {
		logger.Info("Reconciling the DynamoComponentDeployment", "serviceName", serviceName, "dynamoComponentDeployment", dynamoComponentDeployment)
		_, dynamoComponentDeployment, err = commonController.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*nvidiacomv1alpha1.DynamoComponentDeployment, bool, error) {
			return dynamoComponentDeployment, false, nil
		})
		if err != nil {
			logger.Error(err, "failed to sync the DynamoComponentDeployment")
			return "", "", "", fmt.Errorf("failed to sync the DynamoComponentDeployment: %w", err)
		}
		resources = append(resources, dynamoComponentDeployment)
	}

	return r.checkResourcesReadiness(resources)
}

func (r *DynamoGraphDeploymentReconciler) FinalizeResource(ctx context.Context, dynamoDeployment *nvidiacomv1alpha1.DynamoGraphDeployment) error {
	// for now doing nothing
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoGraphDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	ctrlBuilder := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoGraphDeployment{}, builder.WithPredicates(
			predicate.GenerationChangedPredicate{},
		)).
		Named("dynamographdeployment").
		Owns(&nvidiacomv1alpha1.DynamoComponentDeployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config))
	if r.Config.Grove.Enabled {
		ctrlBuilder = ctrlBuilder.Owns(&grovev1alpha1.PodCliqueSet{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the pod gang set
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		}))
	}
	return ctrlBuilder.Complete(r)
}

func (r *DynamoGraphDeploymentReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}
