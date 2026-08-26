/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch
// +kubebuilder:rbac:groups=resource.k8s.io,resources=deviceclasses,verbs=get;list;watch

func deploymentEventFilter(
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commonController.RuntimeConfig,
) predicate.Predicate {
	return predicate.NewPredicateFuncs(func(obj client.Object) bool {
		if _, ok := obj.(*resourcev1.DeviceClass); ok {
			return true
		}
		return commonController.NamespaceAllowed(config, runtimeConfig, obj, obj.GetNamespace())
	})
}

// componentReferencesDRAClaim reports whether the non-nil component references
// the named ResourceClaim or ResourceClaimTemplate from its main container.
func componentReferencesDRAClaim(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	objectName string,
	template bool,
) bool {
	if component.PodTemplate == nil {
		return false
	}
	mainContainer := dynamo.GetMainContainer(component)
	if mainContainer == nil || len(mainContainer.Resources.Claims) == 0 {
		return false
	}

	containerClaimNames := make(map[string]struct{}, len(mainContainer.Resources.Claims))
	for _, claim := range mainContainer.Resources.Claims {
		containerClaimNames[claim.Name] = struct{}{}
	}

	for _, podClaim := range component.PodTemplate.Spec.ResourceClaims {
		if _, ok := containerClaimNames[podClaim.Name]; !ok {
			continue
		}
		if template && podClaim.ResourceClaimTemplateName != nil && *podClaim.ResourceClaimTemplateName == objectName {
			return true
		}
		if !template && podClaim.ResourceClaimName != nil && *podClaim.ResourceClaimName == objectName {
			return true
		}
	}
	return false
}

// componentUsesDRAClaims reports whether the non-nil component's main
// container references any DRA claims.
func componentUsesDRAClaims(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return len(dynamo.GetMainContainerResources(component).Claims) > 0
}

func (r *DynamoComponentDeploymentReconciler) mapResourceClaimToDCDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	return r.mapDRAClaimToDCDRequests(ctx, obj, false)
}

func (r *DynamoComponentDeploymentReconciler) mapResourceClaimTemplateToDCDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	return r.mapDRAClaimToDCDRequests(ctx, obj, true)
}

func (r *DynamoComponentDeploymentReconciler) mapDRAClaimToDCDRequests(
	ctx context.Context,
	obj client.Object,
	template bool,
) []ctrl.Request {
	deployments := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := r.List(ctx, deployments, client.InNamespace(obj.GetNamespace())); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoComponentDeployments for DRA claim event")
		return nil
	}

	requests := make([]ctrl.Request, 0)
	for i := range deployments.Items {
		deployment := &deployments.Items[i]
		if componentReferencesDRAClaim(&deployment.Spec.DynamoComponentDeploymentSharedSpec, obj.GetName(), template) {
			requests = append(requests, ctrl.Request{NamespacedName: types.NamespacedName{
				Namespace: deployment.Namespace,
				Name:      deployment.Name,
			}})
		}
	}
	return requests
}

func (r *DynamoComponentDeploymentReconciler) mapDeviceClassToDCDRequests(
	ctx context.Context,
	_ client.Object,
) []ctrl.Request {
	deployments := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := r.List(ctx, deployments); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoComponentDeployments for DeviceClass event")
		return nil
	}

	requests := make([]ctrl.Request, 0)
	for i := range deployments.Items {
		deployment := &deployments.Items[i]
		if commonController.NamespaceAllowed(r.Config, r.RuntimeConfig, deployment, deployment.Namespace) &&
			componentUsesDRAClaims(&deployment.Spec.DynamoComponentDeploymentSharedSpec) {
			requests = append(requests, ctrl.Request{NamespacedName: types.NamespacedName{
				Namespace: deployment.Namespace,
				Name:      deployment.Name,
			}})
		}
	}
	return requests
}

func (r *DynamoGraphDeploymentReconciler) mapResourceClaimToDGDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	return r.mapDRAClaimToDGDRequests(ctx, obj, false)
}

func (r *DynamoGraphDeploymentReconciler) mapResourceClaimTemplateToDGDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	return r.mapDRAClaimToDGDRequests(ctx, obj, true)
}

func (r *DynamoGraphDeploymentReconciler) mapDRAClaimToDGDRequests(
	ctx context.Context,
	obj client.Object,
	template bool,
) []ctrl.Request {
	deployments := &nvidiacomv1beta1.DynamoGraphDeploymentList{}
	if err := r.List(ctx, deployments, client.InNamespace(obj.GetNamespace())); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoGraphDeployments for DRA claim event")
		return nil
	}

	requests := make([]ctrl.Request, 0)
	for i := range deployments.Items {
		deployment := &deployments.Items[i]
		for componentIndex := range deployment.Spec.Components {
			if componentReferencesDRAClaim(&deployment.Spec.Components[componentIndex], obj.GetName(), template) {
				requests = append(requests, ctrl.Request{NamespacedName: types.NamespacedName{
					Namespace: deployment.Namespace,
					Name:      deployment.Name,
				}})
				break
			}
		}
	}
	return requests
}

func (r *DynamoGraphDeploymentReconciler) mapDeviceClassToDGDRequests(
	ctx context.Context,
	_ client.Object,
) []ctrl.Request {
	deployments := &nvidiacomv1beta1.DynamoGraphDeploymentList{}
	if err := r.List(ctx, deployments); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoGraphDeployments for DeviceClass event")
		return nil
	}

	requests := make([]ctrl.Request, 0)
	for i := range deployments.Items {
		deployment := &deployments.Items[i]
		if !commonController.NamespaceAllowed(r.Config, r.RuntimeConfig, deployment, deployment.Namespace) {
			continue
		}
		for componentIndex := range deployment.Spec.Components {
			if componentUsesDRAClaims(&deployment.Spec.Components[componentIndex]) {
				requests = append(requests, ctrl.Request{NamespacedName: types.NamespacedName{
					Namespace: deployment.Namespace,
					Name:      deployment.Name,
				}})
				break
			}
		}
	}
	return requests
}
