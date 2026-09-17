/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	lpxTopologyBindingRefIndex       = "lpx.topologyBinding"
	lpxResourceClaimRefIndex         = "lpx.resourceClaim"
	lpxResourceClaimTemplateRefIndex = "lpx.resourceClaimTemplate"
	lpxDeviceClassRefIndex           = "lpx.deviceClass"
)

// lpxTopologyBindingReferences indexes non-nil DGD objects by their LPX topology policy.
func lpxTopologyBindingReferences(obj client.Object) []string {
	source := obj.(*nvidiacomv1beta1.DynamoGraphDeployment)
	if !source.HasLPXComponent() || source.Spec.Experimental == nil || source.Spec.Experimental.KvTransferPolicy == nil {
		return nil
	}
	name := source.Spec.Experimental.KvTransferPolicy.ClusterTopologyName
	if name == "" {
		return nil
	}
	return []string{name}
}

// lpxDRAClaimReferences indexes non-nil DGD objects by consumed GPU-role claims.
// Extraction uses only source intent, including references not yet created.
func lpxDRAClaimReferences(template bool) client.IndexerFunc {
	return func(obj client.Object) []string {
		// Index dependencies from every independently authored role.
		source := obj.(*nvidiacomv1beta1.DynamoGraphDeployment)
		references := sets.New[string]()
		for index := range source.Spec.Components {
			component := &source.Spec.Components[index]
			if !component.IsLPX() {
				continue
			}
			for _, role := range component.Roles {
				if role.PodTemplate == nil {
					continue
				}
				pod := &role.PodTemplate.Spec
				consumed := sets.New[string]()
				for _, container := range dra.AllContainers(pod) {
					for _, claim := range container.Resources.Claims {
						consumed.Insert(claim.Name)
					}
				}
				for _, claim := range pod.ResourceClaims {
					if !consumed.Has(claim.Name) {
						continue
					}
					name := claim.ResourceClaimName
					if template {
						name = claim.ResourceClaimTemplateName
					}
					if name != nil && *name != "" {
						references.Insert(*name)
					}
				}
			}
		}
		return references.UnsortedList()
	}
}

// lpxDeviceClassReferences indexes claims and templates, never fetched DGD state.
// A claim-only edit therefore updates this index without requiring a DGD edit.
func lpxDeviceClassReferences(obj client.Object) []string {
	var spec *resourcev1.ResourceClaimSpec
	switch resource := obj.(type) {
	case *resourcev1.ResourceClaim:
		spec = &resource.Spec
	case *resourcev1.ResourceClaimTemplate:
		spec = &resource.Spec.Spec
	default:
		return nil
	}

	// Every alternative can become the selected device request.
	classes := sets.New[string]()
	for _, request := range spec.Devices.Requests {
		if request.Exactly != nil && request.Exactly.DeviceClassName != "" {
			classes.Insert(request.Exactly.DeviceClassName)
		}
		for _, alternative := range request.FirstAvailable {
			if alternative.DeviceClassName != "" {
				classes.Insert(alternative.DeviceClassName)
			}
		}
	}
	return classes.UnsortedList()
}

func (r *graphReconciler) indexedLPXDependencyRequests(ctx context.Context, dependency client.Object, index string) []ctrl.Request {
	// Cached field queries load only the LPX sources referencing this dependency.
	sources := &nvidiacomv1beta1.DynamoGraphDeploymentList{}
	if err := r.List(ctx, sources, client.InNamespace(dependency.GetNamespace()), client.MatchingFields{index: dependency.GetName()}); err != nil {
		ctrl.LoggerFrom(ctx).Error(err, "Unable to list LPX sources for dependency", "index", index, "dependency", client.ObjectKeyFromObject(dependency))
		return nil
	}

	// Respect each source's namespace filter and enqueue its exact owned materializations.
	requests := make([]ctrl.Request, 0, len(sources.Items))
	for i := range sources.Items {
		source := &sources.Items[i]
		if commoncontroller.NamespaceAllowed(r.Config, r.runtimeConfig, source, source.Namespace) {
			requests = append(requests, r.mapLPXSourceToRequests(ctx, source)...)
		}
	}
	return requests
}

func (r *graphReconciler) mapLPXDeviceClassToRequests(ctx context.Context, class *resourcev1.DeviceClass) []ctrl.Request {
	// DeviceClass names live on claims/templates, so resolve both indexed hops.
	claims := &resourcev1.ResourceClaimList{}
	if err := r.List(ctx, claims, client.MatchingFields{lpxDeviceClassRefIndex: class.Name}); err != nil {
		ctrl.LoggerFrom(ctx).Error(err, "Unable to list LPX ResourceClaim dependencies", "deviceClass", class.Name)
		return nil
	}
	templates := &resourcev1.ResourceClaimTemplateList{}
	if err := r.List(ctx, templates, client.MatchingFields{lpxDeviceClassRefIndex: class.Name}); err != nil {
		ctrl.LoggerFrom(ctx).Error(err, "Unable to list LPX ResourceClaimTemplate dependencies", "deviceClass", class.Name)
		return nil
	}

	// Multiple claims and engines can refer to the same class in one graph.
	consumers := sets.New[ctrl.Request]()
	for i := range claims.Items {
		for _, request := range r.indexedLPXDependencyRequests(ctx, &claims.Items[i], lpxResourceClaimRefIndex) {
			consumers.Insert(request)
		}
	}
	for i := range templates.Items {
		for _, request := range r.indexedLPXDependencyRequests(ctx, &templates.Items[i], lpxResourceClaimTemplateRefIndex) {
			consumers.Insert(request)
		}
	}
	return consumers.UnsortedList()
}
