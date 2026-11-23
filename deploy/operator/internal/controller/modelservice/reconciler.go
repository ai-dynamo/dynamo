// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelservice

import (
	"context"
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	coreapply "k8s.io/client-go/applyconfigurations/core/v1"
	metaapply "k8s.io/client-go/applyconfigurations/meta/v1"
	"k8s.io/client-go/util/csaupgrade"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// LabelModelDiscovery identifies the resource family owned by this reconciler.
const LabelModelDiscovery = "nvidia.com/dynamo-model-discovery"

const fieldManager = "dynamo-model-discovery"

// Reconciler owns model-discovery Services for eligible DGDs.
type Reconciler struct {
	Client client.Client
}

// Reconcile converges a complete non-nil DGD's model Services and returns their names.
// The caller selects this reconciler using the immutable parent origin gate.
func (r *Reconciler) Reconcile(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment) (map[string]struct{}, error) {
	// Resolve names before any write so separate Service families cannot overwrite each other.
	models := sets.New[string]()
	componentNames := sets.New[string]()
	for _, component := range dgd.Spec.Components {
		componentNames.Insert(strings.ToLower(strings.ReplaceAll(dgd.Name+"-"+component.ComponentName, ".", "-")))
		if component.ModelRef != nil && component.ModelRef.Name != "" {
			models.Insert(component.ModelRef.Name)
		}
	}
	services := make([]*corev1.Service, 0, models.Len())
	for _, model := range sets.List(models) {
		desired := Render(Params{Namespace: dgd.Namespace, ModelName: model, GraphName: dgd.Name, Annotations: dgd.Spec.Annotations, ReadableName: true})
		if componentNames.Has(desired.Name) {
			return nil, fmt.Errorf("model-discovery Service %q conflicts with a component Service; rename that component", desired.Name)
		}
		desired.Labels[LabelModelDiscovery] = "true"
		services = append(services, desired)
	}

	// Retiring model Services also reserve their names until provider cutover completes.
	existing := &corev1.ServiceList{}
	if err := r.Client.List(ctx, existing, client.InNamespace(dgd.Namespace)); err != nil {
		return nil, err
	}
	for i := range existing.Items {
		service := &existing.Items[i]
		if metav1.IsControlledBy(service, dgd) && isModelDiscovery(service) && componentNames.Has(service.Name) {
			return nil, fmt.Errorf("component Service %q conflicts with an existing model-discovery Service; rename that component", service.Name)
		}
	}

	// Applying desired Services does not retire models still served by an old revision.
	names := make(map[string]struct{}, len(services))
	for _, desired := range services {
		if err := controllerutil.SetControllerReference(dgd, desired, r.Client.Scheme()); err != nil {
			return nil, err
		}
		if err := r.apply(ctx, dgd, desired); err != nil {
			return nil, fmt.Errorf("reconcile model-discovery Service %s: %w", desired.Name, err)
		}
		names[desired.Name] = struct{}{}
	}
	return names, nil
}

// DeleteStale retires this graph's obsolete model Services after provider cutover.
// dgd is non-nil; an empty desiredNames map retires every model Service of the graph.
func (r *Reconciler) DeleteStale(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, desiredNames map[string]struct{}) error {
	services := &corev1.ServiceList{}
	if err := r.Client.List(ctx, services, client.InNamespace(dgd.Namespace)); err != nil {
		return err
	}
	for i := range services.Items {
		service := &services.Items[i]
		if !metav1.IsControlledBy(service, dgd) || !isModelDiscovery(service) {
			continue
		}
		if _, desired := desiredNames[service.Name]; desired {
			continue
		}
		uid, version := service.UID, service.ResourceVersion
		if err := r.Client.Delete(ctx, service, client.Preconditions{UID: &uid, ResourceVersion: &version}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

// The field manager identifies this family even if its identifying label drifts.
func isModelDiscovery(service *corev1.Service) bool {
	if service.Labels[LabelModelDiscovery] == "true" {
		return true
	}
	for _, entry := range service.ManagedFields {
		if entry.Manager == fieldManager && entry.Subresource == "" {
			return true
		}
	}
	return false
}

func (r *Reconciler) apply(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, desired *corev1.Service) error {
	// Explicit Create prevents a stale cache miss from adopting a same-name object.
	observed := &corev1.Service{}
	if err := r.Client.Get(ctx, client.ObjectKeyFromObject(desired), observed); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		observed = desired.DeepCopy()
		if err := r.Client.Create(ctx, observed, client.FieldOwner(fieldManager)); err != nil {
			return err
		}
	}
	if err := commoncontroller.CheckControllerOwnership(observed, dgd, r.Client.Scheme()); err != nil {
		return err
	}

	if !isModelDiscovery(observed) {
		return fmt.Errorf("Service %q is not managed as model discovery", observed.Name)
	}

	// Upgrade only this explicit manager's Create attribution, never a legacy shared manager.
	patch, err := csaupgrade.UpgradeManagedFieldsPatch(observed, sets.New(fieldManager), fieldManager)
	if err != nil {
		return err
	}
	if patch != nil {
		if err := r.Client.Patch(ctx, observed, client.RawPatch(types.JSONPatchType, patch)); err != nil {
			return err
		}
	}

	// Construct the fields this controller owns, including explicit false presence.
	owner := metav1.GetControllerOf(desired)
	port := desired.Spec.Ports[0]
	configuration := coreapply.Service(desired.Name, desired.Namespace).
		WithUID(observed.UID).WithResourceVersion(observed.ResourceVersion).
		WithLabels(desired.Labels).WithAnnotations(desired.Annotations).
		WithOwnerReferences(metaapply.OwnerReference().WithAPIVersion(owner.APIVersion).WithKind(owner.Kind).
			WithName(owner.Name).WithUID(owner.UID).WithController(true).WithBlockOwnerDeletion(true)).
		WithSpec(coreapply.ServiceSpec().WithClusterIP(corev1.ClusterIPNone).WithPublishNotReadyAddresses(false).
			WithSelector(desired.Spec.Selector).
			WithPorts(coreapply.ServicePort().WithName(port.Name).WithPort(port.Port).WithProtocol(port.Protocol).WithTargetPort(port.TargetPort)))
	return r.Client.Apply(ctx, configuration, client.FieldOwner(fieldManager), client.ForceOwnership)
}
