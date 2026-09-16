/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"maps"
	"sort"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
)

type disaggregatedSetStableResourcesReconciler struct {
	client   client.Client
	renderer *dcdWorkloadRenderer
}

const managedServiceMetadataAnnotation = "nvidia.com/dynamo-managed-service-metadata"

type managedServiceMetadata struct {
	Labels      []string `json:"labels,omitempty"`
	Annotations []string `json:"annotations,omitempty"`
}

func newDisaggregatedSetStableResourcesReconciler(
	k8sClient client.Client,
	renderer *dcdWorkloadRenderer,
) *disaggregatedSetStableResourcesReconciler {
	return &disaggregatedSetStableResourcesReconciler{
		client:   k8sClient,
		renderer: renderer,
	}
}

func (r *disaggregatedSetStableResourcesReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	normalizedComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	selection disaggregatedSetSelection,
	targetRevision string,
	targetReady bool,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) (map[string]struct{}, error) {
	desiredServiceNames := map[string]struct{}{}
	componentNames := make([]string, 0, len(normalizedComponents))
	for componentName := range normalizedComponents {
		componentNames = append(componentNames, componentName)
	}
	sort.Strings(componentNames)
	for _, componentName := range componentNames {
		component := normalizedComponents[componentName]
		serviceName := dynamo.GetDCDResourceName(dgd, componentName, "")
		if dcd := dcds[componentName]; dcd != nil {
			serviceName = dcd.Name
		}
		if _, selected := selection.componentToRole[componentName]; selected {
			serviceName = dynamo.GetDCDResourceName(dgd, componentName, rollingUpdateCtx.NewWorkerHash)
		}
		service, deleted, err := r.renderer.generateServiceForDGDComponent(ctx, dgd, component, componentName, serviceName)
		if err != nil {
			return nil, fmt.Errorf("failed to render component service for %q: %w", componentName, err)
		}
		if deleted {
			continue
		}
		if roleName, selected := selection.componentToRole[componentName]; selected && targetRevision != "" {
			existing := &corev1.Service{}
			existingErr := r.client.Get(ctx, types.NamespacedName{Name: service.Name, Namespace: service.Namespace}, existing)
			if existingErr != nil && !apierrors.IsNotFound(existingErr) {
				return nil, fmt.Errorf("failed to get existing component service for %q: %w", componentName, existingErr)
			}
			setDesiredDisaggregatedSetServiceSelector(service, existing, existingErr == nil, disaggregatedSetName(dgd), roleName, targetRevision, targetReady)
		}
		if err := r.syncDGDStableService(ctx, dgd, service); err != nil {
			return nil, fmt.Errorf("failed to reconcile component service for %q: %w", componentName, err)
		}
		desiredServiceNames[service.Name] = struct{}{}
	}

	modelNames := map[string]struct{}{}
	for _, component := range normalizedComponents {
		if component.ModelRef == nil || component.ModelRef.Name == "" {
			continue
		}
		modelNames[component.ModelRef.Name] = struct{}{}
	}
	modelNamesSorted := make([]string, 0, len(modelNames))
	for modelName := range modelNames {
		modelNamesSorted = append(modelNamesSorted, modelName)
	}
	sort.Strings(modelNamesSorted)
	for _, modelName := range modelNamesSorted {
		annotations := maps.Clone(dgd.Spec.Annotations)
		service := dynamo.GenerateModelServiceForGraph(dgd.Namespace, modelName, dgd.Name, annotations)
		if err := r.syncDGDStableService(ctx, dgd, service); err != nil {
			return nil, fmt.Errorf("failed to reconcile model service for %q: %w", modelName, err)
		}
		desiredServiceNames[service.Name] = struct{}{}
	}
	return desiredServiceNames, nil
}

func (r *disaggregatedSetStableResourcesReconciler) DeleteStale(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desiredServiceNames map[string]struct{},
) error {
	serviceList := &corev1.ServiceList{}
	if err := r.client.List(ctx, serviceList, client.InNamespace(dgd.Namespace)); err != nil {
		return fmt.Errorf("failed to list DisaggregatedSet Services: %w", err)
	}
	for i := range serviceList.Items {
		service := &serviceList.Items[i]
		if !isControlledByBetaDGD(service, dgd) {
			continue
		}
		if _, desired := desiredServiceNames[service.Name]; desired {
			continue
		}
		deleteOptions := []client.DeleteOption{}
		if service.UID != "" {
			deleteOptions = append(deleteOptions, client.Preconditions{UID: &service.UID})
		}
		if err := r.client.Delete(ctx, service, deleteOptions...); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to delete stale Service %s/%s: %w", service.Namespace, service.Name, err)
		}
	}
	return nil
}

func setDisaggregatedSetServiceSelector(service *corev1.Service, setName, roleName, revision string) {
	service.Spec.Selector = map[string]string{
		disaggregatedsetv1.SetNameLabelKey:  setName,
		disaggregatedsetv1.RoleLabelKey:     roleName,
		disaggregatedsetv1.RevisionLabelKey: revision,
	}
}

func setDesiredDisaggregatedSetServiceSelector(
	service *corev1.Service,
	existingService *corev1.Service,
	hasExistingService bool,
	setName string,
	roleName string,
	revision string,
	targetReady bool,
) {
	if targetReady || !hasExistingService {
		setDisaggregatedSetServiceSelector(service, setName, roleName, revision)
		return
	}
	service.Spec.Selector = maps.Clone(existingService.Spec.Selector)
}

func (r *disaggregatedSetStableResourcesReconciler) syncDGDStableService(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *corev1.Service,
) error {
	key := types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}
	existing := &corev1.Service{}
	if err := r.client.Get(ctx, key, existing); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		created := desired.DeepCopy()
		var metadataErr error
		created.Labels, created.Annotations, metadataErr = reconcileDGDStableServiceMetadata(nil, desired)
		if metadataErr != nil {
			return metadataErr
		}
		setDGDControllerOwnerReference(dgd, created)
		return r.client.Create(ctx, created)
	}

	if owner := metav1.GetControllerOf(existing); owner != nil && !ownerReferenceMatchesDGD(owner, dgd) {
		if owner.APIVersion != nvidiacomv1beta1.GroupVersion.String() || owner.Kind != dynamoComponentDeploymentKind {
			return fmt.Errorf("Service %s/%s is controlled by %s/%s %q", existing.Namespace, existing.Name, owner.APIVersion, owner.Kind, owner.Name)
		}
		currentOwner := &nvidiacomv1beta1.DynamoComponentDeployment{}
		if err := r.client.Get(ctx, types.NamespacedName{Name: owner.Name, Namespace: existing.Namespace}, currentOwner); err != nil {
			return fmt.Errorf("failed to get current Service owner %s/%s: %w", existing.Namespace, owner.Name, err)
		}
		if !isControlledByBetaDGD(currentOwner, dgd) {
			return fmt.Errorf("Service %s/%s is controlled by unrelated DynamoComponentDeployment %s", existing.Namespace, existing.Name, currentOwner.Name)
		}
	}

	updated := existing.DeepCopy()
	updated.Spec = *desired.Spec.DeepCopy()
	normalizeDGDStableServiceSpec(&updated.Spec)
	var metadataErr error
	updated.Labels, updated.Annotations, metadataErr = reconcileDGDStableServiceMetadata(existing, desired)
	if metadataErr != nil {
		return metadataErr
	}
	setDGDControllerOwnerReference(dgd, updated)
	updated.Spec.ClusterIP = existing.Spec.ClusterIP
	updated.Spec.ClusterIPs = existing.Spec.ClusterIPs
	updated.Spec.IPFamilies = existing.Spec.IPFamilies
	updated.Spec.IPFamilyPolicy = existing.Spec.IPFamilyPolicy
	updated.Spec.HealthCheckNodePort = existing.Spec.HealthCheckNodePort
	if equality.Semantic.DeepEqual(existing.Spec, updated.Spec) &&
		equality.Semantic.DeepEqual(existing.Labels, updated.Labels) &&
		equality.Semantic.DeepEqual(existing.Annotations, updated.Annotations) &&
		equality.Semantic.DeepEqual(existing.OwnerReferences, updated.OwnerReferences) {
		return nil
	}
	return r.client.Update(ctx, updated)
}

func reconcileDGDStableServiceMetadata(
	existing *corev1.Service,
	desired *corev1.Service,
) (map[string]string, map[string]string, error) {
	labels := map[string]string{}
	annotations := map[string]string{}
	previous := managedServiceMetadata{}
	if existing != nil {
		maps.Copy(labels, existing.Labels)
		maps.Copy(annotations, existing.Annotations)
		if raw := existing.Annotations[managedServiceMetadataAnnotation]; raw != "" {
			if err := json.Unmarshal([]byte(raw), &previous); err != nil {
				return nil, nil, fmt.Errorf("decode managed Service metadata inventory: %w", err)
			}
		}
	}
	for _, key := range previous.Labels {
		delete(labels, key)
	}
	for _, key := range previous.Annotations {
		delete(annotations, key)
	}

	desiredAnnotations := maps.Clone(desired.Annotations)
	delete(desiredAnnotations, managedServiceMetadataAnnotation)
	maps.Copy(labels, desired.Labels)
	maps.Copy(annotations, desiredAnnotations)

	next := managedServiceMetadata{
		Labels:      sortedMetadataKeys(desired.Labels),
		Annotations: sortedMetadataKeys(desiredAnnotations),
	}
	raw, err := json.Marshal(next)
	if err != nil {
		return nil, nil, fmt.Errorf("encode managed Service metadata inventory: %w", err)
	}
	annotations[managedServiceMetadataAnnotation] = string(raw)
	return labels, annotations, nil
}

func sortedMetadataKeys(metadata map[string]string) []string {
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func normalizeDGDStableServiceSpec(spec *corev1.ServiceSpec) {
	if spec.Type == "" {
		spec.Type = corev1.ServiceTypeClusterIP
	}
	if spec.SessionAffinity == "" {
		spec.SessionAffinity = corev1.ServiceAffinityNone
	}
	if spec.SessionAffinity == corev1.ServiceAffinityNone {
		spec.SessionAffinityConfig = nil
	}
	if spec.Type == corev1.ServiceTypeNodePort || spec.Type == corev1.ServiceTypeLoadBalancer {
		if spec.ExternalTrafficPolicy == "" {
			spec.ExternalTrafficPolicy = corev1.ServiceExternalTrafficPolicyTypeCluster
		}
	}
	if spec.InternalTrafficPolicy == nil &&
		(spec.Type == corev1.ServiceTypeClusterIP ||
			spec.Type == corev1.ServiceTypeNodePort ||
			spec.Type == corev1.ServiceTypeLoadBalancer) {
		spec.InternalTrafficPolicy = ptr.To(corev1.ServiceInternalTrafficPolicyCluster)
	}
	for i := range spec.Ports {
		port := &spec.Ports[i]
		if port.Protocol == "" {
			port.Protocol = corev1.ProtocolTCP
		}
		if port.TargetPort == intstr.FromInt32(0) || port.TargetPort == intstr.FromString("") {
			port.TargetPort = intstr.FromInt32(port.Port)
		}
	}
	if spec.Type == corev1.ServiceTypeLoadBalancer && spec.AllocateLoadBalancerNodePorts == nil {
		spec.AllocateLoadBalancerNodePorts = ptr.To(true)
	}
}
