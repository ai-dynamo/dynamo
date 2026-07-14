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
	"fmt"
	"sort"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
)

var disaggregatedSetGVK = schema.GroupVersionKind{
	Group:   "disaggregatedset.x-k8s.io",
	Version: "v1",
	Kind:    "DisaggregatedSet",
}

type disaggregatedSetSelection struct {
	componentToRole map[string]string
	desiredReplicas map[string]int32
}

func newDisaggregatedSetObject() *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(disaggregatedSetGVK)
	return obj
}

func (r *DynamoGraphDeploymentReconciler) wantsDisaggregatedSet(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	if dgd == nil || dgd.Annotations == nil {
		return false
	}
	return strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableDisaggregatedSet]) == consts.KubeLabelValueTrue
}

func (r *DynamoGraphDeploymentReconciler) shouldUseDisaggregatedSet(dgd *nvidiacomv1beta1.DynamoGraphDeployment) (bool, string) {
	if !r.wantsDisaggregatedSet(dgd) {
		return false, ""
	}
	if r.RuntimeConfig == nil {
		return false, "runtime config is not initialized"
	}
	if !r.RuntimeConfig.DisaggregatedSetEnabled {
		return false, "DisaggregatedSet API is not available"
	}
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return false, reason
	}
	if len(selection.componentToRole) < 2 {
		return false, "DisaggregatedSet requires at least two eligible multinode worker roles"
	}
	return true, ""
}

func selectDisaggregatedSetComponents(dgd *nvidiacomv1beta1.DynamoGraphDeployment) (disaggregatedSetSelection, string) {
	selection := disaggregatedSetSelection{
		componentToRole: make(map[string]string),
		desiredReplicas: make(map[string]int32),
	}
	if dgd == nil {
		return selection, "DynamoGraphDeployment is nil"
	}

	usedRoles := make(map[string]struct{})
	zeroReplicas := 0
	positiveReplicas := 0
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !isDisaggregatedSetEligibleComponent(component) {
			continue
		}
		if component.ScalingAdapter != nil {
			return selection, fmt.Sprintf("component %q uses scalingAdapter, but DisaggregatedSet does not support scale subresource integration", component.ComponentName)
		}
		roleName := disaggregatedSetRoleName(component, usedRoles)
		usedRoles[roleName] = struct{}{}
		selection.componentToRole[component.ComponentName] = roleName

		desiredReplicas := desiredComponentReplicas(component)
		selection.desiredReplicas[component.ComponentName] = desiredReplicas
		if desiredReplicas == 0 {
			zeroReplicas++
		} else {
			positiveReplicas++
		}
	}

	if len(selection.componentToRole) == 0 {
		return selection, "no eligible multinode worker roles found"
	}
	if zeroReplicas > 0 && positiveReplicas > 0 {
		return selection, "DisaggregatedSet requires replicas to be zero for all selected roles or positive for all selected roles"
	}
	return selection, ""
}

func isDisaggregatedSetEligibleComponent(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return component != nil && component.GetNumberOfNodes() > 1 && dynamo.IsWorkerComponent(string(component.ComponentType))
}

func desiredComponentReplicas(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) int32 {
	if component == nil || component.Replicas == nil {
		return 1
	}
	return *component.Replicas
}

func disaggregatedSetRoleName(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, used map[string]struct{}) string {
	preferred := strings.ToLower(string(component.ComponentType))
	if preferred != consts.ComponentTypePrefill && preferred != consts.ComponentTypeDecode {
		preferred = ""
	}
	if preferred == "" || roleNameUsed(preferred, used) {
		preferred = dynamo.NormalizeKubeResourceName(component.ComponentName)
	}
	roleName := preferred
	for i := 2; roleNameUsed(roleName, used); i++ {
		roleName = fmt.Sprintf("%s-%d", truncateDNSLabel(preferred, 61), i)
	}
	return roleName
}

func roleNameUsed(roleName string, used map[string]struct{}) bool {
	_, ok := used[roleName]
	return ok
}

func truncateDNSLabel(value string, maxLength int) string {
	if len(value) <= maxLength {
		return value
	}
	return strings.TrimRight(value[:maxLength], "-")
}

func disaggregatedSetName(dgd *nvidiacomv1beta1.DynamoGraphDeployment) string {
	return dynamo.NormalizeKubeResourceName(dgd.Name)
}

func (r *DynamoGraphDeploymentReconciler) reconcileDisaggregatedSetResources(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error) {
	resources := []Resource{}
	logger := log.FromContext(ctx)

	rollingUpdateCtx, err := r.buildRollingUpdateContext(ctx, dynamoDeployment)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to build rolling update context: %w", err)
	}

	existingRestartAnnotations, err := r.getExistingRestartAnnotationsDCD(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "failed to get existing restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing restart annotations: %w", err)
	}

	dynamoComponentsDeployments, err := dynamo.GenerateDynamoComponentsDeployments(
		dynamoDeployment,
		restartState,
		existingRestartAnnotations,
		rollingUpdateCtx,
	)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to generate DynamoComponentDeployments for DisaggregatedSet path: %w", err)
	}

	selection, reason := selectDisaggregatedSetComponents(dynamoDeployment)
	if reason != "" {
		return ReconcileResult{}, fmt.Errorf("failed to select DisaggregatedSet roles: %s", reason)
	}

	desiredDS, err := r.generateDisaggregatedSet(ctx, dynamoDeployment, dynamoComponentsDeployments, selection)
	if err != nil {
		return ReconcileResult{}, err
	}
	dsModified, syncedDS, err := r.syncDisaggregatedSet(ctx, dynamoDeployment, desiredDS)
	if err != nil {
		return ReconcileResult{}, err
	}

	if err := r.reconcileDisaggregatedSetSideResources(ctx, dynamoDeployment, dynamoComponentsDeployments, selection); err != nil {
		return ReconcileResult{}, err
	}

	syncedDSResource, err := commoncontroller.NewResourceWithComponentStatuses(
		syncedDS,
		func() (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
			if dsModified {
				_, _, statuses := checkDisaggregatedSetReadiness(syncedDS, selection)
				return false, "DisaggregatedSet spec was updated; waiting for controller status", statuses
			}
			return checkDisaggregatedSetReadiness(syncedDS, selection)
		},
	)
	if err != nil {
		return ReconcileResult{}, err
	}
	resources = append(resources, syncedDSResource)

	dsReady, _, _ := checkDisaggregatedSetReadiness(syncedDS, selection)
	for _, key := range sortedDCDKeys(dynamoComponentsDeployments) {
		dcd := dynamoComponentsDeployments[key]
		if _, selected := selection.componentToRole[key]; selected {
			continue
		}
		if err := applyDCDCheckpointStartupPolicy(dcd, checkpointInfos[key]); err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", key, err)
		}
		if err := r.preserveExistingDCDBackendFramework(ctx, dcd); err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to preserve existing DynamoComponentDeployment backendFramework: %w", err)
		}
		_, syncedDCD, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*nvidiacomv1beta1.DynamoComponentDeployment, bool, error) {
			return dcd, false, nil
		})
		if err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to sync non-DisaggregatedSet DynamoComponentDeployment %s: %w", dcd.Name, err)
		}
		resources = append(resources, syncedDCD)
	}

	if dsReady {
		if err := r.deleteOwnedSelectedDCDs(ctx, dynamoDeployment, selection); err != nil {
			return ReconcileResult{}, err
		}
	}

	return r.checkResourcesReadiness(resources), nil
}

func (r *DynamoGraphDeploymentReconciler) generateDisaggregatedSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	selection disaggregatedSetSelection,
) (*unstructured.Unstructured, error) {
	ds := newDisaggregatedSetObject()
	ds.SetName(disaggregatedSetName(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.SetLabels(map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoSelector:            disaggregatedSetName(dgd),
	})
	if ownerRef := dgdControllerOwnerReference(dgd); ownerRef != nil {
		ds.SetOwnerReferences([]metav1.OwnerReference{*ownerRef})
	}

	dcdReconciler := &DynamoComponentDeploymentReconciler{
		Client:                r.Client,
		Recorder:              r.Recorder,
		Config:                r.Config,
		RuntimeConfig:         r.RuntimeConfig,
		DockerSecretRetriever: r.DockerSecretRetriever,
	}

	roles := make([]any, 0, len(selection.componentToRole))
	for i := range dgd.Spec.Components {
		componentName := dgd.Spec.Components[i].ComponentName
		roleName, ok := selection.componentToRole[componentName]
		if !ok {
			continue
		}
		dcd := dcds[componentName]
		if dcd == nil {
			return nil, fmt.Errorf("generated DynamoComponentDeployment missing for selected component %q", componentName)
		}
		lws, _, err := dcdReconciler.generateLeaderWorkerSet(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
		if err != nil {
			return nil, fmt.Errorf("failed to generate LeaderWorkerSet template for DisaggregatedSet role %q: %w", roleName, err)
		}
		lwsSpec, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&lws.Spec)
		if err != nil {
			return nil, fmt.Errorf("failed to convert LeaderWorkerSet spec for DisaggregatedSet role %q: %w", roleName, err)
		}
		roleMetadata := map[string]any{}
		if len(lws.Labels) > 0 {
			roleMetadata["labels"] = stringMapToAny(lws.Labels)
		}
		if len(lws.Annotations) > 0 {
			roleMetadata["annotations"] = stringMapToAny(lws.Annotations)
		}
		roles = append(roles, map[string]any{
			"name":     roleName,
			"metadata": roleMetadata,
			"spec":     lwsSpec,
		})
	}
	if len(roles) < 2 {
		return nil, fmt.Errorf("DisaggregatedSet requires at least two roles, got %d", len(roles))
	}
	ds.Object["spec"] = map[string]any{"roles": roles}
	return ds, nil
}

func (r *DynamoGraphDeploymentReconciler) reconcileDisaggregatedSetSideResources(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	selection disaggregatedSetSelection,
) error {
	if err := dynamo.ReconcileModelServicesForComponents(ctx, r, dgd, selectedComponentsByName(dgd, selection), dgd.Namespace); err != nil {
		return fmt.Errorf("failed to reconcile DisaggregatedSet model services: %w", err)
	}
	if err := r.adoptSelectedModelServices(ctx, dgd, selection); err != nil {
		return err
	}

	dcdReconciler := &DynamoComponentDeploymentReconciler{
		Client:                r.Client,
		Recorder:              r.Recorder,
		Config:                r.Config,
		RuntimeConfig:         r.RuntimeConfig,
		DockerSecretRetriever: r.DockerSecretRetriever,
	}
	for _, componentName := range sortedSelectionComponentNames(selection) {
		dcd := dcds[componentName]
		if dcd == nil {
			return fmt.Errorf("generated DynamoComponentDeployment missing for selected component %q", componentName)
		}
		_, syncedService, err := commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*corev1.Service, bool, error) {
			return dcdReconciler.generateService(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
		})
		if err != nil {
			return fmt.Errorf("failed to reconcile DisaggregatedSet component service for %q: %w", componentName, err)
		}
		if syncedService != nil {
			if err := r.ensureControlledByDGD(ctx, dgd, syncedService); err != nil {
				return fmt.Errorf("failed to adopt DisaggregatedSet component service %s/%s: %w", syncedService.Namespace, syncedService.Name, err)
			}
		}
	}
	return nil
}

func selectedComponentsByName(dgd *nvidiacomv1beta1.DynamoGraphDeployment, selection disaggregatedSetSelection) map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	components := map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{}
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if _, selected := selection.componentToRole[component.ComponentName]; selected {
			components[component.ComponentName] = component
		}
	}
	return components
}

func sortedSelectionComponentNames(selection disaggregatedSetSelection) []string {
	names := make([]string, 0, len(selection.componentToRole))
	for componentName := range selection.componentToRole {
		names = append(names, componentName)
	}
	sort.Strings(names)
	return names
}

func (r *DynamoGraphDeploymentReconciler) adoptSelectedModelServices(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, selection disaggregatedSetSelection) error {
	seen := map[string]struct{}{}
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if _, selected := selection.componentToRole[component.ComponentName]; !selected {
			continue
		}
		if component.ModelRef == nil || component.ModelRef.Name == "" {
			continue
		}
		serviceName := dynamo.GenerateServiceName(component.ModelRef.Name)
		if _, ok := seen[serviceName]; ok {
			continue
		}
		seen[serviceName] = struct{}{}
		service := &corev1.Service{}
		err := r.Get(ctx, types.NamespacedName{Name: serviceName, Namespace: dgd.Namespace}, service)
		if apierrors.IsNotFound(err) {
			continue
		}
		if err != nil {
			return fmt.Errorf("failed to get DisaggregatedSet model service %s/%s: %w", dgd.Namespace, serviceName, err)
		}
		if err := r.ensureControlledByDGD(ctx, dgd, service); err != nil {
			return fmt.Errorf("failed to adopt DisaggregatedSet model service %s/%s: %w", service.Namespace, service.Name, err)
		}
	}
	return nil
}

func stringMapToAny(in map[string]string) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func sortedDCDKeys(dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment) []string {
	keys := make([]string, 0, len(dcds))
	for key := range dcds {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func (r *DynamoGraphDeploymentReconciler) syncDisaggregatedSet(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, desired *unstructured.Unstructured) (bool, *unstructured.Unstructured, error) {
	current := newDisaggregatedSetObject()
	key := types.NamespacedName{Name: desired.GetName(), Namespace: desired.GetNamespace()}
	err := r.Get(ctx, key, current)
	if apierrors.IsNotFound(err) {
		if err := r.Create(ctx, desired); err != nil {
			return false, nil, fmt.Errorf("failed to create DisaggregatedSet %s/%s: %w", desired.GetNamespace(), desired.GetName(), err)
		}
		return true, desired, nil
	}
	if err != nil {
		return false, nil, fmt.Errorf("failed to get DisaggregatedSet %s/%s: %w", desired.GetNamespace(), desired.GetName(), err)
	}
	if !isControlledByBetaDGD(current, dgd) {
		return false, nil, fmt.Errorf("refusing to update DisaggregatedSet %s/%s because it is not controlled by DynamoGraphDeployment %s/%s", desired.GetNamespace(), desired.GetName(), dgd.Namespace, dgd.Name)
	}
	original := current.DeepCopy()
	current.SetLabels(desired.GetLabels())
	current.SetAnnotations(desired.GetAnnotations())
	current.SetOwnerReferences(desired.GetOwnerReferences())
	current.Object["spec"] = desired.Object["spec"]
	if equality.Semantic.DeepEqual(original.Object["spec"], current.Object["spec"]) &&
		equality.Semantic.DeepEqual(original.GetLabels(), current.GetLabels()) &&
		equality.Semantic.DeepEqual(original.GetAnnotations(), current.GetAnnotations()) &&
		equality.Semantic.DeepEqual(original.GetOwnerReferences(), current.GetOwnerReferences()) {
		return false, current, nil
	}
	if err := r.Patch(ctx, current, client.MergeFrom(original)); err != nil {
		return false, nil, fmt.Errorf("failed to patch DisaggregatedSet %s/%s: %w", current.GetNamespace(), current.GetName(), err)
	}
	return true, current, nil
}

func (r *DynamoGraphDeploymentReconciler) deleteDisaggregatedSetIfExists(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error {
	ds := newDisaggregatedSetObject()
	key := types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}
	if err := r.Get(ctx, key, ds); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get stale DisaggregatedSet %s/%s: %w", dgd.Namespace, disaggregatedSetName(dgd), err)
	}
	if !isControlledByBetaDGD(ds, dgd) {
		return fmt.Errorf("refusing to delete DisaggregatedSet %s/%s because it is not controlled by DynamoGraphDeployment %s/%s", dgd.Namespace, disaggregatedSetName(dgd), dgd.Namespace, dgd.Name)
	}
	if err := r.Delete(ctx, ds); err != nil {
		return fmt.Errorf("failed to delete stale DisaggregatedSet %s/%s: %w", dgd.Namespace, disaggregatedSetName(dgd), err)
	}
	return nil
}

func (r *DynamoGraphDeploymentReconciler) deleteOwnedSelectedDCDs(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, selection disaggregatedSetSelection) error {
	dcds, err := r.listOwnedSelectedDCDs(ctx, dgd, selection)
	if err != nil {
		return err
	}
	for i := range dcds {
		dcd := &dcds[i]
		if err := r.Delete(ctx, dcd); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to delete selected DynamoComponentDeployment %s/%s: %w", dcd.Namespace, dcd.Name, err)
		}
	}
	return nil
}

func (r *DynamoGraphDeploymentReconciler) listOwnedSelectedDCDs(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, selection disaggregatedSetSelection) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := r.List(ctx, dcdList, client.InNamespace(dgd.Namespace), client.MatchingLabels{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
	}); err != nil {
		return nil, fmt.Errorf("failed to list DynamoComponentDeployments for DisaggregatedSet cleanup: %w", err)
	}
	selectedDCDs := []nvidiacomv1beta1.DynamoComponentDeployment{}
	for _, dcd := range dcdList.Items {
		if !isControlledByBetaDGD(&dcd, dgd) {
			continue
		}
		componentName := dynamo.GetDCDComponentName(&dcd)
		if _, selected := selection.componentToRole[componentName]; selected {
			selectedDCDs = append(selectedDCDs, dcd)
		}
	}
	return selectedDCDs, nil
}

func (r *DynamoGraphDeploymentReconciler) ensureControlledByDGD(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, obj client.Object) error {
	if dgdControllerOwnerReference(dgd) == nil || isControlledByBetaDGD(obj, dgd) {
		return nil
	}
	controllerOwner := metav1.GetControllerOf(obj)
	if controllerOwner != nil {
		if controllerOwner.APIVersion != nvidiacomv1beta1.GroupVersion.String() || controllerOwner.Kind != "DynamoComponentDeployment" {
			return fmt.Errorf("resource is controlled by %s/%s %q", controllerOwner.APIVersion, controllerOwner.Kind, controllerOwner.Name)
		}
		dcd := &nvidiacomv1beta1.DynamoComponentDeployment{}
		if err := r.Get(ctx, types.NamespacedName{Name: controllerOwner.Name, Namespace: obj.GetNamespace()}, dcd); err != nil {
			return fmt.Errorf("failed to verify current DynamoComponentDeployment owner %s/%s: %w", obj.GetNamespace(), controllerOwner.Name, err)
		}
		if !isControlledByBetaDGD(dcd, dgd) {
			return fmt.Errorf("current DynamoComponentDeployment owner %s/%s is not controlled by DynamoGraphDeployment %s/%s", dcd.Namespace, dcd.Name, dgd.Namespace, dgd.Name)
		}
	}
	original := obj.DeepCopyObject().(client.Object)
	setDGDControllerOwnerReference(dgd, obj)
	if equality.Semantic.DeepEqual(original.GetOwnerReferences(), obj.GetOwnerReferences()) {
		return nil
	}
	if err := r.Patch(ctx, obj, client.MergeFrom(original)); err != nil {
		return fmt.Errorf("failed to update owner references: %w", err)
	}
	return nil
}

func dgdControllerOwnerReference(dgd *nvidiacomv1beta1.DynamoGraphDeployment) *metav1.OwnerReference {
	if dgd == nil || dgd.UID == "" {
		return nil
	}
	return &metav1.OwnerReference{
		APIVersion:         nvidiacomv1beta1.GroupVersion.String(),
		Kind:               "DynamoGraphDeployment",
		Name:               dgd.Name,
		UID:                dgd.UID,
		Controller:         ptr.To(true),
		BlockOwnerDeletion: ptr.To(true),
	}
}

func setDGDControllerOwnerReference(dgd *nvidiacomv1beta1.DynamoGraphDeployment, obj client.Object) {
	ownerRef := dgdControllerOwnerReference(dgd)
	if ownerRef == nil {
		return
	}
	ownerRefs := make([]metav1.OwnerReference, 0, len(obj.GetOwnerReferences())+1)
	for _, ref := range obj.GetOwnerReferences() {
		if ptr.Deref(ref.Controller, false) {
			continue
		}
		if ref.APIVersion == ownerRef.APIVersion && ref.Kind == ownerRef.Kind && ref.Name == ownerRef.Name {
			continue
		}
		ownerRefs = append(ownerRefs, ref)
	}
	ownerRefs = append(ownerRefs, *ownerRef)
	obj.SetOwnerReferences(ownerRefs)
}

func isControlledByBetaDGD(obj client.Object, dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	if obj == nil || dgd == nil {
		return false
	}
	if dgd.UID != "" {
		return metav1.IsControlledBy(obj, dgd)
	}
	controllerOwner := metav1.GetControllerOf(obj)
	return controllerOwner != nil &&
		controllerOwner.APIVersion == nvidiacomv1beta1.GroupVersion.String() &&
		controllerOwner.Kind == "DynamoGraphDeployment" &&
		controllerOwner.Name == dgd.Name
}

func checkDisaggregatedSetReadiness(ds *unstructured.Unstructured, selection disaggregatedSetSelection) (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
	statuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus, len(selection.componentToRole))
	roleStatuses := disaggregatedSetRoleStatuses(ds)
	notReadyReasons := []string{}
	for componentName, roleName := range selection.componentToRole {
		desiredReplicas := selection.desiredReplicas[componentName]
		componentStatus := nvidiacomv1beta1.ComponentReplicaStatus{
			ComponentKind:  nvidiacomv1beta1.ComponentKindLeaderWorkerSet,
			ComponentNames: []string{fmt.Sprintf("%s/%s", ds.GetName(), roleName)},
		}
		roleStatus, found := roleStatuses[roleName]
		if found {
			componentStatus.Replicas = nestedInt32(roleStatus, "replicas")
			componentStatus.UpdatedReplicas = nestedInt32(roleStatus, "updatedReplicas")
			readyReplicas := nestedInt32(roleStatus, "readyReplicas")
			componentStatus.ReadyReplicas = &readyReplicas
		}
		statuses[componentName] = componentStatus
		if desiredReplicas == 0 {
			continue
		}
		if !found {
			notReadyReasons = append(notReadyReasons, fmt.Sprintf("%s role %q has no status yet", componentName, roleName))
			continue
		}
		if componentStatus.Replicas < desiredReplicas ||
			componentStatus.UpdatedReplicas < desiredReplicas ||
			componentStatus.ReadyReplicas == nil ||
			*componentStatus.ReadyReplicas < desiredReplicas {
			notReadyReasons = append(notReadyReasons, fmt.Sprintf(
				"%s role %q replicas not ready (desired=%d replicas=%d updated=%d ready=%d)",
				componentName,
				roleName,
				desiredReplicas,
				componentStatus.Replicas,
				componentStatus.UpdatedReplicas,
				ptr.Deref(componentStatus.ReadyReplicas, 0),
			))
		}
	}
	if current, reason := disaggregatedSetStatusObserved(ds); !current {
		return false, reason, statuses
	}
	if len(notReadyReasons) > 0 {
		sort.Strings(notReadyReasons)
		return false, strings.Join(notReadyReasons, "; "), statuses
	}
	return true, "All DisaggregatedSet roles are ready", statuses
}

func checkDisaggregatedSetComponentReady(ds *unstructured.Unstructured, selection disaggregatedSetSelection, componentName string) (bool, string) {
	roleName, selected := selection.componentToRole[componentName]
	if !selected {
		return false, fmt.Sprintf("component %q is not managed by DisaggregatedSet", componentName)
	}
	desiredReplicas := selection.desiredReplicas[componentName]
	componentSelection := disaggregatedSetSelection{
		componentToRole: map[string]string{componentName: roleName},
		desiredReplicas: map[string]int32{componentName: desiredReplicas},
	}
	ready, reason, _ := checkDisaggregatedSetReadiness(ds, componentSelection)
	return ready, reason
}

func disaggregatedSetStatusObserved(ds *unstructured.Unstructured) (bool, string) {
	if ds == nil || ds.GetGeneration() == 0 {
		return true, ""
	}
	if observedGeneration, found := nestedInt64FromObject(ds.Object, "status", "observedGeneration"); found && observedGeneration < ds.GetGeneration() {
		return false, fmt.Sprintf("DisaggregatedSet status has not observed generation %d (observedGeneration=%d)", ds.GetGeneration(), observedGeneration)
	}
	conditions, found, _ := unstructured.NestedSlice(ds.Object, "status", "conditions")
	if !found {
		return true, ""
	}
	for _, item := range conditions {
		condition, ok := item.(map[string]any)
		if !ok {
			continue
		}
		observedGeneration, ok := nestedInt64(condition, "observedGeneration")
		if !ok || observedGeneration >= ds.GetGeneration() {
			continue
		}
		conditionType, _ := condition["type"].(string)
		return false, fmt.Sprintf("DisaggregatedSet condition %q has not observed generation %d (observedGeneration=%d)", conditionType, ds.GetGeneration(), observedGeneration)
	}
	return true, ""
}

func disaggregatedSetRoleStatuses(ds *unstructured.Unstructured) map[string]map[string]any {
	out := map[string]map[string]any{}
	roleStatuses, found, _ := unstructured.NestedSlice(ds.Object, "status", "roleStatuses")
	if !found {
		return out
	}
	for _, item := range roleStatuses {
		roleStatus, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name, ok := roleStatus["name"].(string)
		if !ok || name == "" {
			continue
		}
		out[name] = roleStatus
	}
	return out
}

func nestedInt32(obj map[string]any, key string) int32 {
	value, _ := nestedInt64(obj, key)
	return int32(value)
}

func nestedInt64FromObject(obj map[string]any, fields ...string) (int64, bool) {
	value, found, err := unstructured.NestedFieldNoCopy(obj, fields...)
	if err != nil || !found {
		return 0, false
	}
	return int64Value(value)
}

func nestedInt64(obj map[string]any, key string) (int64, bool) {
	return int64Value(obj[key])
}

func int64Value(value any) (int64, bool) {
	switch v := value.(type) {
	case int32:
		return int64(v), true
	case int64:
		return v, true
	case int:
		return int64(v), true
	case float64:
		return int64(v), true
	default:
		return 0, false
	}
}

func disaggregatedSetStatusChanged(oldObj, newObj client.Object) bool {
	oldDS, okOld := oldObj.(*unstructured.Unstructured)
	newDS, okNew := newObj.(*unstructured.Unstructured)
	if !okOld || !okNew {
		return false
	}
	return oldDS.GetGeneration() != newDS.GetGeneration() || !equality.Semantic.DeepEqual(oldDS.Object["status"], newDS.Object["status"])
}

func (r *DynamoGraphDeploymentReconciler) getUpdatedInProgressForDisaggregatedSet(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
	logger := log.FromContext(ctx)
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		logger.V(1).Info("failed to select DisaggregatedSet components for restart progress", "reason", reason)
		return inProgress
	}

	ds := newDisaggregatedSetObject()
	dsErr := r.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, ds)
	if dsErr != nil && !apierrors.IsNotFound(dsErr) {
		logger.V(1).Info("failed to get DisaggregatedSet for restart progress", "error", dsErr)
	}

	updatedInProgress := make([]string, 0, len(inProgress))
	for _, componentName := range inProgress {
		if _, selected := selection.componentToRole[componentName]; !selected {
			isFullyUpdated, reason := r.checkComponentFullyUpdated(ctx, dgd, componentName)
			if !isFullyUpdated {
				logger.V(1).Info("component not fully updated", "componentName", componentName, "reason", reason)
				updatedInProgress = append(updatedInProgress, componentName)
			}
			continue
		}

		if dsErr != nil {
			reason := "resource not found"
			if !apierrors.IsNotFound(dsErr) {
				reason = dsErr.Error()
			}
			logger.V(1).Info("DisaggregatedSet component not fully updated", "componentName", componentName, "reason", reason)
			updatedInProgress = append(updatedInProgress, componentName)
			continue
		}

		isFullyUpdated, reason := checkDisaggregatedSetComponentReady(ds, selection, componentName)
		if !isFullyUpdated {
			logger.V(1).Info("DisaggregatedSet component not fully updated", "componentName", componentName, "reason", reason)
			updatedInProgress = append(updatedInProgress, componentName)
		}
	}
	return updatedInProgress
}
