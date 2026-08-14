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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"sort"
	"strconv"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

var disaggregatedSetGVK = schema.GroupVersionKind{
	Group:   "disaggregatedset.x-k8s.io",
	Version: "v1",
	Kind:    "DisaggregatedSet",
}

const (
	maxDisaggregatedSetRoles       = 10
	disaggregatedSetRevisionLength = 8
	maxDisaggregatedSetNameLength  = 31
	// The name budget assumes at most two slice-index digits (slices <= 99).
	// The transitional pathway always renders one slice; when a future
	// grouping API raises the slice cardinality, extend this budget first.
	maxDisaggregatedSetSliceIndexLength = len("99")
	disaggregatedSetServiceSuffixLength = len("-prv")
	maxDisaggregatedSetRoleNameLength   = 63 - maxDisaggregatedSetNameLength - maxDisaggregatedSetSliceIndexLength - disaggregatedSetRevisionLength - 3 - disaggregatedSetServiceSuffixLength
	disaggregatedSetNameHashLength      = 8
	dynamoGraphDeploymentKind           = "DynamoGraphDeployment"
	dynamoComponentDeploymentKind       = "DynamoComponentDeployment"
	resourceNotFoundReason              = "resource not found"
)

type disaggregatedSetSelection struct {
	componentToRole map[string]string
	desiredReplicas map[string]int32
}

type disaggregatedSetChildIdentity struct {
	slice int
	role  string
}

func newDisaggregatedSetObject() *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(disaggregatedSetGVK)
	return obj
}

func disaggregatedSetEligibilityReason(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	gate features.Gate,
) string {
	if dgd == nil {
		return "DynamoGraphDeployment is nil"
	}
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return reason
	}
	if len(selection.componentToRole) < 2 {
		return "DisaggregatedSet requires at least two eligible multinode worker roles"
	}
	if gate == nil || !gate.Enabled(features.LWS) {
		for i := range dgd.Spec.Components {
			component := &dgd.Spec.Components[i]
			if component.GetNumberOfNodes() <= 1 {
				continue
			}
			if _, selected := selection.componentToRole[component.ComponentName]; !selected {
				return fmt.Sprintf(
					"multinode component %q is not eligible for DisaggregatedSet and requires LeaderWorkerSet support",
					component.ComponentName,
				)
			}
		}
	}
	return ""
}

// coalesceDisaggregatedSetRestartState treats all selected DS roles as one
// restart unit. A DisaggregatedSet revision covers the complete role list, so
// annotating only one selected role would still roll every role.
func coalesceDisaggregatedSetRestartState(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
) *dynamo.RestartState {
	if restartState == nil || dynamo.IsParallelRestart(dgd) {
		return restartState
	}
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return restartState
	}
	selectedRestarting := false
	for componentName := range selection.componentToRole {
		if restartState.ShouldAnnotateComponent(componentName) {
			selectedRestarting = true
			break
		}
	}
	if !selectedRestarting {
		return restartState
	}
	for componentName := range selection.componentToRole {
		restartState.ComponentsToAnnotate[componentName] = true
	}
	return restartState
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
			return selection, fmt.Sprintf(
				"component %q uses scalingAdapter, but DisaggregatedSet does not support scale subresource integration",
				component.ComponentName,
			)
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
	if len(selection.componentToRole) > maxDisaggregatedSetRoles {
		return selection, fmt.Sprintf("DisaggregatedSet supports at most %d roles", maxDisaggregatedSetRoles)
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
	preferred = truncateDNSLabelWithHash(preferred, maxDisaggregatedSetRoleNameLength)
	roleName := preferred
	for i := 2; roleNameUsed(roleName, used); i++ {
		suffix := fmt.Sprintf("-%d", i)
		roleName = truncateDNSLabel(preferred, maxDisaggregatedSetRoleNameLength-len(suffix)) + suffix
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

func truncateDNSLabelWithHash(value string, maxLength int) string {
	if len(value) <= maxLength {
		return value
	}
	hash := sha256.Sum256([]byte(value))
	hashText := hex.EncodeToString(hash[:])[:disaggregatedSetNameHashLength]
	if maxLength <= len(hashText) {
		return hashText[:maxLength]
	}
	suffix := "-" + hashText
	prefix := strings.TrimRight(value[:maxLength-len(suffix)], "-")
	return prefix + suffix
}

func disaggregatedSetName(dgd *nvidiacomv1beta1.DynamoGraphDeployment) string {
	return truncateDNSLabelWithHash(dynamo.NormalizeKubeResourceName(dgd.Name), maxDisaggregatedSetNameLength)
}

func (r *disaggregatedSetWorkloadsReconciler) reconcileDisaggregatedSetResources(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error) {
	resources := []Resource{}
	logger := log.FromContext(ctx)

	rollingUpdateCtx, err := r.buildRollingUpdateContext(ctx, dgd)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to build rolling update context: %w", err)
	}
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return ReconcileResult{}, fmt.Errorf("failed to select DisaggregatedSet roles: %s", reason)
	}

	existingRestartAnnotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
	if err != nil {
		logger.Error(err, "failed to get existing restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing restart annotations: %w", err)
	}
	existingDSRestartAnnotations, err := r.getExistingRestartAnnotationsDisaggregatedSet(ctx, dgd, selection)
	if err != nil {
		logger.Error(err, "failed to get existing DisaggregatedSet restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing DisaggregatedSet restart annotations: %w", err)
	}
	maps.Copy(existingRestartAnnotations, existingDSRestartAnnotations)

	dcds, err := dynamo.GenerateDynamoComponentsDeployments(
		dgd,
		restartState,
		existingRestartAnnotations,
		rollingUpdateCtx,
	)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to generate DynamoComponentDeployments for DisaggregatedSet path: %w", err)
	}

	checkpointGated, err := r.applyDisaggregatedSetCheckpointStartupPolicies(dcds, checkpointInfos, selection)
	if err != nil {
		return ReconcileResult{}, err
	}

	desiredDS, err := r.generateDisaggregatedSet(ctx, dgd, dcds, selection)
	if err != nil {
		return ReconcileResult{}, err
	}
	dsModified, syncedDS, err := r.syncDisaggregatedSet(ctx, dgd, desiredDS)
	if err != nil {
		return ReconcileResult{}, err
	}

	targetReady, dsReason, dsStatuses, err := r.checkDisaggregatedSetReadiness(ctx, syncedDS, selection)
	if err != nil {
		return ReconcileResult{}, err
	}
	dsReady := targetReady && !dsModified && !checkpointGated
	targetRevision, err := disaggregatedSetTargetRevision(syncedDS)
	if err != nil {
		return ReconcileResult{}, err
	}
	selectedServiceNames, err := r.reconcileDisaggregatedSetSideResources(
		ctx,
		dgd,
		dcds,
		selection,
		targetRevision,
		dsReady,
	)
	if err != nil {
		return ReconcileResult{}, err
	}

	syncedDSResource, err := commoncontroller.NewResourceWithComponentStatuses(
		syncedDS,
		func() (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
			if dsModified {
				return false, "DisaggregatedSet spec was updated; waiting for controller status", dsStatuses
			}
			if checkpointGated {
				return false, "DisaggregatedSet roles are waiting for checkpoint readiness", dsStatuses
			}
			return dsReady, dsReason, dsStatuses
		},
	)
	if err != nil {
		return ReconcileResult{}, err
	}
	resources = append(resources, syncedDSResource)

	nonSelectedResources, err := r.reconcileDisaggregatedSetNonSelectedDCDs(ctx, dgd, dcds, selection)
	if err != nil {
		return ReconcileResult{}, err
	}
	resources = append(resources, nonSelectedResources...)
	desiredServiceNames, err := r.reconcileDisaggregatedSetServiceOwnership(ctx, dgd, dcds, selection, selectedServiceNames)
	if err != nil {
		return ReconcileResult{}, err
	}

	if dsReady {
		if err := r.deleteOwnedSelectedDCDs(ctx, dgd, selection); err != nil {
			return ReconcileResult{}, err
		}
	}

	result := r.checkResourcesReadiness(resources)
	if result.State == nvidiacomv1beta1.DGDStateSuccessful {
		if err := r.deleteStaleDisaggregatedSetServices(ctx, dgd, desiredServiceNames); err != nil {
			return ReconcileResult{}, err
		}
	}
	return result, nil
}

func (r *disaggregatedSetWorkloadsReconciler) getExistingRestartAnnotationsDisaggregatedSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) (map[string]string, error) {
	ds := newDisaggregatedSetObject()
	key := types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}
	if err := r.Get(ctx, key, ds); err != nil {
		if apierrors.IsNotFound(err) {
			return map[string]string{}, nil
		}
		return nil, fmt.Errorf("failed to get DisaggregatedSet %s: %w", key, err)
	}
	return restartAnnotationsFromDisaggregatedSet(ds, selection)
}

func restartAnnotationsFromDisaggregatedSet(
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
) (map[string]string, error) {
	restartAnnotations := make(map[string]string)
	if ds == nil {
		return restartAnnotations, nil
	}
	spec, found, err := unstructured.NestedMap(ds.Object, "spec")
	if err != nil {
		return nil, fmt.Errorf("failed to read DisaggregatedSet spec: %w", err)
	}
	if !found {
		return restartAnnotations, nil
	}
	typedSpec := disaggregatedsetv1.DisaggregatedSetSpec{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(spec, &typedSpec); err != nil {
		return nil, fmt.Errorf("failed to decode DisaggregatedSet spec: %w", err)
	}
	roleToComponent := make(map[string]string, len(selection.componentToRole))
	for componentName, roleName := range selection.componentToRole {
		roleToComponent[roleName] = componentName
	}
	for i := range typedSpec.Roles {
		role := &typedSpec.Roles[i]
		componentName, selected := roleToComponent[role.Name]
		if !selected {
			continue
		}
		if role.Spec.LeaderWorkerTemplate.LeaderTemplate != nil {
			if timestamp := role.Spec.LeaderWorkerTemplate.LeaderTemplate.Annotations[consts.RestartAnnotation]; timestamp != "" {
				restartAnnotations[componentName] = timestamp
				continue
			}
		}
		if timestamp := role.Spec.LeaderWorkerTemplate.WorkerTemplate.Annotations[consts.RestartAnnotation]; timestamp != "" {
			restartAnnotations[componentName] = timestamp
		}
	}
	return restartAnnotations, nil
}

func (r *disaggregatedSetWorkloadsReconciler) applyDisaggregatedSetCheckpointStartupPolicies(
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
	selection disaggregatedSetSelection,
) (bool, error) {
	for _, componentName := range sortedDCDKeys(dcds) {
		if err := r.applyDCDCheckpointStartupPolicy(dcds[componentName], checkpointInfos[componentName]); err != nil {
			return false, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", componentName, err)
		}
	}

	gateSelectedRoles := false
	for componentName := range selection.componentToRole {
		info := checkpointInfos[componentName]
		if info != nil &&
			info.Enabled &&
			info.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint &&
			!info.Ready {
			gateSelectedRoles = true
			break
		}
	}
	for componentName := range selection.componentToRole {
		dcd := dcds[componentName]
		if dcd == nil {
			return false, fmt.Errorf("generated DynamoComponentDeployment missing for selected component %q", componentName)
		}
		if gateSelectedRoles {
			dcd.Spec.Replicas = ptr.To(int32(0))
		}
		selection.desiredReplicas[componentName] = desiredComponentReplicas(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
	}
	return gateSelectedRoles, nil
}

func (r *disaggregatedSetWorkloadsReconciler) reconcileDisaggregatedSetNonSelectedDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	selection disaggregatedSetSelection,
) ([]Resource, error) {
	resources := []Resource{}
	for _, componentName := range sortedDCDKeys(dcds) {
		dcd := dcds[componentName]
		if _, selected := selection.componentToRole[componentName]; selected {
			continue
		}
		if err := r.preserveExistingDCDBackendFramework(ctx, dcd); err != nil {
			return nil, fmt.Errorf("failed to preserve existing DynamoComponentDeployment backendFramework: %w", err)
		}
		_, syncedDCD, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*nvidiacomv1beta1.DynamoComponentDeployment, bool, error) {
			return dcd, false, nil
		})
		if err != nil {
			return nil, fmt.Errorf("failed to sync non-DisaggregatedSet DynamoComponentDeployment %s: %w", dcd.Name, err)
		}
		dcds[componentName] = syncedDCD
		resources = append(resources, syncedDCD)
	}
	return resources, nil
}

func (r *disaggregatedSetWorkloadsReconciler) generateDisaggregatedSet(
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
		renderDCD := dcd.DeepCopy()
		if ownerRef := dgdControllerOwnerReference(dgd); ownerRef != nil {
			renderDCD.SetOwnerReferences([]metav1.OwnerReference{*ownerRef})
		}
		role, err := r.buildDisaggregatedSetRole(ctx, renderDCD)
		if err != nil {
			return nil, fmt.Errorf("failed to build DisaggregatedSet role %q: %w", roleName, err)
		}
		role["name"] = roleName
		roles = append(roles, role)
	}
	if len(roles) < 2 {
		return nil, fmt.Errorf("DisaggregatedSet requires at least two roles, got %d", len(roles))
	}
	ds.Object["spec"] = map[string]any{"roles": roles}
	return ds, nil
}

func disaggregatedSetTargetRevision(ds *unstructured.Unstructured) (string, error) {
	typedDS := &disaggregatedsetv1.DisaggregatedSet{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(ds.Object, typedDS); err != nil {
		return "", fmt.Errorf("failed to decode DisaggregatedSet for target revision: %w", err)
	}
	return disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles), nil
}

func setDisaggregatedSetServiceSelector(service *corev1.Service, setName, roleName, revision string) {
	// Dynamo's stable component Service intentionally omits the slice label so it
	// aggregates every slice for this role and revision. LWS owns separate
	// slice-local <lws-name>-prv Services for intra-slice discovery.
	service.Spec.Selector = map[string]string{
		disaggregatedsetv1.SetNameLabelKey:  setName,
		disaggregatedsetv1.RoleLabelKey:     roleName,
		disaggregatedsetv1.RevisionLabelKey: revision,
	}
}

func isDisaggregatedSetServiceSelector(service *corev1.Service) bool {
	if service == nil {
		return false
	}
	return service.Spec.Selector[disaggregatedsetv1.SetNameLabelKey] != "" &&
		service.Spec.Selector[disaggregatedsetv1.RoleLabelKey] != "" &&
		service.Spec.Selector[disaggregatedsetv1.RevisionLabelKey] != ""
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

// buildDisaggregatedSetRole reuses the shared DCD workload renderer.
func (r *disaggregatedSetWorkloadsReconciler) buildDisaggregatedSetRole(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (map[string]any, error) {
	leaderPodTemplateSpec, workerPodTemplateSpec, err := r.renderer.renderMultinodePodTemplateSpecs(ctx, dcd)
	if err != nil {
		return nil, err
	}

	desiredReplicas := int32(1)
	if dcd.Spec.Replicas != nil {
		desiredReplicas = *dcd.Spec.Replicas
	}
	groupSize := dcd.GetNumberOfNodes()

	lwsSpec := leaderworkersetv1.LeaderWorkerSetSpec{
		Replicas:      &desiredReplicas,
		StartupPolicy: leaderworkersetv1.LeaderCreatedStartupPolicy,
		RolloutStrategy: leaderworkersetv1.RolloutStrategy{
			Type: leaderworkersetv1.RollingUpdateStrategyType,
		},
		LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
			LeaderTemplate: leaderPodTemplateSpec,
			WorkerTemplate: *workerPodTemplateSpec,
			Size:           &groupSize,
			RestartPolicy:  leaderworkersetv1.RecreateGroupOnPodRestart,
		},
	}
	lwsSpecUnstructured, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&lwsSpec)
	if err != nil {
		return nil, fmt.Errorf("failed to convert LeaderWorkerSet spec: %w", err)
	}
	return map[string]any{"spec": lwsSpecUnstructured}, nil
}

func (r *disaggregatedSetWorkloadsReconciler) reconcileDisaggregatedSetSideResources(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	selection disaggregatedSetSelection,
	targetRevision string,
	targetReady bool,
) (map[string]struct{}, error) {
	desiredServiceNames := map[string]struct{}{}
	if err := dynamo.ReconcileModelServicesForComponents(ctx, r, dgd, selectedComponentsByName(dgd, selection), dgd.Namespace); err != nil {
		return nil, fmt.Errorf("failed to reconcile DisaggregatedSet model services: %w", err)
	}
	if err := r.adoptSelectedModelServices(ctx, dgd, selection); err != nil {
		return nil, err
	}

	for _, componentName := range sortedSelectionComponentNames(selection) {
		dcd := dcds[componentName]
		if dcd == nil {
			return nil, fmt.Errorf("generated DynamoComponentDeployment missing for selected component %q", componentName)
		}
		serviceKey := types.NamespacedName{Name: dynamo.NormalizeKubeResourceName(dcd.Name), Namespace: dcd.Namespace}
		existingService := &corev1.Service{}
		existingServiceErr := r.Get(ctx, serviceKey, existingService)
		if existingServiceErr != nil && !apierrors.IsNotFound(existingServiceErr) {
			return nil, fmt.Errorf("failed to get existing DisaggregatedSet component service for %q: %w", componentName, existingServiceErr)
		}
		_, syncedService, err := commoncontroller.SyncResource(ctx, r, dgd, func(context.Context) (*corev1.Service, bool, error) {
			service, deleted, err := r.renderer.generateService(ctx, dcd)
			if err != nil || deleted || targetRevision == "" {
				return service, deleted, err
			}
			setDesiredDisaggregatedSetServiceSelector(
				service,
				existingService,
				existingServiceErr == nil,
				disaggregatedSetName(dgd),
				selection.componentToRole[componentName],
				targetRevision,
				targetReady,
			)
			return service, false, nil
		})
		if err != nil {
			return nil, fmt.Errorf("failed to reconcile DisaggregatedSet component service for %q: %w", componentName, err)
		}
		if syncedService != nil {
			desiredServiceNames[syncedService.Name] = struct{}{}
			if err := r.ensureControlledByDGD(ctx, dgd, syncedService); err != nil {
				return nil, fmt.Errorf("failed to adopt DisaggregatedSet component service %s/%s: %w", syncedService.Namespace, syncedService.Name, err)
			}
		}
	}
	return desiredServiceNames, nil
}

type disaggregatedSetDesiredServiceOwner struct {
	dgd bool
	dcd *nvidiacomv1beta1.DynamoComponentDeployment
}

func (r *disaggregatedSetWorkloadsReconciler) reconcileDisaggregatedSetServiceOwnership(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	selection disaggregatedSetSelection,
	selectedServiceNames map[string]struct{},
) (map[string]struct{}, error) {
	owners := make(map[string]disaggregatedSetDesiredServiceOwner, len(dcds))
	for serviceName := range selectedServiceNames {
		owners[serviceName] = disaggregatedSetDesiredServiceOwner{dgd: true}
	}

	for componentName, dcd := range dcds {
		if dcd == nil {
			continue
		}
		if _, selected := selection.componentToRole[componentName]; selected {
			continue
		}
		owners[dynamo.NormalizeKubeResourceName(dcd.Name)] = disaggregatedSetDesiredServiceOwner{dcd: dcd}
	}

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ModelRef == nil || component.ModelRef.Name == "" {
			continue
		}
		serviceName := dynamo.GenerateServiceName(component.ModelRef.Name)
		if _, selected := selection.componentToRole[component.ComponentName]; selected {
			owners[serviceName] = disaggregatedSetDesiredServiceOwner{dgd: true}
			continue
		}
		if _, exists := owners[serviceName]; exists {
			continue
		}
		if dcd := dcds[component.ComponentName]; dcd != nil {
			owners[serviceName] = disaggregatedSetDesiredServiceOwner{dcd: dcd}
		}
	}

	serviceNames := make([]string, 0, len(owners))
	for serviceName := range owners {
		serviceNames = append(serviceNames, serviceName)
	}
	sort.Strings(serviceNames)
	for _, serviceName := range serviceNames {
		service := &corev1.Service{}
		if err := r.Get(ctx, types.NamespacedName{Name: serviceName, Namespace: dgd.Namespace}, service); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return nil, fmt.Errorf("failed to get desired Service %s/%s: %w", dgd.Namespace, serviceName, err)
		}
		owner := owners[serviceName]
		if owner.dgd {
			if err := r.ensureControlledByDGD(ctx, dgd, service); err != nil {
				return nil, fmt.Errorf("failed to assign Service %s/%s to DynamoGraphDeployment: %w", service.Namespace, service.Name, err)
			}
			continue
		}
		if owner.dcd != nil {
			if err := r.ensureControlledByDCD(ctx, dgd, owner.dcd, service); err != nil {
				return nil, fmt.Errorf("failed to assign Service %s/%s to DynamoComponentDeployment %s: %w", service.Namespace, service.Name, owner.dcd.Name, err)
			}
		}
	}

	desiredServiceNames := make(map[string]struct{}, len(owners))
	for serviceName := range owners {
		desiredServiceNames[serviceName] = struct{}{}
	}
	return desiredServiceNames, nil
}

func selectedComponentsByName(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
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

func (r *disaggregatedSetWorkloadsReconciler) adoptSelectedModelServices(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) error {
	selected := selectedComponentsByName(dgd, selection)
	serviceNames := make([]string, 0, len(selected))
	for _, component := range selected {
		if component.ModelRef == nil || component.ModelRef.Name == "" {
			continue
		}
		serviceNames = append(serviceNames, dynamo.GenerateServiceName(component.ModelRef.Name))
	}
	sort.Strings(serviceNames)
	for _, serviceName := range serviceNames {
		service := &corev1.Service{}
		if err := r.Get(ctx, types.NamespacedName{Name: serviceName, Namespace: dgd.Namespace}, service); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return fmt.Errorf("failed to get selected model service %s/%s: %w", dgd.Namespace, serviceName, err)
		}
		ok, err := r.canAdoptModelServiceForDisaggregatedSet(ctx, dgd, service)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		if err := r.ensureControlledByDGD(ctx, dgd, service); err != nil {
			return fmt.Errorf("failed to adopt model service %s/%s: %w", service.Namespace, service.Name, err)
		}
	}
	return nil
}

func (r *disaggregatedSetWorkloadsReconciler) canAdoptModelServiceForDisaggregatedSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	service *corev1.Service,
) (bool, error) {
	owner := metav1.GetControllerOf(service)
	if owner == nil || isControlledByBetaDGD(service, dgd) {
		return true, nil
	}
	if owner.APIVersion != nvidiacomv1beta1.GroupVersion.String() || owner.Kind != dynamoComponentDeploymentKind {
		return false, fmt.Errorf("model service %s/%s is controlled by %s/%s %q", service.Namespace, service.Name, owner.APIVersion, owner.Kind, owner.Name)
	}
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{}
	if err := r.Get(ctx, types.NamespacedName{Name: owner.Name, Namespace: service.Namespace}, dcd); err != nil {
		return false, fmt.Errorf("failed to verify model service owner %s/%s: %w", service.Namespace, owner.Name, err)
	}
	return isControlledByBetaDGD(dcd, dgd), nil
}

func sortedDCDKeys(dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment) []string {
	keys := make([]string, 0, len(dcds))
	for key := range dcds {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func (r *disaggregatedSetWorkloadsReconciler) syncDisaggregatedSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *unstructured.Unstructured,
) (bool, *unstructured.Unstructured, error) {
	current := newDisaggregatedSetObject()
	key := types.NamespacedName{Name: desired.GetName(), Namespace: desired.GetNamespace()}
	err := r.Get(ctx, key, current)
	if apierrors.IsNotFound(err) {
		if err := r.Create(ctx, desired); err != nil {
			return false, nil, fmt.Errorf("failed to create DisaggregatedSet %s: %w", key, err)
		}
		return true, desired, nil
	}
	if err != nil {
		return false, nil, fmt.Errorf("failed to get DisaggregatedSet %s: %w", key, err)
	}
	if !isControlledByBetaDGD(current, dgd) {
		return false, nil, fmt.Errorf(
			"refusing to reconcile DisaggregatedSet %s because it is not controlled by DynamoGraphDeployment %s/%s",
			key,
			dgd.Namespace,
			dgd.Name,
		)
	}

	original := current.DeepCopy()
	labels := maps.Clone(current.GetLabels())
	if labels == nil {
		labels = map[string]string{}
	}
	maps.Copy(labels, desired.GetLabels())
	current.SetLabels(labels)
	annotations := maps.Clone(current.GetAnnotations())
	if annotations == nil && len(desired.GetAnnotations()) > 0 {
		annotations = map[string]string{}
	}
	maps.Copy(annotations, desired.GetAnnotations())
	current.SetAnnotations(annotations)
	setDGDControllerOwnerReference(dgd, current)
	current.Object["spec"] = desired.Object["spec"]
	if disaggregatedSetDesiredStateEqual(original, current) {
		return false, current, nil
	}

	// Let the API server apply structural defaults before deciding whether the
	// controller-owned desired state differs from the live object.
	if err := r.Patch(ctx, current, client.MergeFrom(original), client.DryRunAll); err != nil {
		return false, nil, fmt.Errorf("failed to dry-run patch DisaggregatedSet %s: %w", key, err)
	}
	if disaggregatedSetDesiredStateEqual(original, current) {
		return false, current, nil
	}
	if err := r.Patch(ctx, current, client.MergeFrom(original)); err != nil {
		return false, nil, fmt.Errorf("failed to patch DisaggregatedSet %s: %w", key, err)
	}
	return true, current, nil
}

func disaggregatedSetDesiredStateEqual(a, b *unstructured.Unstructured) bool {
	if a == nil || b == nil {
		return a == b
	}
	return equality.Semantic.DeepEqual(a.Object["spec"], b.Object["spec"]) &&
		equality.Semantic.DeepEqual(a.GetLabels(), b.GetLabels()) &&
		equality.Semantic.DeepEqual(a.GetAnnotations(), b.GetAnnotations()) &&
		equality.Semantic.DeepEqual(a.GetOwnerReferences(), b.GetOwnerReferences())
}

func (r *disaggregatedSetWorkloadsReconciler) deleteOwnedSelectedDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) error {
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

func (r *disaggregatedSetWorkloadsReconciler) deleteStaleDisaggregatedSetServices(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desiredServiceNames map[string]struct{},
) error {
	serviceList := &corev1.ServiceList{}
	if err := r.List(ctx, serviceList, client.InNamespace(dgd.Namespace), client.MatchingLabels{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
	}); err != nil {
		return fmt.Errorf("failed to list DisaggregatedSet Services: %w", err)
	}

	for i := range serviceList.Items {
		service := &serviceList.Items[i]
		if !isControlledByBetaDGD(service, dgd) {
			continue
		}
		// Only the Services this pathway adopts are candidates: per-component
		// discovery Services carry the component label and headless model
		// Services carry the base-model hash label.
		componentName := service.Labels[consts.KubeLabelDynamoComponent]
		modelHash := service.Labels[consts.KubeLabelDynamoBaseModelHash]
		if componentName == "" && modelHash == "" {
			continue
		}
		if _, desired := desiredServiceNames[service.Name]; desired {
			continue
		}
		if err := r.Delete(ctx, service); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to delete stale Service %s/%s: %w", service.Namespace, service.Name, err)
		}
	}
	return nil
}

func (r *disaggregatedSetWorkloadsReconciler) listOwnedSelectedDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
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

func (r *disaggregatedSetWorkloadsReconciler) ensureControlledByDGD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	obj client.Object,
) error {
	if dgdControllerOwnerReference(dgd) == nil || isControlledByBetaDGD(obj, dgd) {
		return nil
	}
	controllerOwner := metav1.GetControllerOf(obj)
	if controllerOwner != nil {
		if controllerOwner.APIVersion != nvidiacomv1beta1.GroupVersion.String() || controllerOwner.Kind != dynamoComponentDeploymentKind {
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

func (r *disaggregatedSetWorkloadsReconciler) ensureControlledByDCD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	obj client.Object,
) error {
	ownerRef := dcdControllerOwnerReference(dcd)
	if ownerRef == nil {
		return fmt.Errorf("DynamoComponentDeployment %s/%s has no UID", dcd.Namespace, dcd.Name)
	}
	if metav1.IsControlledBy(obj, dcd) {
		return nil
	}
	if controllerOwner := metav1.GetControllerOf(obj); controllerOwner != nil && !ownerReferenceMatchesDGD(controllerOwner, dgd) {
		if controllerOwner.APIVersion == nvidiacomv1beta1.GroupVersion.String() && controllerOwner.Kind == dynamoComponentDeploymentKind {
			currentOwner := &nvidiacomv1beta1.DynamoComponentDeployment{}
			if err := r.Get(ctx, types.NamespacedName{Name: controllerOwner.Name, Namespace: obj.GetNamespace()}, currentOwner); err != nil {
				return fmt.Errorf("failed to verify current DynamoComponentDeployment owner %s/%s: %w", obj.GetNamespace(), controllerOwner.Name, err)
			}
			if isControlledByBetaDGD(currentOwner, dgd) {
				return nil
			}
		}
		return fmt.Errorf("resource is controlled by %s/%s %q", controllerOwner.APIVersion, controllerOwner.Kind, controllerOwner.Name)
	}

	original := obj.DeepCopyObject().(client.Object)
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
	if err := r.Patch(ctx, obj, client.MergeFrom(original)); err != nil {
		return fmt.Errorf("failed to update owner references: %w", err)
	}
	return nil
}

func dcdControllerOwnerReference(dcd *nvidiacomv1beta1.DynamoComponentDeployment) *metav1.OwnerReference {
	if dcd == nil || dcd.UID == "" {
		return nil
	}
	return &metav1.OwnerReference{
		APIVersion:         nvidiacomv1beta1.GroupVersion.String(),
		Kind:               dynamoComponentDeploymentKind,
		Name:               dcd.Name,
		UID:                dcd.UID,
		Controller:         ptr.To(true),
		BlockOwnerDeletion: ptr.To(true),
	}
}

func ownerReferenceMatchesDGD(owner *metav1.OwnerReference, dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	return owner != nil &&
		dgd != nil &&
		owner.APIVersion == nvidiacomv1beta1.GroupVersion.String() &&
		owner.Kind == dynamoGraphDeploymentKind &&
		owner.Name == dgd.Name &&
		owner.UID != "" &&
		owner.UID == dgd.UID
}

func dgdControllerOwnerReference(dgd *nvidiacomv1beta1.DynamoGraphDeployment) *metav1.OwnerReference {
	if dgd == nil || dgd.UID == "" {
		return nil
	}
	return &metav1.OwnerReference{
		APIVersion:         nvidiacomv1beta1.GroupVersion.String(),
		Kind:               dynamoGraphDeploymentKind,
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
		controllerOwner.Kind == dynamoGraphDeploymentKind &&
		controllerOwner.Name == dgd.Name
}

func checkDisaggregatedSetReadiness(
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
) (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
	sliceCount := disaggregatedSetSliceCount(ds)
	statuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus, len(selection.componentToRole))
	roleStatuses := disaggregatedSetRoleStatuses(ds)
	notReadyReasons := []string{}
	for componentName, roleName := range selection.componentToRole {
		desiredReplicas := selection.desiredReplicas[componentName] * sliceCount
		componentStatus := nvidiacomv1beta1.ComponentReplicaStatus{
			ComponentKind: nvidiacomv1beta1.ComponentKindLeaderWorkerSet,
		}
		roleStatus, found := roleStatuses[roleName]
		if found {
			componentStatus.Replicas = nestedInt32(roleStatus, "replicas")
			componentStatus.UpdatedReplicas = nestedInt32(roleStatus, "updatedReplicas")
			readyReplicas := nestedInt32(roleStatus, "readyReplicas")
			componentStatus.ReadyReplicas = &readyReplicas
		}
		statuses[componentName] = componentStatus
		if !found {
			notReadyReasons = append(notReadyReasons, fmt.Sprintf("%s role %q has no status yet", componentName, roleName))
			continue
		}
		if desiredReplicas == 0 {
			if componentStatus.Replicas != 0 || componentStatus.UpdatedReplicas != 0 || ptr.Deref(componentStatus.ReadyReplicas, 0) != 0 {
				notReadyReasons = append(notReadyReasons, fmt.Sprintf("%s role %q has not scaled to zero", componentName, roleName))
			}
			continue
		}
		if componentStatus.Replicas != desiredReplicas ||
			componentStatus.UpdatedReplicas != desiredReplicas ||
			componentStatus.ReadyReplicas == nil ||
			*componentStatus.ReadyReplicas != desiredReplicas {
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

func disaggregatedSetSliceCount(ds *unstructured.Unstructured) int32 {
	if ds == nil {
		return 1
	}
	value, found := nestedInt64FromObject(ds.Object, "spec", "slices")
	if !found || value < 1 {
		return 1
	}
	return int32(value)
}

func disaggregatedSetChildSlice(labels map[string]string) (int, bool) {
	value := labels[disaggregatedsetv1.SliceLabelKey]
	if value == "" {
		return 0, true
	}
	slice, err := strconv.Atoi(value)
	return slice, err == nil && slice >= 0
}

func (r *disaggregatedSetWorkloadsReconciler) checkDisaggregatedSetReadiness(
	ctx context.Context,
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
) (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus, error) {
	children := &leaderworkersetv1.LeaderWorkerSetList{}
	if err := r.List(ctx, children, client.InNamespace(ds.GetNamespace()), client.MatchingLabels{
		disaggregatedsetv1.SetNameLabelKey: ds.GetName(),
	}); err != nil {
		return false, "", nil, fmt.Errorf("failed to list DisaggregatedSet child LeaderWorkerSets: %w", err)
	}
	targetRevision, err := disaggregatedSetTargetRevision(ds)
	if err != nil {
		return false, "", nil, err
	}
	targetByIdentity := make(map[disaggregatedSetChildIdentity][]*leaderworkersetv1.LeaderWorkerSet)
	childrenByRole := make(map[string][]*leaderworkersetv1.LeaderWorkerSet)
	sliceCount := int(disaggregatedSetSliceCount(ds))
	for i := range children.Items {
		child := &children.Items[i]
		if !metav1.IsControlledBy(child, ds) {
			continue
		}
		roleName := child.Labels[disaggregatedsetv1.RoleLabelKey]
		childrenByRole[roleName] = append(childrenByRole[roleName], child)
		slice, validSlice := disaggregatedSetChildSlice(child.Labels)
		if !validSlice || slice >= sliceCount || child.Labels[disaggregatedsetv1.RevisionLabelKey] != targetRevision {
			continue
		}
		identity := disaggregatedSetChildIdentity{slice: slice, role: roleName}
		targetByIdentity[identity] = append(targetByIdentity[identity], child)
	}
	if len(disaggregatedSetRoleStatuses(ds)) > 0 && disaggregatedSetStatusHasObservation(ds) {
		ready, reason, statuses := checkDisaggregatedSetReadiness(ds, selection)
		if !ready {
			return ready, reason, statuses, nil
		}
	}
	ready, reason, statuses := checkDisaggregatedSetChildLWSReadiness(selection, sliceCount, targetByIdentity, childrenByRole)
	return ready, reason, statuses, nil
}

func checkDisaggregatedSetChildLWSReadiness(
	selection disaggregatedSetSelection,
	sliceCount int,
	targetByIdentity map[disaggregatedSetChildIdentity][]*leaderworkersetv1.LeaderWorkerSet,
	childrenByRole map[string][]*leaderworkersetv1.LeaderWorkerSet,
) (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
	statuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus, len(selection.componentToRole))
	notReadyReasons := []string{}
	for componentName, roleName := range selection.componentToRole {
		desiredReplicas := selection.desiredReplicas[componentName]
		status := nvidiacomv1beta1.ComponentReplicaStatus{ComponentKind: nvidiacomv1beta1.ComponentKindLeaderWorkerSet}
		children := childrenByRole[roleName]
		sort.Slice(children, func(i, j int) bool { return children[i].Name < children[j].Name })
		readyReplicas := int32(0)
		for _, roleChild := range children {
			status.ComponentNames = append(status.ComponentNames, roleChild.Name)
			status.Replicas += roleChild.Status.Replicas
			readyReplicas += roleChild.Status.ReadyReplicas
		}
		status.ReadyReplicas = ptr.To(readyReplicas)
		for slice := range sliceCount {
			identity := disaggregatedSetChildIdentity{slice: slice, role: roleName}
			targets := targetByIdentity[identity]
			if len(targets) != 1 {
				notReadyReasons = append(notReadyReasons, fmt.Sprintf(
					"%s role %q slice %d has %d target LeaderWorkerSets, expected 1",
					componentName, roleName, slice, len(targets),
				))
				continue
			}
			child := targets[0]
			status.UpdatedReplicas += child.Status.UpdatedReplicas
			if child.Status.ObservedGeneration < child.Generation {
				notReadyReasons = append(notReadyReasons, fmt.Sprintf("%s child LeaderWorkerSet %q has not observed generation %d", componentName, child.Name, child.Generation))
				continue
			}
			if child.Status.Replicas != desiredReplicas || child.Status.UpdatedReplicas != desiredReplicas || child.Status.ReadyReplicas != desiredReplicas {
				notReadyReasons = append(notReadyReasons, fmt.Sprintf(
					"%s child LeaderWorkerSet %q for slice %d replicas not ready (desired=%d replicas=%d updated=%d ready=%d)",
					componentName,
					child.Name,
					slice,
					desiredReplicas,
					child.Status.Replicas,
					child.Status.UpdatedReplicas,
					child.Status.ReadyReplicas,
				))
			}
		}
		statuses[componentName] = status
	}

	notReadyReasons = append(notReadyReasons, staleDisaggregatedSetChildLWSNotReadyReasons(targetByIdentity, childrenByRole)...)
	if len(notReadyReasons) > 0 {
		sort.Strings(notReadyReasons)
		return false, strings.Join(notReadyReasons, "; "), statuses
	}
	return true, "All DisaggregatedSet child LeaderWorkerSets are ready", statuses
}

func staleDisaggregatedSetChildLWSNotReadyReasons(
	targetByIdentity map[disaggregatedSetChildIdentity][]*leaderworkersetv1.LeaderWorkerSet,
	childrenByRole map[string][]*leaderworkersetv1.LeaderWorkerSet,
) []string {
	targets := make(map[*leaderworkersetv1.LeaderWorkerSet]struct{})
	for _, children := range targetByIdentity {
		if len(children) == 1 {
			targets[children[0]] = struct{}{}
		}
	}
	notReadyReasons := []string{}
	for roleName, children := range childrenByRole {
		sort.Slice(children, func(i, j int) bool { return children[i].Name < children[j].Name })
		for _, child := range children {
			if _, target := targets[child]; target {
				continue
			}
			if ptr.Deref(child.Spec.Replicas, 1) == 0 &&
				child.Status.Replicas == 0 &&
				child.Status.UpdatedReplicas == 0 &&
				child.Status.ReadyReplicas == 0 {
				continue
			}
			notReadyReasons = append(notReadyReasons, fmt.Sprintf(
				"stale role %q child LeaderWorkerSet %q has not scaled to zero",
				roleName,
				child.Name,
			))
		}
	}
	sort.Strings(notReadyReasons)
	return notReadyReasons
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

func disaggregatedSetStatusHasObservation(ds *unstructured.Unstructured) bool {
	if ds == nil || ds.GetGeneration() == 0 {
		return true
	}
	if _, found := nestedInt64FromObject(ds.Object, "status", "observedGeneration"); found {
		return true
	}
	conditions, found, _ := unstructured.NestedSlice(ds.Object, "status", "conditions")
	if !found {
		return false
	}
	for _, item := range conditions {
		condition, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if _, found := nestedInt64(condition, "observedGeneration"); found {
			return true
		}
	}
	return false
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
	return oldDS.GetGeneration() != newDS.GetGeneration() ||
		!equality.Semantic.DeepEqual(oldDS.Object["status"], newDS.Object["status"]) ||
		!equality.Semantic.DeepEqual(oldDS.GetLabels(), newDS.GetLabels()) ||
		!equality.Semantic.DeepEqual(oldDS.GetOwnerReferences(), newDS.GetOwnerReferences())
}

func leaderWorkerSetStatusChanged(oldObj, newObj client.Object) bool {
	oldLWS, okOld := oldObj.(*leaderworkersetv1.LeaderWorkerSet)
	newLWS, okNew := newObj.(*leaderworkersetv1.LeaderWorkerSet)
	if !okOld || !okNew {
		return false
	}
	return oldLWS.Generation != newLWS.Generation ||
		!equality.Semantic.DeepEqual(oldLWS.Status, newLWS.Status) ||
		!equality.Semantic.DeepEqual(oldLWS.GetLabels(), newLWS.GetLabels()) ||
		!equality.Semantic.DeepEqual(oldLWS.GetOwnerReferences(), newLWS.GetOwnerReferences())
}

type disaggregatedSetWatchMapper struct {
	reader client.Reader
}

func newDisaggregatedSetWatchMapper(reader client.Reader) *disaggregatedSetWatchMapper {
	return &disaggregatedSetWatchMapper{reader: reader}
}

func (r *disaggregatedSetWatchMapper) MapChildLWSToDGD(ctx context.Context, obj client.Object) []ctrl.Request {
	setName := obj.GetLabels()[disaggregatedsetv1.SetNameLabelKey]
	if setName == "" {
		return nil
	}
	ds := newDisaggregatedSetObject()
	if err := r.reader.Get(ctx, types.NamespacedName{Name: setName, Namespace: obj.GetNamespace()}, ds); err != nil {
		if !apierrors.IsNotFound(err) {
			log.FromContext(ctx).Error(err, "failed to map DisaggregatedSet child LeaderWorkerSet", "leaderWorkerSet", obj.GetName())
		}
		return nil
	}
	owner := metav1.GetControllerOf(ds)
	if owner == nil || owner.APIVersion != nvidiacomv1beta1.GroupVersion.String() || owner.Kind != dynamoGraphDeploymentKind {
		return nil
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Name: owner.Name, Namespace: ds.GetNamespace()}}}
}

func (r *disaggregatedSetWorkloadsReconciler) getUpdatedInProgressForDisaggregatedSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inProgress []string,
) []string {
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
	dsReady := false
	dsReason := resourceNotFoundReason
	if dsErr == nil {
		var err error
		dsReady, dsReason, _, err = r.checkDisaggregatedSetReadiness(ctx, ds, selection)
		if err != nil {
			dsReason = err.Error()
		}
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
			reason := resourceNotFoundReason
			if !apierrors.IsNotFound(dsErr) {
				reason = dsErr.Error()
			}
			logger.V(1).Info("DisaggregatedSet component not fully updated", "componentName", componentName, "reason", reason)
			updatedInProgress = append(updatedInProgress, componentName)
			continue
		}

		if !dsReady {
			logger.V(1).Info("DisaggregatedSet component not fully updated", "componentName", componentName, "reason", dsReason)
			updatedInProgress = append(updatedInProgress, componentName)
		}
	}
	return updatedInProgress
}

func (r *disaggregatedSetWorkloadsReconciler) buildRollingUpdateContext(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (dynamo.RollingUpdateContext, error) {
	return r.rollout.buildRollingUpdateContext(ctx, dgd)
}

func (r *disaggregatedSetWorkloadsReconciler) getExistingRestartAnnotationsDCD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (map[string]string, error) {
	return r.componentWorkloads.getExistingRestartAnnotationsDCD(ctx, dgd)
}

func (r *disaggregatedSetWorkloadsReconciler) preserveExistingDCDBackendFramework(
	ctx context.Context,
	desired *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	return r.componentWorkloads.preserveExistingBackendFramework(ctx, desired)
}

func (r *disaggregatedSetWorkloadsReconciler) checkResourcesReadiness(resources []Resource) ReconcileResult {
	return checkResourcesReadiness(resources)
}

func (r *disaggregatedSetWorkloadsReconciler) applyDCDCheckpointStartupPolicy(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
	return r.componentWorkloads.applyCheckpointStartupPolicy(dcd, checkpointInfo)
}

func (r *disaggregatedSetWorkloadsReconciler) checkComponentFullyUpdated(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) (bool, string) {
	return r.componentRestartProgress.checkComponentFullyUpdated(ctx, dgd, componentName)
}
