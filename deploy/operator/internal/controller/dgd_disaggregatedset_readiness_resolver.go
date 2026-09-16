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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

type disaggregatedSetReadiness struct {
	Ready             bool
	Reason            string
	ComponentStatuses map[string]nvidiacomv1beta1.ComponentReplicaStatus
}

type disaggregatedSetReadinessResolver struct {
	reader client.Reader
}

func newDisaggregatedSetReadinessResolver(reader client.Reader) *disaggregatedSetReadinessResolver {
	return &disaggregatedSetReadinessResolver{reader: reader}
}

func (r *disaggregatedSetReadinessResolver) Resolve(
	ctx context.Context,
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
) (disaggregatedSetReadiness, error) {
	children := &leaderworkersetv1.LeaderWorkerSetList{}
	if err := r.reader.List(ctx, children, client.InNamespace(ds.GetNamespace())); err != nil {
		return disaggregatedSetReadiness{}, fmt.Errorf("failed to list DisaggregatedSet child LeaderWorkerSets: %w", err)
	}

	targetRevision, err := disaggregatedSetTargetRevision(ds)
	if err != nil {
		return disaggregatedSetReadiness{}, err
	}

	targetByIdentity := make(map[disaggregatedSetChildIdentity][]*leaderworkersetv1.LeaderWorkerSet)
	childrenByRole := make(map[string][]*leaderworkersetv1.LeaderWorkerSet)
	sliceCount := int(disaggregatedSetSliceCount(ds))
	expectedIdentityByName := make(map[string]disaggregatedSetChildIdentity, len(selection.componentToRole)*sliceCount)
	for _, roleName := range selection.componentToRole {
		for slice := range sliceCount {
			identity := disaggregatedSetChildIdentity{slice: slice, role: roleName}
			name := disaggregatedsetutils.GenerateName(ds.GetName(), slice, targetRevision, roleName)
			expectedIdentityByName[name] = identity
		}
	}

	for i := range children.Items {
		child := &children.Items[i]
		if !metav1.IsControlledBy(child, ds) {
			continue
		}
		identity, target := expectedIdentityByName[child.Name]
		if !target {
			roleName := child.Labels[disaggregatedsetv1.RoleLabelKey]
			childrenByRole[roleName] = append(childrenByRole[roleName], child)
			continue
		}
		childrenByRole[identity.role] = append(childrenByRole[identity.role], child)
		targetByIdentity[identity] = append(targetByIdentity[identity], child)
	}

	if len(disaggregatedSetRoleStatuses(ds)) > 0 && disaggregatedSetStatusHasObservation(ds) {
		ready, reason, statuses := checkDisaggregatedSetReadiness(ds, selection)
		if !ready {
			return disaggregatedSetReadiness{
				Ready:             ready,
				Reason:            reason,
				ComponentStatuses: statuses,
			}, nil
		}
	}

	ready, reason, statuses := checkDisaggregatedSetChildLWSReadiness(selection, sliceCount, targetByIdentity, childrenByRole)
	return disaggregatedSetReadiness{
		Ready:             ready,
		Reason:            reason,
		ComponentStatuses: statuses,
	}, nil
}

func (r *disaggregatedSetReadinessResolver) updatedInProgressForDisaggregatedSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inProgress []string,
	componentRestartProgress *componentRestartProgressResolver,
) []string {
	logger := log.FromContext(ctx)
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		logger.V(1).Info("failed to select DisaggregatedSet components for restart progress", "reason", reason)
		return inProgress
	}

	ds := newDisaggregatedSetObject()
	dsErr := r.reader.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, ds)
	if dsErr != nil && !apierrors.IsNotFound(dsErr) {
		logger.V(1).Info("failed to get DisaggregatedSet for restart progress", "error", dsErr)
	}

	dsReady := false
	dsReason := resourceNotFoundReason
	if dsErr == nil {
		readiness, err := r.Resolve(ctx, ds, selection)
		if err != nil {
			dsReason = err.Error()
		} else {
			dsReady = readiness.Ready
			dsReason = readiness.Reason
		}
	}

	updatedInProgress := make([]string, 0, len(inProgress))
	for _, componentName := range inProgress {
		if _, selected := selection.componentToRole[componentName]; !selected {
			isFullyUpdated, reason := componentRestartProgress.checkComponentFullyUpdated(ctx, dgd, componentName)
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

func (r *disaggregatedSetWorkloadsReconciler) checkDisaggregatedSetReadiness(
	ctx context.Context,
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
) (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus, error) {
	readiness, err := r.readiness.Resolve(ctx, ds, selection)
	if err != nil {
		return false, "", nil, err
	}
	return readiness.Ready, readiness.Reason, readiness.ComponentStatuses, nil
}

func checkDisaggregatedSetReadiness(
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
) (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
	sliceCount := disaggregatedSetSliceCount(ds)
	statuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus, len(selection.componentToRole))
	notReadyReasons := []string{}
	for componentName, roleName := range selection.componentToRole {
		desiredReplicas := selection.desiredReplicas[componentName] * sliceCount
		componentStatus := nvidiacomv1beta1.ComponentReplicaStatus{
			ComponentKind: nvidiacomv1beta1.ComponentKindLeaderWorkerSet,
		}
		roleStatus, found := disaggregatedSetRoleStatuses(ds)[roleName]
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
