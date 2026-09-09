// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"strconv"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// dgdLPXHandoff owns only the generated child and its status projection. Build,
// Grove and scheduler lifecycles are exclusively the LPX controller's concern.
type dgdLPXHandoff struct{ client.Client }

func (r *dgdLPXHandoff) Reconcile(ctx context.Context, source *v1beta1.DynamoGraphDeployment) (*v1alpha1.LPXGraphDeployment, error) {
	child := &v1alpha1.LPXGraphDeployment{}
	err := r.Get(ctx, client.ObjectKeyFromObject(source), child)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	exists := err == nil
	if exists && !exactLPXSourceOwner(child, source) {
		return nil, fmt.Errorf("LPXGraphDeployment %q belongs to a different source; adoption is not supported", child.Name)
	}
	if !source.HasLPXComponent() {
		if !exists {
			return nil, nil
		}
		return child, r.deleteChild(ctx, child)
	}
	if exists && !child.DeletionTimestamp.IsZero() {
		return child, nil
	}

	restart := dynamo.LPXRestartToken(source, child.Annotations[dynamo.LPXRestartAnnotation])
	revision, err := dynamo.LPXInputRevision(source, restart)
	if err != nil {
		return nil, err
	}
	if exists && child.Spec.InputRevision == revision &&
		child.Annotations[dynamo.LPXRestartAnnotation] == restart &&
		child.Annotations[dynamo.LPXPCSNameAnnotation] == dynamo.PCSNameForLPX(source) {
		return child, nil
	}
	if !exists {
		child = &v1alpha1.LPXGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: source.Name, Namespace: source.Namespace},
		}
		if err := ctrl.SetControllerReference(source, child, r.Scheme()); err != nil {
			return nil, err
		}
	}
	child.Spec.InputRevision = revision
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, dynamo.LPXPCSNameAnnotation, dynamo.PCSNameForLPX(source))
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, dynamo.LPXRestartAnnotation, restart)
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, lpx.DGDGenerationAnnotation, strconv.FormatInt(source.Generation, 10))
	if exists {
		err = r.Update(ctx, child)
	} else {
		err = r.Create(ctx, child)
	}
	return child, err
}

func exactLPXSourceOwner(child *v1alpha1.LPXGraphDeployment, source *v1beta1.DynamoGraphDeployment) bool {
	owner := metav1.GetControllerOf(child)
	return child.Name == source.Name && child.Namespace == source.Namespace &&
		owner != nil && owner.Kind == "DynamoGraphDeployment" && owner.APIVersion == v1beta1.GroupVersion.String() &&
		owner.Name == source.Name && owner.UID == source.UID
}

func (r *dgdLPXHandoff) deleteChild(ctx context.Context, child *v1alpha1.LPXGraphDeployment) error {
	if !child.DeletionTimestamp.IsZero() {
		return nil
	}
	uid, version := child.UID, child.ResourceVersion
	return client.IgnoreNotFound(r.Delete(ctx, child, &client.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid, ResourceVersion: &version}}))
}

func (r *dgdLPXHandoff) Finalize(ctx context.Context, source *v1beta1.DynamoGraphDeployment) error {
	child := &v1alpha1.LPXGraphDeployment{}
	if err := r.Get(ctx, client.ObjectKeyFromObject(source), child); err != nil {
		return client.IgnoreNotFound(err)
	}
	if !exactLPXSourceOwner(child, source) {
		return fmt.Errorf("refusing to delete a foreign LPXGraphDeployment %q", child.Name)
	}
	if err := r.deleteChild(ctx, child); err != nil {
		return err
	}
	return fmt.Errorf("waiting for LPXGraphDeployment %q to finish cleanup", child.Name)
}

// resolveRestartProgress composes child-owned LPX and ordinary Grove observations.
func (p *groveProgram) resolveRestartProgress(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, inProgress []string) []string {
	ordinary := make([]string, 0, len(inProgress))
	pending := make(map[string]bool, len(inProgress))
	var childComponents map[string]v1beta1.ComponentReplicaStatus
	childObserved := false

	// Partition live members and observe the shared LPX child only when requested.
	for _, name := range inProgress {
		component := dgd.GetComponentByName(name)
		if component == nil {
			continue
		}
		if !component.IsLPX() {
			ordinary = append(ordinary, name)
			continue
		}
		if !childObserved {
			childComponents = observeLPXRestart(ctx, p.lpx.Client, dgd)
			childObserved = true
		}
		if !childComponents[name].Ready {
			pending[name] = true
		}
	}

	// Native pending or failed observations cannot prevent LPX progress.
	if len(ordinary) > 0 {
		for _, name := range p.restartProgress.Resolve(ctx, dgd, ordinary) {
			pending[name] = true
		}
	}

	// Retain the original authored restart order across the two observations.
	remaining := make([]string, 0, len(inProgress))
	for _, name := range inProgress {
		if pending[name] {
			remaining = append(remaining, name)
		}
	}
	return remaining
}

// observeLPXRestart returns component status from one current, ready restart observation.
// A failed read or incomplete child leaves every requested member pending.
func observeLPXRestart(ctx context.Context, reader client.Reader, source *v1beta1.DynamoGraphDeployment) map[string]v1beta1.ComponentReplicaStatus {
	child := &v1alpha1.LPXGraphDeployment{}
	if err := reader.Get(ctx, client.ObjectKeyFromObject(source), child); err != nil ||
		child.Status.ObservedGeneration != child.Generation || !child.DeletionTimestamp.IsZero() ||
		dynamo.ValidateLPXSource(child, source) != nil || source.Spec.Restart == nil ||
		child.Annotations[dynamo.LPXRestartAnnotation] != source.Spec.Restart.ID {
		return nil
	}
	ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
	if ready == nil || ready.Status != metav1.ConditionTrue || ready.ObservedGeneration != child.Generation {
		return nil
	}
	return child.Status.Components
}

// projectLPXChildStatus projects the child returned by a successful handoff of source.
// Its identity is already reconciled; results and conditions still require current generations.
// child may be nil or deleting; all other pointer inputs must be non-nil.
func projectLPXChildStatus(source *v1beta1.DynamoGraphDeployment, child *v1alpha1.LPXGraphDeployment, result *ReconcileResult, status *v1beta1.DynamoGraphDeploymentStatus) {
	status.LPX = nil
	components := lpx.Components(source)
	if len(components) == 0 && child == nil {
		return
	}
	if result.ComponentStatus == nil {
		result.ComponentStatus = make(map[string]v1beta1.ComponentReplicaStatus)
	}
	serving := lpx.ServingComponent(source)
	for _, component := range components {
		kind := v1beta1.ComponentKindPodCliqueScalingGroup
		if component != serving {
			kind = v1beta1.ComponentKindPodClique
		}
		result.ComponentStatus[component.ComponentName] = v1beta1.ComponentReplicaStatus{
			ComponentKind: kind,
		}
	}
	// Result payloads belong only to a completely observed revision.
	current := len(components) > 0 && child != nil && child.DeletionTimestamp.IsZero()
	observed := current && child.Status.ObservedGeneration == child.Generation
	if observed {
		status.LPX = &v1beta1.DynamoGraphDeploymentLPXStatus{
			ModelDownload: child.Status.ModelDownload.DeepCopy(),
			Placement:     child.Status.Placement.DeepCopy(),
		}
		// Public attempt observation is relative to the DGD, not its private child.
		if status.LPX.Placement != nil && status.LPX.Placement.LPXAttempt != nil {
			status.LPX.Placement.LPXAttempt.ObservedGeneration = source.Generation
		}
		// Project only current authored members from the single observed child.
		for _, component := range components {
			if observedStatus, found := child.Status.Components[component.ComponentName]; found {
				result.ComponentStatus[component.ComponentName] = *observedStatus.DeepCopy()
			}
		}
	}

	// Reconciliation errors advance their condition, not the complete result.
	// Surface that current failure without exposing stale payloads or hiding an ordinary failure.
	if result.State != v1beta1.DGDStateFailed && current {
		failed := meta.FindStatusCondition(child.Status.Conditions, "Failed")
		if failed != nil && failed.Status == metav1.ConditionTrue && failed.ObservedGeneration == child.Generation {
			result.State = v1beta1.DGDStateFailed
			result.Reason, result.Message = Reason(failed.Reason), Message(failed.Message)
			return
		}
	}
	if observed {
		// The child publishes aggregate readiness and member status atomically.
		ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
		if ready != nil && ready.ObservedGeneration == child.Generation && ready.Status == metav1.ConditionTrue {
			return
		}
		if result.State != v1beta1.DGDStateFailed && ready != nil && ready.ObservedGeneration == child.Generation {
			result.State = v1beta1.DGDStatePending
			result.Reason, result.Message = Reason(ready.Reason), Message(ready.Message)
			return
		}
	}
	if result.State != v1beta1.DGDStateFailed {
		result.State = v1beta1.DGDStatePending
		result.Reason = "LPXChildPending"
		result.Message = "Waiting for the current LPX engine revision and readiness"
	}
}
