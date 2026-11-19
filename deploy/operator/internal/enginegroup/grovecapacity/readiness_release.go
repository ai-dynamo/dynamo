/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package grovecapacity

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Prepare initializes only missing gates; it never reopens a gate closed for retirement.
func (a *Adapter) Prepare(ctx context.Context) error {
	// Only the explicitly selected readiness-steering realization owns this condition.
	if !a.ReadinessScaleIn {
		return nil
	}
	pods, err := a.pods(ctx)
	if err != nil {
		return err
	}
	state := journalState{}
	if _, err := a.Journal.Load(ctx, &state); err != nil {
		return err
	}
	for i := range pods {
		pod := &pods[i]
		if !hasRetentionGate(pod) {
			return fmt.Errorf("Pod %s lacks the Engine Group retention readiness gate", pod.Name)
		}
		if !pod.DeletionTimestamp.IsZero() || slices.ContainsFunc(pod.Status.Conditions, func(condition corev1.PodCondition) bool {
			return condition.Type == corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)
		}) {
			continue
		}
		// Lost status cannot reopen a victim whose retirement intent is already durable.
		retained := state.Release == nil || !slices.ContainsFunc(state.Release.Fence.CapacityRefs, func(ref enginegroup.CapacityRef) bool {
			return ref.Name == pod.Name && ref.UID == enginegroup.PodUID(pod.UID)
		})
		if err := a.setRetention(ctx, pod, retained); err != nil {
			return err
		}
	}
	return nil
}

func (a *Adapter) applyReadinessRelease(ctx context.Context, target enginegroup.CapacityTarget, digest string, state journalState, snapshot kubejournal.Snapshot) (enginegroup.ApplyResult, error) {
	// Freeze the complete expected outcome before observing mutable workload state.
	release, failure := readinessReleaseIntent(target)
	if failure != nil {
		return enginegroup.ApplyResult{Rejection: failure}, nil
	}
	fence := release.Fence
	ref := fence.CapacityRefs[0]
	clique, err := a.clique(ctx)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	wanted := release.Replicas
	if clique.Spec.MinAvailable == nil || wanted < *clique.Spec.MinAvailable {
		return rejected("RuntimeMinimumBreached", "planned shrink must not fall below the member clique's minAvailable"), nil
	}
	pods, err := a.pods(ctx)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	state.ReleaseFences, err = observeReleaseFences(state, pods, clique.Spec.Replicas)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if state.Release != nil && state.AppliedRevision != target.ControlRevision && !hasReleaseFence(state.ReleaseFences, state.Release.Fence) {
		return rejected("ReleaseInProgress", "an accepted release must finish before a different release target"), nil
	}

	// Validate all retained slots and UIDs using the same capacity mapping as growth.
	retainedPods := make([]corev1.Pod, 0, len(pods))
	var victim *corev1.Pod
	for i := range pods {
		if pods[i].Name == ref.Name && enginegroup.PodUID(pods[i].UID) == ref.UID {
			victim = &pods[i]
		} else {
			retainedPods = append(retainedPods, pods[i])
		}
	}
	if failure := a.validateTarget(target, retainedPods); failure != nil {
		return enginegroup.ApplyResult{}, fmt.Errorf("%w: invalid retained capacity during release: %s", enginegroup.ErrRecoveryRequired, failure.Message)
	}
	if state.AppliedRevision == target.ControlRevision && (state.Release == nil || !reflect.DeepEqual(*state.Release, release)) {
		return enginegroup.ApplyResult{}, fmt.Errorf("accepted readiness release lost its exact expected outcome")
	}
	if victim == nil && (state.AppliedRevision != target.ControlRevision || !hasReleaseFence(state.ReleaseFences, fence)) {
		return enginegroup.ApplyResult{}, fmt.Errorf("%w: the authorized victim disappeared before release was acknowledged", enginegroup.ErrRecoveryRequired)
	}
	if victim != nil && !hasRetentionGate(victim) {
		return rejected("MissingRetentionGate", "the selected victim has no Engine Group retention readiness gate"), nil
	}

	// Persist acceptance before any readiness or Scale mutation; retries keep the identical expected UID set.
	if state.AppliedRevision != target.ControlRevision {
		state.AppliedRevision, state.TargetDigest, state.Release = target.ControlRevision, digest, &release
		if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
			return enginegroup.ApplyResult{}, err
		}
	}
	if victim == nil {
		return enginegroup.ApplyResult{}, nil
	}
	if !allRetainedPodsReady(retainedPods) {
		return enginegroup.ApplyResult{}, nil // Retry observation; no count write while another candidate can win.
	}
	if !retentionClosed(victim) {
		return enginegroup.ApplyResult{}, a.setRetention(ctx, victim, false)
	}
	if podAvailable(victim) {
		return enginegroup.ApplyResult{}, nil // The kubelet must first reflect the closed gate in PodReady.
	}

	// Readiness is steering, not authorization: a concurrent failure can still change Grove's choice.
	return enginegroup.ApplyResult{}, a.scaleReadinessRelease(ctx, clique, wanted)
}

func allRetainedPodsReady(pods []corev1.Pod) bool {
	// Only the selected victim may be unavailable when Grove chooses a scale-in victim.
	for i := range pods {
		if pods[i].Status.Phase != corev1.PodRunning || pods[i].Spec.NodeName == "" || !podAvailable(&pods[i]) {
			return false
		}
	}
	return true
}

func readinessReleaseIntent(target enginegroup.CapacityTarget) (releaseIntent, *enginegroup.Failure) {
	// This workaround requires one retired suffix allocation, not a batch of ambiguous victims.
	if len(target.ReleaseFences) != 1 || len(target.ReleaseFences[0].CapacityRefs) != 1 {
		return releaseIntent{}, failure("UnsupportedReleaseSet", "readiness-based scale-in releases exactly one one-Pod allocation")
	}
	fence := target.ReleaseFences[0]
	ref := fence.CapacityRefs[0]
	if ref.Name == "" || ref.UID == "" || fence.TransitionID != target.TransitionID || fence.AuthorizingTopologyGeneration <= 0 ||
		fence.SlotID != enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", len(target.Replicas))) ||
		fence.ReplicaID != enginegroup.ReplicaID(fmt.Sprintf("replica-%d", len(target.Replicas))) {
		return releaseIntent{}, failure("InvalidReleaseFence", "release requires the exact retired suffix UID, operation, and topology generation")
	}
	retained := make([]enginegroup.CapacityRef, 0, len(target.Replicas))
	for _, replica := range target.Replicas {
		if replica.Incarnation == nil || len(replica.Incarnation.CapacityRefs) != 1 {
			return releaseIntent{}, failure("UnpinnedReleaseTarget", "every retained allocation must have its exact current incarnation")
		}
		retained = append(retained, replica.Incarnation.CapacityRefs[0])
	}
	return releaseIntent{Fence: fence, Retained: retained, Replicas: int32(len(target.Replicas))}, nil
}

func (a *Adapter) scaleReadinessRelease(ctx context.Context, clique client.Object, wanted int32) error {
	// Fence both clique incarnation and the sole authorized cardinality change at the Scale write.
	scale := &autoscalingv1.Scale{}
	if err := a.Client.SubResource("scale").Get(ctx, clique, scale); err != nil {
		return err
	}
	if scale.UID != a.CliqueUID || (scale.Spec.Replicas != wanted && scale.Spec.Replicas != wanted+1) {
		return fmt.Errorf("member-clique scale changed outside the accepted single-victim release")
	}
	if scale.Spec.Replicas == wanted {
		return nil
	}
	scale.Spec.Replicas = wanted
	return a.Client.SubResource("scale").Update(ctx, clique, client.WithSubResourceBody(scale))
}

func (a *Adapter) setRetention(ctx context.Context, pod *corev1.Pod, retained bool) error {
	// Strategic merge preserves unowned conditions omitted by the compact cache; RV also fences Pod recreation.
	condition := corev1.PodCondition{
		Type:   corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate),
		Status: corev1.ConditionTrue, Reason: "Retained", LastTransitionTime: metav1.Now(),
	}
	if !retained {
		condition.Status, condition.Reason = corev1.ConditionFalse, "NativeRetirementCommitted"
	}
	patch, err := json.Marshal(map[string]any{
		"metadata": map[string]any{"uid": pod.UID, "resourceVersion": pod.ResourceVersion},
		"status":   map[string]any{"conditions": []corev1.PodCondition{condition}},
	})
	if err != nil {
		return err
	}
	return a.Client.Status().Patch(ctx, pod.DeepCopy(), client.RawPatch(types.StrategicMergePatchType, patch))
}

func hasRetentionGate(pod *corev1.Pod) bool {
	return slices.ContainsFunc(pod.Spec.ReadinessGates, func(gate corev1.PodReadinessGate) bool {
		return gate.ConditionType == corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)
	})
}

func retentionClosed(pod *corev1.Pod) bool {
	return slices.ContainsFunc(pod.Status.Conditions, func(condition corev1.PodCondition) bool {
		return condition.Type == corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate) && condition.Status == corev1.ConditionFalse
	})
}
