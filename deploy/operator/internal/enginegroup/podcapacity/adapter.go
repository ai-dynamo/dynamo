/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package podcapacity provides the direct-Pod capacity adapter used to prove
// the Engine Group lifecycle before a workload-manager implementation lands.
package podcapacity

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// PodBuilder materializes one profile-specific joining allocation from the
// running primary. It is the only POC component that interprets engine launch
// arguments.
type PodBuilder interface {
	Build(primary *corev1.Pod, target enginegroup.CapacityReplicaTarget) (*corev1.Pod, error)
}

// Adapter converges one-pod-per-replica growth using ordinary Kubernetes Pods.
// It intentionally rejects release: Grove (or another workload manager) will
// own production placement and exact-victim deletion.
type Adapter struct {
	Client    client.Client
	Namespace string
	GroupName string
	GroupUID  types.UID
	Builder   PodBuilder
	Journal   kubejournal.Store
}

type journalState struct {
	AppliedRevision int64                      `json:"appliedRevision"`
	TargetDigest    string                     `json:"targetDigest"`
	ReleaseFences   []enginegroup.ReleaseFence `json:"releaseFences,omitempty"`
}

// NewAdapter constructs a direct-Pod adapter for one Engine Group.
func NewAdapter(
	kubeClient client.Client,
	namespace string,
	groupName string,
	groupUID types.UID,
	builder PodBuilder,
) *Adapter {
	return &Adapter{
		Client:    kubeClient,
		Namespace: namespace,
		GroupName: groupName,
		GroupUID:  groupUID,
		Builder:   builder,
		Journal:   kubejournal.NewStore(kubeClient, namespace, groupName, groupUID, "capacity"),
	}
}

// Observe returns exact Pod incarnations and the last durably accepted target.
func (a *Adapter) Observe(ctx context.Context, _ enginegroup.GroupID) (enginegroup.CapacityObservation, error) {
	pods, err := a.listPods(ctx)
	if err != nil {
		return enginegroup.CapacityObservation{}, err
	}
	state := journalState{}
	found, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.CapacityObservation{}, err
	}
	if !found {
		state = journalState{}
	}

	allocations := make([]enginegroup.CapacityAllocation, 0, len(pods))
	for i := range pods {
		incarnation, err := podIncarnation(&pods[i])
		if err != nil {
			return enginegroup.CapacityObservation{}, err
		}
		allocations = append(allocations, enginegroup.CapacityAllocation{
			Incarnation: incarnation,
			Available:   podAvailable(&pods[i]),
			Health:      podAllocationHealth(&pods[i]),
		})
	}
	sort.Slice(allocations, func(i, j int) bool {
		return allocations[i].Incarnation.ReplicaID < allocations[j].Incarnation.ReplicaID
	})
	return enginegroup.CapacityObservation{
		AppliedRevision: state.AppliedRevision,
		Allocations:     allocations,
		ReleaseFences:   append([]enginegroup.ReleaseFence(nil), state.ReleaseFences...),
	}, nil
}

// Apply durably accepts and converges an absolute growth-only target.
func (a *Adapter) Apply(
	ctx context.Context,
	_ enginegroup.GroupID,
	target enginegroup.CapacityTarget,
) (enginegroup.ApplyResult, error) {
	if a.Builder == nil {
		return enginegroup.ApplyResult{}, fmt.Errorf("direct-Pod capacity builder is required")
	}
	if len(target.ReleaseFences) != 0 {
		return rejected("ReleaseUnsupported", "the direct-Pod proof adapter supports growth only"), nil
	}
	digest, err := targetDigest(target)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	state := journalState{}
	found, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if found {
		if failure := validateRevision(target.ControlRevision, digest, state); failure != nil {
			return enginegroup.ApplyResult{Rejection: failure}, nil
		}
	}

	pods, err := a.listPods(ctx)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	podsByReplica, err := indexPods(pods)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if failure := validateGrowthTarget(target, podsByReplica); failure != nil {
		return enginegroup.ApplyResult{Rejection: failure}, nil
	}

	// Persist acceptance before creating anything. A restart replays this exact
	// target; an equal revision with another payload is rejected.
	if !found || state.AppliedRevision != target.ControlRevision || state.TargetDigest != digest {
		state = journalState{AppliedRevision: target.ControlRevision, TargetDigest: digest}
		if err := a.Journal.Save(ctx, state); err != nil {
			return enginegroup.ApplyResult{}, err
		}
	}

	primary, err := primaryPod(pods)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	for _, replica := range target.Replicas {
		if _, present := podsByReplica[replica.ReplicaID]; present {
			continue
		}
		pod, err := a.Builder.Build(primary, replica)
		if err != nil {
			return enginegroup.ApplyResult{}, fmt.Errorf("build Pod for replica %q: %w", replica.ReplicaID, err)
		}
		applyIdentity(pod, a, replica)
		if err := a.Client.Create(ctx, pod); err != nil {
			return enginegroup.ApplyResult{}, fmt.Errorf("create Pod %s/%s: %w", pod.Namespace, pod.Name, err)
		}
	}
	return enginegroup.ApplyResult{}, nil
}

func (a *Adapter) listPods(ctx context.Context) ([]corev1.Pod, error) {
	list := &corev1.PodList{}
	if err := a.Client.List(ctx, list,
		client.InNamespace(a.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoEngineGroup: a.GroupName},
	); err != nil {
		return nil, fmt.Errorf("list Engine Group Pods: %w", err)
	}
	return list.Items, nil
}

func applyIdentity(pod *corev1.Pod, adapter *Adapter, target enginegroup.CapacityReplicaTarget) {
	pod.Namespace = adapter.Namespace
	pod.Labels[consts.KubeLabelDynamoEngineGroup] = adapter.GroupName
	pod.Labels[consts.KubeLabelDynamoEngineGroupReplica] = string(target.ReplicaID)
	pod.Labels[consts.KubeLabelDynamoEngineGroupSlot] = string(target.SlotID)
	pod.Labels[consts.KubeLabelDynamoEngineGroupRole] = consts.KubeLabelDynamoEngineGroupRoleJoiner
	pod.Labels[consts.KubeLabelDynamoScaleRepresentative] = consts.KubeLabelDynamoScaleRepresentativeYes
	controller := true
	blockDeletion := true
	pod.OwnerReferences = []metav1.OwnerReference{{
		APIVersion:         "nvidia.com/v1beta1",
		Kind:               "DynamoGraphDeploymentEngineGroup",
		Name:               adapter.GroupName,
		UID:                adapter.GroupUID,
		Controller:         &controller,
		BlockOwnerDeletion: &blockDeletion,
	}}
}

func indexPods(pods []corev1.Pod) (map[enginegroup.ReplicaID]*corev1.Pod, error) {
	indexed := make(map[enginegroup.ReplicaID]*corev1.Pod, len(pods))
	for i := range pods {
		replicaID := enginegroup.ReplicaID(pods[i].Labels[consts.KubeLabelDynamoEngineGroupReplica])
		if replicaID == "" {
			return nil, fmt.Errorf("Engine Group Pod %s has no replica identity", pods[i].Name)
		}
		if _, duplicate := indexed[replicaID]; duplicate {
			return nil, fmt.Errorf("multiple Pods claim Engine Group replica %q", replicaID)
		}
		indexed[replicaID] = &pods[i]
	}
	return indexed, nil
}

func primaryPod(pods []corev1.Pod) (*corev1.Pod, error) {
	var primary *corev1.Pod
	for i := range pods {
		if pods[i].Labels[consts.KubeLabelDynamoEngineGroupRole] != consts.KubeLabelDynamoEngineGroupRolePrimary {
			continue
		}
		if primary != nil {
			return nil, fmt.Errorf("multiple primary Pods found for Engine Group")
		}
		primary = &pods[i]
	}
	if primary == nil {
		return nil, fmt.Errorf("Engine Group primary Pod is missing")
	}
	return primary, nil
}

func podIncarnation(pod *corev1.Pod) (enginegroup.ReplicaIncarnation, error) {
	replicaID := enginegroup.ReplicaID(pod.Labels[consts.KubeLabelDynamoEngineGroupReplica])
	slotID := enginegroup.CapacitySlotID(pod.Labels[consts.KubeLabelDynamoEngineGroupSlot])
	if replicaID == "" || slotID == "" || pod.UID == "" {
		return enginegroup.ReplicaIncarnation{}, fmt.Errorf("Pod %s has incomplete Engine Group identity", pod.Name)
	}
	return enginegroup.ReplicaIncarnation{
		ReplicaID:          replicaID,
		SlotID:             slotID,
		RuntimeIncarnation: enginegroup.RuntimeIncarnationID(pod.UID),
		CapacityRefs: []enginegroup.CapacityRef{{
			Name: pod.Name,
			UID:  enginegroup.PodUID(pod.UID),
		}},
	}, nil
}

func samePodIncarnation(pod *corev1.Pod, want enginegroup.ReplicaIncarnation) bool {
	got, err := podIncarnation(pod)
	return err == nil && enginegroup.SameIncarnation(got, want)
}

// podAllocationHealth describes this growth-only profile; it does not infer packed survivor health.
func podAllocationHealth(pod *corev1.Pod) enginegroup.AllocationHealth {
	if pod.Status.Phase == corev1.PodFailed {
		return enginegroup.AllocationHealthFailed
	}
	if podAvailable(pod) {
		return enginegroup.AllocationHealthHealthy
	}
	return enginegroup.AllocationHealthUnknown
}

func podAvailable(pod *corev1.Pod) bool {
	if pod.DeletionTimestamp != nil {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func targetDigest(target enginegroup.CapacityTarget) (string, error) {
	encoded, err := json.Marshal(target)
	if err != nil {
		return "", fmt.Errorf("marshal capacity target: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

func validateRevision(revision int64, digest string, state journalState) *enginegroup.Failure {
	switch {
	case revision < state.AppliedRevision:
		return terminalFailure("StaleCapacityRevision", "capacity target revision is older than the accepted revision")
	case revision == state.AppliedRevision && digest != state.TargetDigest:
		return terminalFailure("ConflictingCapacityRevision", "capacity target revision has a different payload")
	default:
		return nil
	}
}

func validateGrowthTarget(
	target enginegroup.CapacityTarget,
	pods map[enginegroup.ReplicaID]*corev1.Pod,
) *enginegroup.Failure {
	targets := make(map[enginegroup.ReplicaID]enginegroup.CapacityReplicaTarget, len(target.Replicas))
	for _, replica := range target.Replicas {
		if replica.ReplicaID == "" || replica.SlotID == "" {
			return terminalFailure("InvalidCapacityTarget", "replica and slot identities are required")
		}
		if _, duplicate := targets[replica.ReplicaID]; duplicate {
			return terminalFailure("InvalidCapacityTarget", "capacity target contains a duplicate replica identity")
		}
		if replica.Incarnation != nil && replica.Bootstrap != nil {
			return terminalFailure("InvalidCapacityTarget", "a replica cannot contain both exact incarnation and bootstrap intent")
		}
		targets[replica.ReplicaID] = replica

		pod, present := pods[replica.ReplicaID]
		switch {
		case replica.Incarnation != nil && (!present || !samePodIncarnation(pod, *replica.Incarnation)):
			return terminalFailure("MissingExactIncarnation", "an exact accepted Pod incarnation is missing or was replaced")
		case !present && (replica.Bootstrap == nil || replica.Bootstrap.Mode != enginegroup.BootstrapModeJoin):
			return terminalFailure("UnsupportedBootstrap", "a missing allocation requires Join bootstrap intent")
		}
	}
	for replicaID := range pods {
		if _, retained := targets[replicaID]; !retained {
			return terminalFailure("ReleaseUnsupported", "the direct-Pod proof adapter cannot remove existing allocations")
		}
	}
	return nil
}

func terminalFailure(reason, message string) *enginegroup.Failure {
	return &enginegroup.Failure{
		Classification: enginegroup.FailureClassificationTerminal,
		Reason:         reason,
		Message:        message,
	}
}

func rejected(reason, message string) enginegroup.ApplyResult {
	return enginegroup.ApplyResult{Rejection: terminalFailure(reason, message)}
}
