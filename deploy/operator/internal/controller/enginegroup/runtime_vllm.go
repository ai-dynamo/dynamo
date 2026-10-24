/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ResolveVLLMGroveCapacity binds the native declaration to the exact DGD-owned physical world.
// kube and group must not be nil. Resolution is read-only and never waits for an engine API.
// This is not a complete RuntimeProvider: serving verification and worker admission remain required.
func ResolveVLLMGroveCapacity(ctx context.Context, kube client.Client, group *api.DynamoGraphDeploymentEngineGroup) (*VLLMGroveCapacity, error) {
	// A logical name or mutable label alone cannot establish native execution authority.
	owner := metav1.GetControllerOf(group)
	if group.UID == "" || owner == nil || owner.APIVersion != api.GroupVersion.String() || owner.Kind != api.DynamoGraphDeploymentGVK.Kind ||
		group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != string(dynamo.BackendFrameworkVLLM) ||
		group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex] != "0" {
		return nil, fmt.Errorf("vLLM runtime requires one explicitly declared DGD-owned world")
	}
	dgd := &api.DynamoGraphDeployment{}
	if err := kube.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: owner.Name}, dgd); err != nil {
		return nil, fmt.Errorf("observe vLLM parent declaration: %w", err)
	}
	if dgd.UID != owner.UID || !dgd.DeletionTimestamp.IsZero() || dgd.Spec.ProviderOverride != nil ||
		(dgd.Spec.BackendFramework != "" && dgd.Spec.BackendFramework != string(dynamo.BackendFrameworkVLLM)) {
		return nil, fmt.Errorf("%w: vLLM parent lifetime or launch authority changed", domain.ErrRecoveryRequired)
	}
	component, err := vllmParentComponent(dgd, group)
	if err != nil {
		return nil, err
	}
	source, profile, err := dynamo.ResolveVLLMComponentEngineGroup(component, dgd.Spec.Env)
	if err != nil {
		return nil, err
	}
	if profile.Geometry.Fingerprint != group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] {
		return nil, fmt.Errorf("%w: vLLM declaration no longer matches the frozen profile", domain.ErrRecoveryRequired)
	}

	// Verify every owner link rather than trusting the stored clique name after recreation.
	pcs := &grove.PodCliqueSet{}
	if err := kube.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components)}, pcs); err != nil {
		return nil, fmt.Errorf("observe vLLM PodCliqueSet: %w", err)
	}
	if !metav1.IsControlledBy(pcs, dgd) || !pcs.DeletionTimestamp.IsZero() {
		return nil, fmt.Errorf("%w: vLLM PodCliqueSet has a different owner or is terminating", domain.ErrRecoveryRequired)
	}
	clique, err := resolveGroveMemberClique(ctx, kube, group, pcs, 0)
	if err != nil {
		return nil, err
	}
	if clique == nil || group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] == "" {
		return nil, fmt.Errorf("%w: waiting for the vLLM member clique binding", ErrRuntimeUnavailable)
	}
	if clique.Name != group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] ||
		string(clique.UID) != group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] {
		return nil, fmt.Errorf("%w: vLLM member clique incarnation changed", domain.ErrRecoveryRequired)
	}
	capacity := &VLLMGroveCapacity{
		Grove: &grovecapacity.Adapter{
			Client: kube, Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID, GroupName: group.Name,
			Journal:          kubejournal.NewStore(kube, group.Namespace, group.Name, group.UID, "grove-capacity-"+string(clique.UID)),
			ReadinessScaleIn: true,
		},
		Group: group, Source: source,
	}
	bindings, err := capacity.capacityBindings()
	if err != nil {
		return nil, err
	}

	// The exact rank-zero Pod supplies rendezvous before any participant exposes its API.
	observation, err := capacity.Grove.Observe(ctx, domain.GroupID(group.UID))
	if err != nil {
		return nil, err
	}
	for _, allocation := range observation.Allocations {
		if allocation.SlotID != "slot-0" {
			continue
		}
		if accepted, bound := bindings.Slots[allocation.SlotID]; bound && accepted != allocation.Pod.UID {
			return nil, fmt.Errorf("%w: rank-zero slot belongs to a previous Pod UID", domain.ErrRecoveryRequired)
		}
		pod := &corev1.Pod{}
		err := kube.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: allocation.Pod.Name}, pod)
		if err != nil {
			return nil, err
		}
		if domain.PodUID(pod.UID) != allocation.Pod.UID {
			return nil, fmt.Errorf("%w: rank-zero Pod changed during rendezvous resolution", domain.ErrRecoveryRequired)
		}
		// Rank zero cannot be retired independently of this world.
		if !pod.DeletionTimestamp.IsZero() {
			continue
		}
		if pod.Status.PodIP == "" || net.ParseIP(pod.Status.PodIP) == nil {
			return nil, fmt.Errorf("%w: waiting for rank-zero Pod IP", ErrRuntimeUnavailable)
		}
		return capacity, nil
	}
	return nil, fmt.Errorf("%w: waiting for the vLLM rank-zero Pod", ErrRuntimeUnavailable)
}

func vllmParentComponent(dgd *api.DynamoGraphDeployment, group *api.DynamoGraphDeploymentEngineGroup) (*api.DynamoComponentDeploymentSharedSpec, error) {
	// The native profile must still belong to the same canonical component/world declaration.
	name := group.Labels[consts.KubeLabelDynamoComponent]
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ComponentName == name && component.EngineGroup != nil &&
			group.Name == dynamo.EngineGroupNameForComponent(dgd.Name, name, 0) {
			return component, nil
		}
	}
	return nil, fmt.Errorf("vLLM Engine Group has no matching parent component")
}

// ResolveVLLMMembership composes native membership control with an exact physical world.
// capacity and httpClient must not be nil. Resolution is read-only; native API lifetimes and epochs
// remain authoritative in the adapter's later observations, not in Pod readiness.
func ResolveVLLMMembership(ctx context.Context, capacity *VLLMGroveCapacity, httpClient *http.Client) (*vllmruntime.MembershipAdapter, error) {
	// Bind the same immutable profile and physical lifetime used by capacity acceptance.
	profile, err := dynamo.ResolveVLLMLaunchProfile(capacity.Source)
	if err != nil {
		return nil, err
	}
	processes, err := resolveVLLMProcesses(ctx, capacity, httpClient)
	if err != nil {
		return nil, err
	}
	layout := vllmruntime.ReplicaLayout{NativeMembersPerReplica: profile.NativeMembersPerReplica}
	minimum, err := layout.NativeSize(profile.InitialReplicas)
	if err != nil {
		return nil, err
	}
	maximum, err := layout.NativeSize(profile.MaximumReplicas)
	if err != nil {
		return nil, err
	}
	return &vllmruntime.MembershipAdapter{
		Processes: processes, ProfileFingerprint: profile.Geometry.Fingerprint,
		Layout: layout, MinDataParallelSize: minimum, MaxDataParallelSize: maximum,
		Journal: kubejournal.NewStore(capacity.Grove.Client, capacity.Group.Namespace, capacity.Group.Name, capacity.Group.UID, "vllm-membership-"+string(capacity.Grove.CliqueUID)),
	}, nil
}

func resolveVLLMProcesses(ctx context.Context, capacity *VLLMGroveCapacity, httpClient *http.Client) ([]vllmruntime.Process, error) {
	// Allocation readiness and API reachability are independent; neither is inferred from the other.
	observation, err := capacity.Observe(ctx, domain.GroupID(capacity.Group.UID))
	if err != nil {
		return nil, err
	}
	port, _, err := dynamo.ResolveVLLMControlPorts(capacity.Source)
	if err != nil {
		return nil, err
	}
	processes := make([]vllmruntime.Process, 0, len(observation.Allocations))
	for _, allocation := range observation.Allocations {
		pod, err := capacity.resolvePod(ctx, allocation.Incarnation)
		if err != nil {
			return nil, err
		}
		// A retiring endpoint may disappear; membership still requires every current rank, and traffic retains proven drains.
		if !pod.DeletionTimestamp.IsZero() {
			continue
		}
		// The launch contract supplies process identity; a Grove slot is not a native rank.
		if len(allocation.Incarnation.Members) != 1 {
			return nil, fmt.Errorf("packed vLLM launch requires a distinct control endpoint for every native process")
		}
		member := allocation.Incarnation.Members[0]
		rank, err := strconv.ParseInt(strings.TrimPrefix(string(member.ID), "dp-"), 10, 32)
		if err != nil {
			return nil, err
		}
		baseURL := "http://" + net.JoinHostPort(pod.Status.PodIP, strconv.Itoa(int(port)))
		constructor := vllmruntime.NewClient
		if dynamo.IsVLLMWorkerLaunch(capacity.Source) {
			constructor = vllmruntime.NewWorkerClient
		}
		native, err := constructor(baseURL, httpClient)
		if err != nil {
			return nil, err
		}
		processes = append(processes, vllmruntime.Process{
			Rank: int32(rank), PodUID: domain.PodUID(pod.UID), Membership: domain.ReplicaMembership{
				ReplicaID: allocation.Incarnation.ReplicaID, Members: []domain.NativeMemberIncarnation{member},
			}, Client: native,
		})
	}
	sort.Slice(processes, func(i, j int) bool { return processes[i].Rank < processes[j].Rank })
	return processes, nil
}

func (a *VLLMGroveCapacity) resolvePod(ctx context.Context, incarnation domain.ReplicaIncarnation) (*corev1.Pod, error) {
	// The engine capacity observation has proved one Pod per allocation and rejected process restarts.
	ref := incarnation.CapacityRefs[0]
	pod := &corev1.Pod{}
	if err := a.Grove.Client.Get(ctx, types.NamespacedName{Namespace: a.Group.Namespace, Name: ref.Name}, pod); err != nil {
		return nil, fmt.Errorf("observe vLLM native process Pod: %w", err)
	}
	if domain.PodUID(pod.UID) != ref.UID {
		return nil, fmt.Errorf("%w: vLLM Pod changed during process resolution", domain.ErrRecoveryRequired)
	}
	bindings, err := a.capacityBindings()
	if err != nil {
		return nil, err
	}
	if accepted, bound := bindings.Slots[incarnation.SlotID]; bound && accepted != ref.UID {
		return nil, fmt.Errorf("%w: vLLM slot %s belongs to a previous Pod UID", domain.ErrRecoveryRequired, incarnation.SlotID)
	}
	if !pod.DeletionTimestamp.IsZero() {
		return pod, nil
	}
	if pod.Status.PodIP == "" {
		return nil, fmt.Errorf("%w: vLLM Pod %s is awaiting its Pod IP", ErrRuntimeUnavailable, pod.Name)
	}
	if net.ParseIP(pod.Status.PodIP) == nil || pod.Spec.RestartPolicy != corev1.RestartPolicyNever {
		return nil, fmt.Errorf("vLLM process requires a valid Pod IP and restartPolicy Never")
	}
	return pod, nil
}
