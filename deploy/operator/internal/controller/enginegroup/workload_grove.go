/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"slices"
	"strconv"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// GroveWorkloadProvider owns native Grove resource identity and physical lifecycle behavior.
// Client must not be nil. It never interprets engine commands or assigns native members.
type GroveWorkloadProvider struct {
	Client client.Client
}

// +kubebuilder:rbac:groups="",resources=pods,verbs=patch
// +kubebuilder:rbac:groups="",resources=pods/status,verbs=patch
// +kubebuilder:rbac:groups=grove.io,resources=podcliques,verbs=get;list;watch
// +kubebuilder:rbac:groups=grove.io,resources=podcliquesets;podcliquescalinggroups,verbs=get;list;watch
// +kubebuilder:rbac:groups=grove.io,resources=podcliques/scale,verbs=get;update;patch

// Resolve proves the complete owner chain before exposing provider-neutral physical capacity.
func (p *GroveWorkloadProvider) Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (WorkloadBinding, error) {
	owner := metav1.GetControllerOf(group)
	if group.UID == "" || owner == nil || owner.Kind != api.DynamoGraphDeploymentGVK.Kind || owner.APIVersion != api.GroupVersion.String() {
		return WorkloadBinding{}, fmt.Errorf("Grove runtime requires a DGD-owned Engine Group")
	}
	dgd := &api.DynamoGraphDeployment{}
	if err := p.Client.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: owner.Name}, dgd); err != nil {
		return WorkloadBinding{}, fmt.Errorf("observe workload parent declaration: %w", err)
	}
	if dgd.UID != owner.UID || !dgd.DeletionTimestamp.IsZero() || dgd.Spec.ProviderOverride != nil {
		return WorkloadBinding{}, fmt.Errorf("%w: workload parent lifetime or launch authority changed", domain.ErrRecoveryRequired)
	}
	pcs := &grove.PodCliqueSet{}
	if err := p.Client.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.PCSNameForDGD(dgd, (*api.DynamoComponentDeploymentSharedSpec).ManagedByExternalController)}, pcs); err != nil {
		return WorkloadBinding{}, fmt.Errorf("observe PodCliqueSet: %w", err)
	}
	if !metav1.IsControlledBy(pcs, dgd) || !pcs.DeletionTimestamp.IsZero() {
		return WorkloadBinding{}, fmt.Errorf("%w: PodCliqueSet has a different owner or is terminating", domain.ErrRecoveryRequired)
	}
	literal := group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex]
	index, err := strconv.ParseInt(literal, 10, 32)
	if err != nil || index < 0 || strconv.FormatInt(index, 10) != literal {
		return WorkloadBinding{}, fmt.Errorf("Engine Group requires a canonical world index")
	}
	clique, err := resolveGroveMemberClique(ctx, p.Client, group, pcs, int32(index))
	if err != nil {
		return WorkloadBinding{}, err
	}
	name := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique]
	uid := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]
	if clique == nil || name == "" || uid == "" {
		return WorkloadBinding{}, fmt.Errorf("%w: waiting for a Grove member-clique binding", ErrRuntimeUnavailable)
	}
	if clique.Name != name || string(clique.UID) != uid {
		return WorkloadBinding{}, fmt.Errorf("%w: member clique incarnation changed", domain.ErrRecoveryRequired)
	}
	return WorkloadBinding{
		ID: clique.UID,
		Capacity: &grovecapacity.Adapter{
			Client: p.Client, Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID, GroupName: group.Name,
			Journal: runtimeJournal(p.Client, group, clique.UID, "grove-capacity"),
			ReadinessScaleIn: slices.ContainsFunc(clique.Spec.PodSpec.ReadinessGates, func(gate corev1.PodReadinessGate) bool {
				return gate.ConditionType == corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)
			}),
		},
		Fence: groveWorldFenceStore(p.Client, group, clique.UID),
	}, nil
}
