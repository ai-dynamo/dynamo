/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/workload"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// WorkloadBinding identifies one physical lifetime independently of its provider's native resources.
// ID is opaque outside the provider. Fence retains stop and restart evidence for this exact lifetime.
type WorkloadBinding struct {
	ID       types.UID
	Capacity workload.Capacity
	Fence    kubejournal.Store
}

// WorkloadProvider owns physical identity, lifecycle fences and binding persistence, not engine membership.
// Resolve is read-only. Reconcile returns either pending work or a complete candidate copy of group with
// only provider-owned binding metadata changed. Accept revalidates and persists that candidate after
// the controller has validated its engine profile. AuthorizeRestart proves the exact predecessor stopped.
type WorkloadProvider interface {
	Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (WorkloadBinding, error)
	Reconcile(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (*api.DynamoGraphDeploymentEngineGroup, bool, error)
	Accept(ctx context.Context, group, candidate *api.DynamoGraphDeploymentEngineGroup) error
	AuthorizeRestart(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, previous types.UID) error
}

// worldFence retains physical stop proof and restart verification without provider-native identities.
type worldFence struct {
	WorldID             types.UID
	PreviousID          types.UID
	Pods                []worldPodFence
	VerificationFailure *domain.Failure
}

type worldPodFence struct {
	Ref     domain.CapacityRef
	SlotID  domain.CapacitySlotID
	Stopped bool
}

func (r *Reconciler) reconcileWorkload(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (bool, error) {
	// Injected runtimes may have no separately managed physical workload.
	if r.WorkloadProvider == nil {
		return false, nil
	}
	candidate, pending, err := r.WorkloadProvider.Reconcile(ctx, group)
	if err != nil || candidate == nil {
		return pending, err
	}

	// Validate the new engine binding before accepting a new physical lifetime.
	runtime, err := r.RuntimeProvider.Resolve(ctx, candidate)
	if err != nil {
		return true, err
	}
	if err := validateEngineGroupRuntime(runtime); err != nil {
		return true, err
	}
	if group.Status.Profile != nil && runtime.Profile.Fingerprint != group.Status.Profile.Fingerprint {
		return true, errors.New("replacement world has a different immutable profile")
	}
	if err := r.WorkloadProvider.Accept(ctx, group, candidate); err != nil {
		return true, err
	}
	return true, nil
}

// runtimeJournal scopes adapter evidence to the physical world, never merely to a logical name.
func runtimeJournal(kubeClient client.Client, group *api.DynamoGraphDeploymentEngineGroup, worldID types.UID, suffix string) kubejournal.Store {
	return kubejournal.NewStore(kubeClient, group.Namespace, group.Name, group.UID, suffix+"-"+string(worldID))
}
