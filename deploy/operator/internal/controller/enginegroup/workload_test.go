//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// lifecycleTestWorkload exercises controller sequencing without any provider-native API types.
type lifecycleTestWorkload struct {
	candidate  *api.DynamoGraphDeploymentEngineGroup
	pending    bool
	reconciles int
	accepts    int
}

func (p *lifecycleTestWorkload) Resolve(context.Context, *api.DynamoGraphDeploymentEngineGroup) (WorkloadBinding, error) {
	return WorkloadBinding{}, nil
}

func (p *lifecycleTestWorkload) Reconcile(context.Context, *api.DynamoGraphDeploymentEngineGroup) (*api.DynamoGraphDeploymentEngineGroup, bool, error) {
	p.reconciles++
	return p.candidate, p.pending, nil
}

func (p *lifecycleTestWorkload) Accept(context.Context, *api.DynamoGraphDeploymentEngineGroup, *api.DynamoGraphDeploymentEngineGroup) error {
	p.accepts++
	return nil
}

func (p *lifecycleTestWorkload) AuthorizeRestart(context.Context, *api.DynamoGraphDeploymentEngineGroup, types.UID) error {
	return errors.New("no restart authority")
}

func TestWorkloadLifecycleRunsWhenEngineCannotResolve(t *testing.T) {
	t.Log("protect a logical group with an unresolved engine and an independently pending physical lifecycle")
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	group := &api.DynamoGraphDeploymentEngineGroup{Name: "world", Namespace: "test", UID: "group-uid",
		Finalizers: []string{engineGroupFinalizer}, Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1}}
	kube := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(group).WithObjects(group).Build()
	physical := &lifecycleTestWorkload{pending: true}
	controller := &Reconciler{Client: kube, WorkloadProvider: physical,
		RuntimeProvider: failingEngineGroupRuntimeProvider{err: ErrRuntimeUnavailable}}

	t.Log("reconcile the physical fence without constructing or invoking any native adapter")
	result, err := controller.Reconcile(t.Context(), reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)})
	require.NoError(t, err)
	assert.Positive(t, result.RequeueAfter)
	assert.Equal(t, 1, physical.reconciles)
	assert.Zero(t, physical.accepts)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), group))
	assert.Nil(t, group.Status.Topology)
	assert.Nil(t, group.Status.Operation)
	assert.Contains(t, group.Finalizers, engineGroupFinalizer)
}

func TestWorkloadCandidateRequiresEngineProfileValidation(t *testing.T) {
	for _, scenario := range []string{"matching profile", "different profile", "unresolved engine"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("propose a new physical lifetime independently of engine resolution")
			group := &api.DynamoGraphDeploymentEngineGroup{Name: "world", UID: "group-uid",
				Status: api.DynamoGraphDeploymentEngineGroupStatus{Profile: &api.EngineGroupProfileStatus{Fingerprint: "profile-v1"}}}
			physical := &lifecycleTestWorkload{candidate: group.DeepCopy(), pending: true}
			controller := &Reconciler{WorkloadProvider: physical,
				RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: newEngineGroupControllerTestBackend(1)}}
			switch scenario {
			case "different profile":
				group.Status.Profile.Fingerprint = "another-profile"
			case "unresolved engine":
				controller.RuntimeProvider = failingEngineGroupRuntimeProvider{err: ErrRuntimeUnavailable}
			}

			t.Log("accept the binding only after successful engine resolution and immutable-profile validation")
			pending, err := controller.reconcileWorkload(t.Context(), group)
			assert.True(t, pending)
			if scenario == "matching profile" {
				require.NoError(t, err)
				assert.Equal(t, 1, physical.accepts)
			} else {
				require.Error(t, err)
				assert.Zero(t, physical.accepts)
			}
		})
	}
}
