// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
)

func TestComponentRoleStatusNativeConversion(t *testing.T) {
	t.Log("Publish named roles, declared argv, known empty values, and unknown bindings")
	roles := []v1beta1.ComponentRoleStatus{{Name: "worker", Containers: []v1beta1.ComponentContainerStatus{{
		Name: "main", Image: "engine:test", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"},
		Env: map[string]v1beta1.EnvironmentValueStatus{"EMPTY": {Value: new("")}, "MODE": {Value: new("kv")}, "RANK": {Unresolved: true}},
	}}}, {Name: "leader"}}
	dgd := &v1beta1.DynamoGraphDeployment{Status: v1beta1.DynamoGraphDeploymentStatus{ObservedGeneration: 7,
		Components: map[string]v1beta1.ComponentReplicaStatus{"engine": {Roles: roles}},
	}}
	dcd := &v1beta1.DynamoComponentDeployment{Status: v1beta1.DynamoComponentDeploymentStatus{
		ObservedGeneration: 7, Component: &v1beta1.ComponentReplicaStatus{Roles: roles},
	}}

	t.Log("Both DGD and DCD carry roles natively through alpha, without preservation annotations")
	alphaDGD := &DynamoGraphDeployment{}
	require.NoError(t, alphaDGD.ConvertFrom(dgd))
	require.Equal(t, roles, alphaDGD.Status.Services["engine"].Roles)
	alphaDCD := &DynamoComponentDeployment{}
	require.NoError(t, alphaDCD.ConvertFrom(dcd))
	require.Equal(t, roles, alphaDCD.Status.Service.Roles)
	require.Empty(t, alphaDGD.Annotations)
	require.Empty(t, alphaDCD.Annotations)
	gotDGD := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, alphaDGD.ConvertTo(gotDGD))
	require.Equal(t, dgd.Status, gotDGD.Status)
	gotDCD := &v1beta1.DynamoComponentDeployment{}
	require.NoError(t, alphaDCD.ConvertTo(gotDCD))
	require.Equal(t, dcd.Status, gotDCD.Status)

	t.Log("Live alpha removals are authoritative, not restored from hidden data")
	status := alphaDGD.Status.Services["engine"]
	status.Roles = nil
	alphaDGD.Status.Services["engine"] = status
	require.NoError(t, alphaDGD.ConvertTo(gotDGD))
	require.Nil(t, gotDGD.Status.Components["engine"].Roles)
	alphaDCD.Status.Service.Roles = nil
	require.NoError(t, alphaDCD.ConvertTo(gotDCD))
	require.Nil(t, gotDCD.Status.Component.Roles)
}
