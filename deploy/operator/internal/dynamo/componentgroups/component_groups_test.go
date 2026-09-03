/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package componentgroups

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestComponentGroupsLookup(t *testing.T) {
	t.Log("Build an indexed view over two declared component groups")
	groups := New(&v1beta1.DynamoGraphDeploymentExperimentalSpec{
		ComponentGroups: []v1beta1.ComponentGroupSpec{
			{Name: "runtime", Components: []v1beta1.ComponentGroupComponentSpec{{Name: "frontend"}, {Name: "worker"}}},
			{Name: "planner", Components: []v1beta1.ComponentGroupComponentSpec{{Name: "planner"}}},
		},
	})

	t.Log("Resolve component ownership without normalizing component names")
	group, ok := groups.GroupForComponent("worker")
	require.True(t, ok)
	assert.Equal(t, "runtime", group.Name)
	assert.True(t, groups.IsGrouped("frontend"))
	assert.False(t, groups.IsGrouped("Worker"))

	t.Log("Resolve group identity for Kubernetes resource lookups")
	assert.True(t, groups.HasGroup("runtime"))
	assert.False(t, groups.HasGroup("missing"))
}

func TestComponentGroupsNilExperimental(t *testing.T) {
	t.Log("Treat an absent experimental section as an empty group set")
	groups := New(nil)
	assert.Empty(t, groups)
	assert.False(t, groups.IsGrouped("worker"))
}
