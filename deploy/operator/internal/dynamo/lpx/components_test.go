/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
)

func TestComponentGroups(t *testing.T) {
	for _, test := range []struct {
		name       string
		conductors []bool
		want       map[string][]string
	}{
		{name: "no LPX components", want: map[string][]string{}},
		{name: "single workload", conductors: []bool{true}, want: map[string][]string{"component-0": {"component-0"}}},
		{name: "two independent workloads", conductors: []bool{true, true}, want: map[string][]string{"component-0": {"component-0"}, "component-1": {"component-1"}}},
		{name: "shared conductor first", conductors: []bool{true, false}, want: map[string][]string{"component-0": {"component-0", "component-1"}}},
		{name: "shared conductor last", conductors: []bool{false, true}, want: map[string][]string{"component-1": {"component-0", "component-1"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author LPX members beside an ordinary frontend")
			dgd := newSelectedTestDGD(t, "graph", dynamov1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: dynamov1beta1.ComponentTypeFrontend})
			for i, conductor := range test.conductors {
				component := testLPXComponent(fmt.Sprintf("component-%d", i), "build", dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("agent")})
				if conductor {
					component.Roles = append(component.Roles, dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor")})
				}
				dgd.Spec.Components = append(dgd.Spec.Components, component)
			}

			t.Log("Group every LPX member by its conductor without mutating the graph")
			before := dgd.DeepCopy()
			require.Equal(t, test.want, ComponentGroups(dgd))
			require.Equal(t, before, dgd)

			t.Log("Reordering authored components preserves group identity and membership")
			slices.Reverse(dgd.Spec.Components)
			require.Equal(t, test.want, ComponentGroups(dgd))
		})
	}
}

// singleGroupComponents reads the one admitted workload in a test fixture.
func singleGroupComponents(t *testing.T, dgd *dynamov1beta1.DynamoGraphDeployment) []string {
	t.Helper()
	groups := ComponentGroups(dgd)
	require.Len(t, groups, 1)
	for _, components := range groups {
		return components
	}
	return nil
}
