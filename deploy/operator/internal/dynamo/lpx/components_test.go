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
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestLPXSchedulerSelectionUsesComponents(t *testing.T) {
	t.Log("Select components by their declared type, independently of build or templates")
	canonicalLPX := []dynamov1beta1.DynamoComponentDeploymentSharedSpec{{ComponentType: dynamov1beta1.ComponentTypeLPX}}

	t.Log("Define default-selection scenarios")
	tests := []struct {
		name        string
		annotations map[string]string
		components  []dynamov1beta1.DynamoComponentDeploymentSharedSpec
		want        bool
	}{
		{name: "canonical LPX", components: canonicalLPX, want: true},
		{
			name: "component provider",
			annotations: map[string]string{
				commonconsts.KubeAnnotationWorkloadProvider: commonconsts.WorkloadProviderComponent,
			},
			components: canonicalLPX,
			want:       true,
		},
		{
			name: "Grove opt-out",
			annotations: map[string]string{
				commonconsts.KubeAnnotationEnableGrove: "FALSE",
			},
			components: canonicalLPX,
			want:       true,
		},
		{name: "non-LPX", components: []dynamov1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: dynamov1beta1.ComponentRoleLPXAgent}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Construct the DGD selection input")
			dgd := &dynamov1beta1.DynamoGraphDeployment{
				Annotations: test.annotations,
				Spec: dynamov1beta1.DynamoGraphDeploymentSpec{
					Components: test.components,
				},
			}

			t.Log("Resolve and verify scheduler selection")
			selected := dgd.HasLPXComponent()
			require.Equal(t, test.want, selected)
		})
	}
}

func TestComponentGroups(t *testing.T) {
	const (
		conductor = "conductor"
		sidecar   = "agent runtime sidecar"
		draft     = "draft"
	)
	for _, test := range []struct {
		name    string
		serving []string
		want    map[string][]string
	}{
		{name: "no LPX components", want: map[string][]string{}},
		{name: "single workload", serving: []string{conductor}, want: map[string][]string{"component-0": {"component-0"}}},
		{name: "two independent workloads", serving: []string{conductor, conductor}, want: map[string][]string{"component-0": {"component-0"}, "component-1": {"component-1"}}},
		{name: "three independent workloads", serving: []string{conductor, conductor, conductor}, want: map[string][]string{"component-0": {"component-0"}, "component-1": {"component-1"}, "component-2": {"component-2"}}},
		{name: "shared conductor first", serving: []string{conductor, draft}, want: map[string][]string{"component-0": {"component-0", "component-1"}}},
		{name: "shared conductor last", serving: []string{draft, conductor}, want: map[string][]string{"component-1": {"component-0", "component-1"}}},
		{name: "agent runtime sidecar beside a conductor workload", serving: []string{sidecar, conductor}, want: map[string][]string{"component-0": {"component-0"}, "component-1": {"component-1"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author LPX members beside an ordinary frontend")
			dgd := newSelectedTestDGD(t, "graph", dynamov1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: dynamov1beta1.ComponentTypeFrontend})
			for i, serving := range test.serving {
				agent := testLPXPodTemplate("agent")
				component := testLPXComponent(fmt.Sprintf("component-%d", i), "build", dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXAgent, PodTemplate: agent})
				switch serving {
				case conductor:
					component.Roles = append(component.Roles, dynamov1beta1.ComponentRoleSpec{Name: dynamov1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor")})
				case sidecar:
					agent.Spec.InitContainers = []corev1.Container{{Name: commonconsts.RuntimeContainerName, Image: "sidecar", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)}}
				}
				dgd.Spec.Components = append(dgd.Spec.Components, component)
			}

			t.Log("Group every LPX member by its serving component without mutating the graph")
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
