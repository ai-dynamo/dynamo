/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestSelectedRolePodSpecsMaterializeFamilyResourceOnlyOnAgentMainContainer(t *testing.T) {
	t.Parallel()

	t.Log("Construct a mixed-resource PodSpec spanning every container resource location")
	lpuResources := corev1.ResourceList{
		v2LPUResourceName: resource.MustParse("8"),
		corev1.ResourceName("lpu.nvidia.com/devices"): resource.MustParse("1"),
		v3LPUResourceName:                     resource.MustParse("16"),
		corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
		corev1.ResourceCPU:                    resource.MustParse("2"),
	}
	base := corev1.PodSpec{
		Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchExpressions: []corev1.NodeSelectorRequirement{{
					Key: corev1.LabelHostname, Operator: corev1.NodeSelectorOpIn, Values: []string{"lpu-node-a"},
				}}}},
			},
		}},
		InitContainers: []corev1.Container{{
			Name: "init", Resources: corev1.ResourceRequirements{
				Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
			},
		}},
		Containers: []corev1.Container{{
			Name: "main", Image: "runtime",
			Resources: corev1.ResourceRequirements{
				Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
			},
		}},
		EphemeralContainers: []corev1.EphemeralContainer{{
			EphemeralContainerCommon: corev1.EphemeralContainerCommon{
				Name: "debug", Resources: corev1.ResourceRequirements{Limits: lpuResources.DeepCopy()},
			},
		}},
		Resources: &corev1.ResourceRequirements{
			Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
		},
	}

	t.Log("Define V2 and V3 family-specific resource expectations")
	tests := []struct {
		name               string
		family             BuildFamily
		expectedResource   corev1.ResourceName
		expectedQuantity   resource.Quantity
		unexpectedResource corev1.ResourceName
	}{
		{
			name: "V2 XT8888", family: BuildFamilyXT,
			expectedResource: v2LPUResourceName, expectedQuantity: resource.MustParse("8"),
			unexpectedResource: v3LPUResourceName,
		},
		{
			name: "V3 HX", family: BuildFamilyHX,
			expectedResource: v3LPUResourceName, expectedQuantity: resource.MustParse("16"),
			unexpectedResource: v2LPUResourceName,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Shape LPU resources independently for the conductor and Agent")
			conductor, agent := base.DeepCopy(), base.DeepCopy()
			stripLPUResources(conductor)
			configureAgentScheduling(agent, test.family)

			t.Log("Verify the input is immutable and affinity follows its owning role")
			require.Equal(t, lpuResources, base.Containers[0].Resources.Limits, "input PodSpec must remain unchanged")
			require.Equal(t, base.Affinity, agent.Affinity)

			t.Log("Remove generic and wrong-family LPU resources from both roles")
			for _, spec := range []*corev1.PodSpec{conductor, agent} {
				for _, resources := range lpuResourceLists(spec) {
					require.NotContains(t, resources, corev1.ResourceName("lpu.nvidia.com/devices"))
					require.NotContains(t, resources, test.unexpectedResource)
					require.Contains(t, resources, corev1.ResourceName("nvidia.com/gpu"))
					require.Contains(t, resources, corev1.ResourceCPU)
				}
			}
			for _, resources := range lpuResourceLists(conductor) {
				require.NotContains(t, resources, test.expectedResource)
			}
			for index, resources := range lpuResourceLists(agent) {
				if index == 2 || index == 3 {
					continue
				}
				require.NotContains(t, resources, test.expectedResource)
			}
			t.Log("Materialize only the selected family resource on the main Agent container")
			require.Equal(t, test.expectedQuantity, agent.Containers[0].Resources.Requests[test.expectedResource])
			require.Equal(t, test.expectedQuantity, agent.Containers[0].Resources.Limits[test.expectedResource])
		})
	}
}

func lpuResourceLists(spec *corev1.PodSpec) []corev1.ResourceList {
	lists := []corev1.ResourceList{
		spec.InitContainers[0].Resources.Limits,
		spec.InitContainers[0].Resources.Requests,
		spec.Containers[0].Resources.Limits,
		spec.Containers[0].Resources.Requests,
		spec.EphemeralContainers[0].Resources.Limits,
		spec.Resources.Limits,
		spec.Resources.Requests,
	}
	return lists
}
