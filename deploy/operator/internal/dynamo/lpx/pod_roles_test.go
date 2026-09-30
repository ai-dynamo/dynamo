/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestAgentSchedulingPreservesAuthoredResources(t *testing.T) {
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
		Containers: []corev1.Container{
			{
				Name: "sidecar", Image: "helper",
				Resources: corev1.ResourceRequirements{
					Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
				},
			},
			{
				Name: "main", Image: "runtime",
				Resources: corev1.ResourceRequirements{
					Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
				},
			},
		},
		EphemeralContainers: []corev1.EphemeralContainer{{
			EphemeralContainerCommon: corev1.EphemeralContainerCommon{
				Name: "debug", Resources: corev1.ResourceRequirements{Limits: lpuResources.DeepCopy()},
			},
		}},
		Resources: &corev1.ResourceRequirements{
			Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
		},
	}

	t.Log("Preserve unrelated resources for each family, including initially absent resource maps")
	tests := []struct {
		name             string
		family           BuildFamily
		emptyResources   bool
		expectedResource corev1.ResourceName
		expectedQuantity resource.Quantity
	}{
		{
			name: "XT", family: BuildFamilyXT,
			expectedResource: v2LPUResourceName, expectedQuantity: resource.MustParse("8"),
		},
		{
			name: "HX", family: BuildFamilyHX,
			expectedResource: v3LPUResourceName, expectedQuantity: resource.MustParse("16"),
		},
		{
			name: "XT absent resource maps", family: BuildFamilyXT, emptyResources: true,
			expectedResource: v2LPUResourceName, expectedQuantity: resource.MustParse("8"),
		},
		{
			name: "HX absent resource maps", family: BuildFamilyHX, emptyResources: true,
			expectedResource: v3LPUResourceName, expectedQuantity: resource.MustParse("16"),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Expect only the selected resource on main to change")
			agent := base.DeepCopy()
			if test.emptyResources {
				agent.Containers[1].Resources = corev1.ResourceRequirements{}
			}
			want := base.DeepCopy()
			if test.emptyResources {
				want.Containers[1].Resources = corev1.ResourceRequirements{
					Requests: corev1.ResourceList{}, Limits: corev1.ResourceList{},
				}
			}
			want.Containers[1].Resources.Requests[test.expectedResource] = test.expectedQuantity
			want.Containers[1].Resources.Limits[test.expectedResource] = test.expectedQuantity

			t.Log("Bind the family device and retain placement and every other authored resource")
			configureAgentScheduling(agent, test.family)
			require.Empty(t, cmp.Diff(want, agent))
		})
	}
}
