/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestEngineGroupGPUAllocation(t *testing.T) {
	for _, scenario := range []string{"dedicated GPUs", "sidecar GPUs", "dynamic claims", "host networking"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("declare physical allocation independently of any engine or workload provider")
			pod := &corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName,
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
					consts.KubeResourceGPUNvidia: resource.MustParse("4"),
				}},
			}}}
			switch scenario {
			case "sidecar GPUs":
				pod.Containers = append(pod.Containers, corev1.Container{Name: "sidecar",
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
						consts.KubeResourceGPUNvidia: resource.MustParse("1"),
					}},
				})
			case "dynamic claims":
				pod.ResourceClaims = []corev1.PodResourceClaim{{Name: "accelerators"}}
			case "host networking":
				pod.HostNetwork = true
			}
			before := pod.DeepCopy()

			t.Log("resolve only fixed, dedicated GPU capacity without mutating the declaration")
			gpus, err := resolveEngineGroupGPUAllocation(pod, &pod.Containers[0])
			if scenario == "dedicated GPUs" {
				require.NoError(t, err)
				assert.Equal(t, int64(4), gpus)
			} else {
				require.Error(t, err)
			}
			assert.Equal(t, before, pod)
		})
	}
}
