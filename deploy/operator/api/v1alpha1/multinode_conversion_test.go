/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/utils/ptr"
)

func TestMultinodeSpecConversionRoundTrip(t *testing.T) {
	claimTemplateName := "worker-devices"
	seconds := int64(30)
	src := &v1beta1.MultinodeSpec{
		NodeCount: 4,
		Worker: &v1beta1.MultinodeWorkerSpec{
			PodTemplateOverrides: &v1beta1.MultinodePodTemplateOverrides{
				Metadata: &v1beta1.MultinodePodTemplateMetadataOverrides{
					Labels:      ptr.To(map[string]string{"role": "worker"}),
					Annotations: ptr.To(map[string]string{"source": "manual"}),
				},
				Spec: &v1beta1.MultinodePodSpecOverrides{
					NodeSelector: ptr.To(map[string]string{"pool": "worker"}),
					Tolerations: ptr.To([]corev1.Toleration{{
						Key:               "dedicated",
						TolerationSeconds: &seconds,
					}}),
					ResourceClaims: ptr.To([]corev1.PodResourceClaim{{
						Name:                      "devices",
						ResourceClaimTemplateName: &claimTemplateName,
					}}),
					ImagePullSecrets: ptr.To([]corev1.LocalObjectReference{{Name: "registry"}}),
					Containers: []v1beta1.MultinodeContainerOverride{{
						Name:    v1beta1.MainContainerName,
						Image:   ptr.To("runtime:1.4.0"),
						Command: ptr.To([]string{"python3", "-m", "dynamo.vllm"}),
						Args:    ptr.To([]string{"--node-rank=1", "--headless"}),
						Env: ptr.To([]corev1.EnvVar{{
							Name: "POD_NAME",
							ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.name",
							}},
						}}),
						Resources: &v1beta1.MultinodeContainerResourceOverrides{
							Claims: ptr.To([]corev1.ResourceClaim{{Name: "devices", Request: "gpu"}}),
						},
					}},
				},
			},
		},
	}

	alpha := &MultinodeSpec{}
	ConvertToMultinodeSpec(src, alpha)
	got := &v1beta1.MultinodeSpec{}
	ConvertFromMultinodeSpec(alpha, got)
	if !apiequality.Semantic.DeepEqual(got, src) {
		t.Fatalf("multinode round trip changed value:\n got: %#v\nwant: %#v", got, src)
	}

	(*alpha.Worker.PodTemplateOverrides.Metadata.Labels)["role"] = "changed"
	if (*src.Worker.PodTemplateOverrides.Metadata.Labels)["role"] != "worker" {
		t.Fatal("conversion aliased metadata maps across API versions")
	}
}

func TestMultinodeSpecConversionPreservesExplicitEmptyOverrides(t *testing.T) {
	src := &v1beta1.MultinodeSpec{
		NodeCount: 2,
		Worker: &v1beta1.MultinodeWorkerSpec{PodTemplateOverrides: &v1beta1.MultinodePodTemplateOverrides{
			Metadata: &v1beta1.MultinodePodTemplateMetadataOverrides{
				Labels:      ptr.To(map[string]string{}),
				Annotations: ptr.To(map[string]string{}),
			},
			Spec: &v1beta1.MultinodePodSpecOverrides{
				NodeSelector:     ptr.To(map[string]string{}),
				Tolerations:      ptr.To([]corev1.Toleration{}),
				ResourceClaims:   ptr.To([]corev1.PodResourceClaim{}),
				ImagePullSecrets: ptr.To([]corev1.LocalObjectReference{}),
				Containers: []v1beta1.MultinodeContainerOverride{{
					Name:      v1beta1.MainContainerName,
					Command:   ptr.To([]string{}),
					Args:      ptr.To([]string{}),
					Env:       ptr.To([]corev1.EnvVar{}),
					Resources: &v1beta1.MultinodeContainerResourceOverrides{Claims: ptr.To([]corev1.ResourceClaim{})},
				}},
			},
		}},
	}

	alpha := &MultinodeSpec{}
	ConvertToMultinodeSpec(src, alpha)
	got := &v1beta1.MultinodeSpec{}
	ConvertFromMultinodeSpec(alpha, got)
	if !apiequality.Semantic.DeepEqual(got, src) {
		t.Fatalf("explicit empty overrides changed during round trip:\n got: %#v\nwant: %#v", got, src)
	}
	if got.Worker.PodTemplateOverrides.Spec.ResourceClaims == nil ||
		got.Worker.PodTemplateOverrides.Spec.Containers[0].Args == nil ||
		got.Worker.PodTemplateOverrides.Metadata.Labels == nil {
		t.Fatal("explicit empty overrides became omitted")
	}
}

func TestMultinodeSpecConversionWithoutWorkerOverrides(t *testing.T) {
	src := &v1beta1.MultinodeSpec{NodeCount: 2}
	alpha := &MultinodeSpec{}
	ConvertToMultinodeSpec(src, alpha)
	got := &v1beta1.MultinodeSpec{}
	ConvertFromMultinodeSpec(alpha, got)
	if !apiequality.Semantic.DeepEqual(got, src) {
		t.Fatalf("multinode round trip = %#v, want %#v", got, src)
	}
}
