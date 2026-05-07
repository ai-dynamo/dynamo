/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func baseDGD(services map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec) *v1alpha1.DynamoGraphDeployment {
	return &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec:       v1alpha1.DynamoGraphDeploymentSpec{Services: services},
	}
}

func rawBetaDGD(t testing.TB, src *v1alpha1.DynamoGraphDeployment) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	dst := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("convert test DGD to v1beta1: %v", err)
	}
	return dst
}

func TestComputeV1beta1DGDWorkersSpecHash_Deterministic(t *testing.T) {
	dgd := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: commonconsts.ComponentTypePrefill, Replicas: ptr.To(int32(2))},
		"decode":  {ComponentType: commonconsts.ComponentTypeDecode, Replicas: ptr.To(int32(3))},
	})
	h1 := ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd))
	h2 := ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd))
	assert.Equal(t, h1, h2)
	assert.Len(t, h1, 8)
}

func TestComputeLegacyAlphaDGDWorkersSpecHash_MatchesV1Alpha1Hash(t *testing.T) {
	alpha := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: commonconsts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
			Resources: &v1alpha1.Resources{
				Requests: &v1alpha1.ResourceItem{CPU: "1", Memory: "1Gi"},
			},
			Labels:      map[string]string{"resource-label": "ignored-by-legacy-hash"},
			Annotations: map[string]string{"resource-annotation": "ignored-by-legacy-hash"},
		},
	})
	alpha.Annotations = map[string]string{"nvidia.com/current-worker-hash": "old-alpha-hash"}
	beta := &v1beta1.DynamoGraphDeployment{}
	assert.NoError(t, alpha.ConvertTo(beta))

	legacyHash, err := ComputeLegacyAlphaDGDWorkersSpecHash(beta)
	assert.NoError(t, err)
	expectedLegacyHash, err := v1alpha1.ComputeDGDWorkersSpecHash(alpha)
	assert.NoError(t, err)
	assert.Equal(t, expectedLegacyHash, legacyHash)
	assert.NotEqual(t, ComputeV1beta1DGDWorkersSpecHash(beta), legacyHash)
}

func TestComputeV1beta1DGDWorkersSpecHash_IgnoresNonWorkers(t *testing.T) {
	withFrontend := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: commonconsts.ComponentTypeWorker},
		"frontend": {ComponentType: commonconsts.ComponentTypeFrontend},
	})
	withoutFrontend := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: commonconsts.ComponentTypeWorker},
	})
	assert.Equal(t, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, withFrontend)), ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, withoutFrontend)))
}

func TestComputeV1beta1DGDWorkersSpecHash_NoWorkers(t *testing.T) {
	dgd := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"frontend": {ComponentType: commonconsts.ComponentTypeFrontend},
	})
	h := ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd))
	assert.Len(t, h, 8)
}

func TestComputeV1beta1DGDWorkersSpecHash_ChangesOnPodAffectingFields(t *testing.T) {
	base := func() *v1alpha1.DynamoGraphDeployment {
		return baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: commonconsts.ComponentTypeWorker},
		})
	}
	baseHash := ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, base()))

	// Image change (via Resources)
	dgd := base()
	dgd.Spec.Services["worker"].Resources = &v1alpha1.Resources{
		Requests: &v1alpha1.ResourceItem{CPU: "2"},
	}
	assert.NotEqual(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd)), "resource change should change hash")

	// Env change
	dgd2 := base()
	dgd2.Spec.Services["worker"].Envs = []corev1.EnvVar{{Name: "FOO", Value: "bar"}}
	assert.NotEqual(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd2)), "env change should change hash")

	// SharedMemory change
	dgd3 := base()
	dgd3.Spec.Services["worker"].SharedMemory = &v1alpha1.SharedMemorySpec{
		Size: resource.MustParse("1Gi"),
	}
	assert.NotEqual(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd3)), "shared memory change should change hash")

	// GlobalDynamoNamespace change
	dgd4 := base()
	dgd4.Spec.Services["worker"].GlobalDynamoNamespace = true
	assert.NotEqual(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd4)), "global dynamo namespace change should change hash")

	// Converted v1alpha1 ExtraPodMetadata lands in podTemplate metadata.
	dgd5 := base()
	dgd5.Spec.Services["worker"].ExtraPodMetadata = &v1alpha1.ExtraPodMetadata{
		Labels: map[string]string{"rollout": "required"},
	}
	assert.NotEqual(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd5)), "extra pod metadata change should change hash")

	// Native v1beta1 podTemplate metadata is also pod-affecting.
	dgd6 := betaDGD(t, base())
	dgd6.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{"rollout": "required"},
		},
	}
	assert.NotEqual(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(dgd6), "podTemplate metadata change should change hash")
}

func TestComputeV1beta1DGDWorkersSpecHash_StableOnExcludedFields(t *testing.T) {
	base := func() *v1alpha1.DynamoGraphDeployment {
		return baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: commonconsts.ComponentTypeWorker},
		})
	}
	baseHash := ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, base()))

	tests := []struct {
		name   string
		mutate func(*v1alpha1.DynamoGraphDeployment)
	}{
		{"replicas", func(d *v1alpha1.DynamoGraphDeployment) {
			d.Spec.Services["worker"].Replicas = ptr.To(int32(99))
		}},
		{"serviceName", func(d *v1alpha1.DynamoGraphDeployment) {
			d.Spec.Services["worker"].ServiceName = "changed"
		}},
		{"componentType", func(d *v1alpha1.DynamoGraphDeployment) {
			// Change to another worker type — still included in hash but componentType is stripped
			d.Spec.Services["worker"].ComponentType = commonconsts.ComponentTypePrefill
		}},
		{"dynamoNamespace", func(d *v1alpha1.DynamoGraphDeployment) {
			d.Spec.Services["worker"].DynamoNamespace = ptr.To("changed")
		}},
		{"ingress", func(d *v1alpha1.DynamoGraphDeployment) {
			d.Spec.Services["worker"].Ingress = &v1alpha1.IngressSpec{Enabled: true}
		}},
		{"scalingAdapter", func(d *v1alpha1.DynamoGraphDeployment) {
			d.Spec.Services["worker"].ScalingAdapter = &v1alpha1.ScalingAdapter{}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := base()
			tt.mutate(dgd)
			assert.Equal(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd)), "excluded field %s should not change hash", tt.name)
		})
	}
}

func TestComputeV1beta1DGDWorkersSpecHash_StableOnPreservedAlphaResourceMetadata(t *testing.T) {
	base := func() *v1alpha1.DynamoGraphDeployment {
		return baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: commonconsts.ComponentTypeWorker},
		})
	}
	baseHash := ComputeV1beta1DGDWorkersSpecHash(rawBetaDGD(t, base()))

	tests := []struct {
		name   string
		mutate func(*v1alpha1.DynamoGraphDeployment)
	}{
		{"annotations", func(d *v1alpha1.DynamoGraphDeployment) {
			d.Spec.Services["worker"].Annotations = map[string]string{"foo": "bar"}
		}},
		{"labels", func(d *v1alpha1.DynamoGraphDeployment) {
			d.Spec.Services["worker"].Labels = map[string]string{"foo": "bar"}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := base()
			tt.mutate(dgd)
			assert.Equal(t, baseHash, ComputeV1beta1DGDWorkersSpecHash(rawBetaDGD(t, dgd)), "preserved alpha resource metadata should not change hash")
		})
	}
}

func TestComputeV1beta1DGDWorkersSpecHash_EnvOrderMatters(t *testing.T) {
	dgd1 := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: commonconsts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "B", Value: "2"}, {Name: "A", Value: "1"}},
		},
	})
	dgd2 := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: commonconsts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "A", Value: "1"}, {Name: "B", Value: "2"}},
		},
	})
	assert.NotEqual(t, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd1)), ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd2)))
}

func TestComputeV1beta1DGDWorkersSpecHash_AllWorkerTypes(t *testing.T) {
	// All three worker types are included
	dgd := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"w": {ComponentType: commonconsts.ComponentTypeWorker},
		"p": {ComponentType: commonconsts.ComponentTypePrefill},
		"d": {ComponentType: commonconsts.ComponentTypeDecode},
	})
	// Changing any one of them changes the hash
	base := ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd))
	dgd.Spec.Services["p"].Envs = []corev1.EnvVar{{Name: "X", Value: "1"}}
	assert.NotEqual(t, base, ComputeV1beta1DGDWorkersSpecHash(betaDGD(t, dgd)))
}

func TestStripNonPodTemplateFields(t *testing.T) {
	spec := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ServiceName:      "svc",
		ComponentType:    commonconsts.ComponentTypeWorker,
		SubComponentType: "sub",
		DynamoNamespace:  ptr.To("ns"),
		Replicas:         ptr.To(int32(3)),
		Autoscaling:      &v1alpha1.Autoscaling{}, //nolint:staticcheck // SA1019: testing backward compatibility with deprecated field
		ScalingAdapter:   &v1alpha1.ScalingAdapter{},
		Ingress:          &v1alpha1.IngressSpec{Enabled: true},
		ModelRef:         &v1alpha1.ModelReference{Name: "m"},
		EPPConfig:        &v1alpha1.EPPConfig{},
		Annotations:      map[string]string{"a": "b"},
		Labels:           map[string]string{"c": "d"},
		// Pod-affecting fields
		Envs:                  []corev1.EnvVar{{Name: "Z"}, {Name: "A"}},
		GlobalDynamoNamespace: true,
	}
	stripped := stripNonPodTemplateFields(betaComponent(t, spec))

	// Excluded fields are zeroed
	assert.Empty(t, stripped.ComponentType)
	assert.Nil(t, stripped.Replicas)
	assert.Nil(t, stripped.ScalingAdapter)
	assert.Nil(t, stripped.ModelRef)
	assert.Nil(t, stripped.EPPConfig)

	// Included fields are preserved
	if assert.NotNil(t, stripped.PodTemplate) {
		assert.Equal(t, map[string]string{"a": "b"}, stripped.PodTemplate.Annotations)
		assert.Equal(t, map[string]string{"c": "d"}, stripped.PodTemplate.Labels)
	}
	assert.True(t, stripped.GlobalDynamoNamespace)
	main := GetMainContainer(&stripped)
	if assert.NotNil(t, main) {
		assert.Len(t, main.Env, 2)
		// Envs are not sorted
		assert.Equal(t, "Z", main.Env[0].Name)
		assert.Equal(t, "A", main.Env[1].Name)
	}

	// Original spec is not mutated
	assert.Equal(t, "svc", spec.ServiceName)
}

func TestSortEnvVars(t *testing.T) {
	envs := []corev1.EnvVar{{Name: "C"}, {Name: "A"}, {Name: "B"}}
	sorted := sortEnvVars(envs)
	assert.Equal(t, "A", sorted[0].Name)
	assert.Equal(t, "B", sorted[1].Name)
	assert.Equal(t, "C", sorted[2].Name)
	// Original not mutated
	assert.Equal(t, "C", envs[0].Name)
}
