/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// These tests pin the operator-only upgrade contract for workloads created by
// operator 1.6.x or earlier: rendering and worker hashing must match the published
// 1.6.0 operator so that taking over an unchanged workload does not roll it. The
// expected legacy values were captured by rendering the same fixtures through the
// production code of release/1.6.0 (240b7d359f).

const (
	legacyOrigin  = "1.6.0"
	currentOrigin = "1.7.0"
)

type sidecarProbeRenderCase struct {
	name          string
	origin        string
	authored      func(container *corev1.Container)
	wantReadiness *corev1.Probe
	wantLiveness  *corev1.Probe
}

func frontendSidecarOriginComponent(origin string, authored func(container *corev1.Container)) *v1beta1.DynamoComponentDeploymentSharedSpec {
	sidecar := corev1.Container{
		Name:  "sidecar-frontend",
		Image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0",
		Args:  []string{"-m", "dynamo.frontend", "--router-mode", "direct"},
	}
	authored(&sidecar)
	annotations := map[string]string{}
	if origin != "" {
		annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion] = origin
	}
	return &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:   "worker",
		ComponentType:   v1beta1.ComponentTypeDecode,
		FrontendSidecar: new("sidecar-frontend"),
		PodTemplate: &corev1.PodTemplateSpec{
			Annotations: annotations,
			Spec: corev1.PodSpec{Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0",
					Args:  []string{"python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B"},
				},
				sidecar,
			}},
		},
	}
}

func frontendHTTPGet(path string) corev1.ProbeHandler {
	return corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{
		Path: path,
		Port: intstr.FromString(commonconsts.DynamoContainerPortName),
	}}
}

func TestFrontendSidecarProbesByOrigin(t *testing.T) {
	partialProbes := func(container *corev1.Container) {
		container.ReadinessProbe = &corev1.Probe{ProbeHandler: frontendHTTPGet("/health"), InitialDelaySeconds: 30}
		container.LivenessProbe = &corev1.Probe{ProbeHandler: frontendHTTPGet("/live"), PeriodSeconds: 30}
	}
	timingOnlyReadiness := func(container *corev1.Container) {
		container.ReadinessProbe = &corev1.Probe{InitialDelaySeconds: 45}
	}
	completeProbes := func(container *corev1.Container) {
		container.ReadinessProbe = &corev1.Probe{
			ProbeHandler: frontendHTTPGet("/health"), InitialDelaySeconds: 30, PeriodSeconds: 10, TimeoutSeconds: 3, FailureThreshold: 3,
		}
		container.LivenessProbe = &corev1.Probe{
			ProbeHandler: frontendHTTPGet("/live"), InitialDelaySeconds: 15, PeriodSeconds: 30, TimeoutSeconds: 1, FailureThreshold: 3,
		}
	}

	// The 1.6.0 operator field-merged authored probes into these frontend defaults.
	legacyMergedReadiness := &corev1.Probe{
		ProbeHandler: frontendHTTPGet("/health"), InitialDelaySeconds: 30, PeriodSeconds: 10, TimeoutSeconds: 3, FailureThreshold: 3,
	}
	legacyMergedLiveness := &corev1.Probe{
		ProbeHandler: frontendHTTPGet("/live"), InitialDelaySeconds: 15, PeriodSeconds: 30, TimeoutSeconds: 1, FailureThreshold: 3,
	}
	defaultReadiness := &corev1.Probe{
		ProbeHandler: frontendHTTPGet("/health"), InitialDelaySeconds: 10, PeriodSeconds: 10, TimeoutSeconds: 3, FailureThreshold: 3,
	}
	defaultLiveness := &corev1.Probe{
		ProbeHandler: frontendHTTPGet("/live"), InitialDelaySeconds: 15, PeriodSeconds: 10, TimeoutSeconds: 1, FailureThreshold: 3,
	}

	cases := []sidecarProbeRenderCase{
		{
			name:   "legacy origin keeps the 1.6.0 field merge for partial probes",
			origin: legacyOrigin, authored: partialProbes,
			wantReadiness: legacyMergedReadiness, wantLiveness: legacyMergedLiveness,
		},
		{
			name:   "absent origin keeps the 1.6.0 field merge for partial probes",
			origin: "", authored: partialProbes,
			wantReadiness: legacyMergedReadiness, wantLiveness: legacyMergedLiveness,
		},
		{
			name:   "legacy origin completes a timing-only probe from the default handler",
			origin: legacyOrigin, authored: timingOnlyReadiness,
			wantReadiness: &corev1.Probe{
				ProbeHandler: frontendHTTPGet("/health"), InitialDelaySeconds: 45, PeriodSeconds: 10, TimeoutSeconds: 3, FailureThreshold: 3,
			},
			wantLiveness: defaultLiveness,
		},
		{
			name:   "current origin replaces partial probes with the authored value",
			origin: currentOrigin, authored: partialProbes,
			wantReadiness: &corev1.Probe{ProbeHandler: frontendHTTPGet("/health"), InitialDelaySeconds: 30},
			wantLiveness:  &corev1.Probe{ProbeHandler: frontendHTTPGet("/live"), PeriodSeconds: 30},
		},
		{
			name:   "current origin keeps a timing-only probe as authored",
			origin: currentOrigin, authored: timingOnlyReadiness,
			wantReadiness: &corev1.Probe{InitialDelaySeconds: 45},
			wantLiveness:  defaultLiveness,
		},
		{
			name:   "complete authored probes render identically under both origins (legacy)",
			origin: legacyOrigin, authored: completeProbes,
			wantReadiness: legacyMergedReadiness, wantLiveness: legacyMergedLiveness,
		},
		{
			name:   "complete authored probes render identically under both origins (current)",
			origin: currentOrigin, authored: completeProbes,
			wantReadiness: legacyMergedReadiness, wantLiveness: legacyMergedLiveness,
		},
		{
			name:   "no authored probes keep the frontend defaults under both origins (legacy)",
			origin: legacyOrigin, authored: func(*corev1.Container) {},
			wantReadiness: defaultReadiness, wantLiveness: defaultLiveness,
		},
		{
			name:   "no authored probes keep the frontend defaults under both origins (current)",
			origin: currentOrigin, authored: func(*corev1.Container) {},
			wantReadiness: defaultReadiness, wantLiveness: defaultLiveness,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Render the worker with its frontend sidecar through the production PodSpec path")
			component := frontendSidecarOriginComponent(tc.origin, tc.authored)
			pod, err := GenerateBasePodSpec(component, BackendFrameworkVLLM, &mockSecretsRetriever{},
				"qwen", "demo", RoleMain, 1, &configv1alpha1.OperatorConfiguration{},
				commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(1))
			require.NoError(t, err)

			t.Log("Compare the sidecar probes with the expected rendering for this origin")
			var sidecar *corev1.Container
			for i := range pod.Containers {
				if pod.Containers[i].Name == "sidecar-frontend" {
					sidecar = &pod.Containers[i]
				}
			}
			require.NotNil(t, sidecar)
			assert.Equal(t, tc.wantReadiness, sidecar.ReadinessProbe)
			assert.Equal(t, tc.wantLiveness, sidecar.LivenessProbe)

			t.Log("Confirm the startup probe default follows the separately gated FrontendStartupProbe")
			if tc.origin == currentOrigin {
				assert.Equal(t, &corev1.Probe{
					ProbeHandler: frontendHTTPGet("/live"), PeriodSeconds: 10, TimeoutSeconds: 1, FailureThreshold: 30,
				}, sidecar.StartupProbe)
			} else {
				assert.Nil(t, sidecar.StartupProbe)
			}
		})
	}
}

func nativeEngineDGD(withCache bool, explicitBackend string) *v1beta1.DynamoGraphDeployment {
	always := corev1.ContainerRestartPolicyAlways
	component := v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		ComponentType: v1beta1.ComponentTypeWorker,
		Replicas:      new(int32(1)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{{
				Name:          commonconsts.RuntimeContainerName,
				Image:         "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0",
				RestartPolicy: &always,
			}},
			Containers: []corev1.Container{{
				Name:    "main",
				Image:   "vllm/vllm-rs:0.3.0",
				Command: []string{"/usr/local/bin/vllm-rs"},
				Args:    []string{"--model", "Qwen/Qwen3-0.6B", "--port", "8000"},
			}},
		}},
	}
	if withCache {
		component.CompilationCache = &v1beta1.CompilationCacheConfig{PVCName: "compile-cache", MountPath: "/root/.cache/vllm"}
	}
	return &v1beta1.DynamoGraphDeployment{
		Name: "qwen", Namespace: "demo",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: legacyOrigin},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: explicitBackend,
			Components:       []v1beta1.DynamoComponentDeploymentSharedSpec{component},
		},
	}
}

func renderedWorkerEnv(t *testing.T, dgd *v1beta1.DynamoGraphDeployment) map[string]string {
	t.Helper()
	component := &dgd.Spec.Components[0]
	backend, err := BackendFrameworkForComponent(component, dgd)
	require.NoError(t, err)
	pod, err := GenerateBasePodSpec(component, backend, &mockSecretsRetriever{},
		dgd.Name, dgd.Namespace, RoleMain, 1, &configv1alpha1.OperatorConfiguration{},
		commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(1))
	require.NoError(t, err)
	for i := range pod.Containers {
		if pod.Containers[i].Name == commonconsts.MainContainerName {
			return envVarsToMap(pod.Containers[i].Env)
		}
	}
	require.Fail(t, "main container not rendered")
	return nil
}

func TestNativeEngineCommandIsNotInferredForWorkloads(t *testing.T) {
	t.Log("Resolve the backend of a native vllm-rs sidecar worker exactly as DCD generation does")
	dgd := nativeEngineDGD(true, "")
	backend, err := BackendFrameworkForComponent(&dgd.Spec.Components[0], dgd)
	require.NoError(t, err)
	assert.Equal(t, BackendFrameworkNoop, backend)

	t.Log("Verify the generated worker DCD records no backend and the engine gets no vLLM cache default")
	dcds, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{NewWorkerHash: "hash"})
	require.NoError(t, err)
	require.Len(t, dcds, 1)
	for _, dcd := range dcds {
		assert.Empty(t, dcd.Spec.BackendFramework)
	}
	assert.NotContains(t, renderedWorkerEnv(t, dgd), "VLLM_CACHE_ROOT")

	t.Log("Opt in through an explicit backendFramework and receive the vLLM cache default")
	explicit := nativeEngineDGD(true, "vllm")
	assert.Equal(t, "/root/.cache/vllm", renderedWorkerEnv(t, explicit)["VLLM_CACHE_ROOT"])
}

func TestNativeEngineWorkerHashMatchesLegacyOperator(t *testing.T) {
	t.Log("Hash a 1.6.0-origin native vllm-rs sidecar worker with and without a compilation cache")
	legacyHash, err := ComputeDGDWorkersSpecHash(nativeEngineDGD(false, ""))
	require.NoError(t, err)
	legacyCacheHash, err := ComputeDGDWorkersSpecHash(nativeEngineDGD(true, ""))
	require.NoError(t, err)

	t.Log("Match the worker hashes the published 1.6.0 operator computed for these fixtures")
	assert.Equal(t, "8312f8aa", legacyHash)
	assert.Equal(t, "66eaf1eb", legacyCacheHash)

	t.Log("Match the hash the 1.6.0 operator computed with an explicit vLLM backend")
	explicitHash, err := ComputeDGDWorkersSpecHash(nativeEngineDGD(true, "vllm"))
	require.NoError(t, err)
	assert.Equal(t, "4a915483", explicitHash)
}
