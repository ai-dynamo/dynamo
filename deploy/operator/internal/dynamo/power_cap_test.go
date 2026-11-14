// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestLivePowerCapSupport(t *testing.T) {
	for _, tc := range []struct {
		name       string
		provider   string
		component  v1beta1.ComponentType
		multinode  bool
		graphCap   bool
		checkpoint string // "automatic" or "ref"
		alphaCap   bool   // sets the cap in the preserved v1alpha1 service annotations
		wantReason string
	}{
		{name: "grove", provider: "grove"},
		{name: "single-node component", provider: "component"},
		{name: "multinode component", provider: "component", multinode: true, wantReason: "live power-cap updates require the grove workload provider or the component provider without multinode components"},
		{name: "unset provider", wantReason: "live power-cap updates require the grove workload provider or the component provider without multinode components"},
		{name: "planner", provider: "grove", component: consts.ComponentTypePlanner, wantReason: "live power-cap updates are unsupported for graphs with a planner component"},
		{name: "LPX", provider: "grove", component: v1beta1.ComponentTypeLPX, wantReason: "live power-cap updates are unsupported for graphs with LPX components"},
		{name: "graph-level cap", provider: "grove", graphCap: true, wantReason: "live power-cap updates are unsupported when spec.annotations sets dynamo.nvidia.com/gpu-power-limit"},
		{name: "v1alpha1 service cap", provider: "grove", alphaCap: true, wantReason: "live power-cap updates are unsupported when the v1alpha1 annotations of service worker set dynamo.nvidia.com/gpu-power-limit"},
		{name: "automatic checkpoint", provider: "grove", checkpoint: "automatic", wantReason: "live power-cap updates are unsupported for worker worker with automatic checkpoints"},
		{name: "checkpointRef", provider: "component", checkpoint: "ref"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Describe a capped worker graph at the requested boundary")
			dgd := &v1beta1.DynamoGraphDeployment{}
			if tc.provider != "" {
				dgd.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: tc.provider}
			}
			worker := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: consts.ComponentTypeWorker}
			if tc.multinode {
				worker.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
			}
			switch tc.checkpoint {
			case "automatic":
				worker.Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true}}
			case "ref":
				worker.Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true, CheckpointRef: new("snapshot")}}
			}
			dgd.Spec.Components = append(dgd.Spec.Components, worker)
			if tc.component != "" {
				dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "other", ComponentType: tc.component})
			}
			if tc.graphCap {
				dgd.Spec.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
			}
			if tc.alphaCap {
				t.Log("Store the graph through v1alpha1 so the service annotation is preserved, not lifted into podTemplate")
				alpha := &v1alpha1.DynamoGraphDeployment{}
				alpha.Annotations = dgd.Annotations
				alpha.Spec.Services = map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"worker": {ComponentType: consts.ComponentTypeWorker, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}},
				}
				dgd = &v1beta1.DynamoGraphDeployment{}
				require.NoError(t, alpha.ConvertTo(dgd))
				require.Nil(t, dgd.Spec.Components[0].PodTemplate)
			}

			t.Log("Report support, or the exclusion that admission surfaces")
			reason, supported := LivePowerCapSupport(dgd)
			require.Equal(t, tc.wantReason == "", supported)
			require.Equal(t, tc.wantReason, reason)
		})
	}
}

func TestWorkerSpecsEqualComparesResolvedRuntimeVersions(t *testing.T) {
	for _, tc := range []struct {
		name      string
		old, new  string
		wantEqual bool
	}{
		{name: "hashed override change", old: "1.5.0", new: "1.6.0"},
		{name: "unhashed pre-1.5 override change", old: "1.4.0", new: "1.4.1", wantEqual: true},
		{name: "unchanged override", old: "1.6.0", new: "1.6.0", wantEqual: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Render two custom-image workers that differ only by runtimeVersionOverride")
			a := &v1beta1.DynamoComponentDeployment{}
			a.Spec.ComponentName, a.Spec.ComponentType = "worker", consts.ComponentTypeWorker
			a.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/custom-runtime:dev"}}}}
			b := a.DeepCopy()
			a.Spec.RuntimeVersionOverride, b.Spec.RuntimeVersionOverride = tc.old, tc.new

			t.Log("Compare exactly the runtime versions the worker hash includes")
			require.Equal(t, tc.wantEqual, WorkerSpecsEqual(a, b))
		})
	}
}
