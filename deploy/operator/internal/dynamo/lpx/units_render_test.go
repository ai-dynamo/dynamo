/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestResolveUnitsFormatPerWorkload(t *testing.T) {
	t.Parallel()
	bundle := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp20, devicesPerNode: 8,
		partitions: testUnitPartitions(8, nil, 0), units: testSingleUnits(testNodesReservation(1, 0))}
	units, err := acquireTestUnitBundle(t, bundle)
	require.NoError(t, err)
	bundle.units = nil
	legacy, err := acquireTestUnitBundle(t, bundle)
	require.NoError(t, err)
	for _, tc := range []struct {
		name          string
		draft, target *Build
		wantErr       bool
	}{
		{"legacy pair", legacy, legacy, false}, {"units pair", units, units, false},
		{"units draft and legacy target", units, legacy, true}, {"legacy draft and units target", legacy, units, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve the complete pair before rendering any live resource")
			dgd := newSelectedTestDGD(t, "pair",
				testLPXComponent("target", "target-build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("runtime")}, v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("runtime")}),
				testLPXComponent("draft", "draft-build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("runtime")}),
			)
			before := dgd.DeepCopy()
			got, err := ResolveWorkload(t.Context(), fake.NewClientBuilder().Build(), dgd, singleGroupComponents(t, dgd), staticModelRegistry{"draft-build": tc.draft, "target-build": tc.target})
			require.Equal(t, before, dgd)
			if tc.wantErr {
				require.ErrorContains(t, err, "mixes builds with and without lpuUnits")
				require.Nil(t, got)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.target == units, got.placesLPUUnits())
		})
	}
}

func TestRenderUnitsHaveNoLegacyConfig(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name                        string
		hybrid, mount, volume, init bool
	}{
		{name: "Quasar"}, {name: "Cyborg and Hydra", hybrid: true},
		{name: "mount alone", mount: true}, {name: "authored volume and mount", mount: true, volume: true},
		{name: "native init mount", mount: true, volume: true, init: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve a units artifact and prepare fresh runtime templates")
			bundle := testUnitBundle{architecture: deploymentcapnpv1.LpuArchitecture_lp20, devicesPerNode: 8, hybrid: tc.hybrid,
				partitions: testUnitPartitions(8, nil, 0), units: testSingleUnits(testNodesReservation(1, 0))}
			build, err := acquireTestUnitBundle(t, bundle)
			require.NoError(t, err)
			projections, err := projectTestUnits(build, nil)
			require.NoError(t, err)
			projections[0].stage = testRenderComponentName
			workload := &Workload{modelProjections: projections, scalingGroupReplicas: 1}
			serving := &v1beta1.DynamoComponentDeploymentSharedSpec{Roles: []v1beta1.ComponentRoleSpec{{Name: v1beta1.ComponentRoleLPXConductor}}}
			plan, err := workload.PlanNodeLocalMaterialization("test-pcs", serving)
			require.NoError(t, err)
			spec := renderTestPodSpec()
			spec.Containers[0].VolumeMounts = slices.DeleteFunc(spec.Containers[0].VolumeMounts, func(m corev1.VolumeMount) bool { return m.Name == lpuConfigVolumeName })
			if tc.mount {
				spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: "/configs"})
			}
			if tc.volume {
				spec.Volumes = append(spec.Volumes, corev1.Volume{Name: lpuConfigVolumeName, EmptyDir: &corev1.EmptyDirVolumeSource{}})
			}
			if tc.init {
				spec.InitContainers = []corev1.Container{*spec.Containers[0].DeepCopy()}
				spec.InitContainers[0].Name = "setup"
				spec.Containers[0].VolumeMounts = slices.DeleteFunc(spec.Containers[0].VolumeMounts, func(m corev1.VolumeMount) bool { return m.Name == lpuConfigVolumeName })
			}
			input := RenderInput{Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: spec}}}
			if tc.hybrid {
				input.Cyborg = renderTestPCS(true).Spec.Template.Cliques[0]
				input.Cyborg.Spec.PodSpec.Containers[0].VolumeMounts = slices.DeleteFunc(input.Cyborg.Spec.PodSpec.Containers[0].VolumeMounts, func(m corev1.VolumeMount) bool { return m.Name == lpuConfigVolumeName })
			} else {
				input.Conductor = &corev1.PodTemplateSpec{Spec: *spec.DeepCopy()}
			}
			got, err := RenderNodeLocal(workload, plan, input)
			if tc.mount {
				require.ErrorContains(t, err, "mounts legacy config")
				return
			}
			require.NoError(t, err)
			require.Empty(t, got.Resources)
			for _, clique := range got.Cliques {
				require.NotContains(t, clique.Annotations, v1alpha1.AnnotationExtraResourcesHash)
				for _, volume := range clique.Spec.PodSpec.Volumes {
					require.NotEqual(t, lpuConfigVolumeName, volume.Name)
				}
				if clique.Name == plan.Agents[0].TemplateName {
					require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: localPartitionIDsEnv, Value: ""})
				}
			}
		})
	}
}
