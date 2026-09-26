/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

const (
	testRenderComponentName = "lpu"
	testAgentTemplateName   = "agt"
	testTargetStageName     = "target"
)

func renderSelectedForTest(pcs *grovev1alpha1.PodCliqueSet, projections []*ModelProjection, input RenderInput) (*grovev1alpha1.PodCliqueSet, error) {
	// Attach test projections to their authored stage without inventing templates.
	for _, projection := range projections {
		projection.stage = testRenderComponentName
	}
	digest, err := workloadSetDigest(projections)
	if err != nil {
		return nil, err
	}
	workload := &Workload{
		modelProjections:     projections,
		digest:               digest,
		scalingGroupReplicas: 1,
	}
	plan, err := workload.PlanNodeLocalMaterialization(pcs.Name)
	if err != nil {
		return nil, err
	}
	if workload.Pipeline() == PipelineLPX {
		input.Cyborg = pcs.Spec.Template.Cliques[0]
	}
	templates, err := RenderNodeLocal(workload, plan, input)
	if err != nil {
		return nil, err
	}
	pcs.Spec.Template.Cliques = templates.Cliques
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{templates.ScalingGroup}
	return pcs, nil
}

func TestRenderResolvesAuthoredMetadataAndMounts(t *testing.T) {
	t.Parallel()

	t.Log("Cover XT configuration mounts and HX hybrid model-storage paths")
	hybrid := newV3CompilerFixture()
	hybrid.compilationMode = manifestcapnp.CompilationMode_lpx
	xtSnapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	hxSnapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	tests := []struct {
		name       string
		family     lpxv1alpha1.TargetFamily
		pipeline   Pipeline
		snapshot   *BuildSnapshot
		configPath string
	}{
		{
			name: "XT config mount", family: lpxv1alpha1.TargetFamilyXt8888, pipeline: PipelineSingle,
			snapshot: xtSnapshot, configPath: "/custom",
		},
		{
			name: "XT omitted config mount", family: lpxv1alpha1.TargetFamilyXt8888, pipeline: PipelineSingle,
			snapshot: xtSnapshot,
		},
		{
			name: "HX omitted config mount", family: lpxv1alpha1.TargetFamilyHx16x8x2x3, pipeline: PipelineSingle,
			snapshot: hxSnapshot,
		},
		{
			name: "HX custom config mount", family: lpxv1alpha1.TargetFamilyHx16x8x2x3, pipeline: PipelineSingle,
			snapshot: hxSnapshot, configPath: "/custom",
		},
		{
			name: "HX hybrid storage", family: lpxv1alpha1.TargetFamilyHx16x8x2x3, pipeline: PipelineLPX,
			snapshot: acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)), configPath: "/configs",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Seed conflicting runtime annotations and a nondefault mount")
			projection := projectRenderFixture(t, test.pipeline, test.snapshot)
			require.NotEqual(t, projection.Digest().String(), projection.CompilerSnapshotDigest())
			pcs := renderTestPCS(test.pipeline == PipelineLPX)
			for _, clique := range pcs.Spec.Template.Cliques {
				clique.Annotations = map[string]string{
					lpxv1alpha1.CompilerSnapshotDigestAnnotation: "stale",
				}
			}
			template := corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
					"user":                         "kept",
					lpxv1alpha1.PodModelAnnotation: "spoofed",
					lpxv1alpha1.CompilerSnapshotDigestAnnotation: "stale",
				}},
				Spec: renderTestPodSpec(),
			}
			template.Spec.Tolerations = []corev1.Toleration{
				{Key: "cluster.example/custom", Operator: corev1.TolerationOpExists},
				{Key: "lpu.nvidia.com/node", Operator: corev1.TolerationOpExists},
			}
			if test.family == lpxv1alpha1.TargetFamilyXt8888 && test.configPath != "" {
				template.Spec.Volumes = append(template.Spec.Volumes, corev1.Volume{
					Name: lpuConfigVolumeName,
					VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
						LocalObjectReference: corev1.LocalObjectReference{Name: "authored-config"}, DefaultMode: ptr.To[int32](0440),
					}},
				})
			}
			template.Spec.Containers[0].VolumeMounts = slices.DeleteFunc(template.Spec.Containers[0].VolumeMounts,
				func(mount corev1.VolumeMount) bool { return mount.Name == lpuConfigVolumeName })
			if test.configPath != "" {
				template.Spec.Containers[0].VolumeMounts = append(template.Spec.Containers[0].VolumeMounts,
					corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: test.configPath})
			}
			if test.pipeline == PipelineLPX {
				template.Spec.Containers[0].VolumeMounts[0].MountPath = "/model-cache"
				pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].VolumeMounts[1].MountPath = "/model-cache"
			}

			t.Log("Render into the fresh PCS without retaining stale runtime identity")
			rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, RenderInput{
				Stages:    map[string]corev1.PodTemplateSpec{testRenderComponentName: template},
				Conductor: template.DeepCopy(),
			})
			require.NoError(t, err)
			require.Same(t, pcs, rendered)
			require.Equal(t, projection.Digest().String(), rendered.Spec.Template.PodCliqueScalingGroupConfigs[0].Annotations[WorkloadDigestAnnotation])
			for _, clique := range rendered.Spec.Template.Cliques {
				require.Equal(t, v1alpha1.LPXSchedulerName, clique.Spec.PodSpec.SchedulerName)
				if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleAgent {
					require.Equal(t, projection.CompilerSnapshotDigest(), clique.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation])
				} else {
					require.NotContains(t, clique.Annotations, lpxv1alpha1.CompilerSnapshotDigestAnnotation)
				}
			}
			agent := namedClique(t, rendered, testAgentTemplateName)
			require.Equal(t, projection.Model(), agent.Annotations[lpxv1alpha1.PodModelAnnotation])
			require.Equal(t, projection.Digest().String(), agent.Annotations[WorkloadDigestAnnotation])
			require.Equal(t, projection.CompilerSnapshotDigest(), agent.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation])

			t.Log("Preserve authored placement and volumes")
			require.Equal(t, template.Spec.Tolerations, agent.Spec.PodSpec.Tolerations)
			require.Subset(t, agent.Spec.PodSpec.Volumes, template.Spec.Volumes)

			t.Log("Resolve runtime mounts using the authored storage path")
			if test.pipeline != PipelineLPX {
				conductor := namedClique(t, rendered, "cond")
				require.Equal(t, "kept", conductor.Annotations["user"])
				require.Equal(t, template.Spec.Containers[0].VolumeMounts, conductor.Spec.PodSpec.Containers[0].VolumeMounts)
				require.Equal(t, template.Spec.Containers[0].VolumeMounts, agent.Spec.PodSpec.Containers[0].VolumeMounts)
			} else {
				cyborg := namedClique(t, rendered, "cond").Spec.PodSpec.Containers[0]
				require.Contains(t, cyborg.VolumeMounts, corev1.VolumeMount{Name: "model-storage", MountPath: "/model-cache"})
				require.Contains(t, cyborg.Env, corev1.EnvVar{
					Name: "GBUILD_MANIFEST_PATH", Value: "/model-cache/model-build/manifest.v2.capnp.bin",
				})
			}
		})
	}
}

func TestRenderMaterializesAgentModelFromBasePodSpec(t *testing.T) {
	t.Parallel()

	t.Log("Project an LPU-only model with template-owned runtime settings")
	snapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	projection := projectRenderFixture(t, PipelineSingle, snapshot)

	t.Log("Construct a base PodSpec with model binding and custom placement")
	modelAnnotationSource := &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
		FieldPath: "metadata.annotations['" + lpxv1alpha1.PodModelAnnotation + "']",
	}}
	conductorPodSpec := renderTestPodSpec()
	conductorPodSpec.NodeSelector = map[string]string{"node-pool": "lpu"}
	conductorPodSpec.Containers[0].Command = []string{"/bin/bash"}
	conductorPodSpec.Containers[0].Args = []string{"-c", "custom-agent"}
	conductorPodSpec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
			NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelHostname, Operator: corev1.NodeSelectorOpIn, Values: []string{"lpu-node-a"},
			}}}},
		},
	}}
	conductorPodSpec.Containers[0].Env = []corev1.EnvVar{{Name: "LPU_MODEL_NAME", ValueFrom: modelAnnotationSource}}

	t.Log("Render conductor and Agent roles from the base PodSpec")
	rendered, err := renderSelectedForTest(renderTestPCS(false), []*ModelProjection{projection}, RenderInput{
		Stages:    map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: conductorPodSpec}},
		Conductor: &corev1.PodTemplateSpec{Spec: *conductorPodSpec.DeepCopy()},
	})
	require.NoError(t, err)

	t.Log("Verify conductor-owned placement and entrypoint behavior")
	conductor := namedClique(t, rendered, "cond")
	require.Equal(t, conductorPodSpec.Affinity, conductor.Spec.PodSpec.Affinity)
	require.Equal(t, conductorPodSpec.NodeSelector, conductor.Spec.PodSpec.NodeSelector)
	require.Equal(t, []string{"/bin/bash"}, conductor.Spec.PodSpec.Containers[0].Command)
	require.Equal(t, []string{"-c", "custom-agent"}, conductor.Spec.PodSpec.Containers[0].Args[:2])
	require.Contains(t, conductor.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "LPU_MODEL_NAME", ValueFrom: modelAnnotationSource,
	})
	require.Equal(t, "agt", testContainerEnvValue(conductor.Spec.PodSpec.Containers[0].Env, allocationEnvVar))
	require.Equal(t, []string{"agt"}, conductor.Spec.StartsAfter)

	t.Log("Verify Agent-owned affinity and logical model binding")
	agent := namedClique(t, rendered, testAgentTemplateName)
	require.Equal(t, conductorPodSpec.Affinity, agent.Spec.PodSpec.Affinity)
	require.Equal(t, conductorPodSpec.NodeSelector, agent.Spec.PodSpec.NodeSelector)
	require.Equal(t, []string{"/bin/bash"}, agent.Spec.PodSpec.Containers[0].Command)
	require.Equal(t, []string{"-c", "custom-agent"}, agent.Spec.PodSpec.Containers[0].Args)
	require.Contains(t, agent.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "LPU_MODEL_NAME", ValueFrom: modelAnnotationSource,
	})
	require.Equal(t, projection.Model(), agent.Annotations[lpxv1alpha1.PodModelAnnotation])
}

func TestRenderSpecDecodeRoleOwnershipAndTemplateSettings(t *testing.T) {
	t.Parallel()

	t.Log("Create distinct draft and target compiler fixtures")
	draftFixture := newV3CompilerFixture()
	draftFixture.pipelineName = runtimeModelDraft
	draftSnapshot := acquireTestSnapshot(t, writeCompilerFixture(t, draftFixture))
	targetFixture := newV3CompilerFixture()
	targetFixture.pipelineName = runtimeModelTarget
	targetSnapshot := acquireTestSnapshot(t, writeCompilerFixture(t, targetFixture))

	t.Log("Keep independent draft, target and conductor templates")
	draft := testLPXComponent(runtimeModelDraft, "draft-build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
	)
	draft.Replicas = ptr.To(int32(2))
	target := testLPXComponent(testTargetStageName, "target-build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
	)
	source := newSelectedTestDGD(t, "specdecode", draft, target)
	stages := make(map[string]corev1.PodTemplateSpec)
	for _, component := range source.Spec.Components {
		template := component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate
		template.Labels = map[string]string{"owner": component.ComponentName}
		template.Annotations = map[string]string{"owner": component.ComponentName}
		template.Spec.Containers[0].Image = component.ComponentName + "-runtime"
		template.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "AUTHORED_STAGE", Value: component.ComponentName}}
		template.Spec.Tolerations = []corev1.Toleration{{Key: "custom.example/stage", Operator: corev1.TolerationOpEqual, Value: component.ComponentName}}
		stages[component.ComponentName] = *template.DeepCopy()
	}
	conductorTemplate := source.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	conductorTemplate.Labels = map[string]string{"owner": "conductor"}
	conductorTemplate.Annotations = map[string]string{"owner": "conductor"}
	conductorTemplate.Spec.Containers[0].Image = "conductor-runtime"
	conductorTemplate.Spec.Containers[0].Env = []corev1.EnvVar{
		{Name: "NOVA_NODE_NAME_TEMPLATE", Value: "${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-{rack}-{node}.${GROVE_HEADLESS_SERVICE}"},
		{Name: "NOVA_PIPELINE_TYPE", Value: "SpecDecode"},
		{Name: "NOVA_MAX_SWA_DKVC_BLOCKS_DRAFT", Value: "2"},
		{Name: "NOVA_AGENT_CONNECT_TIMEOUT", Value: "30s"},
		{Name: "NOVA_AGENT_SETUP_TIMEOUT", Value: "180s"},
	}
	before := source.DeepCopy()

	t.Log("Resolve and render the authored speculative workload")
	selected, err := ResolveWorkload(t.Context(), source, singleGroupComponents(t, source), staticBuildSnapshotSource{
		"draft-build": draftSnapshot, "target-build": targetSnapshot,
	})
	require.NoError(t, err)
	pcs := renderTestPCS(false)
	plan, err := selected.PlanNodeLocalMaterialization(pcs.Name)
	require.NoError(t, err)
	templates, err := RenderNodeLocal(selected, plan, RenderInput{
		Stages: stages, Conductor: conductorTemplate.DeepCopy(),
	})
	require.NoError(t, err)

	t.Log("Retain each authored component's image and metadata on its Agent cliques")
	pcs.Spec.Template.Cliques = templates.Cliques
	for index, stage := range []string{runtimeModelDraft, runtimeModelDraft, testTargetStageName} {
		agent := namedClique(t, pcs, plan.Agents[index].TemplateName)
		require.Equal(t, stage+"-runtime", agent.Spec.PodSpec.Containers[0].Image)
		require.Equal(t, stage, agent.Labels["owner"])
		require.Equal(t, stage, agent.Annotations["owner"])
	}

	t.Log("Mutate Agent 0 and verify sibling and conductor PodSpecs do not alias it")
	firstAgent := namedClique(t, pcs, plan.Agents[0].TemplateName)
	secondAgent := namedClique(t, pcs, plan.Agents[1].TemplateName)
	conductor := namedClique(t, pcs, plan.ConductorTemplate)
	require.Equal(t, "conductor-runtime", conductor.Spec.PodSpec.Containers[0].Image)
	require.Equal(t, "conductor", conductor.Labels["owner"])
	require.Equal(t, "conductor", conductor.Annotations["owner"])
	secondAgentBefore := secondAgent.Spec.PodSpec.DeepCopy()
	conductorBefore := conductor.Spec.PodSpec.DeepCopy()
	require.NotEmpty(t, firstAgent.Spec.PodSpec.Containers[0].Env)
	require.NotEmpty(t, firstAgent.Spec.PodSpec.Tolerations)
	firstAgent.Spec.PodSpec.Containers[0].Env[0].Name = "MUTATED_AGENT_ENV"
	firstAgent.Spec.PodSpec.Tolerations[0].Key = "mutated-agent-toleration"
	require.Equal(t, secondAgentBefore, &secondAgent.Spec.PodSpec)
	require.Equal(t, conductorBefore, &conductor.Spec.PodSpec)

	t.Log("Preserve runtime settings in the conductor template and share only partition data")
	configMap, ok := templates.Resources[0].(*corev1.ConfigMap)
	require.True(t, ok)
	require.NotContains(t, configMap.Data, "model_config.toml")
	for _, env := range conductorTemplate.Spec.Containers[0].Env {
		require.Contains(t, conductor.Spec.PodSpec.Containers[0].Env, env)
	}
	require.Equal(t, "draft0\ndraft1\ntarget", configMap.Data["partition_models"])
	require.Equal(t, before, source)

	t.Log("Bind conductor allocation and Nova hostnames to the renamed Agent cliques")
	require.Equal(t, "agt0:agt1:agt2", testContainerEnvValue(conductor.Spec.PodSpec.Containers[0].Env, allocationEnvVar))
	require.Equal(t, []string{"agt0", "agt1", "agt2"}, conductor.Spec.StartsAfter)
	require.NotContains(t, configMap.Data, "datacenter.toml")
	nodeNameTemplate := testContainerEnvValue(conductor.Spec.PodSpec.Containers[0].Env, "NOVA_NODE_NAME_TEMPLATE")
	for _, agent := range plan.Agents {
		hostname := strings.NewReplacer(
			"${GROVE_PCSG_NAME}", plan.LPXScalingGroup,
			"${GROVE_PCSG_INDEX}", "0",
			"{rack}", agent.TemplateName,
			"{node}", "0",
			"${GROVE_HEADLESS_SERVICE}", pcs.Name,
		).Replace(nodeNameTemplate)
		require.Equal(t, agent.CliqueName+"-0."+pcs.Name, hostname)
	}
}

func projectRenderFixture(t *testing.T, pipeline Pipeline, snapshot *BuildSnapshot) *ModelProjection {
	t.Helper()
	intent := ModelProjectionInput{
		Pipeline: pipeline,
		Models:   []string{"default"}, RuntimeBuildRef: "model-build", BuildSnapshot: normalizeTestSnapshot(t, snapshot),
	}
	projectionBatch, err := appendModelProjections(nil, intent)
	require.NoError(t, err)
	return projectionBatch[0]
}

func renderTestPCS(hybrid bool) *grovev1alpha1.PodCliqueSet {
	pcs := &grovev1alpha1.PodCliqueSet{}
	pcs.Name = "test-pcs"
	pcs.Namespace = "test"
	if !hybrid {
		return pcs
	}
	one := int32(1)
	pcs.Spec.Template.Cliques = []*grovev1alpha1.PodCliqueTemplateSpec{{
		Name:   "cond",
		Labels: map[string]string{"kai.scheduler/queue": "legacy-queue"},
		Spec: grovev1alpha1.PodCliqueSpec{
			RoleName: "cond", Replicas: 1, MinAvailable: &one,
			PodSpec: corev1.PodSpec{
				SchedulerName: corev1.DefaultSchedulerName,
				Volumes: []corev1.Volume{
					{Name: "model-storage", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "model-storage"}}},
				},
				Containers: []corev1.Container{{Name: "main", Image: "cyborg", Env: []corev1.EnvVar{
					{Name: "SERVER_HOSTS_FILE", Value: "/tmp/lpu_servers"},
				}, VolumeMounts: []corev1.VolumeMount{
					{Name: lpuConfigVolumeName, MountPath: "/configs"},
					{Name: "model-storage", MountPath: "/models"},
				}}},
				ResourceClaims: []corev1.PodResourceClaim{{
					Name:                      "candidate",
					ResourceClaimTemplateName: ptr.To("candidate"),
				}},
			},
		},
	}}
	return pcs
}

func renderTestPodSpec() corev1.PodSpec {
	return corev1.PodSpec{
		Volumes: []corev1.Volume{{
			Name: "model-storage",
			VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: "model-storage",
			}},
		},
			{Name: "single-v2-ssh-key", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
			{Name: "ssh-secret", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "ssh-secret"}}},
			{Name: "host-dev", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/dev"}}},
			{Name: "host-sys", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/sys"}}},
		},
		Containers: []corev1.Container{{
			Name: "main", Image: "runtime",
			VolumeMounts: []corev1.VolumeMount{
				{Name: "model-storage", MountPath: "/models"},
				{Name: lpuConfigVolumeName, MountPath: "/configs"},
				{Name: "single-v2-ssh-key", MountPath: "/tmp/dynamo-lpu-ssh"},
				{Name: "ssh-secret", MountPath: "/ssh-pk", ReadOnly: true},
			},
			Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
				corev1.ResourceName("lpu.nvidia.com/devices"): resource.MustParse("1"),
			}},
		}},
	}
}

func namedClique(t *testing.T, pcs *grovev1alpha1.PodCliqueSet, name string) *grovev1alpha1.PodCliqueTemplateSpec {
	t.Helper()
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Name == name {
			return clique
		}
	}
	t.Fatalf("clique %q not found", name)
	return nil
}
