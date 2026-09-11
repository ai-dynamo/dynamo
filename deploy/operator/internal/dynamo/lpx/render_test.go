/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

const (
	testRenderComponentName = "lpu"
	testAgentTemplateName   = "lpu-wkr-m-0"
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
	workload := &SelectedWorkload{
		modelProjections:     projections,
		digest:               digest,
		scalingGroupReplicas: 1,
	}
	plan, err := workload.PlanNodeLocalMaterialization(pcs.Name)
	if err != nil {
		return nil, err
	}
	if workload.BuildFamily() == BuildFamilyXT && workload.Pipeline() == PipelineLPX {
		input.CyborgConfigMap, err = workload.RenderCyborgConfigMap(pcs.Namespace, input.MaterializationName, input.Stages[testRenderComponentName].Spec)
		if err != nil {
			return nil, err
		}
	}
	if _, err := RenderSelectedNodeLocal(pcs, workload, plan, input); err != nil {
		return nil, err
	}
	return pcs, nil
}

func TestRenderResolvesAuthoredMetadataAndMounts(t *testing.T) {
	t.Parallel()

	t.Log("Cover XT configuration mounts and HX hybrid model-storage paths")
	hybrid := newV3CompilerFixture()
	hybrid.compilationMode = manifestcapnp.CompilationMode_lpx
	tests := []struct {
		name     string
		family   lpxv1alpha1.TargetFamily
		pipeline Pipeline
		snapshot *BuildSnapshot
	}{
		{
			name: "XT config mount", family: lpxv1alpha1.TargetFamilyXt8888, pipeline: PipelineSingle,
			snapshot: acquireTestSnapshot(t, writeV2CompilerFixture(t)),
		},
		{
			name: "HX hybrid storage", family: lpxv1alpha1.TargetFamilyHx16x8x2x3, pipeline: PipelineLPX,
			snapshot: acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Seed conflicting runtime annotations and a nondefault mount")
			projection := projectRenderFixture(t, test.family, test.pipeline, test.snapshot)
			pcs := renderTestPCS(test.pipeline == PipelineLPX)
			pcs.Annotations = map[string]string{
				ExecutionBackendAnnotation: "stale",
			}
			for _, clique := range pcs.Spec.Template.Cliques {
				clique.Annotations = map[string]string{ExecutionBackendAnnotation: "stale"}
			}
			template := corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
					"user":                         "kept",
					lpxv1alpha1.PodModelAnnotation: "spoofed",
					ExecutionBackendAnnotation:     "stale",
				}},
				Spec: renderTestPodSpec(),
			}
			if test.family == lpxv1alpha1.TargetFamilyXt8888 {
				template.Spec.Containers[0].VolumeMounts = append(template.Spec.Containers[0].VolumeMounts,
					corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: "/custom"})
			} else {
				template.Spec.Containers[0].VolumeMounts[0].MountPath = "/tmp"
			}

			t.Log("Render into the fresh PCS without retaining stale runtime identity")
			rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, RenderInput{
				MaterializationName: "test-dgd",
				Stages:              map[string]corev1.PodTemplateSpec{testRenderComponentName: template},
				SSHSecretName:       "ssh-secret",
			})
			require.NoError(t, err)
			require.Same(t, pcs, rendered)
			require.Equal(t, projection.Digest().String(), rendered.Annotations[WorkloadDigestAnnotation])
			require.NotContains(t, rendered.Annotations, ExecutionBackendAnnotation)
			for _, clique := range rendered.Spec.Template.Cliques {
				require.NotContains(t, clique.Annotations, ExecutionBackendAnnotation)
			}
			agent := namedClique(t, rendered, testAgentTemplateName)
			require.Equal(t, projection.Model(), agent.Annotations[lpxv1alpha1.PodModelAnnotation])
			require.Equal(t, projection.Digest().String(), agent.Annotations[WorkloadDigestAnnotation])

			t.Log("Resolve runtime mounts using the authored storage path")
			if test.family == lpxv1alpha1.TargetFamilyXt8888 {
				conductor := namedClique(t, rendered, "lpu-ldr")
				require.Equal(t, "kept", conductor.Annotations["user"])
				require.Contains(t, conductor.Spec.PodSpec.Containers[0].VolumeMounts,
					corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: lpuConfigMountPath})
			} else {
				cyborg := namedClique(t, rendered, "lpu-engine-gpu").Spec.PodSpec.Containers[0]
				require.Contains(t, cyborg.VolumeMounts, corev1.VolumeMount{Name: "model-storage", MountPath: "/tmp"})
				require.Contains(t, cyborg.Env, corev1.EnvVar{
					Name: "GBUILD_MANIFEST_PATH", Value: "/tmp/model-build/manifest.v2.capnp.bin",
				})
			}
		})
	}
}

func TestRenderMaterializesAgentModelFromBasePodSpec(t *testing.T) {
	t.Parallel()

	t.Log("Project an LPU-only model with nested runtime settings")
	snapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	projection := projectRenderFixture(t, lpxv1alpha1.TargetFamilyXt8888, PipelineSingle, snapshot)
	projection.configuredBuild.runtimeSettings["tokenizer_path"] = "tokenizer"
	projection.configuredBuild.runtimeSettings["stop_tokens"] = []uint32{1}
	projection.configuredBuild.runtimeSettings["swa"] = map[string]any{"chunked": true}
	projection.configuredBuild.runtimeSettings["scheduler"] = map[string]any{"max_inflight_tasks": int64(3)}
	retainedSetup := map[string]any{"custom": "kept"}
	projection.configuredBuild.runtimeSettings["setup"] = retainedSetup

	t.Log("Render and verify isolated model configuration state")
	firstConfig, err := lpuModelConfig([]*ModelProjection{projection}, "/models")
	require.NoError(t, err)
	require.EqualValues(t, 3, firstConfig["scheduler"].(map[string]any)["max_inflight_tasks"])
	require.NotContains(t, firstConfig["iop"], "scheduler")
	require.NotContains(t, firstConfig["iop"], "setup")

	t.Log("Mutate one rendered config and verify the next render is isolated")
	firstConfig["setup"].(map[string]any)["mutated"] = true
	secondConfig, err := lpuModelConfig([]*ModelProjection{projection}, "/models")
	require.NoError(t, err)
	require.NotContains(t, secondConfig["setup"], "mutated")
	require.Equal(t, map[string]any{"custom": "kept"}, retainedSetup)
	require.Equal(t, map[string]any{"chunked": true}, projection.configuredBuild.runtimeSettings["swa"])

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
	conductorPodSpec.Containers[0].Env = []corev1.EnvVar{{Name: lpuModelNameEnvVar, ValueFrom: modelAnnotationSource}}

	t.Log("Render conductor and Agent roles from the base PodSpec")
	rendered, err := renderSelectedForTest(renderTestPCS(false), []*ModelProjection{projection}, RenderInput{
		MaterializationName: "test-dgd",
		Stages:              map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: conductorPodSpec}},
		SSHSecretName:       "ssh-secret",
	})
	require.NoError(t, err)

	t.Log("Verify conductor-owned placement and entrypoint behavior")
	conductor := namedClique(t, rendered, "lpu-ldr")
	require.Nil(t, conductor.Spec.PodSpec.Affinity)
	require.Equal(t, conductorPodSpec.NodeSelector, conductor.Spec.PodSpec.NodeSelector)
	require.Equal(t, []string{"/bin/sh", "-ec"}, conductor.Spec.PodSpec.Containers[0].Command)
	require.Equal(t, []string{"/bin/bash", "-c", "custom-agent"}, conductor.Spec.PodSpec.Containers[0].Args[4:7])
	require.Equal(t, modelAnnotationSource, conductor.Spec.PodSpec.Containers[0].Env[0].ValueFrom)
	require.Empty(t, conductor.Spec.PodSpec.Containers[0].Env[0].Value)

	t.Log("Verify Agent-owned affinity and logical model binding")
	agent := namedClique(t, rendered, testAgentTemplateName)
	require.Equal(t, conductorPodSpec.Affinity, agent.Spec.PodSpec.Affinity)
	require.Equal(t, conductorPodSpec.NodeSelector, agent.Spec.PodSpec.NodeSelector)
	require.Equal(t, []string{"/bin/bash"}, agent.Spec.PodSpec.Containers[0].Command)
	require.Equal(t, []string{"-c", "custom-agent"}, agent.Spec.PodSpec.Containers[0].Args)
	require.Contains(t, agent.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name:  lpuModelNameEnvVar,
		Value: projection.Model(),
	})
}

func TestRenderSpecDecodeRoleOwnershipAndSharedSettings(t *testing.T) {
	t.Parallel()

	t.Log("Create distinct draft and target compiler fixtures")
	draftFixture := newV3CompilerFixture()
	draftFixture.pipelineName = runtimeModelDraft
	draftSnapshot := acquireTestSnapshot(t, writeCompilerFixture(t, draftFixture))
	targetFixture := newV3CompilerFixture()
	targetFixture.pipelineName = runtimeModelTarget
	targetSnapshot := acquireTestSnapshot(t, writeCompilerFixture(t, targetFixture))

	t.Log("Keep independent draft and target templates, with the conductor seeded by target")
	draft := testLPXComponent(draftStageName, "draft-build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
	)
	draft.Replicas = ptr.To(int32(2))
	draft.LPX.Settings = &apiextensionsv1.JSON{Raw: []byte(`{
		"max_swa_dkvc_blocks_draft":2,
		"setup":{"agent_connect_timeout":"30s","agent_setup_timeout":"180s"}
	}`)}
	target := testLPXComponent(testTargetStageName, "target-build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor},
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
	)
	source := newSelectedTestDGD(t, "specdecode", draft, target)
	stages := make(map[string]corev1.PodTemplateSpec)
	for _, component := range source.Spec.Components {
		template := component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate
		template.Labels = map[string]string{"owner": component.ComponentName}
		template.Annotations = map[string]string{"owner": component.ComponentName}
		template.Spec.Containers[0].Image = component.ComponentName + "-runtime"
		stages[component.ComponentName] = *template.DeepCopy()
	}
	before := source.DeepCopy()

	t.Log("Resolve and render the authored speculative workload")
	selected, err := ResolveSelectedWorkload(t.Context(), source, staticBuildSnapshotSource{
		"draft-build": draftSnapshot, "target-build": targetSnapshot,
	})
	require.NoError(t, err)
	pcs := renderTestPCS(false)
	plan, err := selected.PlanNodeLocalMaterialization(pcs.Name)
	require.NoError(t, err)
	extraResources, err := RenderSelectedNodeLocal(pcs, selected, plan, RenderInput{
		MaterializationName: source.Name, Stages: stages, SSHSecretName: "ssh-secret",
	})
	require.NoError(t, err)

	t.Log("Retain each authored component's image and metadata on its Agent cliques")
	for index, stage := range []string{draftStageName, draftStageName, testTargetStageName} {
		agent := namedClique(t, pcs, plan.Agents[index].TemplateName)
		require.Equal(t, stage+"-runtime", agent.Spec.PodSpec.Containers[0].Image)
		require.Equal(t, stage, agent.Labels["owner"])
		require.Equal(t, stage, agent.Annotations["owner"])
	}

	t.Log("Mutate Agent 0 and verify sibling and conductor PodSpecs do not alias it")
	firstAgent := namedClique(t, pcs, plan.Agents[0].TemplateName)
	secondAgent := namedClique(t, pcs, plan.Agents[1].TemplateName)
	conductor := namedClique(t, pcs, plan.ConductorTemplate)
	require.Equal(t, "target-runtime", conductor.Spec.PodSpec.Containers[0].Image)
	require.Equal(t, testTargetStageName, conductor.Labels["owner"])
	require.Equal(t, testTargetStageName, conductor.Annotations["owner"])
	secondAgentBefore := secondAgent.Spec.PodSpec.DeepCopy()
	conductorBefore := conductor.Spec.PodSpec.DeepCopy()
	require.NotEmpty(t, firstAgent.Spec.PodSpec.Containers[0].Env)
	require.NotEmpty(t, firstAgent.Spec.PodSpec.Tolerations)
	firstAgent.Spec.PodSpec.Containers[0].Env[0].Name = "MUTATED_AGENT_ENV"
	firstAgent.Spec.PodSpec.Tolerations[0].Key = "mutated-agent-toleration"
	require.Equal(t, secondAgentBefore, &secondAgent.Spec.PodSpec)
	require.Equal(t, conductorBefore, &conductor.Spec.PodSpec)

	t.Log("Project draft settings once into the shared SpecDecode configuration")
	configMap, ok := extraResources[0].(*corev1.ConfigMap)
	require.True(t, ok)
	require.Contains(t, configMap.Data["model_config.toml"], "SpecDecode")
	require.Contains(t, configMap.Data["model_config.toml"], "[draft")
	require.Contains(t, configMap.Data["model_config.toml"], "[target")
	require.Contains(t, configMap.Data["model_config.toml"], "num_drafts = 2")
	require.Contains(t, configMap.Data["model_config.toml"], "max_swa_dkvc_blocks_draft = 2")
	require.Equal(t, 1, strings.Count(configMap.Data["model_config.toml"], "agent_connect_timeout = '30s'"))
	require.Equal(t, 1, strings.Count(configMap.Data["model_config.toml"], "agent_setup_timeout = '180s'"))
	require.Equal(t, 1, strings.Count(configMap.Data["model_config.toml"], "[draft.iop]"))
	require.Equal(t, "draft0\ndraft1\ntarget", configMap.Data["partition_models"])
	require.Equal(t, before, source)
}

func projectRenderFixture(t *testing.T, family lpxv1alpha1.TargetFamily, pipeline Pipeline, snapshot *BuildSnapshot) *ModelProjection {
	t.Helper()
	intent := ModelProjectionInput{
		Pipeline: pipeline,
		Models:   []string{"default"}, RuntimeBuildRef: "model-build", BuildSnapshot: normalizeTestSnapshot(t, snapshot),
	}
	if family == lpxv1alpha1.TargetFamilyXt8888 {
		intent.ModelSettings = json.RawMessage(`{"prop_sync":true}`)
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
		Name:   "lpu-engine-gpu",
		Labels: map[string]string{"kai.scheduler/queue": "legacy-queue"},
		Spec: grovev1alpha1.PodCliqueSpec{
			RoleName: "lpu-engine-gpu", Replicas: 1, MinAvailable: &one,
			PodSpec: corev1.PodSpec{
				SchedulerName: corev1.DefaultSchedulerName,
				Containers: []corev1.Container{{Name: "main", Image: "cyborg", Env: []corev1.EnvVar{
					{Name: selectedCyborgServerHostsFileEnv, Value: "/tmp/lpu_servers"},
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
		}},
		Containers: []corev1.Container{{
			Name: "main", Image: "runtime",
			VolumeMounts: []corev1.VolumeMount{{Name: "model-storage", MountPath: "/models"}},
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
