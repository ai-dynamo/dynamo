/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
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

	// Only a hybrid fixture has a conductor: the Cyborg clique of the test PCS.
	var conductorReplicas int32
	if projections[0].pipeline == PipelineHybrid {
		if input.Cyborg == nil {
			input.Cyborg = &CliqueTemplate{Template: pcs.Spec.Template.Cliques[0]}
		}
		conductorReplicas = input.Cyborg.Template.Spec.Replicas
	}
	workload, err := newWorkload(pcs.Name, false, projections, 1, conductorReplicas)
	if err != nil {
		return nil, err
	}
	templates, err := RenderNodeLocal(workload, input)
	if err != nil {
		return nil, err
	}
	pcs.Spec.Template.Cliques = templates.Cliques
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{templates.ScalingGroup}
	return pcs, nil
}

func TestRenderFinalizesModelPathBindings(t *testing.T) {
	for _, pipeline := range []Pipeline{PipelineSingle, PipelineHybrid} {
		t.Run(string(pipeline), func(t *testing.T) {
			t.Log("Retain graph dependencies until model paths are bound to every final LPX role")
			projection := projectRenderFixture(t, pipeline, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
			pcs := renderTestPCS(pipeline == PipelineHybrid)
			template := corev1.PodTemplateSpec{Spec: renderTestPodSpec()}
			template.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "LPX_MODEL_PATH", Value: "$(MODEL)"}}
			pods := []*corev1.PodSpec{&template.Spec}
			if pipeline == PipelineHybrid {
				pod := &pcs.Spec.Template.Cliques[0].Spec.PodSpec
				pod.Containers[0].Env = append(pod.Containers[0].Env, template.Spec.Containers[0].Env...)
				pods = append(pods, pod)
			}
			collectors := make([]*semanticenv.ContainerLayers, len(pods))
			for i, pod := range pods {
				pod.Containers = append(pod.Containers, corev1.Container{Name: "observer", Env: []corev1.EnvVar{{Name: "LPX_MODEL_PATH", Value: "$(MODEL)"}}})
				pod.InitContainers = append(pod.InitContainers, corev1.Container{Name: "prepare", Env: []corev1.EnvVar{{Name: "LPX_MODEL_PATH", Value: "$(MODEL)"}}})
				layers := &semanticenv.ContainerLayers{}
				for _, containers := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
					for j := range containers {
						container := &containers[j]
						layers.Register(container,
							semanticenv.Layer{Source: "spec.env", Env: []corev1.EnvVar{{Name: "MODEL", Value: "$(LPX_MODEL_PATH)/model"}}},
							semanticenv.Layer{Source: "container[name=" + container.Name + "].env", Env: container.Env},
						)
					}
				}
				collectors[i] = layers
			}
			input := RenderInput{
				Stages: map[string]PodTemplate{testRenderComponentName: {Template: template, Environment: collectors[0]}},
			}
			if pipeline == PipelineHybrid {
				input.Cyborg = &CliqueTemplate{Template: pcs.Spec.Template.Cliques[0], Environment: collectors[1]}
			}
			rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, input)
			require.NoError(t, err)
			for _, clique := range rendered.Spec.Template.Cliques {
				for _, containers := range [][]corev1.Container{clique.Spec.PodSpec.Containers, clique.Spec.PodSpec.InitContainers} {
					for _, container := range containers {
						var relevant []string
						for _, env := range container.Env {
							if env.Name == "LPX_MODEL_PATH" || env.Name == "MODEL" {
								relevant = append(relevant, env.Name)
							}
						}
						require.Equal(t, []string{"LPX_MODEL_PATH", "MODEL"}, relevant, "%s/%s", clique.Name, container.Name)
					}
				}
			}
		})
	}
}

func TestRenderResolvesAuthoredMetadataAndMounts(t *testing.T) {
	t.Parallel()

	t.Log("Cover shared configuration overrides and HX hybrid model-storage paths")
	hybrid := newV3CompilerFixture()

	xtSnapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	hxSnapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	tests := []struct {
		name       string
		pipeline   Pipeline
		snapshot   *Build
		configPath string
	}{
		{
			name: "XT config mount", pipeline: PipelineSingle,
			snapshot: xtSnapshot, configPath: "/etc/runtime",
		},
		{
			name: "XT omitted config mount", pipeline: PipelineSingle,
			snapshot: xtSnapshot,
		},
		{
			name: "HX omitted config mount", pipeline: PipelineSingle,
			snapshot: hxSnapshot,
		},
		{
			name: "HX custom config mount", pipeline: PipelineSingle,
			snapshot: hxSnapshot, configPath: "/etc/custom-runtime",
		},
		{
			name: "HX hybrid storage", pipeline: PipelineHybrid,
			snapshot: acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)), configPath: "",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Seed conflicting runtime annotations and a nondefault mount")
			projection := projectRenderFixture(t, test.pipeline, test.snapshot)
			require.NotEqual(t, projection.Digest().String(), projection.CompilerSnapshotDigest())
			pcs := renderTestPCS(test.pipeline == PipelineHybrid)
			for _, clique := range pcs.Spec.Template.Cliques {
				clique.Annotations = map[string]string{
					lpxv1alpha1.CompilerSnapshotDigestAnnotation: "stale",
				}
			}
			template := corev1.PodTemplateSpec{
				Annotations: map[string]string{
					"user":                         "kept",
					lpxv1alpha1.PodModelAnnotation: "spoofed",
					lpxv1alpha1.CompilerSnapshotDigestAnnotation: "stale",
				},
				Spec: renderTestPodSpec(),
			}
			template.Spec.Tolerations = []corev1.Toleration{
				{Key: "cluster.example/custom", Operator: corev1.TolerationOpExists},
				{Key: "lpu.nvidia.com/node", Operator: corev1.TolerationOpExists},
			}
			if test.configPath != "" {
				template.Spec.Volumes = append(template.Spec.Volumes, corev1.Volume{
					Name: "runtime-config",
					ConfigMap: &corev1.ConfigMapVolumeSource{
						Name: "authored-config", DefaultMode: ptr.To[int32](0440),
					},
				})
			}
			if test.configPath != "" {
				template.Spec.Containers[0].VolumeMounts = append(template.Spec.Containers[0].VolumeMounts,
					corev1.VolumeMount{Name: "runtime-config", MountPath: test.configPath})
			}
			if test.pipeline == PipelineHybrid {
				template.Spec.Containers[0].VolumeMounts[0].MountPath = "/model-cache"
				pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].VolumeMounts[0].MountPath = "/model-cache"
			}

			t.Log("Render into the fresh PCS without retaining stale runtime identity")
			rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, RenderInput{
				Stages: map[string]PodTemplate{testRenderComponentName: {Template: template}},
			})
			require.NoError(t, err)
			require.Same(t, pcs, rendered)
			require.Equal(t, projection.Digest().String(), rendered.Spec.Template.PodCliqueScalingGroupConfigs[0].Annotations[WorkloadDigestAnnotation])
			for _, clique := range rendered.Spec.Template.Cliques {
				require.Equal(t, "main", clique.Spec.PodSpec.Containers[0].Name)
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
			if test.pipeline != PipelineHybrid {
				require.Len(t, rendered.Spec.Template.Cliques, 1)
				require.Equal(t, "kept", agent.Annotations["user"])
				require.Equal(t, template.Spec.Containers[0].VolumeMounts, agent.Spec.PodSpec.Containers[0].VolumeMounts)
			} else {
				cyborg := namedClique(t, rendered, "cond").Spec.PodSpec.Containers[0]
				require.Contains(t, cyborg.VolumeMounts, corev1.VolumeMount{Name: "model-storage", MountPath: "/model-cache"})
				require.Contains(t, cyborg.Env, corev1.EnvVar{
					Name: "GBUILD_MANIFEST_PATH", Value: "/model-cache/model-build/deployment.v1.capnp.bin",
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
	podSpec := renderTestPodSpec()
	podSpec.NodeSelector = map[string]string{"node-pool": "lpu"}
	podSpec.Containers[0].Command = []string{"/bin/bash"}
	podSpec.Containers[0].Args = []string{"-c", "custom-agent"}
	podSpec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
			NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchExpressions: []corev1.NodeSelectorRequirement{{
				Key: corev1.LabelHostname, Operator: corev1.NodeSelectorOpIn, Values: []string{"lpu-node-a"},
			}}}},
		},
	}}
	podSpec.Containers[0].Env = []corev1.EnvVar{{Name: "LPU_MODEL_NAME", ValueFrom: modelAnnotationSource}}

	t.Log("Render the Agent role, and no other role, from the base PodSpec")
	rendered, err := renderSelectedForTest(renderTestPCS(false), []*ModelProjection{projection}, RenderInput{
		Stages: map[string]PodTemplate{testRenderComponentName: {Template: corev1.PodTemplateSpec{Spec: podSpec}}},
	})
	require.NoError(t, err)
	require.Len(t, rendered.Spec.Template.Cliques, 1)
	require.Equal(t, []string{testAgentTemplateName}, rendered.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames)

	t.Log("Verify Agent-owned affinity and logical model binding")
	agent := namedClique(t, rendered, testAgentTemplateName)
	require.Equal(t, podSpec.Affinity, agent.Spec.PodSpec.Affinity)
	require.Equal(t, podSpec.NodeSelector, agent.Spec.PodSpec.NodeSelector)
	require.Equal(t, []string{"/bin/bash"}, agent.Spec.PodSpec.Containers[0].Command)
	require.Equal(t, []string{"-c", "custom-agent"}, agent.Spec.PodSpec.Containers[0].Args)
	require.Contains(t, agent.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "LPU_MODEL_NAME", ValueFrom: modelAnnotationSource,
	})
	require.Equal(t, projection.Model(), agent.Annotations[lpxv1alpha1.PodModelAnnotation])
	require.Empty(t, agent.Spec.StartsAfter)
}

func TestRenderSpecDecodeRoleOwnershipAndTemplateSettings(t *testing.T) {
	t.Parallel()

	t.Log("Create distinct draft and target compiler fixtures")
	draftFixture := newV3CompilerFixture()

	draftSnapshot := acquireTestSnapshot(t, writeCompilerFixture(t, draftFixture))
	targetFixture := newV3CompilerFixture()

	targetSnapshot := acquireTestSnapshot(t, writeCompilerFixture(t, targetFixture))

	t.Log("Keep independent draft and target templates; the target serves through its agent runtime sidecar")
	draft := testLPXComponent(runtimeModelDraft, "draft-build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
	)
	draft.Replicas = new(int32(2))
	target := testLPXComponent(testTargetStageName, "target-build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
	)
	target.Roles[0].PodTemplate.Spec.InitContainers = []corev1.Container{
		{Name: commonconsts.RuntimeContainerName, Image: "sidecar", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)},
	}
	source := newSelectedTestDGD(t, "specdecode", draft, target)

	t.Log("Keep role-named sidecars and explicit main-container references in every template")
	for _, component := range source.Spec.Components {
		for _, role := range component.Roles {
			spec := &role.PodTemplate.Spec
			spec.Containers = append(spec.Containers, corev1.Container{
				Name: "agent", Image: "sidecar", Env: []corev1.EnvVar{{
					Name: "MAIN_CPU", ValueFrom: &corev1.EnvVarSource{ResourceFieldRef: &corev1.ResourceFieldSelector{
						ContainerName: "main", Resource: "limits.cpu",
					}},
				}},
			})
			spec.InitContainers = append(spec.InitContainers, corev1.Container{Name: "conductor", Image: "setup"})
			spec.Volumes = append(spec.Volumes, corev1.Volume{Name: "resources",
				DownwardAPI: &corev1.DownwardAPIVolumeSource{Items: []corev1.DownwardAPIVolumeFile{{
					Path: "main-memory", ResourceFieldRef: &corev1.ResourceFieldSelector{
						ContainerName: "main", Resource: "limits.memory",
					},
				}}}})
		}
	}

	t.Log("Assign distinct runtime settings to each component")
	stages := make(map[string]PodTemplate)
	for _, component := range source.Spec.Components {
		template := component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate
		template.Labels = map[string]string{"owner": component.ComponentName}
		template.Annotations = map[string]string{"owner": component.ComponentName}
		template.Spec.Containers[0].Image = component.ComponentName + "-runtime"
		template.Spec.Containers[0].Env = []corev1.EnvVar{
			{Name: "AUTHORED_STAGE", Value: component.ComponentName},
			{Name: "CONTAINER_NAME", Value: "main"},
		}
		template.Spec.Tolerations = []corev1.Toleration{{Key: "custom.example/stage", Operator: corev1.TolerationOpEqual, Value: component.ComponentName}}
		stages[component.ComponentName] = PodTemplate{Template: *template.DeepCopy()}
	}
	before := source.DeepCopy()

	t.Log("Resolve and render the authored speculative workload")
	pcs := renderTestPCS(false)
	selected, err := resolveSoleWorkload(t, fake.NewClientBuilder().Build(), source, staticModelRegistry{
		"draft-build": draftSnapshot, "target-build": targetSnapshot,
	})
	require.NoError(t, err)
	templates, err := RenderNodeLocal(selected, RenderInput{Stages: stages})
	require.NoError(t, err)
	require.Equal(t, before, source)

	t.Log("Render one Agent clique for each model, and no conductor")
	pcs.Spec.Template.Cliques = templates.Cliques
	require.Len(t, templates.Cliques, 3)
	require.Equal(t, []string{"agt0", "agt1", "agt2"}, templates.ScalingGroup.CliqueNames)

	t.Log("Retain each authored component's image, metadata, and companion containers on its Agent cliques")
	for index, stage := range []string{runtimeModelDraft, runtimeModelDraft, testTargetStageName} {
		agent := namedClique(t, pcs, templates.ScalingGroup.CliqueNames[index])
		spec := &agent.Spec.PodSpec
		require.Equal(t, lpxv1alpha1.PodRoleAgent, agent.Annotations[lpxv1alpha1.PodRoleAnnotation])
		require.Empty(t, agent.Spec.StartsAfter)
		require.Equal(t, stage+"-runtime", spec.Containers[0].Image)
		require.Equal(t, stage, agent.Labels["owner"])
		require.Equal(t, stage, agent.Annotations["owner"])
		require.Equal(t, "main", spec.Containers[0].Name)
		require.Equal(t, "main", testContainerEnvValue(spec.Containers[0].Env, "CONTAINER_NAME"))

		// The workload's model paths precede the authored references of every companion container.
		authored := source.GetComponentByName(stage).ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.DeepCopy()
		for _, containers := range [][]corev1.Container{authored.Containers[1:], authored.InitContainers} {
			for i := range containers {
				containers[i].Env = append([]corev1.EnvVar{
					{Name: "LPX_DRAFT_MODEL_PATH", Value: "/models/draft-build"},
					{Name: "LPX_TARGET_MODEL_PATH", Value: "/models/target-build"},
				}, containers[i].Env...)
			}
		}
		require.Equal(t, authored.Containers[1:], spec.Containers[1:])
		require.Equal(t, authored.InitContainers, spec.InitContainers)
		require.Contains(t, spec.Volumes, authored.Volumes[len(authored.Volumes)-1])
	}

	t.Log("Mutate Agent 0 and verify that sibling and target PodSpecs do not alias it")
	firstAgent := namedClique(t, pcs, "agt0")
	secondAgent := namedClique(t, pcs, "agt1")
	targetAgent := namedClique(t, pcs, "agt2")
	secondAgentBefore := secondAgent.Spec.PodSpec.DeepCopy()
	targetAgentBefore := targetAgent.Spec.PodSpec.DeepCopy()
	require.NotEmpty(t, firstAgent.Spec.PodSpec.Containers[0].Env)
	require.NotEmpty(t, firstAgent.Spec.PodSpec.Tolerations)
	firstAgent.Spec.PodSpec.Containers[0].Env[0].Name = "MUTATED_AGENT_ENV"
	firstAgent.Spec.PodSpec.Tolerations[0].Key = "mutated-agent-toleration"
	require.Equal(t, secondAgentBefore, &secondAgent.Spec.PodSpec)
	require.Equal(t, targetAgentBefore, &targetAgent.Spec.PodSpec)
}

func projectRenderFixture(t *testing.T, pipeline Pipeline, snapshot *Build) *ModelProjection {
	t.Helper()
	intent := ModelProjectionInput{
		Pipeline: pipeline,
		Models:   []string{"default"}, RuntimeBuildRef: "model-build", Build: snapshot,
	}
	projectionBatch, err := projectModels(intent)
	require.NoError(t, err)
	return projectionBatch[0]
}

func TestRenderSpecDecodeRequiresTheServingStoragePath(t *testing.T) {
	t.Parallel()

	t.Log("Resolve a speculative pair whose draft mounts model storage at another path")
	draft := testLPXComponent(runtimeModelDraft, "build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}})
	draft.Roles[0].PodTemplate.Spec.Containers[0].VolumeMounts[0].MountPath = "/draft-models"
	target := testLPXComponent(testTargetStageName, "build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}})
	target.Roles[0].PodTemplate.Spec.InitContainers = []corev1.Container{
		{Name: commonconsts.RuntimeContainerName, Image: "sidecar", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)},
	}
	workload, err := resolveSoleWorkload(t, fake.NewClientBuilder().Build(), newSelectedTestDGD(t, "specdecode", draft, target),
		staticModelRegistry{"build": acquireTestSnapshot(t, writeV3CompilerFixture(t))})
	require.NoError(t, err)

	t.Log("Reject the draft, because the serving component owns the model storage path")
	_, err = RenderNodeLocal(workload, RenderInput{Stages: map[string]PodTemplate{
		runtimeModelDraft:   {Template: *draft.Roles[0].PodTemplate},
		testTargetStageName: {Template: *target.Roles[0].PodTemplate},
	}})
	require.EqualError(t, err, `stage draft must use the serving model-storage mount path "/models"`)
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
					{Name: "model-storage", PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "model-storage"}},
				},
				Containers: []corev1.Container{{Name: "main", Image: "cyborg", Env: []corev1.EnvVar{
					{Name: "SERVER_HOSTS_FILE", Value: "/tmp/lpu_servers"},
				}, VolumeMounts: []corev1.VolumeMount{
					{Name: "model-storage", MountPath: "/models"},
				}}},
				ResourceClaims: []corev1.PodResourceClaim{{
					Name:                      "candidate",
					ResourceClaimTemplateName: new("candidate"),
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
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: "model-storage",
			},
		},
			{Name: "single-v2-ssh-key", EmptyDir: &corev1.EmptyDirVolumeSource{}},
			{Name: "ssh-secret", Secret: &corev1.SecretVolumeSource{SecretName: "ssh-secret"}},
			{Name: "host-dev", HostPath: &corev1.HostPathVolumeSource{Path: "/dev"}},
			{Name: "host-sys", HostPath: &corev1.HostPathVolumeSource{Path: "/sys"}},
		},
		Containers: []corev1.Container{{
			Name: "main", Image: "runtime",
			VolumeMounts: []corev1.VolumeMount{
				{Name: "model-storage", MountPath: "/models"},
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
