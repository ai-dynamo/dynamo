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
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testRenderComponentName = "lpu"
	testAgentTemplateName   = "agt"
	testTargetStageName     = "target"
)

// renderSelectedForTest renders projections into pcs and returns it with the
// workload's runtime resources.
func renderSelectedForTest(
	pcs *grovev1alpha1.PodCliqueSet,
	projections []*ModelProjection,
	input RenderInput,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	// Attach test projections to their authored stage without inventing templates.
	for _, projection := range projections {
		projection.stage = testRenderComponentName
	}
	digest, err := workloadSetDigest(projections)
	if err != nil {
		return nil, nil, err
	}
	workload := &Workload{
		modelProjections:     projections,
		digest:               digest,
		scalingGroupReplicas: 1,
	}
	plan, err := workload.PlanNodeLocalMaterialization(pcs.Name)
	if err != nil {
		return nil, nil, err
	}
	if workload.Pipeline() == PipelineHybrid {
		input.Cyborg = pcs.Spec.Template.Cliques[0]
	}
	templates, err := RenderNodeLocal(workload, plan, input)
	if err != nil {
		return nil, nil, err
	}
	pcs.Spec.Template.Cliques = templates.Cliques
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{templates.ScalingGroup}
	return pcs, templates.Resources, nil
}

func TestRenderResolvesAuthoredMetadataAndMounts(t *testing.T) {
	t.Parallel()

	t.Log("Cover shared configuration overrides and HX hybrid model-storage paths")
	hybrid := newV3CompilerFixture()
	hybrid.compilationMode = manifestcapnp.CompilationMode_lpx
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
			snapshot: xtSnapshot, configPath: "/custom",
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
			snapshot: hxSnapshot, configPath: "/custom",
		},
		{
			name: "HX hybrid storage", pipeline: PipelineHybrid,
			snapshot: acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)), configPath: "/configs",
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
					Name: lpuConfigVolumeName,
					ConfigMap: &corev1.ConfigMapVolumeSource{
						Name: "authored-config", DefaultMode: ptr.To[int32](0440),
					},
				})
			}
			template.Spec.Containers[0].VolumeMounts = slices.DeleteFunc(template.Spec.Containers[0].VolumeMounts,
				func(mount corev1.VolumeMount) bool { return mount.Name == lpuConfigVolumeName })
			if test.configPath != "" {
				template.Spec.Containers[0].VolumeMounts = append(template.Spec.Containers[0].VolumeMounts,
					corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: test.configPath})
			}
			if test.pipeline == PipelineHybrid {
				template.Spec.Containers[0].VolumeMounts[0].MountPath = "/model-cache"
				pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].VolumeMounts[1].MountPath = "/model-cache"
			}

			t.Log("Render into the fresh PCS without retaining stale runtime identity")
			rendered, _, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, RenderInput{
				Stages:    map[string]corev1.PodTemplateSpec{testRenderComponentName: template},
				Conductor: template.DeepCopy(),
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
				conductor := namedClique(t, rendered, "cond")
				require.Equal(t, "kept", conductor.Annotations["user"])
				require.Subset(t, conductor.Spec.PodSpec.Volumes, template.Spec.Volumes)
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
	rendered, _, err := renderSelectedForTest(renderTestPCS(false), []*ModelProjection{projection}, RenderInput{
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
	draft.Replicas = new(int32(2))
	target := testLPXComponent(testTargetStageName, "target-build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: renderTestPodSpec()}},
	)
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

	t.Log("Assign distinct runtime settings to each component and conductor")
	stages := make(map[string]corev1.PodTemplateSpec)
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
		stages[component.ComponentName] = *template.DeepCopy()
	}
	conductorTemplate := source.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	conductorTemplate.Labels = map[string]string{"owner": "conductor"}
	conductorTemplate.Annotations = map[string]string{"owner": "conductor"}
	conductorTemplate.Spec.Containers[0].Image = "conductor-runtime"
	conductorResources := corev1.ResourceList{
		v2LPUResourceName: resource.MustParse("3"),
		v3LPUResourceName: resource.MustParse("5"),
		corev1.ResourceName("lpu.nvidia.com/devices"): resource.MustParse("1"),
	}
	conductorTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{
		Requests: conductorResources.DeepCopy(), Limits: conductorResources.DeepCopy(),
	}
	conductorTemplate.Spec.Containers[0].Env = []corev1.EnvVar{
		{Name: "CONTAINER_NAME", Value: "main"},
		{Name: "NOVA_NODE_NAME_TEMPLATE", Value: "${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-{rack}-{node}.${GROVE_HEADLESS_SERVICE}"},
		{Name: "NOVA_PIPELINE_TYPE", Value: "SpecDecode"},
		{Name: "NOVA_MAX_SWA_DKVC_BLOCKS_DRAFT", Value: "2"},
		{Name: "NOVA_AGENT_CONNECT_TIMEOUT", Value: "30s"},
		{Name: "NOVA_AGENT_SETUP_TIMEOUT", Value: "180s"},
	}
	before := source.DeepCopy()

	t.Log("Resolve and render the authored speculative workload")
	selected, err := ResolveWorkload(t.Context(), fake.NewClientBuilder().Build(), source, singleGroupComponents(t, source), staticModelRegistry{
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

	t.Log("Preserve container identity and references while materializing distinct Pod roles")
	for _, clique := range templates.Cliques {
		spec := &clique.Spec.PodSpec
		require.Equal(t, "main", spec.Containers[0].Name)
		require.Equal(t, "main", testContainerEnvValue(spec.Containers[0].Env, "CONTAINER_NAME"))
		require.Equal(t, conductorTemplate.Spec.Containers[1:], spec.Containers[1:])
		require.Equal(t, conductorTemplate.Spec.InitContainers, spec.InitContainers)
		require.Contains(t, spec.Volumes, conductorTemplate.Spec.Volumes[len(conductorTemplate.Spec.Volumes)-1])
		require.Contains(t, []string{lpxv1alpha1.PodRoleAgent, lpxv1alpha1.PodRoleConductor}, clique.Annotations[lpxv1alpha1.PodRoleAnnotation])
	}

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
	require.Equal(t, conductorTemplate.Spec.Containers[0].Resources, conductor.Spec.PodSpec.Containers[0].Resources)
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
	for _, agent := range plan.ForReplica(0).Agents {
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

func projectRenderFixture(t *testing.T, pipeline Pipeline, snapshot *Build) *ModelProjection {
	t.Helper()
	intent := ModelProjectionInput{
		Pipeline: pipeline,
		Models:   []string{"default"}, RuntimeBuildRef: "model-build", Build: snapshot,
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
					{Name: "model-storage", PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "model-storage"}},
				},
				Containers: []corev1.Container{{Name: "main", Image: "cyborg", Env: []corev1.EnvVar{
					{Name: "SERVER_HOSTS_FILE", Value: "/tmp/lpu_servers"},
				}, VolumeMounts: []corev1.VolumeMount{
					{Name: lpuConfigVolumeName, MountPath: "/configs"},
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

func TestRenderHybridUnitRuntimeEnvironment(t *testing.T) {
	t.Parallel()
	bundle := testUnitBundle{
		architecture: deploymentcapnpv1.LpuArchitecture_lp20, devicesPerNode: 8, hybrid: true, chains: [][]uint32{{1, 2}},
		partitions: testUnitPartitions(16, nil, 0, 1, 2, 3),
		units: []testLPUUnit{
			{reservations: []testLPUReservation{testNodesReservation(2, 0)}},
			{reservations: []testLPUReservation{testNodesReservation(2, 1), testNodesReservation(2, 2)}, connectors: []testLPUConnector{{}}},
			{reservations: []testLPUReservation{testNodesReservation(2, 3)}},
		},
	}
	authoredConfig := corev1.Volume{Name: lpuConfigVolumeName, EmptyDir: &corev1.EmptyDirVolumeSource{}}
	undeclaredConfig := "container main mounts volume \"config\", which its pod template does not declare; " +
		"the operator provides /configs only for builds without lpuUnits"
	for _, test := range []struct {
		name      string
		legacy    bool
		selection *v1beta1.LPXLocalPartitions
		// agentConfig and cyborgConfig mount config at /configs in that role;
		// a non-nil volume also declares it.
		agentConfig, cyborgConfig bool
		agentVolume               *corev1.Volume
		wantErr                   string
		wantAgentEnv              []corev1.EnvVar
		wantCyborgEnv             []corev1.EnvVar
		// wantConfigMaps lists the rendered runtime ConfigMaps' name prefixes.
		wantConfigMaps []string
	}{
		{
			name:   "legacy placement keeps the selection on Cyborg only",
			legacy: true, agentConfig: true, cyborgConfig: true,
			selection:      &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{0}},
			wantAgentEnv:   []corev1.EnvVar{},
			wantCyborgEnv:  []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "0"}},
			wantConfigMaps: []string{"test-pcs-decode-", "test-pcs-lpu-"},
		},
		{
			name:   "legacy placement omits an empty selection",
			legacy: true, agentConfig: true, cyborgConfig: true,
			wantAgentEnv:   []corev1.EnvVar{},
			wantCyborgEnv:  []corev1.EnvVar{},
			wantConfigMaps: []string{"test-pcs-decode-", "test-pcs-lpu-"},
		},
		{
			name:          "unit placement publishes an empty selection and the Agent template",
			wantAgentEnv:  []corev1.EnvVar{{Name: localPartitionIDsEnv}},
			wantCyborgEnv: []corev1.EnvVar{{Name: allocationEnvVar, Value: testAgentTemplateName}, {Name: localPartitionIDsEnv}},
		},
		{
			name:          "unit placement publishes local unit roots",
			selection:     &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{0}},
			wantAgentEnv:  []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "0"}},
			wantCyborgEnv: []corev1.EnvVar{{Name: allocationEnvVar, Value: testAgentTemplateName}, {Name: localPartitionIDsEnv, Value: "0"}},
		},
		{
			name:          "unit placement names the Agent template without Agents",
			selection:     &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll},
			wantCyborgEnv: []corev1.EnvVar{{Name: allocationEnvVar, Value: testAgentTemplateName}, {Name: localPartitionIDsEnv, Value: "0,1,3"}},
		},
		{
			name:        "unit placement keeps an authored config volume",
			agentConfig: true, agentVolume: &authoredConfig,
			wantAgentEnv:  []corev1.EnvVar{{Name: localPartitionIDsEnv}},
			wantCyborgEnv: []corev1.EnvVar{{Name: allocationEnvVar, Value: testAgentTemplateName}, {Name: localPartitionIDsEnv}},
		},
		{
			name:        "unit placement rejects an Agent config mount without a volume",
			agentConfig: true,
			wantErr:     "LPX role " + testAgentTemplateName + " " + undeclaredConfig,
		},
		{
			name:         "unit placement rejects a Cyborg config mount without a volume",
			cyborgConfig: true,
			wantErr:      "LPX role cond " + undeclaredConfig,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project the hybrid bundle through its units or the legacy derivation")
			build, err := acquireTestUnitBundle(t, bundle)
			require.NoError(t, err)
			if test.legacy {
				build.lpuUnits = nil
			}
			projections, err := projectTestUnits(build, test.selection)
			require.NoError(t, err)

			t.Log("Mount /configs only in the roles that the case names")
			agentSpec := renderTestPodSpec()
			agentSpec.Containers[0].VolumeMounts = withoutConfigMount(agentSpec.Containers[0].VolumeMounts, test.agentConfig)
			if test.agentVolume != nil {
				agentSpec.Volumes = append(agentSpec.Volumes, *test.agentVolume)
			}
			pcs := renderTestPCS(true)
			cyborgContainer := &pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0]
			cyborgContainer.VolumeMounts = withoutConfigMount(cyborgContainer.VolumeMounts, test.cyborgConfig)

			t.Log("Render the hybrid workload")
			rendered, resources, err := renderSelectedForTest(pcs, projections, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: agentSpec}},
			})
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)

			t.Log("Publish placement variables to hybrid Agents and Cyborg in binding order")
			require.Equal(t, test.wantCyborgEnv, placementEnv(namedClique(t, rendered, "cond").Spec.PodSpec.Containers[0].Env))
			agentIndex := slices.IndexFunc(rendered.Spec.Template.Cliques, func(clique *grovev1alpha1.PodCliqueTemplateSpec) bool {
				return clique.Name == testAgentTemplateName
			})
			if test.wantAgentEnv == nil {
				require.Equal(t, -1, agentIndex)
			} else {
				require.Equal(t, test.wantAgentEnv, placementEnv(rendered.Spec.Template.Cliques[agentIndex].Spec.PodSpec.Containers[0].Env))
			}

			t.Log("Render /configs and bind every role to it only for legacy placement")
			require.Len(t, resources, len(test.wantConfigMaps))
			configMapNames := make([]string, 0, len(resources))
			for index, resource := range resources {
				configMap, ok := resource.(*corev1.ConfigMap)
				require.True(t, ok)
				require.True(t, strings.HasPrefix(configMap.Name, test.wantConfigMaps[index]), configMap.Name)
				configMapNames = append(configMapNames, configMap.Name)
			}
			for _, clique := range rendered.Spec.Template.Cliques {
				volumes := clique.Spec.PodSpec.Volumes
				configIndex := slices.IndexFunc(volumes, func(volume corev1.Volume) bool { return volume.Name == lpuConfigVolumeName })
				switch {
				case test.legacy:
					require.GreaterOrEqual(t, configIndex, 0, clique.Name)
					require.Contains(t, configMapNames, volumes[configIndex].ConfigMap.Name)
					require.NotEmpty(t, clique.Annotations[v1alpha1.AnnotationExtraResourcesHash], clique.Name)
				case clique.Name == testAgentTemplateName && test.agentVolume != nil:
					require.Equal(t, *test.agentVolume, volumes[configIndex])
					require.NotContains(t, clique.Annotations, v1alpha1.AnnotationExtraResourcesHash)
				default:
					require.Equal(t, -1, configIndex, clique.Name)
					require.NotContains(t, clique.Annotations, v1alpha1.AnnotationExtraResourcesHash)
				}
			}
		})
	}
}

func TestRenderSpecDecodeRuntimeConfigFollowsBuildPlacement(t *testing.T) {
	t.Parallel()
	bundle := testUnitBundle{
		architecture: deploymentcapnpv1.LpuArchitecture_lp20, devicesPerNode: 8, chains: [][]uint32{{1, 2}},
		partitions: testUnitPartitions(8, nil, 1, 2),
		units: []testLPUUnit{{
			reservations: []testLPUReservation{testNodesReservation(1, 1), testNodesReservation(1, 2)},
			connectors:   []testLPUConnector{{}},
		}},
	}
	for _, test := range []struct {
		name string
		// legacyTarget drops the target build's units; the draft keeps them.
		legacyTarget bool
		configMount  bool
		wantErr      string
		wantData     map[string]string
	}{
		{
			name: "builds that all record units read no /configs",
		},
		{
			name:         "a target without units reads /configs rows for both builds",
			legacyTarget: true, configMount: true,
			wantData: map[string]string{
				"nodes_per_partition":    "1\n1\n1\n1",
				"partition_ids":          "1\n2\n1\n2",
				"partition_indices":      "0\n1\n0\n1",
				"partition_models":       "draft0\ndraft0\ntarget\ntarget",
				"partition_node_offsets": "0\n1\n0\n1",
				"partition_paths":        "part-1\npart-2\npart-1\npart-2",
			},
		},
		{
			name:        "builds that all record units reject a conductor config mount without a volume",
			configMount: true,
			wantErr: "LPX role cond container main mounts volume \"config\", which its pod template does not declare; " +
				"the operator provides /configs only for builds without lpuUnits",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a draft through its units and a target with or without them")
			draft, err := acquireTestUnitBundle(t, bundle)
			require.NoError(t, err)
			target, err := acquireTestUnitBundle(t, bundle)
			require.NoError(t, err)
			if test.legacyTarget {
				target.lpuUnits = nil
			}
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineSpecDecode, Models: []string{"draft0"}, RuntimeBuildRef: "draft-build", Build: draft,
			})
			require.NoError(t, err)
			projections, err = appendModelProjections(projections, ModelProjectionInput{
				Pipeline: PipelineSpecDecode, Models: []string{runtimeModelTarget}, RuntimeBuildRef: "target-build", Build: target,
			})
			require.NoError(t, err)

			t.Log("Render the conductor and Agents from templates that mount /configs only when the case says so")
			spec := renderTestPodSpec()
			spec.Containers[0].VolumeMounts = withoutConfigMount(spec.Containers[0].VolumeMounts, test.configMount)
			rendered, resources, err := renderSelectedForTest(renderTestPCS(false), projections, RenderInput{
				Stages:    map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: spec}},
				Conductor: &corev1.PodTemplateSpec{Spec: *spec.DeepCopy()},
			})
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)

			t.Log("Render the shared rows and bind the conductor and every Agent to them only when a build lacks units")
			if test.wantData == nil {
				require.Empty(t, resources)
			} else {
				require.Len(t, resources, 1)
				configMap, ok := resources[0].(*corev1.ConfigMap)
				require.True(t, ok)
				require.Equal(t, test.wantData, configMap.Data)
			}
			require.Len(t, rendered.Spec.Template.Cliques, 3)
			for _, clique := range rendered.Spec.Template.Cliques {
				volumes := clique.Spec.PodSpec.Volumes
				configIndex := slices.IndexFunc(volumes, func(volume corev1.Volume) bool { return volume.Name == lpuConfigVolumeName })
				if test.wantData == nil {
					require.Equal(t, -1, configIndex, clique.Name)
					require.NotContains(t, clique.Annotations, v1alpha1.AnnotationExtraResourcesHash)
					continue
				}
				require.GreaterOrEqual(t, configIndex, 0, clique.Name)
				require.Equal(t, resources[0].GetName(), volumes[configIndex].ConfigMap.Name)
				require.NotEmpty(t, clique.Annotations[v1alpha1.AnnotationExtraResourcesHash], clique.Name)
			}
		})
	}
}

// withoutConfigMount returns mounts without the config mount unless keep is set,
// in which case it keeps or adds a mount at /configs.
func withoutConfigMount(mounts []corev1.VolumeMount, keep bool) []corev1.VolumeMount {
	mounts = slices.DeleteFunc(slices.Clone(mounts), func(mount corev1.VolumeMount) bool { return mount.Name == lpuConfigVolumeName })
	if keep {
		mounts = append(mounts, corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: "/configs"})
	}
	return mounts
}

// placementEnv returns the unit placement variables of env in order.
func placementEnv(env []corev1.EnvVar) []corev1.EnvVar {
	return slices.DeleteFunc(slices.Clone(env), func(variable corev1.EnvVar) bool {
		return variable.Name != localPartitionIDsEnv && variable.Name != allocationEnvVar
	})
}
