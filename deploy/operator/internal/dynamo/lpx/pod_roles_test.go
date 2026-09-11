/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func TestSelectedRolePodSpecsMaterializeFamilyResourceOnlyOnAgentMainContainer(t *testing.T) {
	t.Parallel()

	t.Log("Construct a mixed-resource PodSpec spanning every container resource location")
	lpuResources := corev1.ResourceList{
		v2LPUResourceName: resource.MustParse("8"),
		corev1.ResourceName("lpu.nvidia.com/devices"): resource.MustParse("1"),
		v3LPUResourceName:                     resource.MustParse("16"),
		corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
		corev1.ResourceCPU:                    resource.MustParse("2"),
	}
	base := corev1.PodSpec{
		Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchExpressions: []corev1.NodeSelectorRequirement{{
					Key: corev1.LabelHostname, Operator: corev1.NodeSelectorOpIn, Values: []string{"lpu-node-a"},
				}}}},
			},
		}},
		InitContainers: []corev1.Container{{
			Name: "init", Resources: corev1.ResourceRequirements{
				Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
			},
		}},
		Containers: []corev1.Container{{
			Name: "main", Image: "runtime",
			Resources: corev1.ResourceRequirements{
				Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
			},
		}},
		EphemeralContainers: []corev1.EphemeralContainer{{
			EphemeralContainerCommon: corev1.EphemeralContainerCommon{
				Name: "debug", Resources: corev1.ResourceRequirements{Limits: lpuResources.DeepCopy()},
			},
		}},
		Resources: &corev1.ResourceRequirements{
			Limits: lpuResources.DeepCopy(), Requests: lpuResources.DeepCopy(),
		},
	}

	t.Log("Define V2 and V3 family-specific resource expectations")
	tests := []struct {
		name               string
		family             BuildFamily
		expectedResource   corev1.ResourceName
		expectedQuantity   resource.Quantity
		unexpectedResource corev1.ResourceName
	}{
		{
			name: "V2 XT8888", family: BuildFamilyXT,
			expectedResource: v2LPUResourceName, expectedQuantity: resource.MustParse("8"),
			unexpectedResource: v3LPUResourceName,
		},
		{
			name: "V3 HX", family: BuildFamilyHX,
			expectedResource: v3LPUResourceName, expectedQuantity: resource.MustParse("16"),
			unexpectedResource: v2LPUResourceName,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Shape LPU resources independently for the conductor and Agent")
			conductor, agent := base.DeepCopy(), base.DeepCopy()
			stripLPUResources(conductor)
			configureAgentScheduling(agent, test.family)

			t.Log("Verify the input is immutable and affinity follows its owning role")
			require.Equal(t, lpuResources, base.Containers[0].Resources.Limits, "input PodSpec must remain unchanged")
			require.Equal(t, base.Affinity, agent.Affinity)

			t.Log("Remove generic and wrong-family LPU resources from both roles")
			for _, spec := range []*corev1.PodSpec{conductor, agent} {
				for _, resources := range lpuResourceLists(spec) {
					require.NotContains(t, resources, corev1.ResourceName("lpu.nvidia.com/devices"))
					require.NotContains(t, resources, test.unexpectedResource)
					require.Contains(t, resources, corev1.ResourceName("nvidia.com/gpu"))
					require.Contains(t, resources, corev1.ResourceCPU)
				}
			}
			for _, resources := range lpuResourceLists(conductor) {
				require.NotContains(t, resources, test.expectedResource)
			}
			for index, resources := range lpuResourceLists(agent) {
				if index == 2 || index == 3 {
					continue
				}
				require.NotContains(t, resources, test.expectedResource)
			}
			t.Log("Materialize only the selected family resource on the main Agent container")
			require.Equal(t, test.expectedQuantity, agent.Containers[0].Resources.Requests[test.expectedResource])
			require.Equal(t, test.expectedQuantity, agent.Containers[0].Resources.Limits[test.expectedResource])
		})
	}
}

func TestApplyLPUWorkerContainerBaseClampsCPURequestToLimit(t *testing.T) {
	for _, test := range []struct {
		name  string
		limit string
		want  string
	}{
		{name: "lower limit", limit: "16", want: "16"},
		{name: "higher limit", limit: "64", want: "62"},
		{name: "no limit", want: "62"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Configure the worker CPU limit")
			container := corev1.Container{Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{},
				Limits:   corev1.ResourceList{},
			}}
			if test.limit != "" {
				container.Resources.Limits = corev1.ResourceList{corev1.ResourceCPU: resource.MustParse(test.limit)}
			}

			t.Log("Apply the LPX worker runtime defaults")
			applyLPUWorkerContainerBase(&container, false)

			t.Log("Clamp the default request to the configured limit")
			require.True(t, container.Resources.Requests.Cpu().Equal(resource.MustParse(test.want)))
		})
	}
}

func lpuResourceLists(spec *corev1.PodSpec) []corev1.ResourceList {
	lists := []corev1.ResourceList{
		spec.InitContainers[0].Resources.Limits,
		spec.InitContainers[0].Resources.Requests,
		spec.Containers[0].Resources.Limits,
		spec.Containers[0].Resources.Requests,
		spec.EphemeralContainers[0].Resources.Limits,
		spec.Resources.Limits,
		spec.Resources.Requests,
	}
	return lists
}

func TestStripNovaOnlyArgs(t *testing.T) {
	t.Parallel()

	t.Log("Define mixed Nova-leader and worker argument shapes")
	for _, test := range []struct {
		name         string
		args         []string
		want         []string
		wantNovaOnly bool
	}{
		{
			name: "worker args only",
			args: []string{"custom-agent-command"},
			want: []string{"custom-agent-command"},
		},
		{
			name: "split leader arg",
			args: []string{"--instance-model-name", "custom-model-name", "custom-agent-command"},
			want: []string{"custom-agent-command"},
		},
		{
			name: "equals leader arg",
			args: []string{"--instance-model-name=custom-model-name", "custom-agent-command"},
			want: []string{"custom-agent-command"},
		},
		{
			name: "leader args only",
			args: []string{"--instance-model-name", "custom-model-name"},
			want: []string{}, wantNovaOnly: true,
		},
		{
			name: "Nova environment args",
			args: []string{"--agent-env-vars", "FIRST=1", "--agent-env-vars=SECOND=2", "custom-agent-command"},
			want: []string{"custom-agent-command"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Strip Nova-only arguments from a copied input")
			args := append([]string(nil), test.args...)
			got, novaOnly := stripNovaOnlyArgs(args)

			t.Log("Verify worker arguments, classification, and input immutability")
			require.Equal(t, test.want, got)
			require.Equal(t, test.wantNovaOnly, novaOnly)
			require.Equal(t, test.args, args)
		})
	}
}

func TestConfigureNodeLocalLPURuntimeRoles(t *testing.T) {
	t.Log("Start from the supported image-only V3 PodSpec")
	base := corev1.PodSpec{
		HostUsers: ptr.To(false),
		Containers: []corev1.Container{{
			Name:         commonconsts.MainContainerName,
			Image:        "runtime:latest",
			Args:         []string{"--instance-model-name", "test/model", "--agent-env-vars", "PIPELINE_ENV=kept"},
			VolumeMounts: []corev1.VolumeMount{{Name: "tmp", MountPath: "/tmp", ReadOnly: true}},
			Env: []corev1.EnvVar{
				{Name: "CONTAINER_NAME", Value: commonconsts.MainContainerName},
				{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"},
				{Name: "GROQ_V2_RESET_ON_OPEN", Value: "custom"},
				{
					Name: "MAIN_CPU",
					ValueFrom: &corev1.EnvVarSource{ResourceFieldRef: &corev1.ResourceFieldSelector{
						ContainerName: commonconsts.MainContainerName,
						Resource:      "limits.cpu",
					}},
				},
			},
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")},
				Limits:   corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
			},
			LivenessProbe: testExecProbe("check-live"),
			StartupProbe:  testExecProbe("check-started"),
			Lifecycle: &corev1.Lifecycle{
				PreStop: &corev1.LifecycleHandler{Exec: &corev1.ExecAction{Command: []string{"stop-main"}}},
			},
		}},
		Volumes: []corev1.Volume{
			{
				Name: "config",
				VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{},
				},
			},
			{
				Name: "resources",
				VolumeSource: corev1.VolumeSource{
					DownwardAPI: &corev1.DownwardAPIVolumeSource{Items: []corev1.DownwardAPIVolumeFile{{
						Path: "main-memory",
						ResourceFieldRef: &corev1.ResourceFieldSelector{
							ContainerName: commonconsts.MainContainerName,
							Resource:      "limits.memory",
						},
					}}},
				},
			},
			{Name: "tmp", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "read-only-tmp"}}},
		},
	}
	conductor := *base.DeepCopy()
	agent := *base.DeepCopy()

	t.Log("Lower the image into the Nova conductor and privileged SSH worker roles")
	require.ErrorContains(t, configureNodeLocalConductorRuntime(&conductor, BuildFamilyHX, "lpu-wkr-m-0", "ssh-secret"), "writable storage")
	conductor = *base.DeepCopy()
	conductor.Containers[0].VolumeMounts[0].ReadOnly = false
	conductor.Volumes[2].VolumeSource = corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}
	require.NoError(t, configureNodeLocalConductorRuntime(&conductor, BuildFamilyHX, "lpu-wkr-m-0", "ssh-secret"))
	require.NoError(t, configureNodeLocalAgentRuntime(&agent, BuildFamilyHX, false, "ssh-secret"))

	t.Log("Verify the conductor starts Nova with immutable placement and source behavior")
	require.Len(t, conductor.Containers, 1)
	conductorContainer := conductor.Containers[0]
	require.Equal(t, "conductor", conductorContainer.Name)
	require.Equal(t, []string{"/bin/sh", "-ec"}, conductorContainer.Command)
	require.Equal(t, []string{runtimeConfigExpansion + "exec \"$@\"\n", "--", "/configs/datacenter.toml", "/tmp/datacenter.toml", "/bin/nova"}, conductorContainer.Args[:5])
	require.Contains(t, conductorContainer.Args, "--instance-model-name")
	requireFlagValue(t, conductorContainer.Args, "--allocation", "lpu-wkr-m-0")
	require.GreaterOrEqual(t, len(conductorContainer.Args), len(nodeLocalHXAgentEnvironmentArgs))
	staticSuffixIndex := len(conductorContainer.Args) - len(nodeLocalHXAgentEnvironmentArgs)
	require.Equal(t, nodeLocalHXAgentEnvironmentArgs, conductorContainer.Args[staticSuffixIndex:])
	pipelineEnvIndex := slices.Index(conductorContainer.Args, "PIPELINE_ENV=kept")
	require.GreaterOrEqual(t, pipelineEnvIndex, 0)
	require.Less(t, pipelineEnvIndex, staticSuffixIndex)
	require.Equal(t, "1", testContainerEnvValue(conductorContainer.Env, "OMPI_ALLOW_RUN_AS_ROOT"))
	require.Equal(t, "conductor", testContainerEnvValue(conductorContainer.Env, "CONTAINER_NAME"))
	require.Equal(t, "conductor", testContainerEnvSource(conductorContainer.Env, "MAIN_CPU").ResourceFieldRef.ContainerName)
	require.Equal(t, "conductor", conductor.Volumes[1].DownwardAPI.Items[0].ResourceFieldRef.ContainerName)
	require.Equal(t, base.Containers[0].LivenessProbe, conductorContainer.LivenessProbe)
	require.Equal(t, base.Containers[0].StartupProbe, conductorContainer.StartupProbe)
	require.Equal(t, base.Containers[0].Lifecycle, conductorContainer.Lifecycle)
	require.False(t, *conductor.HostUsers)
	require.NotNil(t, testContainerEnvSource(conductorContainer.Env, "POD_IP"))
	require.Contains(t, conductorContainer.VolumeMounts, corev1.VolumeMount{Name: "tmp", MountPath: "/tmp"})
	require.NotNil(t, testVolumeByName(t, conductor.Volumes, "tmp").EmptyDir)
	require.Contains(t, testVolumeMountPaths(conductorContainer.VolumeMounts), conductorSSHKeyDir)
	require.Len(t, conductor.InitContainers, 1)
	require.Equal(t, conductorSSHKeyInitContainerName, conductor.InitContainers[0].Name)
	require.Equal(t, int32(0600), *testVolumeByName(t, conductor.Volumes, runtimeSSHVolumeName).Secret.DefaultMode)
	require.Equal(t, "ssh-secret", testVolumeByName(t, conductor.Volumes, runtimeSSHVolumeName).Secret.SecretName)

	t.Log("Verify the Agent hosts SSHD with local devices and the source resource profile")
	require.Len(t, agent.Containers, 1)
	agentContainer := agent.Containers[0]
	require.Equal(t, lpuAgentContainerName, agentContainer.Name)
	require.Equal(t, []string{"/bin/bash"}, agentContainer.Command)
	require.Equal(t, []string{"-c", lpuWorkerRunCommand}, agentContainer.Args)
	require.Empty(t, testContainerEnvValue(agentContainer.Env, "LPU_RUN_SYSTEM_INIT"))
	require.Equal(t, "cvp", testContainerEnvValue(agentContainer.Env, "LPU_FPGA_RECONFIGURE_METHOD"))
	require.Equal(t, "custom", testContainerEnvValue(agentContainer.Env, "GROQ_V2_RESET_ON_OPEN"))
	require.Equal(t, "0", testContainerEnvValue(agentContainer.Env, "GROQ_PG613_SKIP_HOST_LINK_PROFILE"))
	require.Equal(t, lpuAgentContainerName, testContainerEnvValue(agentContainer.Env, "CONTAINER_NAME"))
	require.Equal(t, lpuAgentContainerName, testContainerEnvSource(agentContainer.Env, "MAIN_CPU").ResourceFieldRef.ContainerName)
	require.Equal(t, lpuAgentContainerName, agent.Volumes[1].DownwardAPI.Items[0].ResourceFieldRef.ContainerName)
	require.Equal(t, &corev1.SecurityContext{
		Privileged:   ptr.To(true),
		RunAsUser:    ptr.To(int64(0)),
		RunAsGroup:   ptr.To(int64(0)),
		RunAsNonRoot: ptr.To(false),
	}, agentContainer.SecurityContext)
	require.True(t, agentContainer.TTY)
	require.True(t, agentContainer.Stdin)
	require.Nil(t, agentContainer.LivenessProbe)
	require.Nil(t, agentContainer.StartupProbe)
	require.Nil(t, agentContainer.Lifecycle)
	require.Nil(t, agent.HostUsers)
	require.True(t, agent.HostIPC)
	require.True(t, agent.HostNetwork)
	require.Equal(t, corev1.DNSClusterFirstWithHostNet, agent.DNSPolicy)
	require.Contains(t, testVolumeMountPaths(agentContainer.VolumeMounts), "/dev")
	require.Contains(t, testVolumeMountPaths(agentContainer.VolumeMounts), "/sys")
	require.Contains(t, testVolumeMountPaths(agentContainer.VolumeMounts), "/ssh-pk")
	require.Contains(t, agentContainer.VolumeMounts, corev1.VolumeMount{Name: "tmp", MountPath: "/tmp"})
	require.NotNil(t, testVolumeByName(t, agent.Volumes, "tmp").EmptyDir)
	require.Equal(t, resource.MustParse("2"), agentContainer.Resources.Requests[corev1.ResourceCPU])
	require.Equal(t, resource.MustParse("4Gi"), agentContainer.Resources.Limits[corev1.ResourceMemory])
	require.NotContains(t, testVolumeNames(agent.Volumes), "hugepages")
	require.Equal(t, int32(0644), *testVolumeByName(t, agent.Volumes, runtimeSSHVolumeName).Secret.DefaultMode)
	require.Len(t, agent.InitContainers, 1)
	initContainer := agent.InitContainers[0]
	require.Equal(t, lp30InitContainerName, initContainer.Name)
	require.Equal(t, "runtime:latest", initContainer.Image)
	require.Equal(t, []string{lp30InitPath}, initContainer.Command)
	require.True(t, *initContainer.SecurityContext.Privileged)
	require.Equal(t, "spec.nodeName", testContainerEnvSource(initContainer.Env, "NODE_NAME").FieldRef.FieldPath)
	require.Contains(t, testVolumeMountPaths(initContainer.VolumeMounts), "/dev")
	require.Contains(t, testVolumeMountPaths(initContainer.VolumeMounts), "/sys")
}

func TestConfigureNodeLocalXTConductorSSHInitUsesMainImage(t *testing.T) {
	t.Log("Place an unrelated sidecar before the XT conductor main container")
	podSpec := corev1.PodSpec{Containers: []corev1.Container{
		{Name: "metrics", Image: "metrics-sidecar", ImagePullPolicy: corev1.PullAlways},
		{Name: commonconsts.MainContainerName, Image: "lpu-runtime", ImagePullPolicy: corev1.PullIfNotPresent},
	}}

	t.Log("Configure the conductor without deriving its runtime identity from list position")
	require.NoError(t, configureNodeLocalConductorRuntime(&podSpec, BuildFamilyXT, "lpu-wkr-m-0", "ssh-secret"))

	t.Log("Run SSH initialization with the resolved conductor image and preserve the sidecar")
	require.Equal(t, "metrics", podSpec.Containers[0].Name)
	require.Equal(t, "metrics-sidecar", podSpec.Containers[0].Image)
	require.Equal(t, "conductor", podSpec.Containers[1].Name)
	require.Len(t, podSpec.InitContainers, 1)
	require.Equal(t, "lpu-runtime", podSpec.InitContainers[0].Image)
	require.Equal(t, corev1.PullIfNotPresent, podSpec.InitContainers[0].ImagePullPolicy)
}

func TestXTSSHSecretNameIsNotUsedAsVolumeName(t *testing.T) {
	const sshSecretName = "mpi.ssh"

	t.Log("Configure each XT runtime role with a valid dotted SSH Secret name")
	base := corev1.PodSpec{Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "lpu-runtime"}}}
	configureAgentScheduling(&base, BuildFamilyXT)
	direct, agent, conductor := base.DeepCopy(), base.DeepCopy(), base.DeepCopy()
	require.NoError(t, configureDirectHybridAgentRuntime(direct, "graph-lpu", sshSecretName))
	require.NoError(t, configureNodeLocalAgentRuntime(agent, BuildFamilyXT, false, sshSecretName))
	require.NoError(t, configureNodeLocalConductorRuntime(conductor, BuildFamilyXT, "lpu-wkr-m-0", sshSecretName))

	for _, test := range []struct {
		name   string
		pod    *corev1.PodSpec
		mounts []corev1.VolumeMount
	}{
		{"direct hybrid agent", direct, direct.Containers[0].VolumeMounts},
		{"node-local agent", agent, agent.Containers[0].VolumeMounts},
		{"node-local conductor", conductor, conductor.InitContainers[0].VolumeMounts},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Keep the configured Secret separate from the fixed volume and mount name")
			sshVolume := testVolumeByName(t, test.pod.Volumes, "ssh-secret")
			require.Equal(t, sshSecretName, sshVolume.Secret.SecretName)
			require.NotContains(t, testVolumeNames(test.pod.Volumes), sshSecretName)
			require.Contains(t, test.mounts, corev1.VolumeMount{Name: "ssh-secret", MountPath: "/ssh-pk", ReadOnly: true})
		})
	}
}

func TestConfigureDirectHybridAgentRuntimePreservesPodOverrides(t *testing.T) {
	t.Log("Define custom entrypoint, probe and Pod security preservation cases")
	customStartup := testExecProbe("custom-startup")
	customReadiness := testExecProbe("custom-ready")
	tests := []struct {
		name          string
		container     corev1.Container
		podSecurity   *corev1.PodSecurityContext
		wantCommand   []string
		wantArgs      []string
		wantStartup   *corev1.Probe
		wantReadiness *corev1.Probe
	}{
		{
			name:          "args",
			container:     corev1.Container{Name: commonconsts.MainContainerName, Args: []string{"custom-agent"}},
			wantCommand:   nil,
			wantArgs:      []string{"custom-agent"},
			wantStartup:   lpuV2StartupProbe(true),
			wantReadiness: lpuV2ReadinessProbe(true),
		},
		{
			name:      "command",
			container: corev1.Container{Name: commonconsts.MainContainerName, Command: []string{"custom-entrypoint"}},
			podSecurity: &corev1.PodSecurityContext{
				RunAsUser:           ptr.To(int64(1000)),
				RunAsGroup:          ptr.To(int64(1000)),
				RunAsNonRoot:        ptr.To(true),
				FSGroup:             ptr.To(int64(1000)),
				FSGroupChangePolicy: ptr.To(corev1.FSGroupChangeOnRootMismatch),
				SupplementalGroups:  []int64{2000},
				Sysctls:             []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_time", Value: "600"}},
				SeccompProfile:      &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault},
			},
			wantCommand:   []string{"custom-entrypoint"},
			wantStartup:   lpuV2StartupProbe(true),
			wantReadiness: lpuV2ReadinessProbe(true),
		},
		{
			name: "user probes",
			container: corev1.Container{
				Name:           commonconsts.MainContainerName,
				Args:           []string{"custom-agent"},
				StartupProbe:   customStartup,
				ReadinessProbe: customReadiness,
			},
			wantCommand:   nil,
			wantArgs:      []string{"custom-agent"},
			wantStartup:   customStartup,
			wantReadiness: customReadiness,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Configure the direct LPU agent runtime from a user-owned entrypoint")
			podSpec := corev1.PodSpec{
				Containers:      []corev1.Container{*test.container.DeepCopy()},
				SecurityContext: test.podSecurity.DeepCopy(),
			}
			configureAgentScheduling(&podSpec, BuildFamilyXT)
			err := configureDirectHybridAgentRuntime(&podSpec, "graph-lpu", "ssh-secret")
			require.NoError(t, err)

			t.Log("Preserve the entrypoint and user probes while supplying defaults when absent")
			agent := podSpec.Containers[0]
			require.Equal(t, test.wantCommand, agent.Command)
			require.Equal(t, test.wantArgs, agent.Args)
			require.Equal(t, test.wantStartup, agent.StartupProbe)
			require.Equal(t, test.wantReadiness, agent.ReadinessProbe)

			t.Log("Preserve orthogonal Pod security settings while using the root runtime identity")
			wantSecurity := test.podSecurity.DeepCopy()
			if wantSecurity == nil {
				wantSecurity = &corev1.PodSecurityContext{}
			}
			wantSecurity.RunAsUser = ptr.To(int64(0))
			wantSecurity.RunAsGroup = ptr.To(int64(0))
			wantSecurity.RunAsNonRoot = ptr.To(false)
			require.Equal(t, wantSecurity, podSpec.SecurityContext)
		})
	}
}

func TestLPXInitContainerNames(t *testing.T) {
	t.Log("Reserve a generated name only in the Pod receiving that init container")
	for _, test := range []struct {
		name string
		add  func(*corev1.PodSpec, *corev1.Container) error
	}{
		{name: "prepare-lp30", add: addLP30InitContainer},
		{name: "prepare-ssh-key", add: func(spec *corev1.PodSpec, main *corev1.Container) error {
			return addConductorSSHKey(spec, main, "ssh-secret")
		}},
	} {
		for _, name := range []string{"prepare-lp30", "prepare-ssh-key"} {
			for _, list := range []string{"containers", "initContainers"} {
				t.Run(test.name+"/"+name+"/"+list, func(t *testing.T) {
					t.Log("Author an independent container with an image and command to preserve")
					spec := corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime"}}}
					authored := corev1.Container{Name: name, Image: "custom", Command: []string{"/custom-init"}}
					if list == "containers" {
						spec.Containers = append(spec.Containers, authored)
					} else {
						spec.InitContainers = []corev1.Container{authored}
					}
					before := spec.DeepCopy()

					t.Log("Reject only the actual generated name, before any mutation")
					err := test.add(&spec, &spec.Containers[0])
					if name == test.name {
						require.ErrorContains(t, err, list)
						require.ErrorContains(t, err, name)
						require.Equal(t, *before, spec)
						return
					}

					t.Log("Preserve a name belonging to the other role and add the intended init")
					require.NoError(t, err)
					require.Equal(t, before.Containers, spec.Containers)
					require.Len(t, spec.InitContainers, len(before.InitContainers)+1)
					for i, container := range before.InitContainers {
						require.Equal(t, container, spec.InitContainers[i])
					}
					require.Equal(t, test.name, spec.InitContainers[len(before.InitContainers)].Name)
				})
			}
		}
	}
}

func TestConfigureNodeLocalLPURuntimeRolesRejectsUnsupportedShape(t *testing.T) {
	t.Log("Define unsupported node-local PodSpec shapes")
	tests := []struct {
		name      string
		mutate    func(*corev1.PodSpec)
		errorText string
	}{
		{
			name: "explicit command",
			mutate: func(spec *corev1.PodSpec) {
				spec.Containers[0].Command = []string{"/custom"}
			},
			errorText: "image-only main container without command",
		},
		{
			name: "sidecar",
			mutate: func(spec *corev1.PodSpec) {
				spec.Containers = append(spec.Containers, corev1.Container{Name: "sidecar"})
			},
			errorText: `requires exactly one "main" container`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Reject input outside the supported image-only V3 contract")
			base := corev1.PodSpec{Containers: []corev1.Container{{Name: commonconsts.MainContainerName}}}
			test.mutate(&base)
			conductor := *base.DeepCopy()
			agent := *base.DeepCopy()

			require.ErrorContains(t, configureNodeLocalConductorRuntime(&conductor, BuildFamilyHX, "agent", "ssh-secret"), test.errorText)
			require.ErrorContains(t, configureNodeLocalAgentRuntime(&agent, BuildFamilyHX, false, "ssh-secret"), test.errorText)
		})
	}
}

func testExecProbe(command string) *corev1.Probe {
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{command}}},
	}
}

func requireFlagValue(t *testing.T, args []string, flag, value string) {
	t.Helper()
	index := slices.Index(args, flag)
	require.GreaterOrEqual(t, index, 0)
	require.Less(t, index+1, len(args))
	require.Equal(t, value, args[index+1])
}

func testContainerEnvValue(env []corev1.EnvVar, name string) string {
	index := slices.IndexFunc(env, func(value corev1.EnvVar) bool { return value.Name == name })
	if index < 0 {
		return ""
	}
	return env[index].Value
}

func testContainerEnvSource(env []corev1.EnvVar, name string) *corev1.EnvVarSource {
	index := slices.IndexFunc(env, func(value corev1.EnvVar) bool { return value.Name == name })
	if index < 0 {
		return nil
	}
	return env[index].ValueFrom
}

func testVolumeMountPaths(mounts []corev1.VolumeMount) []string {
	paths := make([]string, 0, len(mounts))
	for _, mount := range mounts {
		paths = append(paths, mount.MountPath)
	}
	return paths
}

func testVolumeByName(t *testing.T, volumes []corev1.Volume, name string) corev1.VolumeSource {
	t.Helper()
	index := slices.IndexFunc(volumes, func(volume corev1.Volume) bool { return volume.Name == name })
	require.GreaterOrEqual(t, index, 0)
	return volumes[index].VolumeSource
}

func testVolumeNames(volumes []corev1.Volume) []string {
	names := make([]string, 0, len(volumes))
	for _, volume := range volumes {
		names = append(names, volume.Name)
	}
	return names
}
