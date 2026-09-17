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
			applyLPUWorkerContainerBase(&container)

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

func TestConfigureNodeLocalRuntimeBindings(t *testing.T) {
	for _, role := range []string{"conductor", "agent"} {
		t.Run(role, func(t *testing.T) {
			t.Log("Author main-container references and HX resource requirements")
			pod := corev1.PodSpec{
				HostUsers: ptr.To(false),
				Containers: []corev1.Container{{
					Name: commonconsts.MainContainerName,
					Env: []corev1.EnvVar{
						{Name: "CONTAINER_NAME", Value: commonconsts.MainContainerName},
						{Name: "MAIN_CPU", ValueFrom: &corev1.EnvVarSource{ResourceFieldRef: &corev1.ResourceFieldSelector{
							ContainerName: commonconsts.MainContainerName, Resource: "limits.cpu",
						}}},
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")},
						Limits:   corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
					},
				}},
				Volumes: []corev1.Volume{{Name: "resources", VolumeSource: corev1.VolumeSource{
					DownwardAPI: &corev1.DownwardAPIVolumeSource{Items: []corev1.DownwardAPIVolumeFile{{
						Path: "main-memory", ResourceFieldRef: &corev1.ResourceFieldSelector{
							ContainerName: commonconsts.MainContainerName, Resource: "limits.memory",
						},
					}}},
				}}},
			}
			before := pod.DeepCopy()

			t.Log("Bind the selected role and retarget every authored main-container reference")
			if role == "conductor" {
				configureNodeLocalConductorRuntime(&pod, BuildFamilyHX, "agt")
			} else {
				configureNodeLocalAgentRuntime(&pod, BuildFamilyHX)
			}
			container := pod.Containers[0]
			require.Equal(t, role, container.Name)
			require.Equal(t, role, testContainerEnvValue(container.Env, "CONTAINER_NAME"))
			require.Equal(t, role, testContainerEnvSource(container.Env, "MAIN_CPU").ResourceFieldRef.ContainerName)
			require.Equal(t, role, pod.Volumes[0].DownwardAPI.Items[0].ResourceFieldRef.ContainerName)
			require.Equal(t, "status.podIP", testContainerEnvSource(container.Env, "POD_IP").FieldRef.FieldPath)
			require.Equal(t, before.Containers[0].Resources, container.Resources)

			t.Log("Apply worker networking only to the Agent and placement only to the conductor")
			if role == "conductor" {
				require.Equal(t, "agt", testContainerEnvValue(container.Env, "LPX_ALLOCATION"))
				require.Equal(t, before.HostUsers, pod.HostUsers)
				require.False(t, pod.HostIPC)
				require.False(t, pod.HostNetwork)
			} else {
				require.Nil(t, pod.HostUsers)
				require.True(t, pod.HostIPC)
				require.True(t, pod.HostNetwork)
				require.Equal(t, corev1.DNSClusterFirstWithHostNet, pod.DNSPolicy)
			}
		})
	}
}

func TestConfigureDirectHybridAgentRuntimePreservesPodSecurity(t *testing.T) {
	for _, test := range []struct {
		name     string
		security *corev1.PodSecurityContext
	}{
		{name: "omitted"},
		{name: "authored", security: &corev1.PodSecurityContext{
			RunAsUser:           ptr.To(int64(1000)),
			RunAsGroup:          ptr.To(int64(1000)),
			RunAsNonRoot:        ptr.To(true),
			FSGroup:             ptr.To(int64(1000)),
			FSGroupChangePolicy: ptr.To(corev1.FSGroupChangeOnRootMismatch),
			SupplementalGroups:  []int64{2000},
			Sysctls:             []corev1.Sysctl{{Name: "net.ipv4.tcp_keepalive_time", Value: "600"}},
			SeccompProfile:      &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault},
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Configure a direct worker with the authored Pod security context")
			pod := corev1.PodSpec{
				Containers:      []corev1.Container{{Name: commonconsts.MainContainerName}},
				SecurityContext: test.security.DeepCopy(),
			}
			configureAgentScheduling(&pod, BuildFamilyXT)
			configureDirectHybridAgentRuntime(&pod, "graph-lpu")

			t.Log("Change the worker identity to root while preserving every other security field")
			want := test.security.DeepCopy()
			if want == nil {
				want = &corev1.PodSecurityContext{}
			}
			want.RunAsUser = ptr.To(int64(0))
			want.RunAsGroup = ptr.To(int64(0))
			want.RunAsNonRoot = ptr.To(false)
			require.Equal(t, want, pod.SecurityContext)
		})
	}
}

func testExecProbe(command string) *corev1.Probe {
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{command}}},
	}
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
