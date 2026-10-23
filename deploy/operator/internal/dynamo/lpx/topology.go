/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
)

// TopologyPublisherLabel marks the Agent clique whose Pod 0 in each allocation
// publishes the allocation's topology. The topology Service selects it.
const TopologyPublisherLabel = "lpx.nvidia.com/topology-publisher"

const (
	topologyVolumeName = "lpx-topology"
	topologyMountPath  = "/run/lpx-topology"
	// topologyAgentSocket is the Agent's own topology service, read by its publisher sidecar.
	topologyAgentSocket = topologyMountPath + "/agent.sock"
	// topologyConsumerSocket is the local API that the Cyborg sidecar serves to Cyborg.
	topologyConsumerSocket = topologyMountPath + "/topology.sock"
	deploymentEnv          = "LPX_DEPLOYMENT"
	topologyRoleEnv        = "DYN_LPX_TOPOLOGY_ROLE"
	topologyServiceSuffix  = "-topology"
)

// TopologyServiceName names the workload's topology Service: ResourcePrefix
// plus "-topology". A prefix too long for a DNS label is shortened and keeps a
// hash of its full value, so distinct workloads keep distinct names.
func (p *MaterializationPlan) TopologyServiceName() string {
	if len(p.ResourcePrefix)+len(topologyServiceSuffix) <= validation.DNS1123LabelMaxLength {
		return p.ResourcePrefix + topologyServiceSuffix
	}
	digest := sha256.Sum256([]byte(p.ResourcePrefix))
	hash := hex.EncodeToString(digest[:2])
	prefix := p.ResourcePrefix[:validation.DNS1123LabelMaxLength-len(topologyServiceSuffix)-len(hash)-1]
	return strings.TrimRight(prefix, "-") + "-" + hash + topologyServiceSuffix
}

// configureTopologyPublisher binds a fresh Agent PodSpec to topology
// publication: Agent main serves its topology on the shared socket and the
// runtime sidecar of Pod 0 publishes it. Every Pod shares the template, so the
// sidecar takes its rank from the Grove Pod index. podSpec must declare the
// "main" container and the "runtime" init container.
func configureTopologyPublisher(podSpec *corev1.PodSpec, deployment string) error {
	return bindTopology(podSpec, deployment,
		[]corev1.EnvVar{{Name: "LPX_TOPOLOGY_SOCKET", Value: topologyAgentSocket}},
		[]corev1.EnvVar{
			{Name: topologyRoleEnv, Value: "publisher"},
			{Name: "DYN_LPX_TOPOLOGY_AGENT_SOCKET", Value: topologyAgentSocket},
			{Name: "DYN_SIDECAR_NODE_RANK", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: fmt.Sprintf("metadata.labels['%s']", grovecommon.LabelPodCliquePodIndex),
			}}},
		},
	)
}

// configureTopologyConsumer binds a fresh Cyborg PodSpec to topology
// consumption: the runtime sidecar serves the local topology API that Cyborg
// main reads at startup. podSpec must declare the "main" container and the
// "runtime" init container.
func configureTopologyConsumer(podSpec *corev1.PodSpec, deployment string) error {
	return bindTopology(podSpec, deployment,
		[]corev1.EnvVar{{Name: "CYBORG_LPU_TOPOLOGY_SOCKET", Value: topologyConsumerSocket}},
		[]corev1.EnvVar{
			{Name: topologyRoleEnv, Value: "consumer"},
			{Name: "DYN_LPX_TOPOLOGY_SOCKET", Value: topologyConsumerSocket},
		},
	)
}

// bindTopology gives every container the deployment identity, prepends the
// role bindings to main and runtime, and shares the socket directory between
// them. It rejects an authored volume or main/runtime mount that would collide
// with the operator-owned socket directory.
func bindTopology(podSpec *corev1.PodSpec, deployment string, mainEnv, runtimeEnv []corev1.EnvVar) error {
	main := common.FindContainerByName(podSpec.Containers, commonconsts.MainContainerName)
	runtime := common.FindContainerByName(podSpec.InitContainers, commonconsts.RuntimeContainerName)

	// The socket directory belongs to the operator, so authored entries cannot alias it.
	if slices.ContainsFunc(podSpec.Volumes, func(volume corev1.Volume) bool { return volume.Name == topologyVolumeName }) {
		return fmt.Errorf("pod template already defines volume %q, which LPX topology discovery adds", topologyVolumeName)
	}
	for _, container := range []*corev1.Container{main, runtime} {
		if slices.ContainsFunc(container.VolumeMounts, func(mount corev1.VolumeMount) bool { return path.Clean(mount.MountPath) == topologyMountPath }) {
			return fmt.Errorf("container %q already mounts %s, which LPX topology discovery owns", container.Name, topologyMountPath)
		}
	}

	// Kubernetes expands environment references in order; publish identity before authored bindings.
	identity := corev1.EnvVar{Name: deploymentEnv, Value: deployment}
	for _, containers := range [][]corev1.Container{podSpec.Containers, podSpec.InitContainers} {
		for i := range containers {
			container := &containers[i]
			owned := []corev1.EnvVar{identity}
			switch container {
			case main:
				owned = append(owned, mainEnv...)
			case runtime:
				owned = append(owned, runtimeEnv...)
			}
			applyOwnedEnv(container, owned)
		}
	}

	// Both containers reach the sockets through one Pod-local directory.
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: topologyVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	})
	mount := corev1.VolumeMount{Name: topologyVolumeName, MountPath: topologyMountPath}
	main.VolumeMounts = append(main.VolumeMounts, mount)
	runtime.VolumeMounts = append(runtime.VolumeMounts, mount)
	return nil
}
