/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// Preserve rendered workload identities and image-owned executable paths.
const (
	runtimeTemporaryStorageVolumeName = "tmp"
	runtimeTemporaryStorageMountPath  = "/tmp"
	lp30InitContainerName             = "prepare-lp30"
	lp30InitPath                      = "/bin/lp30-agent-v2-init"
)

var lp30WorkerEnvironment = []corev1.EnvVar{
	{Name: "BMC_SSH_ENABLED", Value: "false"},
	{Name: "GROQ_HAL_V2_BACKGROUND_METRICS", Value: "false"},
	{Name: "GROQ_PG613_PORT_CONFIG", Value: "EB3"},
	{Name: "GROQ_PG613_REQUIRE_ALTAIR_PCIE_GEN2", Value: "false"},
	{Name: "GROQ_PG613_SKIP_HOST_LINK_PROFILE", Value: "0"},
	{Name: "GROQ_V2_FPGA_BUS_ADDRESS_ID_0", Value: "0000:9a:00.0"},
	{Name: "GROQ_V2_FPGA_BUS_ADDRESS_ID_1", Value: "0000:7e:00.0"},
	{Name: "GROQ_V2_HEALTH_CHECK_ON_OPEN", Value: "0"},
	{Name: "GROQ_V2_RESET_ON_OPEN", Value: "0"},
	{Name: "LPU_FPGA_AUTOMATIC_JTAG_RECOVERY", Value: "false"},
	{Name: "LPU_FPGA_RECONFIGURE_METHOD", Value: "cvp"},
	{Name: "REDFISH_BMC_ENABLED", Value: "false"},
}

// nodeLocalHXAgentEnvironmentArgs is the immutable Nova environment suffix for HX.
var nodeLocalHXAgentEnvironmentArgs = []string{
	"--agent-env-vars", "BMC_SSH_ENABLED=false",
	"--agent-env-vars", "GROQ_FPGA_VSAP_DRAM_WAIT_CYCLES=0",
	"--agent-env-vars", "GROQ_FPGA_VSAP_SETUP_MPINGO_KERNELS=false",
	"--agent-env-vars", "GROQ_HAL_V2_BACKGROUND_METRICS=false",
	"--agent-env-vars", "GROQ_PG613_PORT_CONFIG=EB3",
	"--agent-env-vars", "GROQ_PG613_REQUIRE_ALTAIR_PCIE_GEN2=false",
	"--agent-env-vars", "GROQ_PG613_SKIP_HOST_LINK_PROFILE=1",
	"--agent-env-vars", "GROQ_V2_FPGA_BUS_ADDRESS_ID_0=0000:9a:00.0",
	"--agent-env-vars", "GROQ_V2_FPGA_BUS_ADDRESS_ID_1=0000:7e:00.0",
	"--agent-env-vars", "GROQ_V2_HEALTH_CHECK_ON_OPEN=0",
	"--agent-env-vars", "GROQ_V2_RESET_ON_OPEN=0",
	"--agent-env-vars", "LPU_FPGA_AUTOMATIC_JTAG_RECOVERY=false",
	"--agent-env-vars", "LPU_FPGA_RECONFIGURE_METHOD=cvp",
	"--agent-env-vars", "LPU_PARTITION_RUNNER_DATAPLANE=pcie",
	"--agent-env-vars", "LPU_PARTITION_RUNNER_PCIE_SOURCE_PORT=33000",
	"--agent-env-vars", "LPU_RUNNER_OUTSTANDING_TASK_DRAIN_TIMEOUT_SECONDS=10",
	"--agent-env-vars", "LPU_VSAP_DIB_BARRIER_BEFORE_PUSH_ENABLED=true",
	"--agent-env-vars", "LPU_VSAP_MAX_TENSORS_PER_CHUNK=4",
	"--agent-env-vars", "REDFISH_BMC_ENABLED=false",
}

// stripNovaOnlyArgs removes conductor-owned Nova CLI flags and reports whether
// every input argument was a well-formed conductor-owned flag.
func stripNovaOnlyArgs(args []string) ([]string, bool) {
	if len(args) == 0 {
		return args, true
	}

	// Parse both accepted flag forms once while retaining worker arguments in order.
	const modelNameFlag = "--instance-model-name"
	const agentEnvFlag = "--agent-env-vars"
	out := make([]string, 0, len(args))
	novaOnly := true
	seenModelName := false
	for i := 0; i < len(args); i++ {
		argument := args[i]
		switch argument {
		case modelNameFlag, agentEnvFlag:
			validValue := i+1 < len(args) && strings.TrimSpace(args[i+1]) != ""
			if argument == modelNameFlag {
				validValue = validValue && !seenModelName
				seenModelName = true
			}
			novaOnly = novaOnly && validValue
			if i+1 < len(args) {
				i++
			}
		default:
			flag, value, hasValue := strings.Cut(argument, "=")
			switch flag {
			case modelNameFlag:
				novaOnly = novaOnly && !seenModelName && hasValue && strings.TrimSpace(value) != ""
				seenModelName = true
			case agentEnvFlag:
				novaOnly = novaOnly && hasValue && strings.TrimSpace(value) != ""
			default:
				novaOnly = false
				out = append(out, argument)
			}
		}
	}
	return out, novaOnly
}

// configureDirectHybridAgentRuntime lowers an LPX-scheduled PodSpec into the
// direct agent runtime used by conductorless hybrid workloads.
func configureDirectHybridAgentRuntime(
	agentPodSpec *corev1.PodSpec,
	lpuConfigMapName, sshSecretName string,
) error {
	if strings.TrimSpace(sshSecretName) == "" {
		return fmt.Errorf("direct hybrid agent runtime requires an MPI SSH secret name")
	}

	agent := findMainContainer(agentPodSpec.Containers)
	// Drop Nova-only flags before applying the direct Agent identity and worker defaults.
	agent.Args, _ = stripNovaOnlyArgs(agent.Args)
	hasCustomStartup := len(agent.Command) != 0 || len(agent.Args) != 0
	agent.Name = lpuAgentContainerName
	applyLPUWorkerContainerBase(agent, hasCustomStartup)
	if len(agent.Args) == 0 && !hasCustomStartup {
		agent.Args = []string{"-c", lpuPartitionRunCommand}
	}

	if !hasCustomStartup || agent.StartupProbe == nil {
		agent.StartupProbe = lpuV2StartupProbe(hasCustomStartup)
	}
	if !hasCustomStartup || agent.ReadinessProbe == nil {
		agent.ReadinessProbe = lpuV2ReadinessProbe(hasCustomStartup)
	}

	agent.Env = append(agent.Env,
		corev1.EnvVar{Name: "GLUE_RDMA_PATH_MTU", Value: "1024"},
		corev1.EnvVar{Name: "NIC_NAME", Value: "mlx5_0"},
		corev1.EnvVar{Name: "UCX_TLS", Value: "tcp"},
		corev1.EnvVar{Name: "RDMA_PORT", Value: fmt.Sprintf("%d", lpuRDMAPort)},
		corev1.EnvVar{Name: "READINESS_PORT", Value: fmt.Sprintf("%d", LPUReadinessPort)},
	)

	addLPUHostDeviceVolumeMounts(agent)
	agent.VolumeMounts = setVolumeMount(agent.VolumeMounts, sshVolumeMount())
	retargetMainContainerReferences(agentPodSpec, agent)

	agent.Env = append(agent.Env,
		corev1.EnvVar{
			Name: "TOPOLOGIES",
			ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{Name: lpuConfigMapName},
				Key:                  "topologies",
			}},
		},
		corev1.EnvVar{
			Name: "GAS_DIR",
			ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{Name: lpuConfigMapName},
				Key:                  "gas_dir",
			}},
		},
	)
	setNodeLocalPodIPEnv(agent, true)
	addRuntimeTemporaryStorage(agentPodSpec, agent, false)
	updateWorkerPodSpec(agentPodSpec)
	agentPodSpec.SecurityContext = &corev1.PodSecurityContext{
		RunAsUser:  ptr.To(int64(0)),
		RunAsGroup: ptr.To(int64(0)),
	}
	if err := addSSHVolume(agentPodSpec, sshSecretName, 0644); err != nil {
		return err
	}
	applyLPUHostDeviceVolumes(agentPodSpec, false)
	agentPodSpec.Volumes = appendVolumeIfMissing(agentPodSpec.Volumes, corev1.Volume{
		Name: "hugepages",
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{
			Medium: corev1.StorageMediumHugePages,
		}},
	})
	return nil
}

// configureNodeLocalConductorRuntime consumes a nonnil conductor PodSpec.
// The caller has validated the runtime SSH secret name.
func configureNodeLocalConductorRuntime(
	conductorPodSpec *corev1.PodSpec,
	targetFamily BuildFamily,
	allocation string,
	sshSecretName string,
) error {
	// Resolve the conductor container before changing its image-only contract.
	conductor, err := runtimeMainContainer(conductorPodSpec, targetFamily)
	if err != nil {
		return fmt.Errorf("node-local LPU conductor: %w", err)
	}

	isXT := targetFamily == BuildFamilyXT
	if !isXT && len(conductor.Command) != 0 {
		return fmt.Errorf("node-local LPU runtime requires an image-only main container without command")
	}

	// Materialize the GPC conductor around the image's Nova binary.
	setNodeLocalPodIPEnv(conductor, isXT)
	if err := addRuntimeConfigStorage(conductorPodSpec, conductor, "datacenter.toml"); err != nil {
		return err
	}
	updateLPUConductorContainer(conductor, allocation)
	if targetFamily == BuildFamilyHX {
		conductor.Args = append(conductor.Args, nodeLocalHXAgentEnvironmentArgs...)
	}
	retargetMainContainerReferences(conductorPodSpec, conductor)
	return addConductorSSHKey(conductorPodSpec, conductor, sshSecretName)
}

// configureNodeLocalAgentRuntime consumes a nonnil Agent PodSpec.
// The caller has validated the runtime SSH secret name.
func configureNodeLocalAgentRuntime(
	agentPodSpec *corev1.PodSpec,
	targetFamily BuildFamily,
	validateXTSingleArgs bool,
	sshSecretName string,
) error {
	// Validate the emitted Agent's own shape before changing its startup contract.
	agent, err := runtimeMainContainer(agentPodSpec, targetFamily)
	if err != nil {
		return fmt.Errorf("node-local LPU agent: %w", err)
	}
	isXT := targetFamily == BuildFamilyXT
	if isXT {
		strippedArgs, novaOnly := stripNovaOnlyArgs(agent.Args)
		if validateXTSingleArgs && len(agent.Command) == 0 && !novaOnly {
			return fmt.Errorf("operator-managed XT single only supports --instance-model-name and --agent-env-vars; set command for custom startup behavior")
		}
		agent.Args = strippedArgs
	} else if len(agent.Command) != 0 {
		return fmt.Errorf("node-local LPU runtime requires an image-only main container without command")
	}
	preserveAgentEntrypoint := isXT && (len(agent.Command) != 0 || len(agent.Args) != 0)
	// Materialize the LPU role as the privileged SSH target that Nova launches.
	setNodeLocalPodIPEnv(agent, isXT)
	agent.Name = lpuAgentContainerName
	retargetMainContainerReferences(agentPodSpec, agent)
	configureNodeLocalAgentWorkerContainer(agent, isXT, preserveAgentEntrypoint)
	addRuntimeTemporaryStorage(agentPodSpec, agent, !isXT)
	agentPodSpec.HostUsers = nil
	updateWorkerPodSpec(agentPodSpec)
	applyLPUHostDeviceVolumes(agentPodSpec, !isXT)
	if err := addSSHVolume(agentPodSpec, sshSecretName, 0644); err != nil {
		return err
	}

	// Expose node-local devices and hugetlbfs through host /dev without an unaccounted HugePages volume.
	addLPUHostDeviceVolumeMounts(agent)
	agent.VolumeMounts = setVolumeMount(agent.VolumeMounts, sshVolumeMount())
	if isXT {
		applyLPUWorkerContainerBase(agent, preserveAgentEntrypoint)
		agent.SecurityContext.RunAsGroup = ptr.To(int64(0))
		agent.SecurityContext.RunAsNonRoot = ptr.To(false)
		agentPodSpec.Volumes = appendVolumeIfMissing(agentPodSpec.Volumes, corev1.Volume{
			Name: "hugepages",
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{
				Medium: corev1.StorageMediumHugePages,
			}},
		})
	} else {
		addLP30InitContainer(agentPodSpec, agent)
	}

	return nil
}

// runtimeMainContainer consumes a fresh selected role with a validated main container.
func runtimeMainContainer(
	podSpec *corev1.PodSpec,
	targetFamily BuildFamily,
) (*corev1.Container, error) {
	// Keep V3's image-only single-container contract unchanged.
	if targetFamily == BuildFamilyHX && len(podSpec.Containers) != 1 {
		return nil, fmt.Errorf("PodSpec requires exactly one %q container", commonconsts.MainContainerName)
	}
	return findMainContainer(podSpec.Containers), nil
}

func configureNodeLocalAgentWorkerContainer(
	container *corev1.Container,
	isXT bool,
	preserveEntrypoint bool,
) {
	// The Agent pod hosts SSHD; Nova starts /bin/agent in a separate MPI process.
	if !preserveEntrypoint {
		runCommand := lpuWorkerRunCommand
		if isXT {
			runCommand = lpuPartitionWorkerRunCommand
		} else {
			container.Command = []string{"/bin/bash"}
		}
		container.Args = []string{"-c", runCommand}
	}

	if isXT {
		if !preserveEntrypoint || container.ReadinessProbe == nil {
			container.ReadinessProbe = lpuSSHProbe()
		}
		setContainerEnv(container, true, corev1.EnvVar{Name: "LPU_RUN_SYSTEM_INIT", Value: "true"})
	} else {
		container.SecurityContext = &corev1.SecurityContext{
			Privileged:   ptr.To(true),
			RunAsUser:    ptr.To(int64(0)),
			RunAsGroup:   ptr.To(int64(0)),
			RunAsNonRoot: ptr.To(false),
		}
		container.TTY = true
		container.Stdin = true

		// Remove health and lifecycle behavior owned by the source image process.
		container.LivenessProbe = nil
		container.StartupProbe = nil
		container.Lifecycle = nil
		container.ReadinessProbe = lpuSSHProbe()
		setContainerEnv(container, true, lp30WorkerEnvironment...)
	}
}

func addLP30InitContainer(podSpec *corev1.PodSpec, agent *corev1.Container) {
	initContainer := corev1.Container{
		Name:            lp30InitContainerName,
		Image:           agent.Image,
		ImagePullPolicy: agent.ImagePullPolicy,
		Command:         []string{lp30InitPath},
		Env: []corev1.EnvVar{{
			Name: "NODE_NAME",
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: "spec.nodeName",
			}},
		}},
		SecurityContext: &corev1.SecurityContext{
			Privileged:   ptr.To(true),
			RunAsUser:    ptr.To(int64(0)),
			RunAsGroup:   ptr.To(int64(0)),
			RunAsNonRoot: ptr.To(false),
		},
	}
	addLPUHostDeviceVolumeMounts(&initContainer)
	podSpec.InitContainers = setContainerByName(podSpec.InitContainers, initContainer)
}

func setNodeLocalPodIPEnv(container *corev1.Container, isXT bool) {
	if isXT &&
		slices.ContainsFunc(container.Env, func(existing corev1.EnvVar) bool { return existing.Name == "POD_IP" }) {
		return
	}
	container.Env = append(container.Env, corev1.EnvVar{
		Name: "POD_IP",
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "status.podIP",
			},
		},
	})
}

// retargetMainContainerReferences requires its non-nil target to point into the non-nil PodSpec's Containers.
func retargetMainContainerReferences(podSpec *corev1.PodSpec, target *corev1.Container) {
	targetName := target.Name
	// Keep discovery identity and explicit resource selectors valid anywhere in the Pod after renaming main.
	for index := range podSpec.Containers {
		container := &podSpec.Containers[index]
		retargetEnvReferences(container.Env, targetName, container == target)
	}
	for index := range podSpec.InitContainers {
		retargetEnvReferences(podSpec.InitContainers[index].Env, targetName, false)
	}
	for index := range podSpec.EphemeralContainers {
		retargetEnvReferences(podSpec.EphemeralContainers[index].Env, targetName, false)
	}

	// Retarget both standalone and projected downwardAPI resource files.
	for index := range podSpec.Volumes {
		volume := &podSpec.Volumes[index]
		if volume.DownwardAPI != nil {
			retargetResourceFieldReferencesInDownwardAPI(volume.DownwardAPI.Items, targetName)
		}
		if volume.Projected == nil {
			continue
		}
		for sourceIndex := range volume.Projected.Sources {
			downwardAPI := volume.Projected.Sources[sourceIndex].DownwardAPI
			if downwardAPI != nil {
				retargetResourceFieldReferencesInDownwardAPI(downwardAPI.Items, targetName)
			}
		}
	}
}

func retargetEnvReferences(env []corev1.EnvVar, targetName string, retargetDiscoveryIdentity bool) {
	// Replace the target's discovery opt-in before retargeting explicit resource selectors.
	for index := range env {
		if retargetDiscoveryIdentity && env[index].Name == "CONTAINER_NAME" {
			env[index] = corev1.EnvVar{Name: "CONTAINER_NAME", Value: targetName}
		}
		source := env[index].ValueFrom
		if source == nil || source.ResourceFieldRef == nil {
			continue
		}
		if source.ResourceFieldRef.ContainerName == commonconsts.MainContainerName {
			source.ResourceFieldRef.ContainerName = targetName
		}
	}
}

func retargetResourceFieldReferencesInDownwardAPI(items []corev1.DownwardAPIVolumeFile, targetName string) {
	// DownwardAPI files require the renamed container's exact Kubernetes name.
	for index := range items {
		selector := items[index].ResourceFieldRef
		if selector != nil && selector.ContainerName == commonconsts.MainContainerName {
			selector.ContainerName = targetName
		}
	}
}

func addRuntimeTemporaryStorage(podSpec *corev1.PodSpec, container *corev1.Container, replaceExisting bool) {
	// Give conductor expansion and Agent setup a pod-lifetime writable workspace.
	volume := corev1.Volume{
		Name: runtimeTemporaryStorageVolumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}
	mount := corev1.VolumeMount{Name: runtimeTemporaryStorageVolumeName, MountPath: runtimeTemporaryStorageMountPath}
	if replaceExisting {
		podSpec.Volumes = setVolumeByName(podSpec.Volumes, volume)
		container.VolumeMounts = setVolumeMount(container.VolumeMounts, mount)
		return
	}
	if !slices.ContainsFunc(container.VolumeMounts, func(mount corev1.VolumeMount) bool {
		return mount.MountPath == runtimeTemporaryStorageMountPath
	}) {
		container.VolumeMounts = append(container.VolumeMounts, mount)
	}
	podSpec.Volumes = appendVolumeIfMissing(podSpec.Volumes, volume)
}
