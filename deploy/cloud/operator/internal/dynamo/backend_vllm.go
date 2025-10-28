package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	VLLMPort                 = "6379"
	dataParallelRPCPort      = "13445"
	tensorParallelSizeFlag   = "--tensor-parallel-size"
	pipelineParallelSizeFlag = "--pipeline-parallel-size"
	dataParallelSizeFlag     = "--data-parallel-size"
)

type VLLMBackend struct{}

func (b *VLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	isMultinode := numberOfNodes > 1

	if isMultinode {
		// Apply multinode-specific argument modifications
		updateVLLMMultinodeArgs(container, role, serviceName, multinodeDeployer)

		// Remove probes for multinode worker and leader
		if role == RoleWorker {
			container.LivenessProbe = nil
			container.ReadinessProbe = nil
			container.StartupProbe = nil
		}
	}

	// Set compilation cache environment variables for VLLM
	cacheDir := ""

	// Check for volumeMounts with useAsCompilationCache=true
	for _, volumeMount := range component.VolumeMounts {
		if volumeMount.UseAsCompilationCache {
			cacheDir = volumeMount.MountPoint
			break
		}
	}

	if cacheDir != "" {
		// Set VLLM cache directory using the environment variable
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "VLLM_CACHE_ROOT",
			Value: cacheDir,
		})

		// Log confirmation that compilation cache is configured for VLLM
		logger := log.Log.WithName("vllm-backend")
		logger.Info("Compilation cache configured and enabled for VLLM backend",
			"backend", "vllm",
			"status", "fully-supported",
			"cache-dir", cacheDir,
			"use-as-compilation-cache", true,
			"env-vars-set", true,
			"env-vars", "VLLM_CACHE_ROOT")
	}
}

func (b *VLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentSharedSpec, serviceName string) {
	// do nothing
}

// updateVLLMMultinodeArgs will inject Ray-specific flags for tensor parallel multinode deployments
// OR data parallel flags for data parallel multinode deployments
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) {
	expandedArgs := getExpandedArgs(container)
	if needsRayDistributedLaunch(expandedArgs, container.Resources) {
		injectRayDistributedLaunchFlags(container, role, serviceName, multinodeDeployer)
	} else if needsDataParallelLaunch(expandedArgs, container.Resources) {
		injectDataParallelLaunchFlags(container, role, serviceName, multinodeDeployer)
	} else {
		logger := log.Log.WithName("vllm-backend")
		logger.Info("No need to inject Ray-specific or data parallel flags for multinode deployments", "args", strings.Join(container.Args, " "))
	}
}

// getExpandedArgs will expand the containers args in the case where
// the args are joined together with spaces as an individual string (i.e. "python3 -m dynamo.vllm")
func getExpandedArgs(container *corev1.Container) []string {
	expandedArgs := []string{}
	for _, arg := range container.Args {
		expandedArgs = append(expandedArgs, strings.Fields(arg)...)
	}
	return expandedArgs
}

func injectRayDistributedLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) {
	switch role {
	case RoleLeader:
		fullCommand := strings.Join(container.Command, " ")
		originalArgs := strings.Join(container.Args, " ")
		// Prepend ray start --head command to existing args
		container.Args = []string{fmt.Sprintf("ray start --head --port=%s && %s %s", VLLMPort, fullCommand, originalArgs)}
	case RoleWorker:
		// Worker nodes only run Ray, completely replace args
		leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
		container.Args = []string{fmt.Sprintf("ray start --address=%s:%s --block", leaderHostname, VLLMPort)}
	}
	container.Command = []string{"/bin/sh", "-c"} // ensure cmd is a shell
}

func injectDataParallelLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) {
	expandedArgs := getExpandedArgs(container)
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	dataParallelSizeLocal := getContainerGPUs(container.Resources) / getWorldSize(expandedArgs)
	var startRank string
	switch role {
	case RoleWorker:
		nodeRank, _ := multinodeDeployer.GetNodeRank()
		startRank = fmt.Sprintf("$(( %d * %s ))", dataParallelSizeLocal, nodeRank)
	case RoleLeader:
		startRank = "0" // leader start rank is always 0
	default:
		startRank = "0"
	}
	flags := []string{
		"--data-parallel-address", leaderHostname,
		"--data-parallel-size-local", strconv.FormatInt(dataParallelSizeLocal, 10),
		"--data-parallel-rpc-port", dataParallelRPCPort,
		"--data-parallel-start-rank", startRank,
	}
	injectFlagsIntoContainerCommand(container, strings.Join(flags, " "), true, "vllm")
}

// if world size (within DP rank) > GPU count, then we need to inject ray
// world size = tensor parallel size * pipeline parallel size
func needsRayDistributedLaunch(expandedArgs []string, resources corev1.ResourceRequirements) bool {
	return getWorldSize(expandedArgs) > getContainerGPUs(resources)
}

func getWorldSize(expandedArgs []string) int64 {
	tensorParallelSize := getFlagValue(expandedArgs, tensorParallelSizeFlag)
	pipelineParallelSize := getFlagValue(expandedArgs, pipelineParallelSizeFlag)
	return tensorParallelSize * pipelineParallelSize
}

// if world size across all DP ranks > GPU count, then we need to inject data parallel multinode coordination
func needsDataParallelLaunch(expandedArgs []string, resources corev1.ResourceRequirements) bool {
	dataParallelSize := getFlagValue(expandedArgs, dataParallelSizeFlag)
	return getWorldSize(expandedArgs)*dataParallelSize > getContainerGPUs(resources)
}

func getFlagValue(expandedArgs []string, flag string) int64 {
	var flagValue int64 = 1
	for i, arg := range expandedArgs {
		if arg == flag && (i+1 < len(expandedArgs)) {
			flagValue, err := strconv.ParseInt(expandedArgs[i+1], 10, 64)
			if err != nil {
				continue
			}
			return flagValue
		}
	}
	return flagValue
}

func getContainerGPUs(resources corev1.ResourceRequirements) int64 {
	var containerGPUs int64 = 1
	// Requests defaults to Limits, doesn't make sense in case where Requests < Limits for gpus
	for name, quantity := range resources.Limits {
		if name.String() == consts.KubeResourceGPUNvidia {
			containerGPUs = quantity.Value()
			break
		}
	}
	return containerGPUs
}
