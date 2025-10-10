package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
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

func (b *VLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
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

func (b *VLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string) {
	// do nothing
}

// data parallel case
// data parallel backend defaults to mp, --data-parallel-backend can be set as ray
// need to set --data-parallel-address to leader address
// --data-parallel-rank can be set, should either not exist or be set to 1
// --data-parall-size-local needs to be set to pod gpu count
// --data-parallel-rpc-port needs to be set to a unique port (set as const)
// --data-parallel-start rank needs to be set to index*dp-size-local

// updateVLLMMultinodeArgs applies Ray-specific modifications for multinode deployments
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) {
	if needsRayDistributedLaunch(container) {
		switch role {
		// determine are we injecting ray
		case RoleLeader:
			if len(container.Args) > 0 {
				// Prepend ray start --head command to existing args
				container.Args = []string{fmt.Sprintf("ray start --head --port=%s && %s", VLLMPort, strings.Join(container.Args, " "))}
			}
		case RoleWorker:
			// Worker nodes only run Ray, completely replace args
			leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
			container.Args = []string{fmt.Sprintf("ray start --address=%s:%s --block", leaderHostname, VLLMPort)}
		}
	} else if needsDataParallelLaunch(container) {
		leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
		dataParallelSizeLocal := getContainerGPUs(container)
		nodeRank, _ := multinodeDeployer.GetNodeRank()
		startRank := fmt.Sprintf("$(( %d * %s ))", dataParallelSizeLocal, nodeRank)
		container.Args = append(container.Args,
			"--data-parallel-address", leaderHostname,
			"--data-parallel-size-local", strconv.FormatInt(dataParallelSizeLocal, 10),
			"--data-parallel-rpc-port", dataParallelRPCPort,
			"--data-parallel-start-rank", startRank,
		)
	}
}

// if world size > GPU count, then we need to inject ray
// world size = tensor parallel size * pipeline parallel size
func needsRayDistributedLaunch(container *corev1.Container) bool {
	tensorParallelSize := getFlagValue(container, tensorParallelSizeFlag)
	pipelineParallelSize := getFlagValue(container, pipelineParallelSizeFlag)
	return tensorParallelSize*pipelineParallelSize > getContainerGPUs(container)
}

func needsDataParallelLaunch(container *corev1.Container) bool {
	dataParallelSize := getFlagValue(container, dataParallelSizeFlag)
	return dataParallelSize > getContainerGPUs(container)
}

func getFlagValue(container *corev1.Container, flag string) int64 {
	var flagValue int64 = 1
	for i, arg := range container.Args {
		if arg == flag && (i+1 < len(container.Args)) {
			flagValue, err := strconv.ParseInt(container.Args[i+1], 10, 64)
			if err != nil {
				continue
			}
			return flagValue
		}
	}
	return flagValue
}

func getContainerGPUs(container *corev1.Container) int64 {
	var containerGPUs int64 = 1
	// Requests defaults to Limits, doesn't make sense in case where Requests < Limits for gpus
	for name, quantity := range container.Resources.Limits {
		if name.String() == "nvidia.com/gpu" { // TODO: use const
			containerGPUs = quantity.Value()
			break
		}
	}
	return containerGPUs
}
