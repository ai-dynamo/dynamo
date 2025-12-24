package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
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
		updateVLLMMultinodeArgs(container, role, serviceName, multinodeDeployer, component.Resources, numberOfNodes)

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
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, resources *v1alpha1.Resources, numberOfNodes int32) {
	expandedArgs := getExpandedArgs(container)
	if needsRayDistributedLaunch(expandedArgs, resources) {
		injectRayDistributedLaunchFlags(container, role, serviceName, multinodeDeployer)
	} else if needsDataParallelLaunch(expandedArgs, resources) {
		injectDataParallelLaunchFlags(container, role, serviceName, multinodeDeployer, resources, numberOfNodes)
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
		// Use Ray executor for multi-node vLLM deployments.
		// vLLM will create a placement group spanning all Ray nodes and spawn workers automatically.
		// DO NOT pass --nnodes or --node-rank - these are only for mp backend.
		// The Ray executor handles multi-node distribution via placement groups.
		vllmMultinodeFlags := "--distributed-executor-backend ray"
		container.Args = []string{fmt.Sprintf("ray start --head --port=%s && %s %s %s", VLLMPort, fullCommand, originalArgs, vllmMultinodeFlags)}
	case RoleWorker:
		// Worker nodes only run Ray agent - vLLM on leader will spawn Ray actors on workers
		leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
		container.Args = []string{fmt.Sprintf("ray start --address=%s:%s --block", leaderHostname, VLLMPort)}
	}
	container.Command = []string{"/bin/sh", "-c"} // ensure cmd is a shell
}

func injectDataParallelLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, resources *v1alpha1.Resources, numberOfNodes int32) {
	expandedArgs := getExpandedArgs(container)
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)

	// Calculate engines per node
	containerGPUs := getContainerGPUs(resources)
	worldSize := getWorldSize(expandedArgs) // TP * PP per engine
	dataParallelSizeLocal := containerGPUs / worldSize

	// Get total DP size from args, or calculate from nodes
	totalDPSize := getFlagValue(expandedArgs, dataParallelSizeFlag)
	if totalDPSize == 1 {
		totalDPSize = dataParallelSizeLocal * int64(numberOfNodes)
	}

	var flags []string
	needsShell := false

	// Helper to check if flag already exists in args
	hasFlag := func(flag string) bool {
		for _, arg := range expandedArgs {
			if arg == flag {
				return true
			}
		}
		return false
	}

	switch role {
	case RoleLeader:
		// Leader runs API server + coordinator + local engines
		// Hybrid LB mode: local DP coordination within node, Dynamo routes between nodes
		flags = []string{"--data-parallel-hybrid-lb"}
		// Only inject --data-parallel-size if not already present (avoids duplicates from profiler)
		if !hasFlag("--data-parallel-size") {
			flags = append(flags, "--data-parallel-size", strconv.FormatInt(totalDPSize, 10))
		}
		flags = append(flags,
			"--data-parallel-size-local", strconv.FormatInt(dataParallelSizeLocal, 10),
			"--data-parallel-start-rank", "0",
			"--data-parallel-address", leaderHostname,
			"--data-parallel-rpc-port", dataParallelRPCPort,
		)

	case RoleWorker:
		// Worker runs API server + coordinator + local engines on its node
		// Hybrid LB mode: local DP coordination within node, Dynamo routes between nodes
		nodeRank, _ := multinodeDeployer.GetNodeRank()
		startRank := fmt.Sprintf("$(( %d * %s ))", dataParallelSizeLocal, nodeRank)
		needsShell = true // Need shell for arithmetic expansion

		flags = []string{"--data-parallel-hybrid-lb"}
		// Only inject --data-parallel-size if not already present (avoids duplicates from profiler)
		if !hasFlag("--data-parallel-size") {
			flags = append(flags, "--data-parallel-size", strconv.FormatInt(totalDPSize, 10))
		}
		flags = append(flags,
			"--data-parallel-size-local", strconv.FormatInt(dataParallelSizeLocal, 10),
			"--data-parallel-start-rank", startRank,
			"--data-parallel-address", leaderHostname,
			"--data-parallel-rpc-port", dataParallelRPCPort,
		)
	}

	injectFlagsIntoContainerCommand(container, strings.Join(flags, " "), needsShell, "vllm")
}

// if world size (within DP rank) > GPU count, then we need to inject ray
// world size = tensor parallel size * pipeline parallel size
func needsRayDistributedLaunch(expandedArgs []string, resources *v1alpha1.Resources) bool {
	containerGPUs := getContainerGPUs(resources)
	if containerGPUs == 0 {
		return false
	}
	return getWorldSize(expandedArgs) > containerGPUs
}

func getWorldSize(expandedArgs []string) int64 {
	tensorParallelSize := getFlagValue(expandedArgs, tensorParallelSizeFlag)
	pipelineParallelSize := getFlagValue(expandedArgs, pipelineParallelSizeFlag)
	return tensorParallelSize * pipelineParallelSize
}

// if world size across all DP ranks > GPU count, then we need to inject data parallel multinode coordination
func needsDataParallelLaunch(expandedArgs []string, resources *v1alpha1.Resources) bool {
	dataParallelSize := getFlagValue(expandedArgs, dataParallelSizeFlag)
	containerGPUs := getContainerGPUs(resources)
	if containerGPUs == 0 {
		return false
	}
	return getWorldSize(expandedArgs)*dataParallelSize > containerGPUs
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

func getContainerGPUs(resources *v1alpha1.Resources) int64 {
	if resources == nil || resources.Limits == nil || resources.Limits.GPU == "" {
		return 0
	}
	if gpus, err := strconv.ParseInt(resources.Limits.GPU, 10, 64); err == nil {
		return gpus
	}
	return 0
}
