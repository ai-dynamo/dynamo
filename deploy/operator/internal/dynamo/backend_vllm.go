package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/featuregate"
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
		updateVLLMMultinodeArgs(container, role, serviceName, multinodeDeployer, component.Resources, numberOfNodes, component.Annotations)

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

// updateVLLMMultinodeArgs dispatches to the appropriate injection function based on
// parallelism strategy (TP/PP distributed vs data-parallel) and executor backend (mp vs ray).
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, resources *v1alpha1.Resources, numberOfNodes int32, annotations map[string]string) {
	expandedArgs := getExpandedArgs(container)
	if needsMultinodeDistributedLaunch(expandedArgs, resources) {
		if shouldUseMpBackend(annotations) {
			injectMpDistributedLaunchFlags(container, role, serviceName, multinodeDeployer, numberOfNodes)
		} else {
			injectRayDistributedLaunchFlags(container, role, serviceName, multinodeDeployer)
		}
	} else if needsDataParallelLaunch(expandedArgs, resources) {
		injectDataParallelLaunchFlags(container, role, serviceName, multinodeDeployer, resources, numberOfNodes)
	} else {
		logger := log.Log.WithName("vllm-backend")
		logger.Info("No need to inject distributed or data parallel flags for multinode deployments", "args", strings.Join(container.Args, " "))
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

// shouldUseMpBackend determines whether to use multiprocessing (mp) or Ray for vLLM
// multi-node distributed launches.
//
// Decision logic:
//  1. Explicit override annotation takes priority (user set "mp" or "ray")
//  2. Operator origin version feature gate: uses featuregate.VLLMMultiprocessing
func shouldUseMpBackend(annotations map[string]string) bool {
	logger := log.Log.WithName("vllm-backend")

	// Step 1: Check explicit override
	if override, exists := annotations[commonconsts.KubeAnnotationVLLMDistributedExecutorBackend]; exists {
		switch strings.ToLower(override) {
		case "mp":
			logger.Info("Using mp backend (explicit override)")
			return true
		case "ray":
			logger.Info("Using ray backend (explicit override)")
			return false
		default:
			logger.Info("Ignoring invalid vllm-distributed-executor-backend annotation value, falling through to version check",
				"value", override)
		}
	}

	// Step 2: Check operator origin version gate
	return featuregate.VLLMMultiprocessing.IsEnabled(annotations)
}

// injectMpDistributedLaunchFlags injects vLLM multiprocessing flags for multi-node TP/PP deployments.
//
// Leader: runs the original vLLM command with --distributed-executor-backend mp,
// --nnodes, --node-rank 0, --master-addr, --master-port
//
// Worker: waits for leader's master port to be listening (TCP port-wait loop),
// then runs the same vLLM command with --headless, --node-rank <rank>, and the same coordination flags
func injectMpDistributedLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, numberOfNodes int32) {
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	mpFlags := fmt.Sprintf("--distributed-executor-backend mp --nnodes %d --master-addr %s --master-port %s",
		numberOfNodes, leaderHostname, commonconsts.VLLMMpMasterPort)

	needsShell := false

	switch role {
	case RoleLeader:
		mpFlags += " --node-rank 0"
	case RoleWorker:
		nodeRank, _ := multinodeDeployer.GetNodeRank()
		needsShell = true // Always need shell for port-wait loop
		mpFlags += fmt.Sprintf(" --node-rank %s --headless", nodeRank)
	}

	injectFlagsIntoContainerCommand(container, mpFlags, needsShell, "vllm")

	// For workers, prepend a port-wait loop to ensure the leader's master port
	// is listening before starting vLLM distributed init.
	if role == RoleWorker {
		injectMpWorkerPortWait(container, leaderHostname)
	}
}

// injectMpWorkerPortWait prepends a TCP port-wait loop to the worker container command.
// The worker waits until the leader's master port (used by PyTorch TCPStore for distributed init)
// is accepting connections before starting the vLLM process.
// Uses Python's socket module since nc/netcat may not be available in the container image.
// Errors are redirected to /tmp/mp-leader-wait.log for debugging (e.g. DNS failures)
// without flooding pod logs with expected connection-refused errors every 2s.
func injectMpWorkerPortWait(container *corev1.Container, leaderHostname string) {
	if len(container.Args) == 0 {
		return
	}

	waitPrefix := fmt.Sprintf(
		`echo 'Waiting for leader master port at %s:%s...' && until python3 -c 'import socket; s=socket.create_connection(("%s", %s), timeout=2); s.close()' 2>>/tmp/mp-leader-wait.log; do sleep 2; done && echo 'Leader master port ready' && `,
		leaderHostname, commonconsts.VLLMMpMasterPort, leaderHostname, commonconsts.VLLMMpMasterPort,
	)

	container.Args[0] = waitPrefix + container.Args[0]
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

// needsMultinodeDistributedLaunch returns true when the model's world size (TP * PP)
// exceeds the GPU count of a single node, requiring multi-node distribution (via mp or ray).
func needsMultinodeDistributedLaunch(expandedArgs []string, resources *v1alpha1.Resources) bool {
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
