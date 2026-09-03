package dynamo

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	SglangPort = "29500"

	maxTCPPort = 65535
)

type SGLangBackend struct{}

// isPythonCommand checks if the command is a Python interpreter
func isPythonCommand(cmd string) bool {
	if cmd == "python" || cmd == "python3" {
		return true
	}
	// Match python with version numbers like python3.11, python2.7, etc.
	// Also support absolute paths like /usr/bin/python3.8, /opt/python/bin/python3.11
	matched, _ := regexp.MatchString(`^(.*/)?(python\d*(\.\d+)*)$`, cmd)
	return matched
}

func (b *SGLangBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUCount ContainerGPUCount) error {
	// Reserve the exporter ports before any early return: a single-node worker
	// is exactly the case that co-locates every rank in one container.
	if err := reserveNixlExporterPorts(container, containerGPUCount); err != nil {
		return err
	}

	if component.CompilationCache != nil {
		logger := log.Log.WithName("sglang-backend")
		logger.Info("Compilation cache configured for SGLang but not yet fully supported",
			"backend", "sglang",
			"status", "partial-support",
			"cache-dir", component.CompilationCache.MountPath,
			"env-vars-set", false,
			"next-steps", "upstream SGLang changes needed")
	}

	// For single node, nothing to do
	if numberOfNodes <= 1 {
		return nil
	}

	// Remove probes for multinode worker
	if role == RoleWorker {
		container.LivenessProbe = nil
		container.ReadinessProbe = nil
		container.StartupProbe = nil
	}

	// Generate the flags to add
	flags, needsShell := b.getMultinodeFlags(numberOfNodes, role, serviceName, multinodeDeployer)
	if flags == "" {
		return nil
	}

	injectFlagsIntoContainerCommand(container, flags, needsShell, "sglang")
	return nil
}

// reserveNixlExporterPorts declares one NIXL exporter port per node-local rank.
// Skips containers without a nixl port or with NIXL_TELEMETRY_ENABLE set off.
func reserveNixlExporterPorts(container *corev1.Container, containerGPUCount ContainerGPUCount) error {
	basePort := findContainerPort(container, commonconsts.DynamoNixlPortName)
	if basePort == nil {
		return nil
	}

	enabled := findEnvVar(container.Env, "NIXL_TELEMETRY_ENABLE")
	if enabled == nil {
		return nil
	}
	// The operator cannot resolve valueFrom, so reserve the range rather than
	// assume telemetry is off: an unused declaration is harmless, a missing one
	// leaves every rank past the base unscrapeable.
	if enabled.ValueFrom == nil && !strings.EqualFold(strings.TrimSpace(enabled.Value), "y") {
		return nil
	}

	containerGPUs, err := containerGPUCount()
	if err != nil {
		return fmt.Errorf("failed to resolve container GPUs: %w", err)
	}

	// Rank i binds NIXL_TELEMETRY_PROMETHEUS_PORT+i, so a literal override moves
	// the whole range: realign `nixl` with it or it advertises a port rank 0
	// never binds.
	if override, ok := literalPort(findEnvVar(container.Env, "NIXL_TELEMETRY_PROMETHEUS_PORT")); ok {
		basePort.ContainerPort = override
	}

	colocatedRanks := min(containerGPUs, int64(commonconsts.DynamoMaxNixlPorts))
	if last := int64(basePort.ContainerPort) + colocatedRanks - 1; last > maxTCPPort {
		return fmt.Errorf(
			"NIXL_TELEMETRY_PROMETHEUS_PORT=%d with %d co-located ranks needs ports %d-%d, which exceeds the maximum port %d",
			basePort.ContainerPort, colocatedRanks, basePort.ContainerPort, last, maxTCPPort)
	}

	for rank := int64(1); rank < colocatedRanks; rank++ {
		name := fmt.Sprintf("%s-%d", commonconsts.DynamoNixlPortName, rank)
		if findContainerPort(container, name) != nil {
			continue
		}
		container.Ports = append(container.Ports, corev1.ContainerPort{
			Protocol:      corev1.ProtocolTCP,
			Name:          name,
			ContainerPort: basePort.ContainerPort + int32(rank),
		})
	}

	return nil
}

// literalPort reads a TCP port written inline on an environment variable. A
// value taken from valueFrom is resolved in the container at startup and is
// reported as absent here, as is a value that is not a usable port: neither can
// be turned into a container port declaration.
func literalPort(env *corev1.EnvVar) (int32, bool) {
	if env == nil || env.ValueFrom != nil {
		return 0, false
	}

	port, err := strconv.Atoi(strings.TrimSpace(env.Value))
	if err != nil || port < 1 || port > 65535 {
		return 0, false
	}
	return int32(port), true
}

func (b *SGLangBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	// do nothing
}

// getMultinodeFlags returns the multinode flags and whether shell interpretation is needed
func (b *SGLangBackend) getMultinodeFlags(numberOfNodes int32, role Role, serviceName string, multinodeDeployer MultinodeDeployer) (string, bool) {
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)

	var nodeRank string
	var needsShell bool

	if role == RoleLeader {
		nodeRank = "0"
		needsShell = false
	} else {
		nodeRank, needsShell = multinodeDeployer.GetNodeRank()
	}
	distInitAddr := fmt.Sprintf("%s:%s", leaderHostname, SglangPort)

	flags := fmt.Sprintf("--dist-init-addr %s --nnodes %d --node-rank %s", distInitAddr, numberOfNodes, nodeRank)
	return flags, needsShell
}
