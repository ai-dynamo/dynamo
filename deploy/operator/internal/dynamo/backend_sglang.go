package dynamo

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	SglangPort = "29500"
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

// reserveNixlExporterPorts declares one NIXL Prometheus exporter port per
// node-local rank on an SGLang worker container.
//
// SGLang runs one scheduler process per node-local GPU and each of those
// processes builds its own NIXL agent, so the single DynamoNixlPortName port
// the worker container declares covers rank 0 only. dynamo.sglang gives rank i
// DynamoNixlPort+i (see components/src/dynamo/sglang/nixl_telemetry.py); a port
// the pod does not declare generates no PodMonitor target, so the whole range
// has to be declared here for those ranks to be scraped.
//
// Two containers are left alone. Those without DynamoNixlPortName - the
// frontend, planner and router share this backend - never run an exporter. So
// do those whose resolved NIXL_TELEMETRY_ENABLE is not "y", which is the
// operator default: reserving there would declare ports nothing binds, and
// resolving the GPU count is not free (a DRA-backed claim needs an API read
// that some render paths cannot perform).
//
// A deployment that supplies NIXL_TELEMETRY_ENABLE indirectly, through envFrom
// or valueFrom, is not recognized here. Those ranks still get distinct ports
// and still reach Ready; only their scrape targets are missing. Set the
// variable inline to have them declared.
func reserveNixlExporterPorts(container *corev1.Container, containerGPUCount ContainerGPUCount) error {
	if findContainerPort(container, commonconsts.DynamoNixlPortName) == nil {
		return nil
	}

	enabled := findEnvVar(container.Env, "NIXL_TELEMETRY_ENABLE")
	if enabled == nil || !strings.EqualFold(strings.TrimSpace(enabled.Value), "y") {
		return nil
	}

	containerGPUs, err := containerGPUCount()
	if err != nil {
		return fmt.Errorf("failed to resolve container GPUs: %w", err)
	}

	colocatedRanks := min(containerGPUs, int64(commonconsts.DynamoMaxNixlPorts))
	for rank := int64(1); rank < colocatedRanks; rank++ {
		name := fmt.Sprintf("%s-%d", commonconsts.DynamoNixlPortName, rank)
		if findContainerPort(container, name) != nil {
			continue
		}
		container.Ports = append(container.Ports, corev1.ContainerPort{
			Protocol:      corev1.ProtocolTCP,
			Name:          name,
			ContainerPort: int32(commonconsts.DynamoNixlPort + rank),
		})
	}

	return nil
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
