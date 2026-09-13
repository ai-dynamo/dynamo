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
// Skips containers without a nixl port, with NIXL_TELEMETRY_ENABLE set off, or
// with an exporter other than Prometheus selected.
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
	// leaves every rank past the base unscrapeable. Name each setting it cannot
	// read, because the diagnostic below is the only place a deployment learns
	// which one left telemetry undecided.
	var sourced []string
	prometheusOn := enabled.ValueFrom == nil
	if !prometheusOn {
		sourced = append(sourced, "NIXL_TELEMETRY_ENABLE")
	}
	if prometheusOn {
		switch strings.ToLower(enabled.Value) {
		case "y", "1", "yes", "on", "true", "enable":
		default:
			return nil
		}
	}

	// Only the Prometheus exporter binds a port per rank, so activate on the
	// same pair of variables nixl_prometheus_base_port() reads. An absent
	// selection or any other literal selection needs no ports at all, while a
	// sourced one joins an unreadable enable value in reserving a range this
	// code cannot rule out.
	exporter := findEnvVar(container.Env, "NIXL_TELEMETRY_EXPORTER")
	if exporter == nil {
		return nil
	}
	if exporter.ValueFrom != nil {
		prometheusOn = false
		sourced = append(sourced, "NIXL_TELEMETRY_EXPORTER")
	} else if exporter.Value != "prometheus" {
		return nil
	}

	containerGPUs, err := containerGPUCount()
	if err != nil {
		return fmt.Errorf("failed to resolve container GPUs: %w", err)
	}

	// Rank i binds NIXL_TELEMETRY_PROMETHEUS_PORT+i, so a literal override moves
	// the whole range: realign `nixl` with it or it advertises a port rank 0
	// never binds.
	override := findEnvVar(container.Env, "NIXL_TELEMETRY_PROMETHEUS_PORT")
	overridden, literal, err := nixlPrometheusPort(override)
	switch {
	case err != nil:
		// A base that is present but unusable is the quietest way to lose the
		// metrics: NIXL binds 0 as an ephemeral port and reports back no port to
		// scrape, so the container starts while the ports declared here and the
		// PodMonitor still name the default range. Say so at admission instead.
		return fmt.Errorf(
			"NIXL_TELEMETRY_PROMETHEUS_PORT is the base of the NIXL exporter range and %w, so the operator "+
				"cannot declare the range it names and Prometheus would scrape ports no rank binds. Set "+
				"NIXL_TELEMETRY_PROMETHEUS_PORT to a port between 1 and %d, or set NIXL_TELEMETRY_ENABLE=n",
			err, maxTCPPort)
	case literal:
		basePort.ContainerPort = overridden
	case override != nil && override.ValueFrom != nil:
		// A sourced base has no conservative fallback the way a sourced enable
		// value does: the container resolves it and binds that range, while the
		// declared ports and the PodMonitor stay on the base written here, so
		// the metrics disappear instead of merely being over-declared. That
		// holds however the enable value is written, so an unreadable one is no
		// reason to accept the base and declare a range nothing binds.
		return fmt.Errorf(
			"NIXL_TELEMETRY_PROMETHEUS_PORT is set through valueFrom, so the operator cannot declare the exporter "+
				"range as %s container ports and Prometheus would scrape a range no rank binds. Set "+
				"NIXL_TELEMETRY_PROMETHEUS_PORT to a literal port, or set NIXL_TELEMETRY_ENABLE=n",
			commonconsts.DynamoNixlPortName)
	}

	// Every co-located rank needs a port of its own, and the runtime refuses a
	// rank past the reserved range rather than share one: truncating the count
	// here would fail startup on the ranks that lost their port. An unreadable
	// telemetry setting is exempt, because refusing a deployment that may not
	// use telemetry at all costs more than the over-declaration above.
	colocatedRanks := containerGPUs
	if colocatedRanks > int64(commonconsts.DynamoMaxNixlPorts) {
		if prometheusOn {
			return fmt.Errorf(
				"%d co-located GPUs each need a NIXL exporter port, but only %d consecutive ports are declared and scraped, "+
					"so the ranks past the %dth would fail to start. Run at most %d ranks per container, "+
					"or set NIXL_TELEMETRY_ENABLE=n",
				colocatedRanks, commonconsts.DynamoMaxNixlPorts, commonconsts.DynamoMaxNixlPorts, commonconsts.DynamoMaxNixlPorts)
		}

		// Admitting the deployment leaves one outcome admission cannot rule
		// out, so say so here rather than let it surface as an unexplained
		// startup failure: settings that resolve to an enabled Prometheus
		// exporter give the ranks past the reserved range no port, and each of
		// those refuses to start.
		log.Log.WithName("sglang-backend").Info(
			"co-located GPUs exceed the NIXL exporter ports the operator declares, and the settings named below are set through valueFrom, "+
				"so the deployment is admitted with the supported range reserved; if they resolve to a Prometheus exporter, the ranks past it fail to start",
			"colocatedGPUs", colocatedRanks,
			"reservedPorts", commonconsts.DynamoMaxNixlPorts,
			"unresolvedSettings", strings.Join(sourced, ", "))

		colocatedRanks = int64(commonconsts.DynamoMaxNixlPorts)
	}

	if last := int64(basePort.ContainerPort) + colocatedRanks - 1; last > maxTCPPort {
		return fmt.Errorf(
			"NIXL_TELEMETRY_PROMETHEUS_PORT=%d with %d co-located ranks needs ports %d-%d, which exceeds the maximum port %d",
			basePort.ContainerPort, colocatedRanks, basePort.ContainerPort, last, maxTCPPort)
	}

	// A rank port already on the pod template was written against whatever base
	// was in effect then, so realign it rather than keep it: the PodMonitor
	// scrapes these by name, and a stale number sends it to a port the rank
	// never binds. Read the base once, because appending invalidates basePort.
	base := basePort.ContainerPort
	for rank := int64(1); rank < colocatedRanks; rank++ {
		name := fmt.Sprintf("%s-%d", commonconsts.DynamoNixlPortName, rank)
		rankPort := base + int32(rank)
		if declared := findContainerPort(container, name); declared != nil {
			declared.ContainerPort = rankPort
			continue
		}
		container.Ports = append(container.Ports, corev1.ContainerPort{
			Protocol:      corev1.ProtocolTCP,
			Name:          name,
			ContainerPort: rankPort,
		})
	}

	return nil
}

// nixlPrometheusPort reads a literal NIXL Prometheus port with NIXL's unsigned
// 16-bit syntax. An unset variable and one taken from valueFrom are both
// reported as absent, because a value resolved in the container at startup
// cannot be turned into a container port declaration here.
func nixlPrometheusPort(env *corev1.EnvVar) (int32, bool, error) {
	if env == nil || env.ValueFrom != nil {
		return 0, false, nil
	}

	value := env.Value
	base := 10
	digits := value
	if strings.HasPrefix(value, "0x") || strings.HasPrefix(value, "0X") {
		base = 16
		digits = value[2:]
	}
	port, err := strconv.ParseUint(digits, base, 64)
	if err != nil {
		return 0, false, fmt.Errorf("is set to %q, which is not a number", env.Value)
	}
	if port < 1 || port > maxTCPPort {
		return 0, false, fmt.Errorf("is set to %d, which is outside the port range 1-%d", port, maxTCPPort)
	}
	return int32(port), true, nil
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
