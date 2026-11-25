package dynamo

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	SglangPort = "29500"

	maxTCPPort = 65535
)

type SGLangBackend struct {
	roleLaunchOwnership roleLaunchOwnership
	environment         *podEnvironment
}

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
	// Older origins keep their port declarations; semantic origins read the final env.
	if b.environment == nil {
		if err := reserveNixlExporterPorts(container, containerGPUCount, nil); err != nil {
			return err
		}
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
	if b.roleLaunchOwnership == roleLaunchOwnedByPodTemplate {
		return nil
	}

	// Generate the flags to add
	flags, needsShell := b.getMultinodeFlags(
		numberOfNodes,
		role,
		serviceName,
		multinodeDeployer,
		usesMultinodeTopologyAliases(GetPodTemplateAnnotations(component)),
	)
	if flags == "" {
		return nil
	}

	injectFlagsIntoContainerCommand(container, flags, needsShell, "sglang")
	return nil
}

// finalizeSGLangExporterPorts reads composed engine environments to validate and
// declare NIXL exporter ports. pod must not be nil; nil engineNames selects none.
// Engines without scrape-port declarations are validated without recreating ports.
func finalizeSGLangExporterPorts(pod *corev1.PodSpec, engineNames []string, containerGPUs ContainerGPUCount) error {
	for i := range pod.Containers {
		container := &pod.Containers[i]
		if !slices.Contains(engineNames, container.Name) {
			continue
		}
		resolved := semanticenv.Replay(container.Env)
		if findContainerPort(container, commonconsts.DynamoNixlPortName) == nil {
			if _, err := resolveNixlExporterPortRange(container.Env, containerGPUs, resolved, commonconsts.DynamoNixlPort); err != nil {
				return err
			}
		} else if err := reserveNixlExporterPorts(container, containerGPUs, resolved); err != nil {
			return err
		}
	}
	return nil
}

// reserveNixlExporterPorts declares one NIXL exporter port per node-local rank.
// Skips containers without a nixl port, with NIXL_TELEMETRY_ENABLE set off, or
// with an exporter other than Prometheus selected. Literal enable values other
// than y retain their existing port declarations for upgrade compatibility.
// A nil resolved map selects the legacy raw reader and its diagnostics.
func reserveNixlExporterPorts(container *corev1.Container, containerGPUCount ContainerGPUCount, resolved map[string]semanticenv.Value) error {
	basePort := findContainerPort(container, commonconsts.DynamoNixlPortName)
	if basePort == nil {
		return nil
	}
	ports, err := resolveNixlExporterPortRange(container.Env, containerGPUCount, resolved, basePort.ContainerPort)
	if err != nil || ports == nil {
		return err
	}

	// Rank ports authored against another base must move too: otherwise Prometheus
	// scrapes stale ports that no rank binds. Appending invalidates basePort.
	basePort.ContainerPort = ports.base
	for rank := int64(1); rank < ports.ranks; rank++ {
		name := fmt.Sprintf("%s-%d", commonconsts.DynamoNixlPortName, rank)
		rankPort := ports.base + int32(rank)
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

type nixlExporterPortRange struct {
	base  int32
	ranks int64
}

// resolveNixlExporterPortRange validates engine inputs independently of optional
// Kubernetes scrape-port declarations. Nil means no static range can be selected.
func resolveNixlExporterPortRange(env []corev1.EnvVar, containerGPUCount ContainerGPUCount, resolved map[string]semanticenv.Value, basePort int32) (*nixlExporterPortRange, error) {
	// Preserve raw lookup semantics for old origins, without expanding or rewriting env.
	ordered := resolved != nil
	if !ordered {
		resolved = make(map[string]semanticenv.Value, 3)
		for _, name := range []string{"NIXL_TELEMETRY_ENABLE", "NIXL_TELEMETRY_EXPORTER", "NIXL_TELEMETRY_PROMETHEUS_PORT"} {
			if value := findEnvVar(env, name); value != nil {
				resolved[name] = semanticenv.Value{Value: value.Value, Known: value.ValueFrom == nil}
			}
		}
	}
	enabled, exists := resolved["NIXL_TELEMETRY_ENABLE"]
	if !exists {
		return nil, nil
	}

	// For unresolved runtime inputs, reserve the range rather than
	// assume telemetry is off: an unused declaration is harmless, a missing one
	// leaves every rank past the base unscrapeable. Name each setting it cannot
	// read, because the diagnostic below is the only place a deployment learns
	// which one left telemetry undecided.
	var sourced []string
	prometheusOn := enabled.Known
	if !prometheusOn {
		sourced = append(sourced, "NIXL_TELEMETRY_ENABLE")
	}

	// Preserve the existing pod template for non-y truthy values. Users opt in
	// to rank port declarations by changing the CR's enable value to y.
	if prometheusOn {
		switch strings.ToLower(enabled.Value) {
		case "y":
		case "1", "yes", "on", "true", "enable":
			return nil, nil
		case "n", "0", "no", "off", "false", "disable":
			return nil, nil
		default:
			return nil, fmt.Errorf(
				"NIXL_TELEMETRY_ENABLE is set to %q, which is not a boolean value recognized by NIXL",
				enabled.Value)
		}
	}

	// Only the Prometheus exporter binds a port per rank, so activate on the
	// same pair of variables nixl_prometheus_base_port() reads. An absent
	// selection or any other literal selection needs no ports at all, while a
	// sourced one joins an unreadable enable value in reserving a range this
	// code cannot rule out.
	exporter, exists := resolved["NIXL_TELEMETRY_EXPORTER"]
	if !exists {
		return nil, nil
	}
	if !exporter.Known {
		prometheusOn = false
		sourced = append(sourced, "NIXL_TELEMETRY_EXPORTER")
	} else if exporter.Value != "prometheus" {
		return nil, nil
	}

	// Rank i binds NIXL_TELEMETRY_PROMETHEUS_PORT+i, so a literal override moves
	// the whole range: realign `nixl` with it or it advertises a port rank 0
	// never binds.
	if override, exists := resolved["NIXL_TELEMETRY_PROMETHEUS_PORT"]; exists {
		// An unknown runtime base is allowed, but cannot determine static scrape ports.
		if !override.Known {
			if ordered {
				return nil, nil
			}
			return nil, fmt.Errorf(
				"NIXL_TELEMETRY_PROMETHEUS_PORT is set through valueFrom, so the operator cannot declare the exporter "+
					"range as %s container ports and Prometheus would scrape a range no rank binds. Set "+
					"NIXL_TELEMETRY_PROMETHEUS_PORT to a literal port, or set NIXL_TELEMETRY_ENABLE=n",
				commonconsts.DynamoNixlPortName)
		}
		overridden, err := nixlPrometheusPort(override.Value)
		if err != nil {
			// Reject a setting that cannot name the fixed range declared below. NIXL
			// throws on malformed and out-of-range values; it accepts zero as an
			// ephemeral port, but the operator cannot declare or scrape that port.
			return nil, fmt.Errorf(
				"NIXL_TELEMETRY_PROMETHEUS_PORT is the base of the NIXL exporter range and %w, so the operator "+
					"cannot declare the range it names and Prometheus would scrape ports no rank binds. Set "+
					"NIXL_TELEMETRY_PROMETHEUS_PORT to a port between 1 and %d, or set NIXL_TELEMETRY_ENABLE=n",
				err, maxTCPPort)
		}
		basePort = overridden
	}

	// Resolve the rank count only when the base is known.
	containerGPUs, err := containerGPUCount()
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container GPUs: %w", err)
	}

	// Every co-located rank needs a port of its own, and the runtime refuses a
	// rank past the reserved range rather than share one: truncating the count
	// here would fail startup on the ranks that lost their port. An unreadable
	// telemetry setting is exempt, because refusing a deployment that may not
	// use telemetry at all costs more than the over-declaration above.
	colocatedRanks := containerGPUs
	if colocatedRanks > int64(commonconsts.DynamoMaxNixlPorts) {
		if prometheusOn {
			return nil, fmt.Errorf(
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
			"co-located GPUs exceed the NIXL exporter ports the operator declares, and the settings named below depend on unresolved runtime inputs, "+
				"so the deployment is admitted with the supported range reserved; if they resolve to a Prometheus exporter, the ranks past it fail to start",
			"colocatedGPUs", colocatedRanks,
			"reservedPorts", commonconsts.DynamoMaxNixlPorts,
			"unresolvedSettings", strings.Join(sourced, ", "))

		colocatedRanks = int64(commonconsts.DynamoMaxNixlPorts)
	}

	if last := int64(basePort) + colocatedRanks - 1; last > maxTCPPort {
		if ordered && !prometheusOn {
			colocatedRanks = int64(maxTCPPort-basePort) + 1
		} else {
			return nil, fmt.Errorf(
				"NIXL_TELEMETRY_PROMETHEUS_PORT=%d with %d co-located ranks needs ports %d-%d, which exceeds the maximum port %d",
				basePort, colocatedRanks, basePort, last, maxTCPPort)
		}
	}
	return &nixlExporterPortRange{base: basePort, ranks: colocatedRanks}, nil
}

// nixlPrometheusPort parses a known NIXL value and requires a fixed nonzero port.
func nixlPrometheusPort(value string) (int32, error) {
	base := 10
	digits := value
	if strings.HasPrefix(value, "0x") || strings.HasPrefix(value, "0X") {
		base = 16
		digits = value[2:]
	}
	port, err := strconv.ParseUint(digits, base, 64)
	if err != nil {
		return 0, fmt.Errorf("is set to %q, which is not a number", value)
	}
	if port == 0 {
		return 0, fmt.Errorf(
			"is set to 0, which asks NIXL for an ephemeral port the operator cannot declare or scrape")
	}
	if port > maxTCPPort {
		return 0, fmt.Errorf("is set to %d, which is outside the port range 1-%d", port, maxTCPPort)
	}
	return int32(port), nil
}

func (b *SGLangBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	// do nothing
}

// getMultinodeFlags returns the multinode flags and whether shell interpretation
// is needed. Topology aliases keep new DGD command lines provider-independent.
func (b *SGLangBackend) getMultinodeFlags(
	numberOfNodes int32,
	role Role,
	serviceName string,
	multinodeDeployer MultinodeDeployer,
	useTopologyAliases bool,
) (string, bool) {
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	if useTopologyAliases {
		leaderHostname = commonconsts.DynamoLeaderAddressEnvVarReference
	}

	var nodeRank string
	var needsShell bool

	if useTopologyAliases {
		nodeRank = commonconsts.DynamoRankEnvVarReference
	} else if role == RoleLeader {
		nodeRank = "0"
		needsShell = false
	} else {
		nodeRank, needsShell = multinodeDeployer.GetNodeRank()
	}
	distInitAddr := fmt.Sprintf("%s:%s", leaderHostname, SglangPort)

	flags := fmt.Sprintf("--dist-init-addr %s --nnodes %d --node-rank %s", distInitAddr, numberOfNodes, nodeRank)
	return flags, needsShell
}
