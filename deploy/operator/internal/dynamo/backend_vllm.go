package dynamo

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	VLLMPort                  = "6379"
	dataParallelRPCPort       = "13445"
	tensorParallelSizeFlag    = "--tensor-parallel-size"
	pipelineParallelSizeFlag  = "--pipeline-parallel-size"
	dataParallelSizeFlag      = "--data-parallel-size"
	dataParallelSizeLocalFlag = "--data-parallel-size-local"
	distributedExecutorFlag   = "--distributed-executor-backend"
	enableElasticEPFlag       = "--enable-elastic-ep"
	dataParallelBackendFlag   = "--data-parallel-backend"
	// dataParallelBackendShortFlag is vLLM's documented short alias for
	// --data-parallel-backend (see the v0.26.0 `vllm serve` CLI reference).
	dataParallelBackendShortFlag = "-dpb"
	dataParallelBackendRay       = "ray"
)

type VLLMBackend struct {
	ParentGraphDeploymentName string
}

func (b *VLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUCount ContainerGPUCount) error {
	// The inter-pod GMS layout (with or without failover) requires the engine
	// to load weights from the dedicated GMS weight-server pod rather than
	// from disk.
	if component.IsInterPodGMSEnabled() {
		if !containerHasArg(container, "--load-format", "gms") {
			injectFlagsIntoContainerCommand(container, "--load-format gms", false, "vllm")
		}
		if component.IsInterPodFailoverEnabled() {
			// DYN_VLLM_GMS_SHADOW_MODE activates vLLM's standby/failover behavior
			// on top of --load-format gms. Standalone inter-pod GMS should not set
			// it: those engines are active clients of the dedicated GMS weight
			// server, not shadow engines waiting for failover activation.
			container.Env = append(container.Env, corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"})
		}
	}

	isMultinode := numberOfNodes > 1
	annotations := GetPodTemplateAnnotations(component)

	if isMultinode {
		containerGPUs, err := containerGPUCount()
		if err != nil {
			return fmt.Errorf("failed to resolve container GPUs: %w", err)
		}

		// Apply multinode-specific argument modifications
		updateVLLMMultinodeArgs(container, role, serviceName, multinodeDeployer, containerGPUs, numberOfNodes, annotations)

		if shouldUseMpBackend(annotations) {
			container.Env = append(container.Env, corev1.EnvVar{
				Name: commonconsts.VLLMNixlSideChannelHostEnvVar,
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "status.podIP",
					},
				},
			})
		}

		// Remove probes for multinode workers.
		if role == RoleWorker {
			container.LivenessProbe = nil
			container.ReadinessProbe = nil
			container.StartupProbe = nil
		}
	} else if role == RoleMain && IsElasticEPRayLaunch(container) {
		// A single-pod elastic-EP component still needs a Ray head, so that
		// follower pods created later have a cluster to join. Only the leader
		// arm applies here: a lone pod is expanded as RoleMain, never RoleWorker.
		if injectElasticEPRayLaunchFlags(container, role, serviceName, multinodeDeployer) {
			// Bind both addresses only when a Ray head was actually injected.
			//
			// Both resolve from status.podIP, which is the point: the Ray head
			// registers under that address and vLLM searches for that address, so
			// the two cannot disagree. POD_IP is what the launch command
			// interpolates into --node-ip-address; VLLM_DP_MASTER_IP is what the
			// engine reads.
			//
			// The engine needs telling because at --data-parallel-size 1 vLLM
			// discards the DP master IP it derives from the Ray node and falls back
			// to VLLM_DP_MASTER_IP, which defaults to 127.0.0.1 — so it looks for a
			// node that does not exist and aborts with "The DP master node (ip:
			// 127.0.0.1) is missing or dead". Neither VLLM_HOST_IP nor
			// --data-parallel-address survives that overwrite; this env var is the
			// only value the fallback reads.
			podIPRef := func() *corev1.EnvVarSource {
				return &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "status.podIP",
					},
				}
			}
			container.Env = append(container.Env,
				corev1.EnvVar{Name: commonconsts.PodIPEnvVar, ValueFrom: podIPRef()},
				corev1.EnvVar{Name: commonconsts.VLLMDPMasterIPEnvVar, ValueFrom: podIPRef()},
			)
		}
	}

	// Set compilation cache environment variables for VLLM
	cacheDir := ""
	if component.CompilationCache != nil {
		cacheDir = component.CompilationCache.MountPath
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

	return nil
}

const (
	waitLeaderConfigMapSuffix = "wait-leader-script"
	waitLeaderScriptKey       = "wait-for-leader.py"
	waitLeaderVolumeName      = "wait-leader-script"
	waitLeaderMountPath       = "/scripts"
)

// WaitLeaderScript is the Python script that verifies leader pod health via
// the K8s API before attempting a TCP connection. It reads LEADER_HOST and
// LEADER_PORT from environment variables so the script content is generic.
const WaitLeaderScript = `import socket, time, json, ssl, urllib.request, os

SA = "/var/run/secrets/kubernetes.io/serviceaccount"
host = os.environ["LEADER_HOST"]
port = int(os.environ["LEADER_PORT"])

def _k8s_ctx():
    return ssl.create_default_context(cafile=f"{SA}/ca.crt")

def _k8s_headers():
    token = open(f"{SA}/token").read()
    return {"Authorization": f"Bearer {token}"}

def _k8s_api():
    ns = open(f"{SA}/namespace").read()
    return f"https://kubernetes.default.svc/api/v1/namespaces/{ns}/pods"

def resolve_leader_ip():
    # gethostbyname is IPv4-only; fall back to IPv6 for IPv6-only clusters.
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return socket.getaddrinfo(host, None, socket.AF_INET6, socket.SOCK_STREAM)[0][4][0]

def leader_pod_is_healthy():
    try:
        ip = resolve_leader_ip()
    except socket.gaierror:
        return False, "DNS resolution failed", None, None
    try:
        req = urllib.request.Request(
            f"{_k8s_api()}?fieldSelector=status.podIP={ip}",
            headers=_k8s_headers(),
        )
        resp = json.loads(urllib.request.urlopen(req, context=_k8s_ctx(), timeout=5).read())
        pods = resp.get("items", [])
        if not pods:
            return False, f"no pod found with IP {ip}", None, ip
        pod = pods[0]
        name = pod["metadata"].get("name", "unknown")
        uid = pod["metadata"].get("uid", "unknown")
        phase = pod.get("status", {}).get("phase")
        deletion_ts = pod["metadata"].get("deletionTimestamp")
        info = f"ip={ip} pod={name} uid={uid} phase={phase} deletionTimestamp={deletion_ts}"
        if deletion_ts:
            return False, f"pod {name} is terminating", info, ip
        if phase != "Running":
            return False, f"pod {name} phase is {phase}", info, ip
        return True, "", info, ip
    except Exception as e:
        # Fall back to TCP-only when the API is unavailable (e.g. 403 no RBAC)
        return True, f"K8s API unavailable ({e}), falling back to TCP", f"ip={ip}", ip

print(f"Waiting for leader master port at {host}:{port}...", flush=True)
time.sleep(5)
start = time.monotonic()
last_status = start
last_err = ""
while True:
    healthy, reason, pod_info, leader_ip = leader_pod_is_healthy()
    if healthy:
        try:
            s = socket.create_connection((leader_ip, port), timeout=2)
            s.close()
            elapsed = time.monotonic() - start
            print(f"Leader master port ready (waited {elapsed:.1f}s) [{pod_info}]", flush=True)
            break
        except Exception as e:
            last_err = f"tcp: {type(e).__name__}: {e} [{pod_info}]"
    else:
        last_err = f"{reason} [{pod_info}]" if pod_info else reason
    now = time.monotonic()
    if now - last_status >= 30:
        print(f"Still waiting for {host}:{port}... ({now - start:.0f}s elapsed, last: {last_err})", flush=True)
        last_status = now
    time.sleep(5)
`

// k8sVarPattern matches Kubernetes $(VAR) env-var expansion syntax.
var k8sVarPattern = regexp.MustCompile(`\$\((\w+)\)`)

// k8sToShellVarSyntax converts Kubernetes $(VAR) references to shell ${VAR}
// so that variables can be expanded by a shell at runtime. Plain $VAR
// references (e.g. from LWS) are already valid shell syntax and left as-is.
func k8sToShellVarSyntax(s string) string {
	return k8sVarPattern.ReplaceAllString(s, `${$1}`)
}

// GetWaitLeaderConfigMapName returns the ConfigMap name for a given DGD.
func GetWaitLeaderConfigMapName(dgdName string) string {
	return fmt.Sprintf("%s-%s", dgdName, waitLeaderConfigMapSuffix)
}

// GenerateWaitLeaderConfigMap creates a ConfigMap containing the wait-for-leader
// Python script. One ConfigMap is created per DGD and owned by the DGD.
func GenerateWaitLeaderConfigMap(dgdName, namespace string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GetWaitLeaderConfigMapName(dgdName),
			Namespace: namespace,
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: dgdName,
			},
		},
		Data: map[string]string{
			waitLeaderScriptKey: WaitLeaderScript,
		},
	}
}

func (b *VLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, _ *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	if !b.shouldInjectVLLMMpWaitLeaderInit(podSpec, numberOfNodes, role) {
		return
	}

	mainContainer := &podSpec.Containers[0]
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	mainImage := mainContainer.Image
	cmName := GetWaitLeaderConfigMapName(b.ParentGraphDeploymentName)

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: waitLeaderVolumeName,
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: cmName,
				},
			},
		},
	})

	// Use sh -c so the shell expands variable references at runtime.
	// Grove/LWS env vars are appended to init containers AFTER our env
	// vars, so Kubernetes $(VAR) expansion (which is order-dependent)
	// cannot resolve them. The shell sees all env vars regardless of
	// definition order.
	shellHostname := k8sToShellVarSyntax(leaderHostname)
	initContainer := corev1.Container{
		Name:  "wait-for-leader-mp",
		Image: mainImage,
		Command: []string{"sh", "-c", fmt.Sprintf(
			`export LEADER_HOST="%s" LEADER_PORT="%s" && exec python3 %s/%s`,
			shellHostname, commonconsts.VLLMMpMasterPort, waitLeaderMountPath, waitLeaderScriptKey)},
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      waitLeaderVolumeName,
				MountPath: waitLeaderMountPath,
				ReadOnly:  true,
			},
		},
	}

	podSpec.InitContainers = append(podSpec.InitContainers, initContainer)
}

func (b *VLLMBackend) shouldInjectVLLMMpWaitLeaderInit(podSpec *corev1.PodSpec, numberOfNodes int32, role Role) bool {
	if b.ParentGraphDeploymentName == "" || numberOfNodes <= 1 || role != RoleWorker || len(podSpec.Containers) == 0 {
		return false
	}

	args := parseVLLMLaunchArgs(getExpandedCommandLine(&podSpec.Containers[0]))
	return args.IsMpDistributedExecutorBackend
}

// updateVLLMMultinodeArgs dispatches to the appropriate injection function based on
// parallelism strategy (TP/PP distributed vs data-parallel) and executor backend (mp vs ray).
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUs int64, numberOfNodes int32, annotations map[string]string) {
	args := parseVLLMLaunchArgs(getExpandedArgs(container))
	needsDistributed := needsTensorParallelMultinodeLaunch(args, containerGPUs)

	if needsDistributed && shouldUseMpBackend(annotations) {
		injectMpDistributedLaunchFlags(container, role, serviceName, multinodeDeployer, numberOfNodes)
	} else if needsDistributed {
		injectRayDistributedLaunchFlags(container, role, serviceName, multinodeDeployer)
	} else if args.IsElasticEPEnabled {
		// Elastic EP requires a single Ray cluster spanning all nodes.
		// The operator's RPC-based DP coordination (--data-parallel-hybrid-lb) is
		// explicitly incompatible with elastic EP — vLLM raises NotImplementedError
		// if both are present. Instead we set up a cross-node Ray cluster:
		//   Leader: ray start --head --block & <tcp-poll-ray-ready> && <vllm cmd>
		//   Worker: <poll /live until 200> && ray start --address=<leader>:6379 --block
		// Note: --data-parallel-size-local is intentionally NOT injected. With the
		// worker's health-gate delaying its Ray join until dynamo.vllm is fully ready,
		// only the leader node is in the Ray cluster when create_dp_placement_groups runs,
		// so vLLM naturally places all initial DP workers on the leader node.
		injectElasticEPRayLaunchFlags(container, role, serviceName, multinodeDeployer)
	} else if needsDataParallelMultinodeLaunch(args, containerGPUs) {
		injectDataParallelLaunchFlags(container, role, serviceName, multinodeDeployer, containerGPUs, numberOfNodes, args)
	} else {
		logger := log.Log.WithName("vllm-backend")
		logger.Info("No need to inject tensor or data parallel flags for multinode deployments", "args", strings.Join(container.Args, " "))
	}
}

// getExpandedArgs will expand the containers args in the case where
// the args are joined together with spaces as an individual string (i.e. "python3 -m dynamo.vllm")
func getExpandedArgs(container *corev1.Container) []string {
	expandedArgs := []string{}
	for _, arg := range container.Args {
		expandedArgs = append(expandedArgs, strings.Fields(arg)...)
	}
	return normalizeVLLMFlags(expandedArgs)
}

// shouldUseMpBackend determines whether to use multiprocessing (mp) or Ray for vLLM
// multi-node distributed launches.
//
// Decision logic:
//  1. Explicit override annotation takes priority (user set "mp" or "ray")
//  2. Operator origin version compatibility gate: uses compatibility.VLLMMultiprocessing
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
	return compatibility.VLLMMultiprocessing.Enabled(annotations)
}

// injectMpDistributedLaunchFlags injects vLLM multiprocessing flags for multi-node TP/PP deployments.
//
// Leader: runs the original vLLM command with --distributed-executor-backend mp,
// --nnodes, --node-rank 0, --master-addr, --master-port
//
// Worker: runs the same vLLM command with --headless, --node-rank <rank>, and the same
// coordination flags. An init container (injected via UpdatePodSpec) handles waiting for
// the leader's master port before the worker's main container starts.
func injectMpDistributedLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, numberOfNodes int32) {
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	mpFlags := fmt.Sprintf("%s mp --nnodes %d --master-addr %s --master-port %s",
		distributedExecutorFlag,
		numberOfNodes, leaderHostname, commonconsts.VLLMMpMasterPort)

	needsShell := false

	switch role {
	case RoleLeader:
		mpFlags += " --node-rank 0"
	case RoleWorker:
		nodeRank, needsShellForRank := multinodeDeployer.GetNodeRank()
		needsShell = needsShellForRank
		mpFlags += fmt.Sprintf(" --node-rank %s --headless", nodeRank)
	}

	injectFlagsIntoContainerCommand(container, mpFlags, needsShell, "vllm")
}

func injectRayDistributedLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) {
	switch role {
	case RoleLeader:
		quotedCmd := make([]string, len(container.Command))
		for i, tok := range container.Command {
			quotedCmd[i] = shellQuoteForBashC(tok)
		}
		fullCommand := strings.Join(quotedCmd, " ")
		quotedArgs := make([]string, len(container.Args))
		for i, arg := range container.Args {
			quotedArgs[i] = shellQuoteForBashC(arg)
		}
		originalArgs := strings.Join(quotedArgs, " ")
		vllmMultinodeFlags := fmt.Sprintf("%s ray", distributedExecutorFlag)
		container.Args = []string{fmt.Sprintf("ray start --head --port=%s && %s %s %s", VLLMPort, fullCommand, originalArgs, vllmMultinodeFlags)}
	case RoleWorker:
		// Worker nodes only run Ray agent - vLLM on leader will spawn Ray actors on workers
		leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
		container.Args = []string{fmt.Sprintf("ray start --address=%s:%s --block", leaderHostname, VLLMPort)}
	}
	container.Command = []string{"/bin/sh", "-c"} // ensure cmd is a shell
}

// injectElasticEPRayLaunchFlags sets up a cross-node Ray cluster for elastic EP.
//
// Elastic EP requires --data-parallel-backend ray so that vLLM's Ray executor
// manages dynamic worker lifecycle. It is explicitly incompatible with
// --data-parallel-hybrid-lb (the operator's normal multinode DP path), because
// elastic EP needs a single API server and core client to coordinate scale up/down.
//
// We reuse the Ray TP/PP topology: leader starts the Ray head and runs vLLM,
// workers join the Ray cluster and expose their GPUs as idle resources.
//
// Worker health-gate: the worker deliberately waits until the leader's /live
// endpoint (DynamoSystemPort 9090) returns HTTP 200 before joining Ray. This is
// critical for correct DP placement:
//   - Port 9090 (system status server) opens EARLY in vLLM startup, before
//     create_dp_placement_groups runs.
//   - GET /live returns 503 during initialization and 200 only after the engine
//     is fully ready (create_dp_placement_groups done, model loaded).
//   - If the worker joins Ray before /live → 200, vLLM's create_dp_placement_groups
//     sees all cluster GPUs (leader + worker) and creates too many placement groups,
//     causing: "AssertionError: Created N DP placement groups, expected dp_size".
//   - Waiting for HTTP 200 ensures the worker joins AFTER placement groups are
//     set, so the leader's GPUs hold all initial DP workers (warm standby).
//
// Note: --data-parallel-size-local is intentionally NOT injected. With the
// health-gate ensuring only the leader is in Ray at vLLM startup, vLLM
// naturally places all --data-parallel-size workers on the leader node.
//
// Leader (or a single-pod RoleMain): ray start --head --port=6379 --block & <tcp-poll-ray-ready 150×2s> && <vllm cmd>
// Worker: <poll /live HTTP until 200> && ray start --address=<leader>:6379 --block
// injectElasticEPRayLaunchFlags returns true when it rewrote the container to
// front the engine with a Ray head, and false when it deliberately left the
// container untouched (see the empty-Command case below), so callers can gate
// side effects such as the VLLM_DP_MASTER_IP injection on whether a Ray head was
// actually set up.
func injectElasticEPRayLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) bool {
	switch role {
	// RoleMain is a component deployed as a single pod; it heads the Ray
	// cluster exactly as a multi-node leader does.
	case RoleLeader, RoleMain:
		// The Ray-head wrapper has to run a concrete executable once the head is
		// up, but an empty Command means the real entrypoint is the image
		// ENTRYPOINT, which the operator cannot see or reconstruct. Rewriting here
		// would emit a shell command with no executable (e.g. `exec --model ...`)
		// and break a pod that Kubernetes would otherwise start from its
		// ENTRYPOINT. Leave that invocation intact and skip the Ray head; a
		// single-pod Ray head needs an explicit Command.
		if len(container.Command) == 0 {
			log.Log.WithName("vllm-backend").Info(
				"elastic-EP Ray head not injected: container has no explicit Command; "+
					"set an explicit command to start the single-pod Ray head",
				"service", serviceName, "role", role)
			return false
		}
		quotedCmd := make([]string, len(container.Command))
		for i, tok := range container.Command {
			quotedCmd[i] = shellQuotePOSIX(tok)
		}
		quotedArgs := make([]string, len(container.Args))
		for i, arg := range container.Args {
			quotedArgs[i] = shellQuotePOSIX(arg)
		}
		vllmCommand := strings.TrimSpace(strings.Join(quotedCmd, " ") + " " + strings.Join(quotedArgs, " "))
		// A single-pod RoleMain leader is an ordinary serving pod that Kubernetes
		// rolls, evicts, and deletes, so exec the engine: it then runs as the
		// container's main process (PID 1) and receives SIGTERM directly for a
		// graceful shutdown, instead of being killed after the grace period with
		// in-flight requests dropped. The backgrounded Ray head continues as its
		// child. The multinode RoleLeader keeps its historical no-exec form so
		// this stays scoped to the new single-pod path.
		if role == RoleMain {
			vllmCommand = "exec " + vllmCommand
		}
		// Name the head's address on the single-pod path instead of letting Ray
		// pick one. vLLM is told the DP master is at status.podIP (see the caller)
		// and then looks for the Ray node registered under that exact address.
		// Ray left to itself chooses an interface by its own heuristic, so on a
		// pod with more than one network the two disagree and the engine aborts
		// with the same "DP master node is missing or dead" the env var exists to
		// prevent. The multinode leader keeps auto-detection: neither side of that
		// pair is pinned, so both run the same heuristic and agree with each other.
		nodeIPFlag := ""
		if role == RoleMain {
			nodeIPFlag = fmt.Sprintf(` --node-ip-address="$%s"`, commonconsts.PodIPEnvVar)
		}
		// Poll Ray head readiness with a bounded retry loop (150 × 2 s = 5 min max).
		// An unbounded `until` loop would spin forever if `ray start --head` crashes
		// silently or the port never opens.
		container.Args = []string{fmt.Sprintf(
			`ray start --head --port=%s%s --block & `+
				`i=0; until python3 -c "import socket; s=socket.create_connection(('127.0.0.1',%s),timeout=1); s.close()" 2>/dev/null; `+
				`do i=$((i+1)); [ "$i" -ge 150 ] && { echo "ERROR: Ray head did not start within 300s" >&2; exit 1; }; sleep 2; done && %s`,
			VLLMPort,
			nodeIPFlag,
			VLLMPort,
			vllmCommand,
		)}
	case RoleWorker:
		leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
		// Health-gate: poll GET /live on DynamoSystemPort (9090) until HTTP 200.
		// /live returns 503 during vLLM initialization and 200 when the engine is
		// fully ready. This ensures the worker joins Ray AFTER create_dp_placement_groups
		// has run (which requires only the leader's GPUs to be in the cluster).
		// Uses Python's urllib (always available) instead of curl.
		// Prerequisite: DYN_SYSTEM_ENABLED=true must be set on the leader pod so
		// that the Dynamo system server listens on port 9090. The operator injects
		// this env var unconditionally via component_worker.go.
		// Bounded at 720 × 15s = 3 hours to cover large models with slow disk I/O.
		// Without a bound, a permanently broken leader leaves the worker looping
		// forever with no Kubernetes liveness probe to detect it (probes are removed
		// from vLLM multinode containers in UpdateContainer).
		healthGate := fmt.Sprintf(
			`i=0; until python3 -c "import urllib.request; urllib.request.urlopen('http://%s:%d/live', timeout=5)" `+
				`2>/dev/null; do `+
				`i=$((i+1)); [ "$i" -ge 720 ] && { echo "ERROR: leader /live did not become ready within 3h" >&2; exit 1; }; `+
				`echo 'waiting for leader dynamo.vllm /live to return 200...'; sleep 15; done`,
			leaderHostname, commonconsts.DynamoSystemPort,
		)
		container.Args = []string{fmt.Sprintf(
			"%s && ray start --address=%s:%s --block",
			healthGate, leaderHostname, VLLMPort,
		)}
	}
	container.Command = []string{"/bin/sh", "-c"}
	return true
}

// IsElasticEPRayLaunch reports whether the container asks for the elastic-EP Ray
// topology.
//
// Elastic EP only works on the Ray data-parallel backend: vLLM's Ray executor is
// what grows and shrinks workers at runtime, and the engine refuses a scale
// request on any other backend. Requiring both flags keeps a Ray head off pods
// that pass --enable-elastic-ep while running the default backend, where it
// would launch a process nothing ever talks to.
//
// Detection scans the full command line (Command + Args) so the flags are found
// whether the manifest carries them in Command or Args, and parseVLLMLaunchArgs
// accepts vLLM's long --data-parallel-backend flag, its documented -dpb alias,
// and underscore/equals spellings of either -- vLLM's argparse treats all of
// these as equivalent, so any of them must trigger Ray-head injection.
func IsElasticEPRayLaunch(container *corev1.Container) bool {
	args := parseVLLMLaunchArgs(getExpandedCommandLine(container))
	return args.IsElasticEPEnabled && args.IsRayDataParallelBackend
}

// getExpandedCommandLine flattens Command and Args and splits any space-joined
// tokens, so flag detection works whether the manifest puts flags in Command or
// Args and whether they are separate list items or a single combined string.
//
// TODO: an "sh -c" script is split on whitespace here (and in getExpandedArgs)
// rather than read the way the shell reads it, so a flag inside a shell comment
// ("# --tensor-parallel-size 4") counts as a real occurrence and can be the one
// parseVLLMLaunchArgs resolves, although vLLM never receives it. Either drop
// everything from a word starting with "#" to the end of its line before
// splitting, or parse the script with a shell parser such as mvdan.cc/sh to get
// exactly the argv vLLM receives (comments, quotes and variables).
func getExpandedCommandLine(container *corev1.Container) []string {
	commandLine := make([]string, 0, len(container.Command)+len(container.Args))
	commandLine = append(commandLine, container.Command...)
	commandLine = append(commandLine, container.Args...)
	expanded := make([]string, 0, len(commandLine))
	for _, arg := range commandLine {
		expanded = append(expanded, strings.Fields(arg)...)
	}
	return normalizeVLLMFlags(expanded)
}

// vllmShortFlagAliases maps vLLM's documented short flag names to their canonical long form.
// Kept here rather than at each reader so a new alias is added once.
var vllmShortFlagAliases = map[string]string{
	"-tp":                        tensorParallelSizeFlag,
	"-pp":                        pipelineParallelSizeFlag,
	"-dp":                        dataParallelSizeFlag,
	"-dpl":                       dataParallelSizeLocalFlag,
	dataParallelBackendShortFlag: dataParallelBackendFlag,
}

// vllmNormalizedFlags is the set of canonical long flags parseVLLMLaunchArgs actually
// looks for. Underscore-to-dash rewriting in
// normalizeVLLMFlags is restricted to this set: vLLM's FlexibleArgumentParser treats "_"
// and "-" as interchangeable in long option names ("--tensor_parallel_size" ==
// "--tensor-parallel-size"), so a token is only rewritten when its dashed form is one of
// these -- an unrelated flag's spelling is left alone.
var vllmNormalizedFlags = map[string]bool{
	tensorParallelSizeFlag:    true,
	pipelineParallelSizeFlag:  true,
	dataParallelSizeFlag:      true,
	dataParallelSizeLocalFlag: true,
	dataParallelBackendFlag:   true,
	enableElasticEPFlag:       true,
	distributedExecutorFlag:   true,
}

// vllmValueFlags is the subset of vllmNormalizedFlags that take a value, and is what
// equals-form splitting is restricted to. Splitting anything outside it would invent a
// flag that vLLM never sees:
//   - An unrelated option carrying one of these as its value must stay whole:
//     "--served-model-name=--enable-elastic-ep" must not become a standalone
//     "--enable-elastic-ep" that IsElasticEPRayLaunch would match.
//   - --enable-elastic-ep itself is absent because vLLM registers it with
//     argparse.BooleanOptionalAction, so it takes no argument and
//     "--enable-elastic-ep=false" is not a request for elastic EP.
var vllmValueFlags = map[string]bool{
	tensorParallelSizeFlag:    true,
	pipelineParallelSizeFlag:  true,
	dataParallelSizeFlag:      true,
	dataParallelSizeLocalFlag: true,
	dataParallelBackendFlag:   true,
	distributedExecutorFlag:   true,
}

// shellControlChars are the POSIX shell control operators and redirection characters.
// strings.Fields has already consumed whitespace, so one of these surviving inside a token
// means the shell ended the word there -- everything from it on is the next command, and
// the engine only ever sees "ray" in "--data-parallel-backend=ray;".
const shellControlChars = ";&|<>()"

// vllmStringValueFlags is the subset of vllmValueFlags whose value parseVLLMLaunchArgs
// compares as a string, and is the only place a shell terminator is trimmed.
//
// Trimming is kept to these two because they are the two whose values were previously
// matched by substring, which tolerated a terminator glued to the value; the numeric flags
// have always gone through strconv.ParseInt, which does not. Restricting the trim the same
// way leaves every numeric flag resolving exactly as it does today.
//
// It is safe here because every legal value of these two is a backend name (ray, mp, uni,
// external_launcher) or a Python dotted import path, none of which can contain a shell
// control character. This is deliberately not a shell parser: a quoted value ("ray") or
// one built by expansion ($(BACKEND)) reads the same as it does today.
var vllmStringValueFlags = map[string]bool{
	dataParallelBackendFlag: true,
	distributedExecutorFlag: true,
}

// trimShellTerminator returns the part of value that the shell would actually pass to the
// program.
func trimShellTerminator(value string) string {
	if i := strings.IndexAny(value, shellControlChars); i >= 0 {
		return value[:i]
	}
	return value
}

// normalizeVLLMFlags standardizes tokenized command-line arguments into a
// single format: "--long-flag" followed by a separate "value" token. It
// expands short aliases (e.g., "-dp" to "--data-parallel-size"), rewrites
// underscore spellings of the flags this package reads to their dashed
// form (e.g., "--tensor_parallel_size" to "--tensor-parallel-size"), splits
// combined pairs (e.g., "--flag=value") for the flags that take a value, and
// drops a shell control operator a manifest glued to the value of a
// vllmStringValueFlags flag.
func normalizeVLLMFlags(expanded []string) []string {
	normalized := make([]string, 0, len(expanded))
	trimNextValue := false
	for _, arg := range expanded {
		if trimNextValue {
			trimNextValue = false
			// A token starting with "-" is the next flag, not the previous
			// flag's value, so let it fall through and be canonicalized.
			if !strings.HasPrefix(arg, "-") {
				normalized = append(normalized, trimShellTerminator(arg))
				continue
			}
		}
		flag, value, hasEquals := strings.Cut(arg, "=")
		if canonical, ok := vllmShortFlagAliases[flag]; ok {
			flag = canonical
		} else if dashed := strings.ReplaceAll(flag, "_", "-"); vllmNormalizedFlags[dashed] {
			flag = dashed
		}
		if hasEquals && !vllmValueFlags[flag] {
			normalized = append(normalized, arg)
			continue
		}
		if hasEquals {
			if vllmStringValueFlags[flag] {
				value = trimShellTerminator(value)
			}
			// No legal value of these flags begins with "--", so a value that
			// does is not a value. Keeping the token whole stops it becoming a
			// standalone flag that an exact-match reader would honor.
			if strings.HasPrefix(value, "--") {
				normalized = append(normalized, arg)
				continue
			}
			normalized = append(normalized, flag, value)
			continue
		}
		normalized = append(normalized, flag)
		// Only --distributed-executor-backend is trimmed in the separated form.
		// It is the one flag whose space-separated spelling was previously read
		// off the unsplit command string, where the substring match spanned the
		// space and tolerated a terminator on the value.
		trimNextValue = flag == distributedExecutorFlag
	}
	return normalized
}

// vllmLaunchArgs is the result of parsing a container's launch command line exactly once,
// so one place interprets vLLM flag semantics (aliases, equals and underscore spellings)
// and two readers cannot disagree.
type vllmLaunchArgs struct {
	TensorParallelSize   int64
	PipelineParallelSize int64
	DataParallelSize     int64
	// HasDataParallelSize distinguishes "--data-parallel-size not present" from
	// "present and equal to vLLM's default of 1" -- callers use this to avoid
	// injecting a duplicate flag when one is already present (e.g. from the profiler).
	HasDataParallelSize            bool
	IsRayDataParallelBackend       bool
	IsElasticEPEnabled             bool
	IsMpDistributedExecutorBackend bool
}

// WorldSize is the number of ranks one engine occupies.
func (a vllmLaunchArgs) WorldSize() int64 {
	return a.TensorParallelSize * a.PipelineParallelSize
}

// parseVLLMLaunchArgs reads an already-normalized list (see getExpandedArgs /
// getExpandedCommandLine) in one pass. Each occurrence of a flag overwrites the field it
// sets, so a repeated flag resolves to its final occurrence, as it does in vLLM's
// argparse. A size value that does not parse as an integer leaves the earlier value in
// place.
func parseVLLMLaunchArgs(expandedArgs []string) vllmLaunchArgs {
	args := vllmLaunchArgs{TensorParallelSize: 1, PipelineParallelSize: 1, DataParallelSize: 1}
	for i, arg := range expandedArgs {
		// Presence flags count even when nothing follows them.
		switch arg {
		case dataParallelSizeFlag:
			args.HasDataParallelSize = true
		case enableElasticEPFlag:
			args.IsElasticEPEnabled = true
		}

		// A trailing flag has no value to apply.
		if i+1 >= len(expandedArgs) {
			continue
		}
		value := expandedArgs[i+1]
		switch arg {
		case tensorParallelSizeFlag:
			args.TensorParallelSize = parseSizeOr(value, args.TensorParallelSize)
		case pipelineParallelSizeFlag:
			args.PipelineParallelSize = parseSizeOr(value, args.PipelineParallelSize)
		case dataParallelSizeFlag:
			args.DataParallelSize = parseSizeOr(value, args.DataParallelSize)
		case dataParallelBackendFlag:
			args.IsRayDataParallelBackend = value == dataParallelBackendRay
		case distributedExecutorFlag:
			args.IsMpDistributedExecutorBackend = value == "mp"
		}
	}
	return args
}

// parseSizeOr returns value parsed as an integer, or current when it does not parse.
func parseSizeOr(value string, current int64) int64 {
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return current
	}
	return parsed
}

func injectDataParallelLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUs int64, numberOfNodes int32, args vllmLaunchArgs) {
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)

	// Calculate engines per node
	dataParallelSizeLocal := containerGPUs / args.WorldSize()

	// Get total DP size from args, or calculate from nodes
	totalDPSize := args.DataParallelSize
	if totalDPSize == 1 {
		totalDPSize = dataParallelSizeLocal * int64(numberOfNodes)
	}

	var flags []string
	needsShell := false

	switch role {
	case RoleLeader:
		// Leader runs API server + coordinator + local engines
		// Hybrid LB mode: local DP coordination within node, Dynamo routes between nodes
		flags = []string{"--data-parallel-hybrid-lb"}
		// Only inject --data-parallel-size if not already present (avoids duplicates from profiler)
		if !args.HasDataParallelSize {
			flags = append(flags, dataParallelSizeFlag, strconv.FormatInt(totalDPSize, 10))
		}
		flags = append(flags,
			dataParallelSizeLocalFlag, strconv.FormatInt(dataParallelSizeLocal, 10),
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
		if !args.HasDataParallelSize {
			flags = append(flags, dataParallelSizeFlag, strconv.FormatInt(totalDPSize, 10))
		}
		flags = append(flags,
			dataParallelSizeLocalFlag, strconv.FormatInt(dataParallelSizeLocal, 10),
			"--data-parallel-start-rank", startRank,
			"--data-parallel-address", leaderHostname,
			"--data-parallel-rpc-port", dataParallelRPCPort,
		)
	}

	injectFlagsIntoContainerCommand(container, strings.Join(flags, " "), needsShell, "vllm")
}

// needsMultinodeDistributedLaunch returns true when the model's world size (TP * PP)
// exceeds the GPU count of one engine container, requiring multi-node distribution (via mp or ray).
func needsTensorParallelMultinodeLaunch(args vllmLaunchArgs, containerGPUs int64) bool {
	if containerGPUs == 0 {
		return false
	}
	return args.WorldSize() > containerGPUs
}

// if world size across all DP ranks > GPU count, then we need to inject data parallel multinode coordination
func needsDataParallelMultinodeLaunch(args vllmLaunchArgs, containerGPUs int64) bool {
	if containerGPUs == 0 {
		return false
	}
	return args.WorldSize()*args.DataParallelSize > containerGPUs
}
