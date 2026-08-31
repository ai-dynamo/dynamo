package dynamo

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
	vllmmutation "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation/vllm"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	VLLMPort                  = "6379"
	dataParallelRPCPort       = "13445"
	tensorParallelSizeFlag    = vllmmutation.TensorParallelSizeFlag
	pipelineParallelSizeFlag  = vllmmutation.PipelineParallelSizeFlag
	dataParallelSizeFlag      = vllmmutation.DataParallelSizeFlag
	dataParallelSizeLocalFlag = vllmmutation.DataParallelSizeLocalFlag
	distributedExecutorFlag   = vllmmutation.DistributedExecutorFlag
	enableElasticEPFlag       = vllmmutation.EnableElasticEPFlag
	dataParallelBackendFlag   = vllmmutation.DataParallelBackendFlag
	// dataParallelBackendShortFlag is vLLM's documented short alias for
	// --data-parallel-backend (see the v0.26.0 `vllm serve` CLI reference).
	dataParallelBackendShortFlag = vllmmutation.DataParallelBackendShortFlag
	dataParallelBackendRay       = vllmmutation.DataParallelBackendRay
)

type VLLMBackend struct {
	ParentGraphDeploymentName string
}

func (b *VLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUCount ContainerGPUCount) error {
	mutations, err := b.containerMutations(
		container,
		numberOfNodes,
		role,
		component,
		serviceName,
		multinodeDeployer,
		containerGPUCount,
	)
	if err != nil {
		return err
	}
	return mutations.Apply(component, role, container)
}

func (b *VLLMBackend) containerMutations(
	container *corev1.Container,
	numberOfNodes int32,
	role Role,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	serviceName string,
	multinodeDeployer MultinodeDeployer,
	containerGPUCount ContainerGPUCount,
) (mutation.EngineMutations, error) {
	commonMutations := vllmmutation.Common()
	cacheMutations := vllmCompilationCacheMutations(component)
	annotations := GetPodTemplateAnnotations(component)

	if numberOfNodes > 1 {
		containerGPUs, err := containerGPUCount()
		if err != nil {
			return nil, fmt.Errorf("failed to resolve container GPUs: %w", err)
		}

		return mutation.Concat(
			commonMutations,
			vllmMultinodeMutations(
				container, role, serviceName, multinodeDeployer, containerGPUs, numberOfNodes, annotations,
			),
			vllmMPSideChannelMutations(annotations),
			vllmmutation.MultinodePodWiring(),
			cacheMutations,
		), nil
	}

	if role == RoleMain && IsElasticEPRayLaunch(container) {
		// A single-pod elastic-EP component still needs a Ray head, so that
		// follower pods created later have a cluster to join. Only the leader
		// arm applies here: a lone pod is expanded as RoleMain, never RoleWorker.
		if command, args, applied := elasticEPRayCommand(container, role, serviceName, multinodeDeployer); applied {
			return mutation.Concat(
				commonMutations,
				vllmmutation.SingleNodeElasticEP(command, args),
				cacheMutations,
			), nil
		}
	}
	return mutation.Concat(commonMutations, cacheMutations), nil
}

func vllmCompilationCacheMutations(component *v1beta1.DynamoComponentDeploymentSharedSpec) mutation.EngineMutations {
	cacheDir := ""
	if component.CompilationCache != nil {
		cacheDir = component.CompilationCache.MountPath
	}

	if cacheDir != "" {
		logger := log.Log.WithName("vllm-backend")
		logger.Info("Compilation cache configured and enabled for VLLM backend",
			"backend", "vllm",
			"status", "fully-supported",
			"cache-dir", cacheDir,
			"use-as-compilation-cache", true,
			"env-vars-set", true,
			"env-vars", "VLLM_CACHE_ROOT")
	}
	return vllmmutation.CompilationCache(cacheDir)
}

func vllmMPSideChannelMutations(annotations map[string]string) mutation.EngineMutations {
	if !shouldUseMpBackend(annotations) {
		return nil
	}
	return vllmmutation.MPSideChannel()
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

def leader_pod_is_healthy():
    try:
        ip = socket.gethostbyname(host)
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

func (b *VLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) error {
	if !b.shouldInjectVLLMMpWaitLeaderInit(podSpec, numberOfNodes, role) {
		return nil
	}

	mainContainer := &podSpec.Containers[0]
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	mainImage := mainContainer.Image
	cmName := GetWaitLeaderConfigMapName(b.ParentGraphDeploymentName)

	volume := corev1.Volume{
		Name: waitLeaderVolumeName,
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: cmName,
				},
			},
		},
	}

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

	return vllmmutation.WaitForLeader(volume, initContainer).Apply(component, role, podSpec)
}

func (b *VLLMBackend) shouldInjectVLLMMpWaitLeaderInit(podSpec *corev1.PodSpec, numberOfNodes int32, role Role) bool {
	if b.ParentGraphDeploymentName == "" || numberOfNodes <= 1 || role != RoleWorker || len(podSpec.Containers) == 0 {
		return false
	}

	return containerCommandLineHasArg(&podSpec.Containers[0], distributedExecutorFlag, "mp")
}

func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUs int64, numberOfNodes int32, annotations map[string]string) {
	mutations := vllmMultinodeMutations(container, role, serviceName, multinodeDeployer, containerGPUs, numberOfNodes, annotations)
	_ = mutations.Apply(nil, role, container)
}

// vllmMultinodeMutations selects the launch mutation for the configured
// parallelism strategy and executor backend.
func vllmMultinodeMutations(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUs int64, numberOfNodes int32, annotations map[string]string) mutation.EngineMutations {
	expandedArgs := getExpandedArgs(container)
	needsDistributed := needsTensorParallelMultinodeLaunch(expandedArgs, containerGPUs)

	if needsDistributed && shouldUseMpBackend(annotations) {
		var workerRank string
		var workerRankNeedsShell bool
		if role == RoleWorker {
			workerRank, workerRankNeedsShell = multinodeDeployer.GetNodeRank()
		}
		return vllmmutation.MP(vllmmutation.MPValues{
			NumberOfNodes:        numberOfNodes,
			LeaderAddress:        multinodeDeployer.GetLeaderHostname(serviceName),
			WorkerRank:           workerRank,
			WorkerRankNeedsShell: workerRankNeedsShell,
		})
	} else if needsDistributed {
		return vllmmutation.Ray(vllmmutation.RayValues{
			Port:          VLLMPort,
			LeaderAddress: multinodeDeployer.GetLeaderHostname(serviceName),
		})
	} else if hasFlag(expandedArgs, enableElasticEPFlag) {
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
		command, args, applied := elasticEPRayCommand(container, role, serviceName, multinodeDeployer)
		if !applied {
			return nil
		}
		return vllmmutation.LaunchCommand(command, args)
	} else if needsDataParallelMultinodeLaunch(expandedArgs, containerGPUs) {
		return vllmmutation.DataParallel(dataParallelLaunchValues(
			container, role, serviceName, multinodeDeployer, containerGPUs, numberOfNodes,
		))
	} else {
		logger := log.Log.WithName("vllm-backend")
		logger.Info("No need to inject tensor or data parallel flags for multinode deployments", "args", strings.Join(container.Args, " "))
	}
	return nil
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

// elasticEPRayCommand returns the cross-node Ray launch command for elastic EP.
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
// The boolean result is false when the command must deliberately remain
// untouched, so callers can gate related environment mutations.
func elasticEPRayCommand(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) ([]string, []string, bool) {
	args := append([]string(nil), container.Args...)
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
			return nil, nil, false
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
		args = []string{fmt.Sprintf(
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
		args = []string{fmt.Sprintf(
			"%s && ray start --address=%s:%s --block",
			healthGate, leaderHostname, VLLMPort,
		)}
	}
	return []string{"/bin/sh", "-c"}, args, true
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
// whether the manifest carries them in Command or Args, and it accepts vLLM's
// long --data-parallel-backend flag and its documented -dpb alias in both the
// "flag value" and "flag=value" spellings — vLLM's argparse treats all of these
// as equivalent, so any of them must trigger Ray-head injection.
func IsElasticEPRayLaunch(container *corev1.Container) bool {
	expanded := getExpandedCommandLine(container)
	return hasFlag(expanded, enableElasticEPFlag) &&
		(hasArg(expanded, dataParallelBackendFlag, dataParallelBackendRay) ||
			hasArg(expanded, dataParallelBackendShortFlag, dataParallelBackendRay))
}

// getExpandedCommandLine flattens Command and Args and splits any space-joined
// tokens, so flag detection works whether the manifest puts flags in Command or
// Args and whether they are separate list items or a single combined string.
func getExpandedCommandLine(container *corev1.Container) []string {
	commandLine := make([]string, 0, len(container.Command)+len(container.Args))
	commandLine = append(commandLine, container.Command...)
	commandLine = append(commandLine, container.Args...)
	expanded := make([]string, 0, len(commandLine))
	for _, arg := range commandLine {
		expanded = append(expanded, strings.Fields(arg)...)
	}
	return expanded
}

// hasFlag returns true if flag exists in expandedArgs.
func hasFlag(expandedArgs []string, flag string) bool {
	for _, arg := range expandedArgs {
		if arg == flag {
			return true
		}
	}
	return false
}

func dataParallelLaunchValues(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, containerGPUs int64, numberOfNodes int32) vllmmutation.DataParallelValues {
	expandedArgs := getExpandedArgs(container)
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)

	// Calculate engines per node
	worldSize := getWorldSize(expandedArgs) // TP * PP per engine
	dataParallelSizeLocal := containerGPUs / worldSize

	// Get total DP size from args, or calculate from nodes
	totalDPSize := getFlagValue(expandedArgs, dataParallelSizeFlag)
	if totalDPSize == 1 {
		totalDPSize = dataParallelSizeLocal * int64(numberOfNodes)
	}

	workerStartRank := ""
	if role == RoleWorker {
		nodeRank, _ := multinodeDeployer.GetNodeRank()
		workerStartRank = fmt.Sprintf("$(( %d * %s ))", dataParallelSizeLocal, nodeRank)
	}
	return vllmmutation.DataParallelValues{
		TotalSize:            totalDPSize,
		OmitTotalSize:        hasFlag(expandedArgs, dataParallelSizeFlag),
		LocalSize:            dataParallelSizeLocal,
		LeaderAddress:        leaderHostname,
		WorkerStartRank:      workerStartRank,
		WorkerRankNeedsShell: true,
		RPCPort:              dataParallelRPCPort,
	}
}

// needsMultinodeDistributedLaunch returns true when the model's world size (TP * PP)
// exceeds the GPU count of one engine container, requiring multi-node distribution (via mp or ray).
func needsTensorParallelMultinodeLaunch(expandedArgs []string, containerGPUs int64) bool {
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
func needsDataParallelMultinodeLaunch(expandedArgs []string, containerGPUs int64) bool {
	dataParallelSize := getFlagValue(expandedArgs, dataParallelSizeFlag)
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
