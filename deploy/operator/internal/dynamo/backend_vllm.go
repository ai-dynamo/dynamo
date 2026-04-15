package dynamo

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/featuregate"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	VLLMPort                     = "6379"
	dataParallelRPCPort          = "13445"
	tensorParallelSizeFlag       = "--tensor-parallel-size"
	pipelineParallelSizeFlag     = "--pipeline-parallel-size"
	dataParallelSizeFlag         = "--data-parallel-size"
	dataParallelSizeLocalFlag    = "--data-parallel-size-local"
	enableElasticEPFlag          = "--enable-elastic-ep"
)

type VLLMBackend struct {
	ParentGraphDeploymentName string
}

func (b *VLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	isMultinode := numberOfNodes > 1

	if isMultinode {
		// Apply multinode-specific argument modifications
		updateVLLMMultinodeArgs(container, role, serviceName, multinodeDeployer, component.Resources, numberOfNodes, component.Annotations)

		if shouldUseMpBackend(component.Annotations) {
			container.Env = append(container.Env, corev1.EnvVar{
				Name: commonconsts.VLLMNixlSideChannelHostEnvVar,
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "status.podIP",
					},
				},
			})
		}

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

func (b *VLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	if numberOfNodes <= 1 || role != RoleWorker || !shouldUseMpBackend(component.Annotations) {
		return
	}

	// Elastic EP workers use a Ray cluster, not the MP coordinator. The worker
	// command (injected by injectElasticEPRayLaunchFlags) already contains an
	// inline health gate that polls DynamoSystemPort (9090) before joining Ray.
	// The MP init container waits on VLLMMpMasterPort (29500), which never opens
	// in the elastic EP path — injecting it would cause the worker to hang forever.
	if component.ExtraPodSpec != nil && component.ExtraPodSpec.MainContainer != nil {
		if hasFlag(getExpandedArgs(component.ExtraPodSpec.MainContainer), enableElasticEPFlag) {
			return
		}
	}

	if len(podSpec.Containers) == 0 || b.ParentGraphDeploymentName == "" {
		return
	}

	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	mainImage := podSpec.Containers[0].Image
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

// updateVLLMMultinodeArgs dispatches to the appropriate injection function based on
// parallelism strategy (TP/PP distributed vs data-parallel) and executor backend (mp vs ray).
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, resources *v1alpha1.Resources, numberOfNodes int32, annotations map[string]string) {
	expandedArgs := getExpandedArgs(container)
	needsDistributed := needsTensorParallelMultinodeLaunch(expandedArgs, resources)

	if needsDistributed && shouldUseMpBackend(annotations) {
		injectMpDistributedLaunchFlags(container, role, serviceName, multinodeDeployer, numberOfNodes)
	} else if needsDistributed {
		injectRayDistributedLaunchFlags(container, role, serviceName, multinodeDeployer)
	} else if hasFlag(expandedArgs, enableElasticEPFlag) {
		// Elastic EP requires a single Ray cluster spanning all nodes.
		// The operator's RPC-based DP coordination (--data-parallel-hybrid-lb) is
		// explicitly incompatible with elastic EP — vLLM raises NotImplementedError
		// if both are present. Instead we set up a cross-node Ray cluster:
		//   Leader: ray start --head && vllm serve ... --data-parallel-size-local 1
		//   Worker: ray start --address=<leader>:6379 --block
		// --data-parallel-size-local 1 tells vLLM's create_dp_placement_groups to
		// allocate exactly 1 DP rank per Ray node (not n_gpus/world_size which would
		// over-allocate on nodes with multiple GPUs).
		injectElasticEPRayLaunchFlags(container, role, serviceName, multinodeDeployer)
	} else if needsDataParallelMultinodeLaunch(expandedArgs, resources) {
		injectDataParallelLaunchFlags(container, role, serviceName, multinodeDeployer, resources, numberOfNodes)
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
// Worker: runs the same vLLM command with --headless, --node-rank <rank>, and the same
// coordination flags. An init container (injected via UpdatePodSpec) handles waiting for
// the leader's master port before the worker's main container starts.
func injectMpDistributedLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer, numberOfNodes int32) {
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
	mpFlags := fmt.Sprintf("--distributed-executor-backend mp --nnodes %d --master-addr %s --master-port %s",
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
		vllmMultinodeFlags := "--distributed-executor-backend ray"
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
// Leader: ray start --head --port=6379 --block & sleep 8 && <vllm cmd>
// Worker: <poll /live HTTP until 200> && ray start --address=<leader>:6379 --block
func injectElasticEPRayLaunchFlags(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) {
	switch role {
	case RoleLeader:
		quotedCmd := make([]string, len(container.Command))
		for i, tok := range container.Command {
			quotedCmd[i] = shellQuoteForBashC(tok)
		}
		quotedArgs := make([]string, len(container.Args))
		for i, arg := range container.Args {
			quotedArgs[i] = shellQuoteForBashC(arg)
		}
		container.Args = []string{fmt.Sprintf(
			"ray start --head --port=%s --block & sleep 8 && %s %s",
			VLLMPort,
			strings.Join(quotedCmd, " "),
			strings.Join(quotedArgs, " "),
		)}
	case RoleWorker:
		leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
		// Health-gate: poll GET /live on DynamoSystemPort (9090) until HTTP 200.
		// /live returns 503 during vLLM initialization and 200 when the engine is
		// fully ready. This ensures the worker joins Ray AFTER create_dp_placement_groups
		// has run (which requires only the leader's GPUs to be in the cluster).
		// Uses Python's urllib (always available) instead of curl.
		healthGate := fmt.Sprintf(
			`until python3 -c "import urllib.request; urllib.request.urlopen('http://%s:%d/live', timeout=5)" `+
				`2>/dev/null; do `+
				`echo 'waiting for leader dynamo.vllm /live to return 200...'; sleep 15; done`,
			leaderHostname, commonconsts.DynamoSystemPort,
		)
		container.Args = []string{fmt.Sprintf(
			"%s && ray start --address=%s:%s --block",
			healthGate, leaderHostname, VLLMPort,
		)}
	}
	container.Command = []string{"/bin/sh", "-c"}
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

	switch role {
	case RoleLeader:
		// Leader runs API server + coordinator + local engines
		// Hybrid LB mode: local DP coordination within node, Dynamo routes between nodes
		flags = []string{"--data-parallel-hybrid-lb"}
		// Only inject --data-parallel-size if not already present (avoids duplicates from profiler)
		if !hasFlag(expandedArgs, dataParallelSizeFlag) {
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
		if !hasFlag(expandedArgs, dataParallelSizeFlag) {
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
// exceeds the GPU count of a single node, requiring multi-node distribution (via mp or ray).
func needsTensorParallelMultinodeLaunch(expandedArgs []string, resources *v1alpha1.Resources) bool {
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
func needsDataParallelMultinodeLaunch(expandedArgs []string, resources *v1alpha1.Resources) bool {
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
