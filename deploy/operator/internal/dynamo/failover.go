/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

const (
	failoverSharedVolumeName       = "failover-shared"
	failoverSharedMountPath        = "/shared"
	failoverLockFile               = "/shared/failover.lock"
	failoverDRAClaimName           = "shared-gpu"
	failoverEngineCount            = 2
	failoverMasterPortStride       = 100 // engine-1 gets master-port + 100
	failoverMasterPortFlag         = "--master-port"
	failoverHarnessVolumeName      = "failover-harness"
	failoverHarnessMountPath       = "/harness"
	failoverHarnessConfigMapSuffix = "failover-harness"
)

func isFailoverEnabled(component *v1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	return component.Failover != nil && component.Failover.Enabled
}

// getGPUCount extracts the GPU count from the component resource spec.
func getGPUCount(component *v1alpha1.DynamoComponentDeploymentSharedSpec) (int, error) {
	if component.Resources == nil {
		return 0, fmt.Errorf("resources must be specified for failover workers")
	}

	gpuStr := ""
	if component.Resources.Limits != nil && component.Resources.Limits.GPU != "" {
		gpuStr = component.Resources.Limits.GPU
	} else if component.Resources.Requests != nil && component.Resources.Requests.GPU != "" {
		gpuStr = component.Resources.Requests.GPU
	}

	if gpuStr == "" {
		return 0, fmt.Errorf("GPU count must be specified for failover workers")
	}

	count, err := strconv.Atoi(gpuStr)
	if err != nil {
		return 0, fmt.Errorf("invalid GPU count %q: %w", gpuStr, err)
	}
	return count, nil
}

// buildFailoverPod transforms a single-container worker pod spec into a
// multi-container failover pod with two engine containers and a GMS weight sidecar.
//
// The transformation:
//  1. Clones the main container into engine-0 and engine-1 with staggered system ports
//  2. Adds a GMS weight sidecar as an init container (restartPolicy: Always)
//  3. Adds a shared emptyDir volume for GMS UDS sockets and the flock file
//  4. Sets up DRA resource claims so all containers share GPU access
//  5. Injects failover-specific env vars (ENGINE_ID, TMPDIR, FAILOVER_LOCK_PATH, etc.)
//  6. Adds GPU toleration for DRA-scheduled pods on tainted nodes
func buildFailoverPod(
	podSpec *corev1.PodSpec,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
	parentName string,
	serviceName string,
	numberOfNodes int32,
	role Role,
	coordinationEndpoint string,
) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for failover transformation")
	}

	gpuCount, err := getGPUCount(component)
	if err != nil {
		return err
	}

	mainContainer := podSpec.Containers[0]

	engines := make([]corev1.Container, failoverEngineCount)
	for i := range failoverEngineCount {
		engines[i] = buildEngineContainer(mainContainer, i, commonconsts.DynamoSystemPort+i)
	}

	gmsSidecar := buildGMSSidecar(mainContainer.Image, gpuCount)

	podSpec.Containers = engines
	podSpec.InitContainers = append(podSpec.InitContainers, gmsSidecar)
	podSpec.Volumes = append(podSpec.Volumes, failoverSharedVolume())

	// DRA replaces normal GPU scheduling, so the default GPU toleration that
	// kubelet/device-plugin would add is lost. Re-add it explicitly.
	podSpec.Tolerations = append(podSpec.Tolerations, corev1.Toleration{
		Key:      commonconsts.KubeResourceGPUNvidia,
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	})

	claimTemplateName := FailoverResourceClaimTemplateName(parentName, serviceName)
	podSpec.ResourceClaims = append(podSpec.ResourceClaims, corev1.PodResourceClaim{
		Name:                      failoverDRAClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	})

	// Inject NNODES for multinode failover so engines know the group size
	if numberOfNodes > 1 {
		nnodesEnv := corev1.EnvVar{Name: "NNODES", Value: strconv.Itoa(int(numberOfNodes))}
		for i := range podSpec.Containers {
			podSpec.Containers[i].Env = append(podSpec.Containers[i].Env, nnodesEnv)
		}
	}

	// Multinode harness injection: wrap engine entrypoints with coordination scripts
	if numberOfNodes > 1 && coordinationEndpoint != "" {
		if err := injectMultinodeHarness(podSpec, parentName, serviceName, numberOfNodes, role, coordinationEndpoint); err != nil {
			return fmt.Errorf("failed to inject multinode harness: %w", err)
		}
	}

	return nil
}

// buildEngineContainer clones the main container with ENGINE_ID and failover env vars.
// Each engine gets a unique system port and named port for probe targeting.
func buildEngineContainer(base corev1.Container, engineID int, systemPort int) corev1.Container {
	engine := *base.DeepCopy()
	engine.Name = fmt.Sprintf("engine-%d", engineID)

	portName := fmt.Sprintf("system-%d", engineID)

	engine.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          portName,
			ContainerPort: int32(systemPort),
		},
	}

	// Env vars to remove: replaced by failover-specific values or intentionally omitted.
	// DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS is omitted to activate Branch 3 in SystemHealth.
	// DYN_DISCOVERY_BACKEND is preserved from the base container (supports both etcd and k8s discovery).
	removeSet := map[string]bool{
		"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": true,
		"DYN_SYSTEM_PORT":                       true,
		"DYN_SYSTEM_ENABLED":                    true,
		"DYN_HEALTH_CHECK_ENABLED":              true,
		"CONTAINER_NAME":                        true,
	}

	var filtered []corev1.EnvVar
	for _, env := range engine.Env {
		if !removeSet[env.Name] {
			filtered = append(filtered, env)
		}
	}

	containerName := fmt.Sprintf("engine-%d", engineID)
	failoverEnvs := []corev1.EnvVar{
		{Name: "ENGINE_ID", Value: strconv.Itoa(engineID)},
		{Name: "CONTAINER_NAME", Value: containerName},
		{Name: "TMPDIR", Value: failoverSharedMountPath},
		{Name: "FAILOVER_LOCK_PATH", Value: failoverLockFile},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
		{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(systemPort)},
		{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"},
		{Name: "VLLM_NIXL_SIDE_CHANNEL_PORT", Value: strconv.Itoa(5600 + engineID)},
		{Name: "DYN_VLLM_KV_EVENT_PORT", Value: strconv.Itoa(20080 + engineID)},
	}
	engine.Env = append(filtered, failoverEnvs...)

	engine.VolumeMounts = append(engine.VolumeMounts, corev1.VolumeMount{
		Name:      failoverSharedVolumeName,
		MountPath: failoverSharedMountPath,
	})

	portRef := intstr.FromString(portName)
	if engine.StartupProbe != nil && engine.StartupProbe.HTTPGet != nil {
		engine.StartupProbe.HTTPGet.Port = portRef
	}
	if engine.LivenessProbe != nil && engine.LivenessProbe.HTTPGet != nil {
		engine.LivenessProbe.HTTPGet.Port = portRef
	}
	if engine.ReadinessProbe != nil && engine.ReadinessProbe.HTTPGet != nil {
		engine.ReadinessProbe.HTTPGet.Port = portRef
	}

	// Stagger --master-port for multinode TP so each engine group uses a
	// distinct torch.distributed TCP store. engine-0 keeps the default (29500),
	// engine-1 gets 29500 + failoverMasterPortStride.
	if engineID > 0 {
		if hasMasterPortFlag(&engine) {
			staggerMasterPort(&engine, engineID)
		} else {
			// No --master-port in args — inject it explicitly
			engine.Args = append(engine.Args, failoverMasterPortFlag, strconv.Itoa(29500+engineID*failoverMasterPortStride))
		}
	}

	removeGPUResources(&engine)
	engine.Resources.Claims = append(engine.Resources.Claims, corev1.ResourceClaim{
		Name: failoverDRAClaimName,
	})

	return engine
}

// hasMasterPortFlag checks if --master-port appears in the container args or command.
func hasMasterPortFlag(container *corev1.Container) bool {
	for _, arg := range container.Args {
		if arg == failoverMasterPortFlag || strings.Contains(arg, failoverMasterPortFlag+" ") {
			return true
		}
	}
	for _, cmd := range container.Command {
		if strings.Contains(cmd, failoverMasterPortFlag+" ") {
			return true
		}
	}
	return false
}

// staggerMasterPort offsets --master-port in the container args by
// engineID * failoverMasterPortStride. This prevents port collisions when
// multiple engine groups run on the same pod in multinode+failover mode.
func staggerMasterPort(container *corev1.Container, engineID int) {
	offset := engineID * failoverMasterPortStride
	staggerFlagValue(container, failoverMasterPortFlag, offset)
}

// staggerFlagValue finds a --flag VALUE pair in container args and adds offset
// to the integer value. Handles both separate-token args (["--flag", "29500"])
// and shell-wrapped args (["sh", "-c", "... --flag 29500 ..."]).
func staggerFlagValue(container *corev1.Container, flag string, offset int) {
	// Try direct args first (non-shell case)
	for i, arg := range container.Args {
		if arg == flag && i+1 < len(container.Args) {
			if port, err := strconv.Atoi(container.Args[i+1]); err == nil {
				container.Args[i+1] = strconv.Itoa(port + offset)
				return
			}
		}
	}

	// Try shell-wrapped args (sh -c "... --flag 29500 ...")
	for i, arg := range container.Args {
		if strings.Contains(arg, flag+" ") {
			parts := strings.Split(arg, flag+" ")
			if len(parts) < 2 {
				continue
			}
			rest := parts[1]
			var portStr string
			for _, ch := range rest {
				if ch >= '0' && ch <= '9' {
					portStr += string(ch)
				} else {
					break
				}
			}
			if port, err := strconv.Atoi(portStr); err == nil {
				newPort := strconv.Itoa(port + offset)
				container.Args[i] = strings.Replace(arg, flag+" "+portStr, flag+" "+newPort, 1)
				return
			}
		}
	}

	// Also check Command for shell-wrapped cases
	for i, cmd := range container.Command {
		if strings.Contains(cmd, flag+" ") {
			parts := strings.Split(cmd, flag+" ")
			if len(parts) < 2 {
				continue
			}
			rest := parts[1]
			var portStr string
			for _, ch := range rest {
				if ch >= '0' && ch <= '9' {
					portStr += string(ch)
				} else {
					break
				}
			}
			if port, err := strconv.Atoi(portStr); err == nil {
				newPort := strconv.Itoa(port + offset)
				container.Command[i] = strings.Replace(cmd, flag+" "+portStr, flag+" "+newPort, 1)
				return
			}
		}
	}
}

// removeGPUResources strips nvidia.com/gpu from container resource limits and requests.
// GPU allocation is handled by DRA when failover is enabled.
func removeGPUResources(container *corev1.Container) {
	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	delete(container.Resources.Limits, gpuResource)
	delete(container.Resources.Requests, gpuResource)
}

// buildGMSSidecar creates the GMS weight server as a sidecar init container
// (restartPolicy: Always). kubelet starts it before regular containers and
// keeps it running for the pod's lifetime.
//
// Each GPU gets its own GMS subprocess via a bash wrapper that forwards
// signals and exits if any child dies. TMPDIR is set so UUID-based sockets
// land in the shared volume.
func buildGMSSidecar(image string, gpuCount int) corev1.Container {
	return corev1.Container{
		Name:          "gms-weights",
		Image:         image,
		Command:       []string{"bash", "-c"},
		Args:          []string{gmsWrapperScript(gpuCount)},
		RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
		Env: []corev1.EnvVar{
			{Name: "TMPDIR", Value: failoverSharedMountPath},
		},
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      failoverSharedVolumeName,
				MountPath: failoverSharedMountPath,
			},
		},
		StartupProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: gmsReadyCheckCommand(gpuCount),
				},
			},
			PeriodSeconds:    2,
			FailureThreshold: 150, // 2s * 150 = 5 min
		},
		Resources: corev1.ResourceRequirements{
			Claims: []corev1.ResourceClaim{
				{Name: failoverDRAClaimName},
			},
		},
	}
}

// gmsWrapperScript generates a bash script that launches one GMS subprocess
// per GPU device, waits for any to exit, then tears down the process group.
func gmsWrapperScript(gpuCount int) string {
	devList := make([]string, gpuCount)
	for i := range gpuCount {
		devList[i] = strconv.Itoa(i)
	}
	return fmt.Sprintf(
		`cleanup() { kill -- -$$ 2>/dev/null; exit 1; }
trap cleanup SIGTERM SIGINT
for dev in %s; do
  python3 -m gpu_memory_service --device "$dev" &
  echo "Started GMS device=$dev pid=$!"
done
wait -n
echo "A GMS subprocess exited, shutting down"
cleanup`, strings.Join(devList, " "))
}

// gmsReadyCheckCommand returns the exec probe command that verifies the
// expected number of GMS UDS sockets exist on the shared volume.
func gmsReadyCheckCommand(gpuCount int) []string {
	return []string{
		"sh", "-c",
		fmt.Sprintf("test $(ls %s/gms_*.sock 2>/dev/null | wc -l) -ge %d", failoverSharedMountPath, gpuCount),
	}
}

func failoverSharedVolume() corev1.Volume {
	return corev1.Volume{
		Name: failoverSharedVolumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}
}

// FailoverResourceClaimTemplateName returns the deterministic name for the
// ResourceClaimTemplate associated with a failover-enabled component.
func FailoverResourceClaimTemplateName(parentName, serviceName string) string {
	return fmt.Sprintf("%s-%s-gpu", parentName, strings.ToLower(serviceName))
}

// GenerateFailoverResourceClaimTemplate builds the ResourceClaimTemplate that
// provides shared GPU access to all containers in a failover pod via DRA.
//
// When failover is not enabled for the component, it returns the template
// skeleton with toDelete=true so that SyncResource cleans up any previously
// created template.
func GenerateFailoverResourceClaimTemplate(
	parentName, namespace, serviceName string,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
) (*resourcev1.ResourceClaimTemplate, bool, error) {
	name := FailoverResourceClaimTemplateName(parentName, serviceName)

	template := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}

	if !isFailoverEnabled(component) {
		return template, true, nil
	}

	gpuCount, err := getGPUCount(component)
	if err != nil {
		return nil, false, fmt.Errorf("failed to get GPU count for ResourceClaimTemplate: %w", err)
	}

	deviceClassName := "gpu.nvidia.com"
	if component.Resources != nil && component.Resources.Limits != nil && component.Resources.Limits.GPUType != "" {
		deviceClassName = component.Resources.Limits.GPUType
	}

	template.Spec = resourcev1.ResourceClaimTemplateSpec{
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{
					{
						Name: "gpus",
						Exactly: &resourcev1.ExactDeviceRequest{
							DeviceClassName: deviceClassName,
							AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
							Count:           int64(gpuCount),
						},
					},
				},
			},
		},
	}

	return template, false, nil
}

// FailoverHarnessConfigMapName returns the deterministic name for the
// harness scripts ConfigMap associated with a failover-enabled multinode component.
func FailoverHarnessConfigMapName(parentName, serviceName string) string {
	return fmt.Sprintf("%s-%s-%s", parentName, strings.ToLower(serviceName), failoverHarnessConfigMapSuffix)
}

// GenerateFailoverHarnessConfigMap builds a ConfigMap containing the multinode
// failover harness scripts (leader + worker). When failover is not enabled or
// the component is single-node, it returns the skeleton with toDelete=true so
// SyncResource cleans up any previously created ConfigMap.
func GenerateFailoverHarnessConfigMap(
	parentName, namespace, serviceName string,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
) (*corev1.ConfigMap, bool, error) {
	name := FailoverHarnessConfigMapName(parentName, serviceName)

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}

	if !isFailoverEnabled(component) || !component.IsMultinode() {
		return cm, true, nil
	}

	cm.Data = map[string]string{
		"harness_leader.sh": harnessLeaderScript,
		"harness_worker.sh": harnessWorkerScript,
	}

	return cm, false, nil
}

// injectMultinodeHarness adds the harness ConfigMap volume and wraps each
// engine container's entrypoint with the appropriate harness script.
// For worker pods, it also strips the wait-for-leader-mp init container
// injected by the VLLM backend — the harness scripts handle formation
// coordination via etcd and include their own TCP store wait.
func injectMultinodeHarness(
	podSpec *corev1.PodSpec,
	parentName, serviceName string,
	numberOfNodes int32,
	role Role,
	coordinationEndpoint string,
) error {
	// Strip the wait-for-leader-mp init container on worker pods.
	// The harness worker script handles the TCP store wait after the
	// etcd "go" signal, avoiding the deadlock where the init container
	// blocks on the leader's TCP store while the leader's harness waits
	// for the worker to register in etcd.
	if role == RoleWorker {
		stripped := make([]corev1.Container, 0, len(podSpec.InitContainers))
		for _, ic := range podSpec.InitContainers {
			if ic.Name != "wait-for-leader-mp" {
				stripped = append(stripped, ic)
			}
		}
		podSpec.InitContainers = stripped
	}

	configMapName := FailoverHarnessConfigMapName(parentName, serviceName)
	execMode := int32(0755)

	// Add harness ConfigMap volume
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: failoverHarnessVolumeName,
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: configMapName,
				},
				DefaultMode: &execMode,
			},
		},
	})

	etcdctl := fmt.Sprintf("etcdctl --endpoints=%s", coordinationEndpoint)

	for i := range podSpec.Containers {
		c := &podSpec.Containers[i]

		// Add volume mount
		c.VolumeMounts = append(c.VolumeMounts, corev1.VolumeMount{
			Name:      failoverHarnessVolumeName,
			MountPath: failoverHarnessMountPath,
			ReadOnly:  true,
		})

		// Determine harness script and node rank based on role
		var harnessScript string
		var nodeRank string
		if role == RoleLeader {
			harnessScript = failoverHarnessMountPath + "/harness_leader.sh"
			nodeRank = "0"
		} else {
			harnessScript = failoverHarnessMountPath + "/harness_worker.sh"
			nodeRank = extractNodeRank(c)
		}

		// Wrap the entrypoint: move original command+args into args for the harness
		originalCmd := append(c.Command, c.Args...)
		c.Command = []string{"bash", harnessScript}
		c.Args = originalCmd

		// Inject multinode harness env vars
		c.Env = append(c.Env,
			corev1.EnvVar{Name: "ETCDCTL", Value: etcdctl},
			corev1.EnvVar{Name: "NODE_RANK", Value: nodeRank},
		)
	}

	return nil
}

// extractNodeRank finds the --node-rank value from container args.
// Returns "1" as a fallback if not found.
func extractNodeRank(container *corev1.Container) string {
	for i, arg := range container.Args {
		if arg == "--node-rank" && i+1 < len(container.Args) {
			return container.Args[i+1]
		}
	}
	// Check env vars
	for _, env := range container.Env {
		if env.Name == "NODE_RANK" {
			return env.Value
		}
	}
	return "1"
}

// Embedded harness scripts for multinode failover coordination via etcd.
var (
	harnessLeaderScript = `#!/bin/bash
set -o pipefail

ENGINE_ID="${ENGINE_ID:-0}"
NNODES="${NNODES:-2}"
ETCDCTL="${ETCDCTL:-etcdctl --endpoints=http://localhost:2379}"
LEASE_TTL="${LEASE_TTL:-5}"
FORMATION_TIMEOUT="${FORMATION_TIMEOUT:-120}"
GROUP="engine-${ENGINE_ID}"
HASH=$(cat /proc/sys/kernel/random/uuid)

ts() { date +%s%3N; }
log() { echo "[$(date +%H:%M:%S.%3N)] [leader/$GROUP] $1"; }

if [ $# -eq 0 ]; then set -- sleep infinity; fi

log "Starting (hash=$HASH, nnodes=$NNODES)"

LEASE_GRANT=$($ETCDCTL lease grant $LEASE_TTL)
log "DEBUG: lease grant output: $LEASE_GRANT"
LEASE_ID=$(echo "$LEASE_GRANT" | awk '/lease/{print $2}')
if [ -z "$LEASE_ID" ]; then
    log "ERROR: Failed to create lease (output was: $LEASE_GRANT)"
    exit 1
fi
log "Lease created: $LEASE_ID (TTL=${LEASE_TTL}s)"

PUT_OUT=$($ETCDCTL put "leaders/$GROUP" "$HASH" --lease="$LEASE_ID" 2>&1)
log "Published leader key (put result: $PUT_OUT)"

# Monitored keepalive
(
    log "DEBUG: keepalive starting for lease $LEASE_ID"
    $ETCDCTL lease keep-alive "$LEASE_ID"
    EXIT_CODE=$?
    log "!!! KEEPALIVE DIED (exit=$EXIT_CODE) !!!"
) &
KEEPALIVE_PID=$!
log "Keepalive PID: $KEEPALIVE_PID"

# No trap/cleanup: when exit 1 runs, PID 1 (this script) exits,
# the container dies, and the kernel kills all remaining processes
# in the container's PID namespace. The lease expires via TTL.

# Formation: polling OK here, no engine running yet
log "Waiting for $((NNODES - 1)) worker(s) to join..."
DEADLINE=$(($(date +%s) + FORMATION_TIMEOUT))
while true; do
    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during formation) !!!"
        exit 1
    fi

    ALL_PRESENT=true
    for rank in $(seq 1 $((NNODES - 1))); do
        VAL=$($ETCDCTL get "groups/$GROUP/$HASH/rank-$rank" --print-value-only 2>&1)
        if [ -z "$VAL" ]; then
            ALL_PRESENT=false
            break
        fi
    done
    if $ALL_PRESENT; then
        log "All workers joined"
        break
    fi
    if [ $(date +%s) -gt $DEADLINE ]; then
        log "ERROR: Formation timeout (${FORMATION_TIMEOUT}s)"
        exit 1
    fi
    sleep 0.5
done

declare -A WORKER_UUIDS
for rank in $(seq 1 $((NNODES - 1))); do
    WORKER_UUIDS[$rank]=$($ETCDCTL get "groups/$GROUP/$HASH/rank-$rank" --print-value-only 2>&1)
    log "Recorded rank-$rank: ${WORKER_UUIDS[$rank]}"
done

# Conditional delay: if flock is held, another engine is active/waking
if ! flock -n /shared/failover.lock -c "exit 0" 2>/dev/null; then
    log "Flock held, delaying 60s for active engine to settle"
    sleep 60
    log "Delay complete"
fi

PUT_OUT=$($ETCDCTL put "groups/$GROUP/$HASH/start" "go" --lease="$LEASE_ID" 2>&1)
log "Sent go signal (put result: $PUT_OUT)"

# Start watches BEFORE engine (persistent streams, no polling)
(
    log "DEBUG: starting leader key watch on leaders/$GROUP"
    $ETCDCTL watch "leaders/$GROUP" 2>&1 | while IFS= read -r line; do
        log "DEBUG: leader watch event: [$line]"
        if [ "$line" = "DELETE" ]; then
            log "WATCH: own leader key deleted (lease expired?)"
            exit 0
        fi
    done
    log "DEBUG: leader watch stream ended (pipe closed or etcdctl exited)"
) &
WATCH_LEADER_PID=$!
log "Leader key watch PID: $WATCH_LEADER_PID"

(
    log "DEBUG: starting worker watch on groups/$GROUP/$HASH/ (prefix)"
    $ETCDCTL watch --prefix "groups/$GROUP/$HASH/" 2>&1 | while IFS= read -r line; do
        log "DEBUG: worker watch event: [$line]"
        if [ "$line" = "DELETE" ]; then
            log "WATCH: worker key deleted under groups/$GROUP/$HASH/"
            exit 0
        fi
    done
    log "DEBUG: worker watch stream ended (pipe closed or etcdctl exited)"
) &
WATCH_WORKERS_PID=$!
log "Worker keys watch PID: $WATCH_WORKERS_PID"

# Start engine
log "Starting engine: $*"
"$@" &
ENGINE_PID=$!
log "Engine PID: $ENGINE_PID"

# Monitor via local PID checks only (no etcdctl get calls)
log "Monitoring (watch-based, no polling)..."
while true; do
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        log "Engine died, exiting ($(ts))"
        exit 1
    fi

    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during monitoring) ($(ts)) !!!"
        exit 1
    fi

    if ! kill -0 $WATCH_LEADER_PID 2>/dev/null; then
        log "DETECTED: leader watch exited (key deleted or stream lost) ($(ts))"
        exit 1
    fi

    if ! kill -0 $WATCH_WORKERS_PID 2>/dev/null; then
        log "DETECTED: worker watch exited (key deleted or stream lost) ($(ts))"
        exit 1
    fi

    sleep 1
done
`

	harnessWorkerScript = `#!/bin/bash
set -o pipefail

ENGINE_ID="${ENGINE_ID:-0}"
NODE_RANK="${NODE_RANK:-1}"
ETCDCTL="${ETCDCTL:-etcdctl --endpoints=http://localhost:2379}"
LEASE_TTL="${LEASE_TTL:-5}"
GROUP="engine-${ENGINE_ID}"
MY_UUID=$(cat /proc/sys/kernel/random/uuid)

ts() { date +%s%3N; }
log() { echo "[$(date +%H:%M:%S.%3N)] [worker/$GROUP/rank-$NODE_RANK] $1"; }

if [ $# -eq 0 ]; then set -- sleep infinity; fi

log "Starting (uuid=$MY_UUID)"

# Wait for leader (polling OK, no engine running yet)
log "Waiting for leader..."
while true; do
    HASH=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>&1)
    [ -n "$HASH" ] && break
    sleep 0.5
done
log "Found leader (hash=$HASH)"

LEASE_GRANT=$($ETCDCTL lease grant $LEASE_TTL)
log "DEBUG: lease grant output: $LEASE_GRANT"
LEASE_ID=$(echo "$LEASE_GRANT" | awk '/lease/{print $2}')
if [ -z "$LEASE_ID" ]; then
    log "ERROR: Failed to create lease (output was: $LEASE_GRANT)"
    exit 1
fi
log "Lease created: $LEASE_ID (TTL=${LEASE_TTL}s)"

PUT_OUT=$($ETCDCTL put "groups/$GROUP/$HASH/rank-$NODE_RANK" "$MY_UUID" --lease="$LEASE_ID" 2>&1)
log "Registered under leader hash (put result: $PUT_OUT)"

(
    log "DEBUG: keepalive starting for lease $LEASE_ID"
    $ETCDCTL lease keep-alive "$LEASE_ID"
    EXIT_CODE=$?
    log "!!! KEEPALIVE DIED (exit=$EXIT_CODE) !!!"
) &
KEEPALIVE_PID=$!
log "Keepalive PID: $KEEPALIVE_PID"

# No trap/cleanup: when exit 1 runs, PID 1 (this script) exits,
# the container dies, and the kernel kills all remaining processes
# in the container's PID namespace. The lease expires via TTL.

# Wait for go signal (polling OK, no engine running yet)
log "Waiting for go signal..."
while true; do
    GO=$($ETCDCTL get "groups/$GROUP/$HASH/start" --print-value-only 2>&1)
    if [ "$GO" = "go" ]; then
        log "Go signal received"
        break
    fi

    CURRENT=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>&1)
    if [ -z "$CURRENT" ]; then
        log "DETECTED: Leader disappeared while waiting for go ($(ts))"
        exit 1
    fi
    if [ "$CURRENT" != "$HASH" ]; then
        log "DETECTED: Leader hash changed while waiting for go ($(ts))"
        exit 1
    fi

    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during wait for go) !!!"
        exit 1
    fi

    sleep 0.5
done

# Start watch BEFORE engine (persistent stream, no polling)
(
    log "DEBUG: starting leader watch on leaders/$GROUP"
    $ETCDCTL watch "leaders/$GROUP" 2>&1 | while IFS= read -r line; do
        log "DEBUG: leader watch event: [$line]"
        if [ "$line" = "DELETE" ]; then
            log "WATCH: leader key deleted"
            exit 0
        fi
    done
    log "DEBUG: leader watch stream ended (pipe closed or etcdctl exited)"
) &
WATCH_LEADER_PID=$!
log "Leader watch PID: $WATCH_LEADER_PID"

# Start engine
log "Starting engine: $*"
"$@" &
ENGINE_PID=$!
log "Engine PID: $ENGINE_PID"

# Monitor via local PID checks only (no etcdctl get calls)
log "Monitoring (watch-based, no polling)..."
while true; do
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        log "Engine died, exiting ($(ts))"
        exit 1
    fi

    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during monitoring) ($(ts)) !!!"
        exit 1
    fi

    if ! kill -0 $WATCH_LEADER_PID 2>/dev/null; then
        log "DETECTED: leader watch exited (key deleted or stream lost) ($(ts))"
        exit 1
    fi

    sleep 1
done
`
)
