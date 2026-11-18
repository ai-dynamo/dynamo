/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"cmp"
	"fmt"
	"net"
	"net/url"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	vllmMasterPortFlag   = "--master-port"
	vllmMasterPortStride = 100

	// The CUDA graph archive is an emptyDir so it never outlives the pod or
	// crosses builds. The path is passed as env rather than a launch flag
	// because the engine args may be shell-wrapped.
	gmsGraphArchiveVolumeName = "gms-graph-archive"
	gmsGraphArchiveMountPath  = "/gms-graph-archive"
	gmsGraphArchiveEnvVar     = "DYN_VLLM_GMS_GRAPH_ARCHIVE"

	vllmEngineModule     = "dynamo.vllm"
	engineLauncherModule = "dynamo.vllm.launcher"
	// OpenBLAS starts one thread per core when it is loaded, and the launcher
	// must stay single-threaded while it imports the engine's modules.
	openBLASThreadsEnvVar = "OPENBLAS_NUM_THREADS"
	// The launcher keeps the uptime, in seconds since boot, until which it
	// counts as live in this file (container-local /tmp, named per engine in
	// case /tmp is shared). Uptime, unlike the wall clock, is never stepped.
	engineLauncherLivenessFileFormat = "/tmp/dynamo-launcher-%s.live"
)

var pythonInterpreter = regexp.MustCompile(`^python[0-9.]*$`)

// buildColdStartFailoverPod prepares snapshot-less failover, supported only for
// vLLM with GMS V0. Engines initialize concurrently in shadow mode and need
// separate collective/KV ports, unlike the restored snapshot engine pair.
// Mutates podSpec, which must not be nil and must already have GMS resources.
// graphArchive may be nil, in which case no CUDA graph archive is rendered.
func buildColdStartFailoverPod(podSpec *corev1.PodSpec, numberOfNodes int32, backendFramework BackendFramework, graphArchive *v1beta1.GraphArchiveSpec) error {
	// Reject other backends before changing the pod.
	if backendFramework != BackendFrameworkVLLM {
		return fmt.Errorf("cold-start failover is currently supported only for vLLM (detected: %s)", backendFramework)
	}
	if err := buildFailoverEnginePair(podSpec); err != nil {
		return err
	}

	// One archive per pod lets replacement engines load the first shadow's CUDA graphs.
	if graphArchive != nil {
		// Defend direct rendering as well as objects that bypassed admission.
		if numberOfNodes != 1 {
			return fmt.Errorf("failover.graphArchive requires a single node setup")
		}
		ensureGraphArchiveVolume(podSpec, graphArchive)
	}

	// Only the two cloned engines receive legacy shadow initialization settings.
	for engineID := range failoverEngineCount {
		c := &podSpec.Containers[engineID]
		c.Env = append(c.Env, corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"})
		c.Env = append(c.Env,
			corev1.EnvVar{Name: "VLLM_NIXL_SIDE_CHANNEL_PORT", Value: strconv.Itoa(5600 + engineID)},
			corev1.EnvVar{Name: "DYN_VLLM_KV_EVENT_PORT", Value: strconv.Itoa(20080 + engineID)},
		)

		// Stagger --master-port for TP so each engine group uses a distinct
		// torch.distributed TCP store. engine-0 keeps the default (29500),
		// engine-1 gets 29500 + stride.
		if engineID > 0 {
			if hasMasterPortFlag(c) {
				staggerMasterPort(c, engineID)
			} else {
				c.Args = append(c.Args, vllmMasterPortFlag, strconv.Itoa(29500+engineID*vllmMasterPortStride))
			}
		}

		if numberOfNodes > 1 {
			c.Env = append(c.Env,
				corev1.EnvVar{Name: "NNODES", Value: strconv.Itoa(int(numberOfNodes))},
			)
		}

		// The operator-managed archive path replaces any value inherited from main.
		if graphArchive != nil {
			if err := mountGraphArchive(c); err != nil {
				return err
			}
			removeEnvVar(c, gmsGraphArchiveEnvVar)
			c.Env = append(c.Env, corev1.EnvVar{Name: gmsGraphArchiveEnvVar, Value: gmsGraphArchiveMountPath})
		}
	}
	return nil
}

// ensureGraphArchiveVolume adds the archive emptyDir to the pod. The volume
// name is operator-owned, so an authored volume with that name is replaced.
// Mutates podSpec; podSpec and graphArchive must not be nil.
func ensureGraphArchiveVolume(podSpec *corev1.PodSpec, graphArchive *v1beta1.GraphArchiveSpec) {
	// Copy the limit so the rendered pod never aliases the component spec.
	volume := corev1.Volume{
		Name:     gmsGraphArchiveVolumeName,
		EmptyDir: &corev1.EmptyDirVolumeSource{},
	}
	if graphArchive.SizeLimit != nil {
		sizeLimit := graphArchive.SizeLimit.DeepCopy()
		volume.EmptyDir.SizeLimit = &sizeLimit
	}

	// Replace by name: two volumes with one name make the pod invalid.
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == gmsGraphArchiveVolumeName {
			podSpec.Volumes[i] = volume
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, volume)
}

// mountGraphArchive mounts the archive volume in an engine container at the
// operator-owned path. Mutates container, which must not be nil.
func mountGraphArchive(container *corev1.Container) error {
	// Drop an inherited mount of the archive volume; refuse another volume at its path.
	mounts := make([]corev1.VolumeMount, 0, len(container.VolumeMounts)+1)
	for _, mount := range container.VolumeMounts {
		if mount.Name == gmsGraphArchiveVolumeName {
			continue
		}
		if mount.MountPath == gmsGraphArchiveMountPath {
			return fmt.Errorf("failover.graphArchive mount path %q is already used by volume %q", gmsGraphArchiveMountPath, mount.Name)
		}
		mounts = append(mounts, mount)
	}
	container.VolumeMounts = append(mounts, corev1.VolumeMount{Name: gmsGraphArchiveVolumeName, MountPath: gmsGraphArchiveMountPath})
	return nil
}

// hasMasterPortFlag checks if --master-port appears in the container args or command.
func hasMasterPortFlag(container *corev1.Container) bool {
	for _, arg := range container.Args {
		if arg == vllmMasterPortFlag || strings.Contains(arg, vllmMasterPortFlag+" ") {
			return true
		}
	}
	for _, cmd := range container.Command {
		if strings.Contains(cmd, vllmMasterPortFlag+" ") {
			return true
		}
	}
	return false
}

func staggerMasterPort(container *corev1.Container, engineID int) {
	offset := engineID * vllmMasterPortStride
	staggerFlagValue(container, vllmMasterPortFlag, offset)
}

// staggerFlagValue finds a --flag VALUE pair in container args and adds offset
// to the integer value. Handles both separate-token args (["--flag", "29500"])
// and shell-wrapped args (["sh", "-c", "... --flag 29500 ..."]).
func staggerFlagValue(container *corev1.Container, flag string, offset int) {
	for i, arg := range container.Args {
		if arg == flag && i+1 < len(container.Args) {
			if port, err := strconv.Atoi(container.Args[i+1]); err == nil {
				container.Args[i+1] = strconv.Itoa(port + offset)
				return
			}
		}
	}

	// Preserve Args precedence when looking inside shell-wrapped launch strings.
	if !staggerEmbeddedFlagValue(container.Args, flag, offset) {
		staggerEmbeddedFlagValue(container.Command, flag, offset)
	}
}

// staggerEmbeddedFlagValue offsets the first embedded flag value it can parse.
func staggerEmbeddedFlagValue(tokens []string, flag string, offset int) bool {
	for i, token := range tokens {
		parts := strings.Split(token, flag+" ")
		if len(parts) < 2 {
			continue
		}

		// Read only the integer prefix, leaving trailing launch arguments intact.
		var portStr strings.Builder
		for _, ch := range parts[1] {
			if ch < '0' || ch > '9' {
				break
			}
			portStr.WriteString(string(ch))
		}
		if port, err := strconv.Atoi(portStr.String()); err == nil {
			tokens[i] = strings.Replace(token, flag+" "+portStr.String(), flag+" "+strconv.Itoa(port+offset), 1)
			return true
		}
	}
	return false
}

// ValidateEngineLauncher checks the requirements of failover.engineLauncher.
// component and fldPath must not be nil.
func ValidateEngineLauncher(component *v1beta1.DynamoComponentDeploymentSharedSpec, fldPath *field.Path, explicitBackendFramework string) field.ErrorList {
	if component.Experimental == nil || component.Experimental.Failover == nil || !component.Experimental.Failover.EngineLauncher {
		return nil
	}
	launcherPath := fldPath.Child("experimental", "failover", "engineLauncher")
	var allErrs field.ErrorList

	// The launcher replaces the cold-started engines of single-node intra-pod failover.
	if !IsIntraPodFailoverEnabled(component) || component.IsInterPodGMSEnabled() {
		allErrs = append(allErrs, field.Forbidden(launcherPath, "requires intra-pod failover"))
	}
	if GetCheckpoint(component) != nil {
		allErrs = append(allErrs, field.Forbidden(launcherPath, "is not supported with a checkpoint: restored engines do not run the container command"))
	}
	if component.GetNumberOfNodes() != 1 {
		allErrs = append(allErrs, field.Forbidden(launcherPath, "requires a single node setup"))
	}

	// Resolve the backend exactly as rendering does, including command inference.
	backendFramework, err := determineBackendFrameworkForComponent(component, explicitBackendFramework)
	if err != nil || backendFramework != BackendFrameworkVLLM {
		allErrs = append(allErrs, field.Forbidden(launcherPath, "requires the vLLM backend"))
	}

	// Every authored main container must reach rendering as a direct
	// dynamo.vllm command with a liveness probe the launcher can run.
	for _, podTemplate := range ComponentPodTemplates(component) {
		main := GetMainContainerFromPodSpec(&podTemplate.Spec)
		if main == nil {
			continue
		}
		// Rendering wraps an elastic-EP Ray launch in a shell that starts the Ray head first.
		if IsElasticEPRayLaunch(main) {
			allErrs = append(allErrs, field.Forbidden(launcherPath, "is not supported with elastic expert parallelism on the Ray backend: the engine is started from a shell wrapper"))
		} else if err := useEngineLauncher(main.DeepCopy()); err != nil {
			allErrs = append(allErrs, field.Forbidden(launcherPath, err.Error()))
		}
		if main.LivenessProbe != nil {
			if err := checkLauncherLivenessProbe(main.LivenessProbe); err != nil {
				allErrs = append(allErrs, field.Forbidden(launcherPath, err.Error()))
			}
		}
	}
	return allErrs
}

// applyEngineLauncher starts both cold-start failover engines through the
// engine launcher. Mutates podSpec, which must not be nil and must already hold
// the engine pair built by buildColdStartFailoverPod. On error podSpec is left
// unchanged.
func applyEngineLauncher(podSpec *corev1.PodSpec, numberOfNodes int32, snapshotEnabled bool) error {
	// Defend direct rendering as well as objects that bypassed admission.
	if snapshotEnabled {
		return fmt.Errorf("failover.engineLauncher is not supported with a checkpoint")
	}
	if numberOfNodes != 1 {
		return fmt.Errorf("failover.engineLauncher requires a single node setup")
	}

	// Rewrite copies first, so that an engine that cannot be rewritten leaves
	// the other one untouched.
	engines := make([]corev1.Container, failoverEngineCount)
	for engineID := range failoverEngineCount {
		c := podSpec.Containers[engineID].DeepCopy()
		launcherArgs, err := moveLivenessProbeToLauncher(c)
		if err == nil {
			err = useEngineLauncher(c, launcherArgs...)
		}
		if err != nil {
			return fmt.Errorf("failover.engineLauncher %w", err)
		}

		if findEnvVar(c.Env, openBLASThreadsEnvVar) == nil {
			c.Env = append(c.Env, corev1.EnvVar{Name: openBLASThreadsEnvVar, Value: "1"})
		}
		engines[engineID] = *c
	}
	copy(podSpec.Containers, engines)
	return nil
}

// moveLivenessProbeToLauncher hands the engine's HTTP liveness probe to the
// launcher and points the container's liveness probe at the launcher instead.
// A replacement engine serves its probe endpoint only once it has started, so
// kubelet probing the engine directly would restart the container during every
// replacement, the cold start the launcher avoids. The launcher runs the same
// checks, kills a hung engine and starts another; kubelet restarts the
// container only when the launcher stops renewing its liveness file.
// Returns the launcher options. Mutates c, which must not be nil.
func moveLivenessProbeToLauncher(c *corev1.Container) ([]string, error) {
	probe := c.LivenessProbe
	if probe == nil {
		return nil, nil
	}
	if err := checkLauncherLivenessProbe(probe); err != nil {
		return nil, err
	}

	// Resolve the endpoint the way kubelet does: a named port on this
	// container, the pod itself unless the probe names a host.
	get := probe.HTTPGet
	port := get.Port.IntVal
	if get.Port.Type == intstr.String {
		named := findContainerPort(c, get.Port.StrVal)
		if named == nil {
			return nil, fmt.Errorf("requires the engine liveness probe's port %q to be a port of container %s", get.Port.StrVal, c.Name)
		}
		port = named.ContainerPort
	}
	livenessURL := launcherLivenessURL(cmp.Or(get.Host, "127.0.0.1"), port, get.Path)

	// Kubernetes defaults for unset probe fields.
	period := cmp.Or(probe.PeriodSeconds, 10)
	timeout := cmp.Or(probe.TimeoutSeconds, 1)
	failureThreshold := cmp.Or(probe.FailureThreshold, 3)
	// The startup probe guards a starting engine; the launcher applies the same
	// budget to every engine it starts.
	startupTimeout := probe.InitialDelaySeconds + period*failureThreshold
	if sp := c.StartupProbe; sp != nil {
		startupTimeout = sp.InitialDelaySeconds + cmp.Or(sp.PeriodSeconds, 10)*cmp.Or(sp.FailureThreshold, 3)
	}
	livenessFile := fmt.Sprintf(engineLauncherLivenessFileFormat, c.Name)

	// Kubelet now checks the launcher's deadline file on the engine probe's
	// schedule. The initial delay is kept: without a startup probe it is what
	// keeps kubelet from checking before the launcher has written the file.
	c.LivenessProbe = &corev1.Probe{
		Exec: &corev1.ExecAction{Command: []string{
			"sh", "-c", fmt.Sprintf(`test "$(cut -d. -f1 /proc/uptime)" -lt "$(cat %s)"`, livenessFile),
		}},
		InitialDelaySeconds:           probe.InitialDelaySeconds,
		PeriodSeconds:                 period,
		TimeoutSeconds:                timeout,
		FailureThreshold:              failureThreshold,
		TerminationGracePeriodSeconds: probe.TerminationGracePeriodSeconds,
	}
	return []string{
		"--liveness-url", livenessURL,
		"--liveness-period", strconv.Itoa(int(period)),
		"--liveness-timeout", strconv.Itoa(int(timeout)),
		"--liveness-failure-threshold", strconv.Itoa(int(failureThreshold)),
		"--startup-timeout", strconv.Itoa(int(startupTimeout)),
		"--liveness-file", livenessFile,
	}, nil
}

// launcherLivenessURL builds the URL of an HTTP probe as kubelet does, so a
// path without a leading slash or with a query string reaches the same
// endpoint.
func launcherLivenessURL(host string, port int32, path string) string {
	u, err := url.Parse(path)
	if err != nil {
		u = &url.URL{Path: path}
	}
	u.Scheme = "http"
	u.Host = net.JoinHostPort(host, strconv.Itoa(int(port)))
	if u.Path == "" {
		u.Path = "/"
	}
	return u.String()
}

// checkLauncherLivenessProbe reports whether the launcher can run probe itself:
// it checks the engine with a plain HTTP GET.
func checkLauncherLivenessProbe(probe *corev1.Probe) error {
	get := probe.HTTPGet
	if get == nil || (get.Scheme != "" && get.Scheme != corev1.URISchemeHTTP) || len(get.HTTPHeaders) > 0 {
		return fmt.Errorf("requires the engine liveness probe to be a plain HTTP GET (no HTTPS, no headers)")
	}
	return nil
}

// useEngineLauncher rewrites `python3 -m dynamo.vllm ARGS` into
// `python3 -m dynamo.vllm.launcher LAUNCHER_ARGS -- ARGS`. The command must be
// set: rendering runs an empty command through a shell. `-m` and the module
// may follow in Command or in Args; any other shape is an error. Mutates
// container, which must not be nil.
func useEngineLauncher(container *corev1.Container, launcherArgs ...string) error {
	tokens := slices.Concat(container.Command, container.Args)
	if len(container.Command) == 0 || len(tokens) < 3 || !pythonInterpreter.MatchString(filepath.Base(tokens[0])) || tokens[1] != "-m" || tokens[2] != vllmEngineModule {
		return fmt.Errorf("requires the main container's command to start with the separate items `python3`, `-m`, `%s`: not a shell string, an interpreter flag or another module", vllmEngineModule)
	}

	// Swap the module; the launcher's own options and `--` follow it, ahead of
	// the engine's arguments.
	launcherTokens := append(slices.Clone(launcherArgs), "--")
	if len(container.Command) >= 3 {
		container.Command[2] = engineLauncherModule
		container.Command = slices.Insert(container.Command, 3, launcherTokens...)
		return nil
	}
	moduleIndex := 2 - len(container.Command)
	container.Args[moduleIndex] = engineLauncherModule
	container.Args = slices.Insert(container.Args, moduleIndex+1, launcherTokens...)
	return nil
}
