package dynamo

import (
	"fmt"
	"path"
	"regexp"
	"strconv"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

const (
	PowerGateExecutable                = "dynamo-power-gate"
	PowerGateSeparator                 = "--"
	PowerGateVolumeName                = "dynamo-power-gate"
	PowerGateVolumeMountPath           = "/var/run/dynamo/power-gate"
	PowerGatePodUIDPath                = "pod-uid"
	PowerGateReportPath                = "report"
	PowerGateTerminationMessagePath    = "/dev/termination-log"
	PowerGateReportAnnotation          = "dynamo.nvidia.com/gpu-power-enforcement-report"
	PowerGateDGDUIDEnv                 = "DYNAMO_POWER_DGD_UID"
	PowerGateComponentEnv              = "DYNAMO_POWER_COMPONENT"
	PowerGateExpectedGPUCountEnv       = "DYNAMO_POWER_EXPECTED_GPU_COUNT"
	PowerGateInGateBoundWattsPerGPUEnv = "DYNAMO_POWER_IN_GATE_BOUND_WATTS_PER_GPU"
	powerGateReportAnnotationFieldPath = "metadata.annotations['" + PowerGateReportAnnotation + "']"
)

// PowerGateInputs are the immutable operator-owned values required by the
// pre-backend enforcement gate.
type PowerGateInputs struct {
	DGDUID                   string
	Component                string
	ExpectedPhysicalGPUCount int32
	InGateBoundWattsPerGPU   int64
}

// ApplyPowerGate structurally wraps the final rendered main-container command.
// It must run after backend rendering and PodTemplate merging.
func ApplyPowerGate(podSpec *corev1.PodSpec, inputs PowerGateInputs) error {
	if podSpec == nil {
		return fmt.Errorf("power gate pod spec is nil")
	}
	if inputs.DGDUID == "" || inputs.Component == "" || inputs.ExpectedPhysicalGPUCount < 1 || inputs.InGateBoundWattsPerGPU < 1 {
		return fmt.Errorf("power gate inputs must contain nonempty identity and positive bounds")
	}

	// Resolve exactly one structurally valid main container before mutation.
	var mainContainer *corev1.Container
	for index := range podSpec.Containers {
		if podSpec.Containers[index].Name != commonconsts.MainContainerName {
			continue
		}
		if mainContainer != nil {
			return fmt.Errorf("power gate found duplicate main containers")
		}
		mainContainer = &podSpec.Containers[index]
	}
	if mainContainer == nil {
		return fmt.Errorf("power gate main container not found")
	}
	if len(mainContainer.Command) == 0 {
		return fmt.Errorf("power gate main container command is empty")
	}
	if mainContainer.Lifecycle != nil &&
		((mainContainer.Lifecycle.PostStart != nil && mainContainer.Lifecycle.PostStart.Exec != nil) ||
			(mainContainer.Lifecycle.PreStop != nil && mainContainer.Lifecycle.PreStop.Exec != nil)) {
		return fmt.Errorf("power gate main container lifecycle exec actions are unsupported")
	}
	if (mainContainer.StartupProbe != nil && mainContainer.StartupProbe.Exec != nil) ||
		(mainContainer.ReadinessProbe != nil && mainContainer.ReadinessProbe.Exec != nil) ||
		(mainContainer.LivenessProbe != nil && mainContainer.LivenessProbe.Exec != nil) {
		return fmt.Errorf("power gate main container exec probes are unsupported")
	}
	if mainContainer.TerminationMessagePath != "" && mainContainer.TerminationMessagePath != PowerGateTerminationMessagePath {
		return fmt.Errorf("power gate main container termination message path %q is unsupported", mainContainer.TerminationMessagePath)
	}
	if mainContainer.TerminationMessagePolicy != "" && mainContainer.TerminationMessagePolicy != corev1.TerminationMessageReadFile {
		return fmt.Errorf("power gate main container termination message policy %q is unsupported", mainContainer.TerminationMessagePolicy)
	}
	if len(podSpec.ResourceClaims) != 0 {
		return fmt.Errorf("power gate does not support Pod resource claims")
	}
	for index := range podSpec.Containers {
		container := &podSpec.Containers[index]
		if len(container.Resources.Claims) != 0 {
			return fmt.Errorf("power gate container %q resource claims are unsupported", container.Name)
		}
		if container.Name != commonconsts.MainContainerName && containerHasPositiveGPUResource(container.Resources) {
			return fmt.Errorf("power gate non-main container %q cannot request GPU resources", container.Name)
		}
	}
	for index := range podSpec.InitContainers {
		container := &podSpec.InitContainers[index]
		if len(container.Resources.Claims) != 0 {
			return fmt.Errorf("power gate init container %q resource claims are unsupported", container.Name)
		}
		if containerHasPositiveGPUResource(container.Resources) {
			return fmt.Errorf("power gate init container %q cannot request GPU resources", container.Name)
		}
	}
	for index := range podSpec.EphemeralContainers {
		container := &podSpec.EphemeralContainers[index]
		if len(container.Resources.Claims) != 0 {
			return fmt.Errorf("power gate ephemeral container %q resource claims are unsupported", container.Name)
		}
		if containerHasPositiveGPUResource(container.Resources) {
			return fmt.Errorf("power gate ephemeral container %q cannot request GPU resources", container.Name)
		}
	}

	// Reject user collisions instead of overwriting operator-owned gate inputs.
	for _, env := range mainContainer.Env {
		if isPowerGateReservedEnv(env.Name) {
			return fmt.Errorf("power gate reserved environment variable %q already exists", env.Name)
		}
	}
	for _, volume := range podSpec.Volumes {
		if volume.Name == PowerGateVolumeName {
			return fmt.Errorf("power gate reserved volume %q already exists", PowerGateVolumeName)
		}
	}
	for index := range podSpec.Containers {
		if containerUsesPowerGateVolume(&podSpec.Containers[index]) {
			return fmt.Errorf("power gate reserved volume reference already exists")
		}
	}
	for index := range podSpec.InitContainers {
		if containerUsesPowerGateVolume(&podSpec.InitContainers[index]) {
			return fmt.Errorf("power gate reserved volume reference already exists")
		}
	}
	for index := range podSpec.EphemeralContainers {
		for _, mount := range podSpec.EphemeralContainers[index].VolumeMounts {
			if mount.Name == PowerGateVolumeName {
				return fmt.Errorf("power gate reserved volume reference already exists")
			}
		}
		for _, device := range podSpec.EphemeralContainers[index].VolumeDevices {
			if device.Name == PowerGateVolumeName {
				return fmt.Errorf("power gate reserved volume reference already exists")
			}
		}
	}
	for _, mount := range mainContainer.VolumeMounts {
		if protectedPowerGatePathOverlaps(mount.MountPath) {
			return fmt.Errorf("power gate reserved mount path overlaps %q", mount.MountPath)
		}
	}
	for _, device := range mainContainer.VolumeDevices {
		if protectedPowerGatePathOverlaps(device.DevicePath) {
			return fmt.Errorf("power gate reserved device path overlaps %q", device.DevicePath)
		}
	}

	// Preserve the rendered args and command tokens while inserting the gate prefix.
	originalCommand := append([]string(nil), mainContainer.Command...)
	mainContainer.Command = append([]string{PowerGateExecutable, PowerGateSeparator}, originalCommand...)
	mainContainer.Env = append(mainContainer.Env,
		corev1.EnvVar{Name: PowerGateDGDUIDEnv, Value: inputs.DGDUID},
		corev1.EnvVar{Name: PowerGateComponentEnv, Value: inputs.Component},
		corev1.EnvVar{Name: PowerGateExpectedGPUCountEnv, Value: strconv.FormatInt(int64(inputs.ExpectedPhysicalGPUCount), 10)},
		corev1.EnvVar{Name: PowerGateInGateBoundWattsPerGPUEnv, Value: strconv.FormatInt(inputs.InGateBoundWattsPerGPU, 10)},
	)
	mainContainer.VolumeMounts = append(mainContainer.VolumeMounts, corev1.VolumeMount{
		Name:      PowerGateVolumeName,
		MountPath: PowerGateVolumeMountPath,
		ReadOnly:  true,
	})
	mainContainer.TerminationMessagePath = PowerGateTerminationMessagePath
	mainContainer.TerminationMessagePolicy = corev1.TerminationMessageReadFile

	// Project only the immutable Pod UID and the Agent-owned report annotation.
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: PowerGateVolumeName,
		VolumeSource: corev1.VolumeSource{
			DownwardAPI: &corev1.DownwardAPIVolumeSource{
				Items: []corev1.DownwardAPIVolumeFile{
					{
						Path: PowerGatePodUIDPath,
						FieldRef: &corev1.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "metadata.uid",
						},
					},
					{
						Path: PowerGateReportPath,
						FieldRef: &corev1.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  powerGateReportAnnotationFieldPath,
						},
					},
				},
			},
		},
	})
	return nil
}

func isPowerGateReservedEnv(name string) bool {
	switch name {
	case PowerGateDGDUIDEnv,
		PowerGateComponentEnv,
		PowerGateExpectedGPUCountEnv,
		PowerGateInGateBoundWattsPerGPUEnv:
		return true
	default:
		return false
	}
}

func containerUsesPowerGateVolume(container *corev1.Container) bool {
	for _, mount := range container.VolumeMounts {
		if mount.Name == PowerGateVolumeName {
			return true
		}
	}
	for _, device := range container.VolumeDevices {
		if device.Name == PowerGateVolumeName {
			return true
		}
	}
	return false
}

func containerHasPositiveGPUResource(resources corev1.ResourceRequirements) bool {
	for _, resourceList := range []corev1.ResourceList{resources.Requests, resources.Limits} {
		for name, quantity := range resourceList {
			normalized := strings.ToLower(string(name))
			resourceSegment := normalized
			if separator := strings.LastIndexByte(normalized, '/'); separator >= 0 {
				resourceSegment = normalized[separator+1:]
			}
			isGPU := resourceSegment == "gpu" ||
				strings.HasPrefix(resourceSegment, "gpu.") ||
				strings.HasPrefix(resourceSegment, "mig-")
			if isGPU && quantity.Sign() > 0 {
				return true
			}
		}
	}
	return false
}

func protectedPowerGatePathOverlaps(candidate string) bool {
	return pathsOverlap(candidate, PowerGateVolumeMountPath) ||
		pathsOverlap(candidate, PowerGateTerminationMessagePath)
}

func pathsOverlap(candidate, protected string) bool {
	cleaned := path.Clean(candidate)
	reserved := path.Clean(protected)
	return cleaned == "/" ||
		cleaned == reserved ||
		strings.HasPrefix(cleaned, reserved+"/") ||
		strings.HasPrefix(reserved, cleaned+"/")
}

/*
 * Flag Injection Strategy for Multinode
 *
 * This code handles the injection of distributed training flags (--dist-init-addr, --nnodes, --node-rank)
 * into container commands for multinode SGLang deployments. The complexity arises from supporting multiple
 * container command patterns and ensuring proper environment variable interpretation.
 *
 * All MultinodeDeployer implementations MUST return Kubernetes env-var
 * expansion syntax ("$(VAR)") from GetLeaderHostname / GetNodeRank. The
 * kubelet substitutes those references in container Args/Command before the
 * container starts, so plain $(VAR) references never require a shell wrapper.
 * Shell wrapping (`sh -c`) is only needed for shell-only constructs that the
 * kubelet does not evaluate - e.g. arithmetic expansion `$(( ... ))` or
 * command substitution - which is signaled by the `needsShell` bool returned
 * from GetNodeRank (Grove's `$((GROVE_PCLQ_POD_INDEX + 1))` is the canonical
 * example).
 *
 * Two main scenarios are handled:
 *
 * 1. Direct Python Command (e.g., Command: ["python3"], Args: ["-m", "sglang", "..."])
 *    - If needsShell is true (shell-only expression such as arithmetic): wrap
 *      the command in "sh -c" with exec so the shell evaluates the expression.
 *    - Otherwise: simply append flags to the Args array; the kubelet expands
 *      any $(VAR) references itself.
 *
 * 2. Non-Python Command (e.g., Command: ["sh"], Args: ["-c", "python3 -m sglang ..."])
 *    - Use regex-based injection to find embedded Python+SGLang commands within args
 *    - Insert flags after the Python command but before any shell operators (|, &, ;)
 */

// shellQuoteForBashC quotes a string so it survives shell interpretation inside sh -c.
// Simple args (flags, paths) pass through unchanged; args containing special characters
// (JSON, env vars, spaces, quotes) are wrapped in double quotes with inner escaping.
func shellQuoteForBashC(s string) string {
	if strings.ContainsAny(s, " \t\n'\"\\{}[]$`!") {
		escaped := s
		escaped = strings.ReplaceAll(escaped, `\`, `\\`) // must be first
		escaped = strings.ReplaceAll(escaped, `"`, `\"`)
		escaped = strings.ReplaceAll(escaped, `$`, `\$`)
		escaped = strings.ReplaceAll(escaped, "`", "\\`")
		escaped = strings.ReplaceAll(escaped, "'", `'"'"'`)
		return `"` + escaped + `"`
	}
	return s
}

// shellSafeToken matches tokens that are literal to the shell in every context
// and therefore need no quoting inside sh -c.
var shellSafeToken = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

// shellQuotePOSIX renders s as exactly one argv token that survives `sh -c`
// unchanged. Tokens built only from shell-neutral characters pass through
// unquoted for readability; everything else — whitespace, quotes, $, ;, |, &,
// globs, and the empty string — is wrapped in single quotes, inside which every
// byte is literal except the single quote itself, which is closed and re-opened
// via the '\” idiom. Unlike shellQuoteForBashC this is argv-preserving: it
// round-trips arbitrary tokens (including empty ones and embedded quotes)
// through the shell without splitting, dropping, or reinterpreting them.
func shellQuotePOSIX(s string) string {
	if shellSafeToken.MatchString(s) {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// containerHasArg reports whether the container already carries the given
// flag/value pair in its Args (either as adjacent tokens "flag", "value" or
// as a single token "flag=value" or "flag value" embedded inside a shell
// string). It is used to make flag injection idempotent.
func containerHasArg(container *corev1.Container, flag, value string) bool {
	if container == nil {
		return false
	}
	return hasArg(container.Args, flag, value)
}

func containerCommandLineHasArg(container *corev1.Container, flag, value string) bool {
	if container == nil {
		return false
	}
	commandLine := make([]string, 0, len(container.Command)+len(container.Args))
	commandLine = append(commandLine, container.Command...)
	commandLine = append(commandLine, container.Args...)
	if hasArg(commandLine, flag, value) {
		return true
	}

	expandedCommandLine := []string{}
	for _, arg := range commandLine {
		expandedCommandLine = append(expandedCommandLine, strings.Fields(arg)...)
	}
	return hasArg(expandedCommandLine, flag, value)
}

func hasArg(args []string, flag, value string) bool {
	joined := flag + " " + value
	equals := flag + "=" + value
	for i, arg := range args {
		if strings.Contains(arg, joined) || strings.Contains(arg, equals) {
			return true
		}
		if arg == flag && i+1 < len(args) && args[i+1] == value {
			return true
		}
	}
	return false
}

func injectFlagsIntoContainerCommand(container *corev1.Container, flags string, needsShell bool, framework string) {
	if len(container.Command) > 0 && isPythonCommand(container.Command[0]) {
		// Direct python command case
		if needsShell {
			// Transform to shell wrapper for env var interpretation.
			// Quote each token individually so paths with spaces or special
			// characters survive shell interpretation.
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
			var shellCommand string
			if len(container.Args) > 0 {
				shellCommand = fmt.Sprintf("exec %s %s %s", fullCommand, originalArgs, flags)
			} else {
				shellCommand = fmt.Sprintf("exec %s %s", fullCommand, flags)
			}
			container.Command = []string{"sh", "-c"}
			container.Args = []string{shellCommand}
		} else {
			flagsSlice := strings.Fields(flags)
			container.Args = append(container.Args, flagsSlice...)
		}
	} else {
		// Non-python command case - try injection on each arg individually
		for i, arg := range container.Args {
			modifiedArg := injectFlagsIntoPythonCommand(arg, flags, framework)
			if modifiedArg != arg { // flags were successfully injected
				container.Args[i] = modifiedArg
				break // stop after first successful injection
			}
		}
	}
}

func injectFlagsIntoPythonCommand(arg, flags string, framework string) string {
	// Regex to match python commands that contain sglang
	// Matches: python, python3, python3.11, etc. followed by sglang-related modules
	pattern := fmt.Sprintf(`(python[0-9.]*\s+[^|&;]*%s[^|&;]*?)(\s|$|[|&;])`, framework)

	re := regexp.MustCompile(pattern)

	// Replace with the command + flags + whatever comes after
	result := re.ReplaceAllStringFunc(arg, func(match string) string {
		// Extract the python command part and the delimiter
		submatches := re.FindStringSubmatch(match)
		if len(submatches) >= 3 {
			pythonCmd := submatches[1]
			delimiter := submatches[2]
			return pythonCmd + " " + flags + delimiter
		}
		return match
	})

	return result
}
