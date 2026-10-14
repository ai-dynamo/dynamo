/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"fmt"
	"maps"
	"net"
	"path"
	"slices"
	"strings"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/imdario/mergo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
)

const relayConnectionVolumePrefix = "relay-connection-"

// relayMainContainer validates primary identity independently of list order.
// relay must be non-nil. The returned container is owned by relay and is read-only.
func relayMainContainer(relay *v1alpha1.DynamoPoolRelay) (*corev1.Container, error) {
	var main *corev1.Container
	seen := map[string]bool{}
	for i := range relay.Spec.PodTemplate.Spec.Containers {
		container := &relay.Spec.PodTemplate.Spec.Containers[i]
		if seen[container.Name] || len(validation.IsDNS1123Label(container.Name)) != 0 {
			return nil, fmt.Errorf("podTemplate requires unique valid container names")
		}
		seen[container.Name] = true
		if strings.TrimSpace(container.Image) == "" {
			return nil, fmt.Errorf("container %s requires an image", container.Name)
		}
		if container.Name == "main" {
			main = container
		}
	}
	for _, container := range relay.Spec.PodTemplate.Spec.InitContainers {
		if seen[container.Name] || container.Name == "main" || len(validation.IsDNS1123Label(container.Name)) != 0 || strings.TrimSpace(container.Image) == "" {
			return nil, fmt.Errorf("init containers require unique names and images and cannot be named main")
		}
		seen[container.Name] = true
	}
	if main == nil {
		return nil, fmt.Errorf("podTemplate requires a main container with an image")
	}
	return main, nil
}

// mergeRelayPodTemplate follows DGD main/sidecar composition while preserving
// generated transport resources. base and relay must be non-nil; relay is not mutated.
func mergeRelayPodTemplate(base *corev1.PodTemplateSpec, relay *v1alpha1.DynamoPoolRelay) error {
	main, err := relayMainContainer(relay)
	if err != nil {
		return err
	}
	main = main.DeepCopy()
	user := relay.Spec.PodTemplate.DeepCopy()
	if !equality.Semantic.DeepEqual(user.ObjectMeta, metav1.ObjectMeta{Labels: user.Labels, Annotations: user.Annotations}) {
		return fmt.Errorf("podTemplate metadata supports only labels and annotations")
	}
	for key, value := range base.Labels {
		if supplied, exists := user.Labels[key]; exists && supplied != value {
			return fmt.Errorf("podTemplate overrides reserved label %s", key)
		}
	}
	if user.Spec.ServiceAccountName != "" || user.Spec.DeprecatedServiceAccount != "" || (user.Spec.RestartPolicy != "" && user.Spec.RestartPolicy != corev1.RestartPolicyAlways) || user.Spec.HostNetwork || len(user.Spec.EphemeralContainers) > 0 {
		return fmt.Errorf("podTemplate cannot override service account, restart policy, host networking or ephemeral containers")
	}
	if len(main.Command) > 0 || len(main.EnvFrom) > 0 {
		return fmt.Errorf("main command and envFrom are operator-managed")
	}

	// Keep generated settings authoritative, including optional transport variables.
	generated := base.Spec.Containers[0]
	reservedEnv := map[string]bool{"DYN_RELAY_NAMESPACES": true, "DYN_RELAY_WATCH_ALL": true, "DYN_RELAY_ENDPOINT_PREFIXES": true, "DYN_CLUSTER_ID": true, "DYN_DC_ID": true, "DYN_RELAY_BIND": true}
	for _, env := range generated.Env {
		reservedEnv[env.Name] = true
	}
	seenEnv := map[string]bool{}
	for _, env := range main.Env {
		if reservedEnv[env.Name] || relayTransportEnv(env.Name) || strings.HasPrefix(env.Name, "DYN_RELAY_CONNECTION_") || strings.HasPrefix(env.Name, "DYN_TCP_RESPONSE_") {
			return fmt.Errorf("main cannot override operator-managed env %s", env.Name)
		}
		if seenEnv[env.Name] {
			return fmt.Errorf("duplicate main env %s", env.Name)
		}
		seenEnv[env.Name] = true
	}
	args, loopback, err := relayMainArgs(main.Args)
	if err != nil {
		return err
	}
	if loopback {
		if main.StartupProbe == nil || main.ReadinessProbe == nil || main.StartupProbe.Exec == nil || main.ReadinessProbe.Exec == nil {
			return fmt.Errorf("loopback bind requires explicit exec startup and readiness probes")
		}
	}
	for _, probe := range []*corev1.Probe{main.StartupProbe, main.ReadinessProbe, main.LivenessProbe} {
		if probe == nil {
			continue
		}
		handlers := 0
		if probe.Exec != nil {
			handlers++
		}
		if probe.HTTPGet != nil {
			handlers++
		}
		if probe.TCPSocket != nil {
			handlers++
		}
		if probe.GRPC != nil {
			handlers++
		}
		if loopback && probe.Exec == nil {
			return fmt.Errorf("loopback main probes must use exec handlers")
		}
		if handlers != 1 {
			return fmt.Errorf("main probes require exactly one handler")
		}
	}

	// User volumes cannot replace generated sources or connection volumes.
	seenVolumes := map[string]bool{}
	for _, volume := range base.Spec.Volumes {
		seenVolumes[volume.Name] = true
	}
	for _, volume := range user.Spec.Volumes {
		if seenVolumes[volume.Name] || strings.HasPrefix(volume.Name, relayConnectionVolumePrefix) || len(validation.IsDNS1123Label(volume.Name)) != 0 {
			return fmt.Errorf("podTemplate volume name conflicts with another or reserved volume")
		}
		seenVolumes[volume.Name] = true
	}
	for _, container := range append(slices.Clone(user.Spec.Containers), user.Spec.InitContainers...) {
		seenPaths := map[string]bool{}
		for _, mount := range container.VolumeMounts {
			if !seenVolumes[mount.Name] || seenPaths[path.Clean(mount.MountPath)] {
				return fmt.Errorf("container %s has an unknown volume or duplicate mount path", container.Name)
			}
			seenPaths[path.Clean(mount.MountPath)] = true
			if mount.Name == relaySourcesVolume || strings.HasPrefix(mount.Name, relayConnectionVolumePrefix) {
				return fmt.Errorf("user containers cannot mount reserved Relay volumes")
			}
			if container.Name != "main" {
				continue
			}
			for _, reserved := range generated.VolumeMounts {
				a, b := path.Clean(mount.MountPath), path.Clean(reserved.MountPath)
				if a == b || strings.HasPrefix(b, a+"/") || strings.HasPrefix(a, b+"/") || a == "/" {
					return fmt.Errorf("main mount shadows a generated sources or connection path")
				}
			}
		}
	}

	// DGD-style container overrides keep probes whole rather than mixing handlers.
	if err := mergo.Merge(&generated, *main.DeepCopy(), mergo.WithOverride); err != nil {
		return err
	}
	generated.Name = "main"
	generated.Env = dynamo.MergeEnvs(base.Spec.Containers[0].Env, main.Env)
	generated.VolumeMounts = append(slices.Clone(base.Spec.Containers[0].VolumeMounts), main.VolumeMounts...)
	generated.Ports = append(slices.Clone(base.Spec.Containers[0].Ports), main.Ports...)
	generated.Args = append(slices.Clone(base.Spec.Containers[0].Args[:4]), args...)
	if main.StartupProbe != nil {
		generated.StartupProbe = main.StartupProbe.DeepCopy()
	}
	if main.ReadinessProbe != nil {
		generated.ReadinessProbe = main.ReadinessProbe.DeepCopy()
	}
	if main.LivenessProbe != nil {
		generated.LivenessProbe = main.LivenessProbe.DeepCopy()
	}

	// Merge Pod settings separately so sidecars never receive main defaults.
	volumes := append(slices.Clone(base.Spec.Volumes), user.Spec.Volumes...)
	sidecars := []corev1.Container{}
	for _, container := range user.Spec.Containers {
		if container.Name != "main" {
			sidecars = append(sidecars, container)
		}
	}
	user.Spec.Containers, user.Spec.Volumes = nil, nil
	if err := mergo.Merge(&base.Spec, user.Spec, mergo.WithOverride); err != nil {
		return err
	}
	base.Spec.Volumes = volumes
	base.Spec.Containers = append([]corev1.Container{generated}, sidecars...)
	base.Annotations = maps.Clone(user.Annotations)
	maps.Copy(base.Labels, user.Labels)
	return nil
}

// relayMainArgs permits only documented non-source CLI overrides. Managed identity
// and sources flags are injected separately, never accepted through user arguments.
func relayMainArgs(input []string) ([]string, bool, error) {
	if len(input) == 0 {
		return []string{"--bind", "0.0.0.0:5561"}, false, nil
	}
	seen := map[string]bool{}
	loopback := false
	for i := 0; i < len(input); i++ {
		flag, value, inline := strings.Cut(input[i], "=")
		if flag != "--bind" && flag != "--expected-unique-blocks" {
			return nil, false, fmt.Errorf("main args allow only --bind and --expected-unique-blocks")
		}
		if seen[flag] {
			return nil, false, fmt.Errorf("duplicate main argument %s", flag)
		}
		seen[flag] = true
		if !inline {
			i++
			if i >= len(input) {
				return nil, false, fmt.Errorf("main argument %s requires a value", flag)
			}
			value = input[i]
		}
		if flag == "--bind" {
			host, port, err := net.SplitHostPort(value)
			ip := net.ParseIP(host)
			if err != nil || port != "5561" || ip == nil || !(ip.IsLoopback() || ip.IsUnspecified()) {
				return nil, false, fmt.Errorf("main --bind must use a loopback or wildcard IP on port 5561")
			}
			loopback = ip.IsLoopback()
		} else {
			for _, c := range value {
				if c < '0' || c > '9' {
					return nil, false, fmt.Errorf("--expected-unique-blocks requires a positive integer")
				}
			}
			if strings.Trim(value, "0") == "" {
				return nil, false, fmt.Errorf("--expected-unique-blocks requires a positive integer")
			}
		}
	}
	result := slices.Clone(input)
	if !seen["--bind"] {
		result = append(result, "--bind", "0.0.0.0:5561")
	}
	return result, loopback, nil
}

// relayServicePorts validates publication targets against the final Pod.
func relayServicePorts(relay *v1alpha1.DynamoPoolRelay, pod *corev1.PodTemplateSpec) ([]corev1.ServicePort, error) {
	ports := []corev1.ServicePort{{Name: "grpc", Port: 5561, TargetPort: intstr.FromString("grpc"), Protocol: corev1.ProtocolTCP}}
	if relay.Spec.Service != nil {
		ports = relay.Spec.Service.DeepCopy().Ports
	}
	named := map[string]corev1.ContainerPort{}
	for _, container := range append(slices.Clone(pod.Spec.Containers), pod.Spec.InitContainers...) {
		for _, port := range container.Ports {
			if _, exists := named[port.Name]; port.Name != "" && exists {
				return nil, fmt.Errorf("Pod container port names must be unique")
			}
			if container.Name != "main" && (port.ContainerPort == 5561 || port.ContainerPort == 5562 || port.ContainerPort == 8081) {
				return nil, fmt.Errorf("sidecar ports conflict with reserved Relay ports")
			}
			if port.Name != "" {
				named[port.Name] = port
			}
		}
	}
	seen := map[string]bool{}
	for i := range ports {
		port := &ports[i]
		if len(validation.IsValidPortName(port.Name)) != 0 || seen[port.Name] || port.Port < 1 || port.Port > 65535 || port.NodePort != 0 {
			return nil, fmt.Errorf("Service ports require unique names, valid port numbers and no nodePort")
		}
		seen[port.Name] = true
		if port.Protocol == "" {
			port.Protocol = corev1.ProtocolTCP
		}
		target := port.TargetPort.IntVal
		if port.TargetPort.Type == intstr.String {
			resolved, ok := named[port.TargetPort.StrVal]
			if !ok {
				return nil, fmt.Errorf("Service target port %s is not declared in the Pod", port.TargetPort.StrVal)
			}
			protocol := resolved.Protocol
			if protocol == "" {
				protocol = corev1.ProtocolTCP
			}
			if protocol != port.Protocol {
				return nil, fmt.Errorf("Service target protocol does not match container port")
			}
			target = resolved.ContainerPort
		} else if target == 0 {
			target = port.Port
			port.TargetPort = intstr.FromInt32(target)
		}
		if target < 1 || target > 65535 || target == 8081 {
			return nil, fmt.Errorf("invalid or reserved sources acknowledgement Service target")
		}
		if port.Name == "grpc" && port.Protocol != corev1.ProtocolTCP {
			return nil, fmt.Errorf("grpc publication requires TCP")
		}
	}
	if !seen["grpc"] {
		return nil, fmt.Errorf("Service requires a publication port named grpc")
	}
	return ports, nil
}
