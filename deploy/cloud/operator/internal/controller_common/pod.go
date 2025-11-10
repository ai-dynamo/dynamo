package controller_common

import (
	"sort"

	corev1 "k8s.io/api/core/v1"
)

// CanonicalizePodSpec sorts the pod spec in a way that is deterministic and easy to reason about.
//
//nolint:gocyclo
func CanonicalizePodSpec(podSpec *corev1.PodSpec) *corev1.PodSpec {
	// Helper function to get EnvFromSource sort key
	envFromKey := func(e corev1.EnvFromSource) string {
		if e.ConfigMapRef != nil {
			return "cm:" + e.ConfigMapRef.Name + ":" + e.Prefix
		}
		if e.SecretRef != nil {
			return "sec:" + e.SecretRef.Name + ":" + e.Prefix
		}
		return "other:" + e.Prefix
	}

	// Helper function to sort container-like fields (works for both Container and EphemeralContainer)
	sortContainerFields := func(env []corev1.EnvVar, envFrom []corev1.EnvFromSource, ports []corev1.ContainerPort, volumeMounts []corev1.VolumeMount, securityContext *corev1.SecurityContext) {
		// Sort env vars by name
		if len(env) > 1 {
			sort.Slice(env, func(i, j int) bool { return env[i].Name < env[j].Name })
		}

		// Sort envFrom by referenced source and prefix
		if len(envFrom) > 1 {
			sort.Slice(envFrom, func(i, j int) bool {
				return envFromKey(envFrom[i]) < envFromKey(envFrom[j])
			})
		}

		// Sort ports by name then port number
		if len(ports) > 1 {
			sort.Slice(ports, func(i, j int) bool {
				if ports[i].Name == ports[j].Name {
					return ports[i].ContainerPort < ports[j].ContainerPort
				}
				return ports[i].Name < ports[j].Name
			})
		}

		// Sort volume mounts by name then mount path
		if len(volumeMounts) > 1 {
			sort.Slice(volumeMounts, func(i, j int) bool {
				if volumeMounts[i].Name == volumeMounts[j].Name {
					return volumeMounts[i].MountPath < volumeMounts[j].MountPath
				}
				return volumeMounts[i].Name < volumeMounts[j].Name
			})
		}

		// Sort security context capability lists
		if securityContext != nil && securityContext.Capabilities != nil {
			if caps := securityContext.Capabilities.Add; len(caps) > 1 {
				sort.Slice(caps, func(i, j int) bool { return string(caps[i]) < string(caps[j]) })
			}
			if caps := securityContext.Capabilities.Drop; len(caps) > 1 {
				sort.Slice(caps, func(i, j int) bool { return string(caps[i]) < string(caps[j]) })
			}
		}
	}

	// Sort regular containers
	for i := range podSpec.Containers {
		c := &podSpec.Containers[i]
		sortContainerFields(c.Env, c.EnvFrom, c.Ports, c.VolumeMounts, c.SecurityContext)
	}
	if len(podSpec.Containers) > 1 {
		sort.Slice(podSpec.Containers, func(i, j int) bool {
			return podSpec.Containers[i].Name < podSpec.Containers[j].Name
		})
	}

	// Sort init containers
	for i := range podSpec.InitContainers {
		c := &podSpec.InitContainers[i]
		sortContainerFields(c.Env, c.EnvFrom, c.Ports, c.VolumeMounts, c.SecurityContext)
	}
	if len(podSpec.InitContainers) > 1 {
		sort.Slice(podSpec.InitContainers, func(i, j int) bool {
			return podSpec.InitContainers[i].Name < podSpec.InitContainers[j].Name
		})
	}

	// Sort ephemeral containers
	for i := range podSpec.EphemeralContainers {
		ec := &podSpec.EphemeralContainers[i]
		sortContainerFields(ec.Env, ec.EnvFrom, ec.Ports, ec.VolumeMounts, ec.SecurityContext)
	}
	if len(podSpec.EphemeralContainers) > 1 {
		sort.Slice(podSpec.EphemeralContainers, func(i, j int) bool {
			return podSpec.EphemeralContainers[i].Name < podSpec.EphemeralContainers[j].Name
		})
	}

	// Sort image pull secrets
	if len(podSpec.ImagePullSecrets) > 1 {
		uniqueSecrets := ensureUniqueImagePullSecrets(podSpec.ImagePullSecrets)
		sort.Slice(uniqueSecrets, func(i, j int) bool {
			return uniqueSecrets[i].Name < uniqueSecrets[j].Name
		})
		podSpec.ImagePullSecrets = uniqueSecrets
	}

	// Sort volumes and their nested items
	sortKeyToPathItems := func(items []corev1.KeyToPath) {
		if len(items) > 1 {
			sort.Slice(items, func(i, j int) bool {
				if items[i].Key == items[j].Key {
					return items[i].Path < items[j].Path
				}
				return items[i].Key < items[j].Key
			})
		}
	}

	for i := range podSpec.Volumes {
		v := &podSpec.Volumes[i]

		// ConfigMap items
		if v.ConfigMap != nil {
			sortKeyToPathItems(v.ConfigMap.Items)
		}

		// Secret items
		if v.Secret != nil {
			sortKeyToPathItems(v.Secret.Items)
		}

		// DownwardAPI items
		if v.DownwardAPI != nil && len(v.DownwardAPI.Items) > 1 {
			sort.Slice(v.DownwardAPI.Items, func(i, j int) bool {
				return v.DownwardAPI.Items[i].Path < v.DownwardAPI.Items[j].Path
			})
		}

		// Projected sources
		if v.Projected != nil {
			// Sort projected sources
			if len(v.Projected.Sources) > 1 {
				sort.Slice(v.Projected.Sources, func(i, j int) bool {
					getProjectionKey := func(p corev1.VolumeProjection) string {
						if p.ConfigMap != nil {
							return "cm:" + p.ConfigMap.Name
						}
						if p.Secret != nil {
							return "sec:" + p.Secret.Name
						}
						if p.DownwardAPI != nil {
							return "downward:"
						}
						if p.ServiceAccountToken != nil {
							return "sat:" + p.ServiceAccountToken.Audience
						}
						return "z:other"
					}
					return getProjectionKey(v.Projected.Sources[i]) < getProjectionKey(v.Projected.Sources[j])
				})
			}

			// Sort nested items for each projection
			for j := range v.Projected.Sources {
				p := &v.Projected.Sources[j]
				if p.ConfigMap != nil {
					sortKeyToPathItems(p.ConfigMap.Items)
				}
				if p.Secret != nil {
					sortKeyToPathItems(p.Secret.Items)
				}
				if p.DownwardAPI != nil && len(p.DownwardAPI.Items) > 1 {
					sort.Slice(p.DownwardAPI.Items, func(i, j int) bool {
						return p.DownwardAPI.Items[i].Path < p.DownwardAPI.Items[j].Path
					})
				}
			}
		}
	}

	// Sort volumes by name
	if len(podSpec.Volumes) > 1 {
		sort.Slice(podSpec.Volumes, func(i, j int) bool {
			return podSpec.Volumes[i].Name < podSpec.Volumes[j].Name
		})
	}

	// Sort tolerations
	if len(podSpec.Tolerations) > 1 {
		sort.Slice(podSpec.Tolerations, func(i, j int) bool {
			a, b := podSpec.Tolerations[i], podSpec.Tolerations[j]

			if a.Key != b.Key {
				return a.Key < b.Key
			}
			if string(a.Operator) != string(b.Operator) {
				return string(a.Operator) < string(b.Operator)
			}
			if a.Value != b.Value {
				return a.Value < b.Value
			}
			if string(a.Effect) != string(b.Effect) {
				return string(a.Effect) < string(b.Effect)
			}

			// Handle TolerationSeconds (could be nil)
			aSec, bSec := int64(0), int64(0)
			if a.TolerationSeconds != nil {
				aSec = *a.TolerationSeconds
			}
			if b.TolerationSeconds != nil {
				bSec = *b.TolerationSeconds
			}
			return aSec < bSec
		})
	}

	// Sort topology spread constraints
	if len(podSpec.TopologySpreadConstraints) > 1 {
		sort.Slice(podSpec.TopologySpreadConstraints, func(i, j int) bool {
			a, b := podSpec.TopologySpreadConstraints[i], podSpec.TopologySpreadConstraints[j]
			if a.TopologyKey != b.TopologyKey {
				return a.TopologyKey < b.TopologyKey
			}
			if string(a.WhenUnsatisfiable) != string(b.WhenUnsatisfiable) {
				return string(a.WhenUnsatisfiable) < string(b.WhenUnsatisfiable)
			}
			return a.MaxSkew < b.MaxSkew
		})
	}

	// Sort host aliases
	if len(podSpec.HostAliases) > 1 {
		// First sort hostnames within each alias
		for i := range podSpec.HostAliases {
			if len(podSpec.HostAliases[i].Hostnames) > 1 {
				sort.Strings(podSpec.HostAliases[i].Hostnames)
			}
		}
		// Then sort aliases by IP
		sort.Slice(podSpec.HostAliases, func(i, j int) bool {
			return podSpec.HostAliases[i].IP < podSpec.HostAliases[j].IP
		})
	}

	// Sort DNS config
	if podSpec.DNSConfig != nil {
		// Sort DNS options
		if len(podSpec.DNSConfig.Options) > 1 {
			sort.Slice(podSpec.DNSConfig.Options, func(i, j int) bool {
				if podSpec.DNSConfig.Options[i].Name == podSpec.DNSConfig.Options[j].Name {
					vi, vj := "", ""
					if podSpec.DNSConfig.Options[i].Value != nil {
						vi = *podSpec.DNSConfig.Options[i].Value
					}
					if podSpec.DNSConfig.Options[j].Value != nil {
						vj = *podSpec.DNSConfig.Options[j].Value
					}
					return vi < vj
				}
				return podSpec.DNSConfig.Options[i].Name < podSpec.DNSConfig.Options[j].Name
			})
		}

		// Sort nameservers and search domains
		if len(podSpec.DNSConfig.Nameservers) > 1 {
			sort.Strings(podSpec.DNSConfig.Nameservers)
		}
		if len(podSpec.DNSConfig.Searches) > 1 {
			sort.Strings(podSpec.DNSConfig.Searches)
		}
	}

	return podSpec
}

func ensureUniqueImagePullSecrets(secrets []corev1.LocalObjectReference) []corev1.LocalObjectReference {
	if len(secrets) == 0 {
		return nil
	}
	uniqueSecrets := make(map[string]corev1.LocalObjectReference)
	for _, secret := range secrets {
		uniqueSecrets[secret.Name] = secret
	}
	uniqueSecretsList := make([]corev1.LocalObjectReference, 0, len(uniqueSecrets))
	for secretName := range uniqueSecrets {
		uniqueSecretsList = append(uniqueSecretsList, corev1.LocalObjectReference{Name: secretName})
	}
	return uniqueSecretsList
}
