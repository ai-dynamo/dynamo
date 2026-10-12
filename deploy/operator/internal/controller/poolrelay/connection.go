/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path"
	"slices"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// relayConnection contains only settings needed to connect to selected workers.
// It never stores resolved Secret payloads; references remain Kubernetes EnvVars.
type relayConnection struct {
	Env     []corev1.EnvVar      `json:"env"`
	Volumes []corev1.Volume      `json:"volumes"`
	Mounts  []corev1.VolumeMount `json:"mounts"`
}

// relayWorkerConnection resolves a supported worker's effective configuration.
// reader, config, dgd and component must be non-nil. The namespace is Kubernetes-local.
func relayWorkerConnection(ctx context.Context, reader client.Reader, config *configv1alpha1.OperatorConfiguration, dgd *v1beta1.DynamoGraphDeployment, component *v1beta1.DynamoComponentDeploymentSharedSpec) (relayConnection, error) {
	// Reject configurations whose sources or environment cannot be derived reliably.
	result := relayConnection{Env: []corev1.EnvVar{}, Volumes: []corev1.Volume{}, Mounts: []corev1.VolumeMount{}}
	if component.GlobalDynamoNamespace {
		return result, fmt.Errorf("globalDynamoNamespace is not supported for DGD selection")
	}
	main := dynamo.GetMainContainer(component)
	if main == nil {
		return result, fmt.Errorf("worker main container is required")
	}
	if err := validateRelayWorkerEnvFrom(ctx, reader, dgd.Namespace, main.EnvFrom); err != nil {
		return result, err
	}
	annotations := dynamo.GetDGDComponentPodAnnotations(dgd, component)
	discovery := dynamo.NewDiscoveryContext(config.Discovery.Backend, annotations)
	backend := string(discovery.Backend)
	if backend == "" {
		backend = "kubernetes"
	}
	mode := string(discovery.Mode)
	if mode == "" {
		mode = "pod"
	}

	// Use the existing operator env merge, preserving graph/component precedence.
	container := corev1.Container{Env: dynamo.MergeEnvs(dgd.Spec.Env, main.Env)}
	dynamo.AddStandardEnvVars(&container, config.Infrastructure)
	dynamo.AddTransportTLSEnvVars(&container, config.Infrastructure)
	defaults := []corev1.EnvVar{
		{Name: "DYN_EVENT_PLANE", Value: "zmq"}, {Name: "DYN_REQUEST_PLANE", Value: "tcp"},
		{Name: "DYN_DISCOVERY_BACKEND", Value: backend}, {Name: "DYN_KUBE_DISCOVERY_MODE", Value: mode},
	}
	effective := dynamo.MergeEnvs(defaults, container.Env)
	resolved := map[string]string{}
	retained := map[string]corev1.EnvVar{}
	for _, entry := range effective {
		if !relayTransportEnv(entry.Name) {
			continue
		}
		value, err := relayEnvValue(ctx, reader, dgd.Namespace, entry)
		if err != nil {
			return result, err
		}
		if strings.Contains(value, "$(") {
			return result, fmt.Errorf("%s uses environment expansion that cannot be inherited independently", entry.Name)
		}
		resolved[entry.Name] = value
		retained[entry.Name] = *entry.DeepCopy()
	}

	// Validate transport choices without including resolved values in diagnostics.
	if resolved["DYN_DISCOVERY_BACKEND"] != "kubernetes" || resolved["DYN_KUBE_DISCOVERY_MODE"] != "pod" {
		return result, fmt.Errorf("Relay requires Kubernetes pod discovery")
	}
	if v := resolved["DYN_EVENT_PLANE"]; v != "zmq" && v != "nats" {
		return result, fmt.Errorf("unsupported DYN_EVENT_PLANE")
	}
	if v := resolved["DYN_REQUEST_PLANE"]; v != "tcp" && v != "nats" {
		return result, fmt.Errorf("unsupported DYN_REQUEST_PLANE")
	}
	if (resolved["NATS_AUTH_USERNAME"] == "") != (resolved["NATS_AUTH_PASSWORD"] == "") {
		return result, fmt.Errorf("NATS_AUTH_USERNAME and NATS_AUTH_PASSWORD must be supplied together")
	}
	if resolved["DYN_EVENT_PLANE"] == "nats" || resolved["DYN_REQUEST_PLANE"] == "nats" {
		if resolved["NATS_SERVER"] == "" {
			return result, fmt.Errorf("NATS transport requires an explicit or operator-provided NATS_SERVER")
		}
	}
	if _, ok := retained["DYN_NAMESPACE"]; ok {
		if resolved["DYN_NAMESPACE"] != dgd.GetDynamoNamespaceForComponent(component) {
			return result, fmt.Errorf("DYN_NAMESPACE override does not match the DGD namespace")
		}
		delete(retained, "DYN_NAMESPACE")
	}

	// Canonical ordering makes compatibility independent of environment declaration order.
	keys := make([]string, 0, len(retained))
	for key := range retained {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		result.Env = append(result.Env, retained[key])
	}
	if err := relayConnectionMounts(ctx, reader, dgd.Namespace, component.PodTemplate.Spec, main.VolumeMounts, resolved, &result); err != nil {
		return result, err
	}
	return result, nil
}

// relayTransportEnv restricts inheritance to transport settings, never backend secrets.
func relayTransportEnv(name string) bool {
	switch name {
	case "DYN_NAMESPACE", "DYN_EVENT_PLANE", "DYN_REQUEST_PLANE", "DYN_DISCOVERY_BACKEND", "DYN_KUBE_DISCOVERY_MODE", "DYN_ZMQ_BROKER_ENABLED", "DYN_ZMQ_BROKER_URL", "NATS_SERVER":
		return true
	}
	return strings.HasPrefix(name, "NATS_AUTH_") || strings.HasPrefix(name, "NATS_TLS_") || strings.HasPrefix(name, "DYN_TCP_TLS_")
}

// relayEnvValue reads only referenced transport keys. Error messages omit values.
// reader must be non-nil. Returned values must not escape into diagnostics or sources files.
func relayEnvValue(ctx context.Context, reader client.Reader, namespace string, entry corev1.EnvVar) (string, error) {
	if entry.ValueFrom == nil {
		return entry.Value, nil
	}
	if ref := entry.ValueFrom.SecretKeyRef; ref != nil {
		var secret corev1.Secret
		if err := reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, &secret); err != nil {
			return "", fmt.Errorf("cannot resolve Secret for %s", entry.Name)
		}
		value, ok := secret.Data[ref.Key]
		if !ok {
			return "", fmt.Errorf("missing Secret key for %s", entry.Name)
		}
		return string(value), nil
	}
	if ref := entry.ValueFrom.ConfigMapKeyRef; ref != nil {
		var configMap corev1.ConfigMap
		if err := reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, &configMap); err != nil {
			return "", fmt.Errorf("cannot resolve ConfigMap for %s", entry.Name)
		}
		value, ok := configMap.Data[ref.Key]
		if !ok {
			return "", fmt.Errorf("missing ConfigMap key for %s", entry.Name)
		}
		return value, nil
	}
	return "", fmt.Errorf("unsupported valueFrom for %s", entry.Name)
}

// revision identifies the declarative connection without exposing Secret contents.
func (c relayConnection) revision() string {
	data, _ := json.Marshal(c)
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

// relayConnectionMounts copies only volumes containing transport files. reader and
// connection must be non-nil. Arbitrary worker filesystem producers are unsupported.
func relayConnectionMounts(ctx context.Context, reader client.Reader, namespace string, pod corev1.PodSpec, mounts []corev1.VolumeMount, values map[string]string, connection *relayConnection) error {
	// Resolve each declared file to its most specific mount; keep mount names canonical.
	selected := map[string]corev1.VolumeMount{}
	selectedItems := map[string]map[string]corev1.KeyToPath{}
	volumes := map[string]corev1.Volume{}
	for _, volume := range pod.Volumes {
		volumes[volume.Name] = volume
	}
	for key, value := range values {
		if value == "" || !(key == "NATS_AUTH_CREDENTIALS_FILE" || strings.HasSuffix(key, "_PATH")) {
			continue
		}
		if !path.IsAbs(value) || path.Clean(value) != value {
			return fmt.Errorf("%s must use an absolute normalized file path", key)
		}
		var chosen *corev1.VolumeMount
		for i := range mounts {
			mount := &mounts[i]
			root := strings.TrimSuffix(mount.MountPath, "/")
			if value != root && !strings.HasPrefix(value, root+"/") {
				continue
			}
			if chosen == nil || len(mount.MountPath) > len(chosen.MountPath) {
				chosen = mount
			}
		}
		if chosen == nil {
			return fmt.Errorf("%s has no corresponding volume mount", key)
		}
		if chosen.Name == relaySourcesVolume {
			return fmt.Errorf("transport volume name is reserved for Relay sources")
		}
		if chosen.SubPathExpr != "" {
			return fmt.Errorf("%s uses unsupported subPathExpr", key)
		}
		volume, ok := volumes[chosen.Name]
		if !ok {
			return fmt.Errorf("%s refers to a missing volume", key)
		}
		if volume.Secret == nil && volume.ConfigMap == nil {
			return fmt.Errorf("%s must be supplied by a Secret or ConfigMap volume", key)
		}

		// Check the requested file exists without retaining its contents.
		relative := strings.TrimPrefix(strings.TrimPrefix(value, strings.TrimSuffix(chosen.MountPath, "/")), "/")
		if chosen.SubPath != "" {
			relative = path.Join(chosen.SubPath, relative)
		}
		var items []corev1.KeyToPath
		available := map[string]bool{}
		if ref := volume.Secret; ref != nil {
			var object corev1.Secret
			if err := reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.SecretName}, &object); err != nil {
				return fmt.Errorf("cannot resolve Secret volume for %s", key)
			}
			items = ref.Items
			for name := range object.Data {
				available[name] = true
			}
		} else {
			ref := volume.ConfigMap
			var object corev1.ConfigMap
			if err := reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, &object); err != nil {
				return fmt.Errorf("cannot resolve ConfigMap volume for %s", key)
			}
			items = ref.Items
			for name := range object.Data {
				available[name] = true
			}
			for name := range object.BinaryData {
				available[name] = true
			}
		}
		fileItem := corev1.KeyToPath{Key: relative, Path: relative}
		fileKey := relative
		if len(items) > 0 {
			fileKey = ""
			for _, item := range items {
				if item.Path == relative {
					fileKey = item.Key
					fileItem = *item.DeepCopy()
					break
				}
			}
		}
		if !available[fileKey] {
			return fmt.Errorf("transport file for %s is missing from its volume", key)
		}
		if previous, ok := selected[chosen.Name]; ok && previous.MountPath != chosen.MountPath {
			return fmt.Errorf("transport volume uses multiple mount paths")
		}
		if selectedItems[chosen.Name] == nil {
			selectedItems[chosen.Name] = map[string]corev1.KeyToPath{}
		}
		selectedItems[chosen.Name][relative] = fileItem
		selected[chosen.Name] = *chosen.DeepCopy()
	}

	// Preserve only required sources and use read-only mounts in the Relay workload.
	names := make([]string, 0, len(selected))
	for name := range selected {
		names = append(names, name)
	}
	slices.Sort(names)
	for index, name := range names {
		mount := selected[name]
		mount.Name = fmt.Sprintf("%s%d", relayConnectionVolumePrefix, index)
		mount.ReadOnly = true
		connection.Mounts = append(connection.Mounts, mount)
		sourceVolume := volumes[name]
		volume := *sourceVolume.DeepCopy()
		volume.Name = mount.Name
		if volume.Secret != nil && volume.Secret.DefaultMode == nil {
			volume.Secret.DefaultMode = ptr.To(int32(420))
		}
		if volume.ConfigMap != nil && volume.ConfigMap.DefaultMode == nil {
			volume.ConfigMap.DefaultMode = ptr.To(int32(420))
		}
		items := make([]corev1.KeyToPath, 0, len(selectedItems[name]))
		for _, item := range selectedItems[name] {
			items = append(items, item)
		}
		slices.SortFunc(items, func(a, b corev1.KeyToPath) int { return strings.Compare(a.Path, b.Path) })
		if volume.Secret != nil {
			volume.Secret.Items = items
			volume.Secret.Optional = ptr.To(false)
		}
		if volume.ConfigMap != nil {
			volume.ConfigMap.Items = items
			volume.ConfigMap.Optional = ptr.To(false)
		}
		connection.Volumes = append(connection.Volumes, *volume.DeepCopy())
	}
	return nil
}

// validateRelayWorkerEnvFrom permits backend-only sources without inheriting their
// contents. Transport settings remain explicit env entries. reader must be non-nil.
func validateRelayWorkerEnvFrom(ctx context.Context, reader client.Reader, namespace string, sources []corev1.EnvFromSource) error {
	for _, source := range sources {
		var keys []string
		var err error
		var optional bool
		switch {
		case source.SecretRef != nil && source.ConfigMapRef == nil:
			ref := source.SecretRef
			optional = ref.Optional != nil && *ref.Optional
			var secret corev1.Secret
			err = reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, &secret)
			for key := range secret.Data {
				keys = append(keys, key)
			}
		case source.ConfigMapRef != nil && source.SecretRef == nil:
			ref := source.ConfigMapRef
			optional = ref.Optional != nil && *ref.Optional
			var configMap corev1.ConfigMap
			err = reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, &configMap)
			for key := range configMap.Data {
				keys = append(keys, key)
			}
		default:
			return fmt.Errorf("worker envFrom must reference exactly one Secret or ConfigMap")
		}

		// Never suppress authorization failures or put source contents in diagnostics.
		if err != nil {
			if optional && apierrors.IsNotFound(err) {
				continue
			}
			return fmt.Errorf("worker envFrom source cannot be read; verify the reference and operator permissions")
		}
		for _, key := range keys {
			if relayTransportEnv(source.Prefix + key) {
				return fmt.Errorf("worker envFrom contains transport settings; declare transport env explicitly")
			}
		}
	}
	return nil
}
