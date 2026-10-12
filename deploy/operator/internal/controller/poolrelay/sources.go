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
	"slices"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const relayMaxSources = 4096
const relayMaxSourcesBytes = 1024 * 1024

const relaySourcesVolume = "relay-sources"
const relaySourcesPath = "/var/run/dynamo-relay/sources/sources.json"

// relaySource binds a physical discovery namespace to a stable exported identity.
// An omitted LogicalNamespace means the physical and logical identities are equal.
type relaySource struct {
	Namespace        string `json:"namespace"`
	LogicalNamespace string `json:"logicalNamespace,omitempty"`
}

// relaySources is the standalone file format. Revision is computed, never written.
type relaySources struct {
	Version            int           `json:"version"`
	ConnectionRevision string        `json:"connectionRevision,omitempty"`
	Sources            []relaySource `json:"sources"`
	Revision           string        `json:"-"`
}

// newRelaySources canonicalizes a source set without mutating the caller's slice.
func newRelaySources(connectionRevision string, entries []relaySource) relaySources {
	// The canonical byte representation matches the Rust reader.
	ordered := append([]relaySource{}, entries...)
	slices.SortFunc(ordered, func(a, b relaySource) int {
		if order := strings.Compare(a.Namespace, b.Namespace); order != 0 {
			return order
		}
		return strings.Compare(a.LogicalNamespace, b.LogicalNamespace)
	})
	document := relaySources{Version: 1, ConnectionRevision: connectionRevision, Sources: ordered}
	data, _ := json.Marshal(document)
	sum := sha256.Sum256(data)
	document.Revision = hex.EncodeToString(sum[:])
	return document
}

// relayDGDSources binds active worker namespaces to the stable DGD identity.
// dgd must be non-nil. Missing status uses the legacy base namespace; published
// status, rather than a desired worker hash, owns the rollout cutover decision.
func relayDGDSources(dgd *v1beta1.DynamoGraphDeployment) []relaySource {
	// Different worker roles may share a namespace or complete cutover separately.
	base := v1beta1.ComputeDynamoNamespace(false, dgd.Namespace, dgd.Name)
	namespaces := make(map[string]struct{})
	for _, component := range dgd.Spec.Components {
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		namespace := dgd.Status.Components[component.ComponentName].RuntimeNamespace
		if namespace == "" {
			namespace = base
		}
		namespaces[namespace] = struct{}{}
	}

	// Preserve the existing source validation for DGDs without supported workers.
	if len(namespaces) == 0 {
		namespaces[base] = struct{}{}
	}
	result := make([]relaySource, 0, len(namespaces))
	for namespace := range namespaces {
		source := relaySource{Namespace: namespace}
		if namespace != base {
			source.LogicalNamespace = base
		}
		result = append(result, source)
	}
	slices.SortFunc(result, func(a, b relaySource) int { return strings.Compare(a.Namespace, b.Namespace) })
	return result
}

// resolveRelayConnections requires one compatible profile across all selected
// worker components. reader and config must be non-nil; dgds are not mutated.
func resolveRelayConnections(ctx context.Context, reader client.Reader, config *configv1alpha1.OperatorConfiguration, dgds []v1beta1.DynamoGraphDeployment) (relayConnection, []relaySource, error) {
	// An empty selection still has a valid empty sources, never an unbounded watch.
	connection := relayConnection{Env: []corev1.EnvVar{
		{Name: "DYN_EVENT_PLANE", Value: "zmq"}, {Name: "DYN_REQUEST_PLANE", Value: "tcp"},
		{Name: "DYN_DISCOVERY_BACKEND", Value: "kubernetes"}, {Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"},
	}, Volumes: []corev1.Volume{}, Mounts: []corev1.VolumeMount{}}
	slices.SortFunc(connection.Env, func(a, b corev1.EnvVar) int { return strings.Compare(a.Name, b.Name) })
	sources := make([]relaySource, 0, len(dgds))
	// Reject ambiguous identities and oversized files before deriving any connection.
	if len(dgds) > relayMaxSources {
		return connection, sources, fmt.Errorf("sources source limit exceeded: maximum %d DGDs", relayMaxSources)
	}
	namespaceOwners := make(map[string]int, len(dgds))
	for i := range dgds {
		dgd := &dgds[i]
		for _, source := range relayDGDSources(dgd) {
			// A logical identity may span worker roles, but never different DGDs.
			for _, namespace := range []string{source.Namespace, source.LogicalNamespace} {
				if namespace == "" {
					continue
				}
				if owner, exists := namespaceOwners[namespace]; exists && owner != i {
					return connection, sources, fmt.Errorf("DGDs %s and %s resolve to the same Dynamo namespace", dgds[owner].Name, dgd.Name)
				}
				namespaceOwners[namespace] = i
			}
			sources = append(sources, source)
			if len(sources) > relayMaxSources {
				return connection, sources, fmt.Errorf("sources source limit exceeded: maximum %d namespaces", relayMaxSources)
			}
		}
	}
	encoded, _ := json.Marshal(newRelaySources(strings.Repeat("0", 64), sources))
	if len(encoded) > relayMaxSourcesBytes {
		return connection, sources, fmt.Errorf("sources file size limit exceeded: maximum %d bytes", relayMaxSourcesBytes)
	}

	// All selected workers must share one connection; never omit an incompatible DGD.
	first := ""
	for i := range dgds {
		dgd := &dgds[i]
		workers := 0
		for j := range dgd.Spec.Components {
			component := &dgd.Spec.Components[j]
			if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
				continue
			}
			workers++

			// Resolve every authored role through the existing component-template contract.
			roles := []dynamo.Role{dynamo.RoleMain}
			if dynamo.HasRolePodTemplates(component) {
				roles = make([]dynamo.Role, 0, len(component.Roles))
				for _, role := range component.Roles {
					roles = append(roles, dynamo.Role(role.Name))
				}
			}
			for _, role := range roles {
				effective, err := dynamo.EffectiveComponentForRole(component, role)
				if err != nil {
					return connection, sources, fmt.Errorf("DGD %s component %s role %s: %w", dgd.Name, component.ComponentName, role, err)
				}
				current, err := relayWorkerConnection(ctx, reader, config, dgd, effective)
				if err != nil {
					return connection, sources, fmt.Errorf("DGD %s component %s role %s: %w", dgd.Name, component.ComponentName, role, err)
				}
				identity := dgd.Name + "/" + component.ComponentName + "/" + string(role)
				if first == "" {
					connection = current
					first = identity
				} else if connection.revision() != current.revision() {
					return connection, sources, fmt.Errorf("incompatible transport configuration between %s and %s", first, identity)
				}
			}
		}
		if workers == 0 {
			return connection, sources, fmt.Errorf("DGD %s has no supported worker components", dgd.Name)
		}
	}
	return connection, sources, nil
}

// retainRelayConnection keeps the running process configuration when selection
// becomes empty. Source withdrawal must not restart a healthy connection.
func retainRelayConnection(ctx context.Context, reader client.Reader, relay *v1alpha1.DynamoPoolRelay, fallback relayConnection) (relayConnection, string, error) {
	var deployment appsv1.Deployment
	if err := reader.Get(ctx, client.ObjectKey{Namespace: relay.Namespace, Name: relayResourceName(relay.Name)}, &deployment); err != nil {
		return fallback, "", client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(&deployment, relay) {
		return fallback, "", fmt.Errorf("existing Deployment is not owned by this DynamoPoolRelay")
	}
	retained := relayConnection{Env: []corev1.EnvVar{}, Volumes: []corev1.Volume{}, Mounts: []corev1.VolumeMount{}}
	revision := ""
	found := false
	for _, container := range deployment.Spec.Template.Spec.Containers {
		if container.Name != "main" {
			continue
		}
		found = true
		for _, entry := range container.Env {
			if entry.Name == "DYN_RELAY_CONNECTION_REVISION" {
				revision = entry.Value
			}
			if entry.Name != "DYN_NAMESPACE" && relayTransportEnv(entry.Name) {
				retained.Env = append(retained.Env, *entry.DeepCopy())
			}
		}
		for _, mount := range container.VolumeMounts {
			if strings.HasPrefix(mount.Name, relayConnectionVolumePrefix) {
				retained.Mounts = append(retained.Mounts, *mount.DeepCopy())
			}
		}
	}
	if !found {
		return fallback, "", fmt.Errorf("existing Deployment has no main container")
	}
	for _, volume := range deployment.Spec.Template.Spec.Volumes {
		if strings.HasPrefix(volume.Name, relayConnectionVolumePrefix) {
			retained.Volumes = append(retained.Volumes, *volume.DeepCopy())
		}
	}
	slices.SortFunc(retained.Env, func(a, b corev1.EnvVar) int { return strings.Compare(a.Name, b.Name) })
	return retained, revision, nil
}
