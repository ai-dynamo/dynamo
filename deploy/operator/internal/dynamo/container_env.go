/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

// ContainerEnvFromOverlay records temporary env entries used while rendering so the output can retain the original envFrom contract.
type ContainerEnvFromOverlay struct {
	originalEnvFrom []corev1.EnvFromSource
	materialized    map[string]struct{}
}

// MaterializeContainerEnvFrom returns a container copy where selected envFrom values are explicit for render-time decisions.
// The input container and referenced Kubernetes objects are not modified.
func MaterializeContainerEnvFrom(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	container *corev1.Container,
	names ...string,
) (*corev1.Container, *ContainerEnvFromOverlay, error) {
	resolved := container.DeepCopy()
	overlay := &ContainerEnvFromOverlay{
		originalEnvFrom: append([]corev1.EnvFromSource(nil), container.EnvFrom...),
		materialized:    make(map[string]struct{}),
	}

	// Explicit env entries already have Kubernetes precedence and need no temporary value.
	wanted := make(map[string]struct{}, len(names))
	for _, name := range names {
		if findEnvVar(container.Env, name) == nil {
			wanted[name] = struct{}{}
		}
	}

	// Resolve sources in declaration order so later envFrom entries retain Kubernetes precedence.
	values := make(map[string]string, len(wanted))
	for _, source := range container.EnvFrom {
		switch {
		case source.ConfigMapRef != nil:
			configMap := &corev1.ConfigMap{}
			err := reader.Get(ctx, types.NamespacedName{Namespace: namespace, Name: source.ConfigMapRef.Name}, configMap)
			if err != nil {
				if apierrors.IsNotFound(err) && source.ConfigMapRef.Optional != nil && *source.ConfigMapRef.Optional {
					continue
				}
				return nil, nil, err
			}
			for name := range wanted {
				if key, ok := envFromKey(source.Prefix, name); ok {
					if value, found := configMap.Data[key]; found {
						values[name] = value
					}
				}
			}
		case source.SecretRef != nil:
			secret := &corev1.Secret{}
			err := reader.Get(ctx, types.NamespacedName{Namespace: namespace, Name: source.SecretRef.Name}, secret)
			if err != nil {
				if apierrors.IsNotFound(err) && source.SecretRef.Optional != nil && *source.SecretRef.Optional {
					continue
				}
				return nil, nil, err
			}
			for name := range wanted {
				if key, ok := envFromKey(source.Prefix, name); ok {
					if value, found := secret.Data[key]; found {
						values[name] = string(value)
					}
				}
			}
		}
	}

	// Hide envFrom during backend rendering and expose only values proven to exist.
	resolved.EnvFrom = nil
	for _, name := range names {
		if value, found := values[name]; found {
			resolved.Env = append(resolved.Env, corev1.EnvVar{Name: name, Value: value})
			overlay.materialized[name] = struct{}{}
		}
	}

	return resolved, overlay, nil
}

// Restore removes render-only env entries and restores the original envFrom sources on the rendered container.
func (o *ContainerEnvFromOverlay) Restore(container *corev1.Container) {
	container.EnvFrom = append([]corev1.EnvFromSource(nil), o.originalEnvFrom...)

	// Remove only entries materialized from envFrom; backend-generated settings remain intact.
	env := container.Env[:0]
	for _, variable := range container.Env {
		if _, temporary := o.materialized[variable.Name]; temporary {
			continue
		}
		env = append(env, variable)
	}
	container.Env = env
}

func envFromKey(prefix, name string) (string, bool) {
	if !strings.HasPrefix(name, prefix) {
		return "", false
	}
	return name[len(prefix):], true
}
