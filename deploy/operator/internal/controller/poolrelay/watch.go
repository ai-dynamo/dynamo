/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"slices"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

const relayDependencyIndex = "dynamo-relay.transport-dependency"

// relayDGDDependencies indexes declared references, including unavailable inputs,
// so creating a missing Secret can unblock configuration resolution.
func relayDGDDependencies(object client.Object) []string {
	dgd := object.(*v1beta1.DynamoGraphDeployment)
	seen := map[string]bool{}
	env := append([]corev1.EnvVar{}, dgd.Spec.Env...)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		for _, template := range dynamo.ComponentPodTemplates(component) {
			for _, container := range template.Spec.Containers {
				if container.Name == "main" {
					env = append(env, container.Env...)
					// Key changes can introduce transport settings into backend-only envFrom.
					for _, source := range container.EnvFrom {
						if source.SecretRef != nil {
							seen["Secret/"+source.SecretRef.Name] = true
						}
						if source.ConfigMapRef != nil {
							seen["ConfigMap/"+source.ConfigMapRef.Name] = true
						}
					}
				}
			}
			for _, volume := range template.Spec.Volumes {
				if volume.Secret != nil {
					seen["Secret/"+volume.Secret.SecretName] = true
				}
				if volume.ConfigMap != nil {
					seen["ConfigMap/"+volume.ConfigMap.Name] = true
				}
			}
		}
	}
	for _, entry := range env {
		if !relayTransportEnv(entry.Name) || entry.ValueFrom == nil {
			continue
		}
		if ref := entry.ValueFrom.SecretKeyRef; ref != nil {
			seen["Secret/"+ref.Name] = true
		}
		if ref := entry.ValueFrom.ConfigMapKeyRef; ref != nil {
			seen["ConfigMap/"+ref.Name] = true
		}
	}
	result := make([]string, 0, len(seen))
	for key := range seen {
		result = append(result, key)
	}
	slices.Sort(result)
	return result
}

// relaysForDGD handles both old and new objects on label updates through the
// controller-runtime map handler, including changes without a new generation.
func (r *relayReconciler) relaysForDGD(ctx context.Context, object client.Object) []reconcile.Request {
	var relays v1alpha1.DynamoPoolRelayList
	if err := r.List(ctx, &relays, client.InNamespace(object.GetNamespace())); err != nil {
		ctrl.LoggerFrom(ctx).Error(err, "Cannot map DGD transition to Relays")
		return nil
	}
	result := make([]reconcile.Request, 0)
	for _, relay := range relays.Items {
		selected := false
		for _, ref := range relay.Spec.Sources.DGDs.Refs {
			if ref.Name == object.GetName() {
				selected = true
				break
			}
		}
		if !selected && relay.Spec.Sources.DGDs.LabelSelector != nil {
			selector, err := metav1.LabelSelectorAsSelector(relay.Spec.Sources.DGDs.LabelSelector)
			selected = err == nil && selector.Matches(labels.Set(object.GetLabels()))
		}
		if selected {
			result = append(result, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(&relay)})
		}
	}
	return result
}

// relaysForDependency follows indexed DGD references rather than requeueing every
// Relay for unrelated namespace Secret or ConfigMap updates.
func (r *relayReconciler) relaysForDependency(ctx context.Context, object client.Object) []reconcile.Request {
	if owner := metav1.GetControllerOf(object); owner != nil && owner.Kind == "DynamoPoolRelay" && owner.APIVersion == "nvidia.com/v1alpha1" {
		return []reconcile.Request{{NamespacedName: client.ObjectKey{Namespace: object.GetNamespace(), Name: owner.Name}}}
	}
	kind := "ConfigMap"
	if _, ok := object.(*corev1.Secret); ok {
		kind = "Secret"
	}
	var dgds v1beta1.DynamoGraphDeploymentList
	if err := r.List(ctx, &dgds, client.InNamespace(object.GetNamespace()), client.MatchingFields{relayDependencyIndex: kind + "/" + object.GetName()}); err != nil {
		ctrl.LoggerFrom(ctx).Error(err, "Cannot map Relay connection dependency")
		return nil
	}
	seen := map[client.ObjectKey]bool{}
	result := make([]reconcile.Request, 0)
	for i := range dgds.Items {
		for _, request := range r.relaysForDGD(ctx, &dgds.Items[i]) {
			if !seen[request.NamespacedName] {
				seen[request.NamespacedName] = true
				result = append(result, request)
			}
		}
	}
	return result
}

// relaysForPod maps sources acknowledgement updates only to the labelled Relay UID.
func (r *relayReconciler) relaysForPod(ctx context.Context, object client.Object) []reconcile.Request {
	uid := object.GetLabels()["nvidia.com/relay-uid"]
	if uid == "" {
		return nil
	}
	var relays v1alpha1.DynamoPoolRelayList
	if err := r.List(ctx, &relays, client.InNamespace(object.GetNamespace())); err != nil {
		return nil
	}
	for _, relay := range relays.Items {
		if string(relay.UID) == uid {
			return []reconcile.Request{{NamespacedName: client.ObjectKeyFromObject(&relay)}}
		}
	}
	return nil
}
