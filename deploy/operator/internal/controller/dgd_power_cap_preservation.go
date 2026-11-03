// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"strings"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type componentPowerCapPreservation struct {
	hash string
	caps map[string]string
}

// resolvePowerCapPreservation proves the exception from owned live DCDs and Deployment templates.
// renderer and dgd must be non-nil; restart may be nil when no restart is requested.
func (r *componentWorkloadsReconciler) resolvePowerCapPreservation(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, restart *dynamo.RestartState, checkpoints map[string]*checkpoint.CheckpointInfo, renderer *dcdWorkloadRenderer) (*componentPowerCapPreservation, error) {
	if len(desiredWorkerPowerCaps(dgd)) == 0 || powerCapRolloutBlocked(dgd.Status.RollingUpdate) {
		return nil, nil
	}
	current := currentWorkerHashes(dgd)
	hash := current.v1
	if hash == "" {
		hash = current.v2
	}
	if hash == "" {
		return nil, nil
	}
	restarts, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
	if err != nil {
		return nil, err
	}
	desired, err := dynamo.GenerateDynamoComponentsDeployments(dgd, restart, restarts, dynamo.RollingUpdateContext{NewWorkerHash: hash})
	if err != nil {
		return nil, err
	}

	// A second observed worker generation invalidates the exception even before rollout status catches up.
	workers := &v1beta1.DynamoComponentDeploymentList{}
	if err := r.syncer.List(ctx, workers, client.InNamespace(dgd.Namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}); err != nil {
		return nil, err
	}
	for i := range workers.Items {
		worker := &workers.Items[i]
		if metav1.IsControlledBy(worker, dgd) && dynamo.IsWorkerComponent(string(worker.Spec.ComponentType)) && dynamo.GetDCDEffectiveWorkerHash(worker) != hash {
			return nil, nil
		}
	}

	return r.resolveWorkerPowerCapPreservation(ctx, dgd, desired, checkpoints, renderer, hash)
}

// resolveWorkerPowerCapPreservation compares every worker's DCD intent and live creation template.
// dgd, renderer, and desired worker entries must be non-nil.
func (r *componentWorkloadsReconciler) resolveWorkerPowerCapPreservation(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, desired map[string]*v1beta1.DynamoComponentDeployment, checkpoints map[string]*checkpoint.CheckpointInfo, renderer *dcdWorkloadRenderer, hash string) (*componentPowerCapPreservation, error) {
	retained := &componentPowerCapPreservation{hash: hash, caps: map[string]string{}}
	changed := false
	for name, candidate := range desired {
		if !dynamo.IsWorkerComponent(string(candidate.Spec.ComponentType)) {
			continue
		}
		live := &v1beta1.DynamoComponentDeployment{}
		if err := r.syncer.Get(ctx, client.ObjectKeyFromObject(candidate), live); err != nil {
			if apierrors.IsNotFound(err) {
				return nil, nil
			}
			return nil, err
		}
		if !metav1.IsControlledBy(live, dgd) || live.DeletionTimestamp != nil {
			return nil, nil
		}
		if err := r.applyCheckpointStartupPolicy(candidate, checkpoints[name]); err != nil {
			return nil, err
		}
		if candidate.Spec.PodTemplate != nil && live.Spec.PodTemplate != nil {
			want, wantPresent := candidate.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]
			old, oldPresent := live.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]
			if wantPresent != oldPresent {
				return nil, nil
			}
			if oldPresent {
				changed = changed || want != old
				retained.caps[name] = old
				candidate.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = old
			}
		}
		if !dynamo.WorkerSpecsEqual(candidate, live) {
			return nil, nil
		}

		// Independently verify the actual creation template; the DCD alone is not rollout proof.
		candidate.OwnerReferences = live.OwnerReferences
		deployment := &appsv1.Deployment{}
		if err := r.syncer.Get(ctx, client.ObjectKeyFromObject(live), deployment); err != nil {
			if apierrors.IsNotFound(err) {
				return nil, nil
			}
			return nil, err
		}
		if !metav1.IsControlledBy(deployment, live) {
			return nil, nil
		}
		template, err := renderer.generatePodTemplateSpec(ctx, candidate, dynamo.RoleMain, renderer.containerGPUCount(ctx, candidate))
		if err != nil {
			return nil, fmt.Errorf("render active worker template: %w", err)
		}
		if !powerCapTemplatesEqual(template, &deployment.Spec.Template) {
			return nil, nil
		}
	}
	if !changed {
		return nil, nil
	}
	return retained, nil
}

// powerCapTemplatesEqual normalizes Kubernetes' built-in Pod defaults, never arbitrary live fields.
// Inputs must be non-nil and remain unmodified. Extra live fields still invalidate preservation.
func powerCapTemplatesEqual(a, b *corev1.PodTemplateSpec) bool {
	x, y := a.DeepCopy(), b.DeepCopy()
	defaultPowerCapTemplate(x)
	defaultPowerCapTemplate(y)
	return apiequality.Semantic.DeepEqual(x, y)
}

func defaultPowerCapTemplate(template *corev1.PodTemplateSpec) {
	spec := &template.Spec
	if spec.RestartPolicy == "" {
		spec.RestartPolicy = corev1.RestartPolicyAlways
	}
	if spec.DNSPolicy == "" {
		spec.DNSPolicy = corev1.DNSClusterFirst
	}
	if spec.SchedulerName == "" {
		spec.SchedulerName = corev1.DefaultSchedulerName
	}
	if spec.TerminationGracePeriodSeconds == nil {
		spec.TerminationGracePeriodSeconds = new(int64(30))
	}
	if spec.SecurityContext == nil {
		spec.SecurityContext = &corev1.PodSecurityContext{}
	}
	for i := range spec.Volumes {
		volume := &spec.Volumes[i]
		if volume.Secret != nil && volume.Secret.DefaultMode == nil {
			volume.Secret.DefaultMode = new(int32(0644))
		}
		if volume.ConfigMap != nil && volume.ConfigMap.DefaultMode == nil {
			volume.ConfigMap.DefaultMode = new(int32(0644))
		}
		if volume.DownwardAPI != nil && volume.DownwardAPI.DefaultMode == nil {
			volume.DownwardAPI.DefaultMode = new(int32(0644))
		}
		if volume.Projected != nil && volume.Projected.DefaultMode == nil {
			volume.Projected.DefaultMode = new(int32(0644))
		}
	}
	for i := range spec.Containers {
		defaultPowerCapContainer(&spec.Containers[i])
	}
	for i := range spec.InitContainers {
		defaultPowerCapContainer(&spec.InitContainers[i])
	}
}

func defaultPowerCapContainer(container *corev1.Container) {
	for i := range container.Env {
		value := container.Env[i].ValueFrom
		if value == nil {
			continue
		}
		if value.FieldRef != nil && value.FieldRef.APIVersion == "" {
			value.FieldRef.APIVersion = "v1"
		}
		if value.ResourceFieldRef != nil && value.ResourceFieldRef.Divisor.IsZero() {
			value.ResourceFieldRef.Divisor = resource.MustParse("1")
		}
	}
	for i := range container.Ports {
		if container.Ports[i].Protocol == "" {
			container.Ports[i].Protocol = corev1.ProtocolTCP
		}
	}
	if container.TerminationMessagePath == "" {
		container.TerminationMessagePath = corev1.TerminationMessagePathDefault
	}
	if container.TerminationMessagePolicy == "" {
		container.TerminationMessagePolicy = corev1.TerminationMessageReadFile
	}
	if container.ImagePullPolicy == "" {
		container.ImagePullPolicy = corev1.PullIfNotPresent
		image, _, _ := strings.Cut(container.Image, "@")
		if !strings.Contains(container.Image, "@") && (!strings.Contains(image[strings.LastIndex(image, "/")+1:], ":") || strings.HasSuffix(image, ":latest")) {
			container.ImagePullPolicy = corev1.PullAlways
		}
	}
	for _, probe := range []*corev1.Probe{container.LivenessProbe, container.ReadinessProbe, container.StartupProbe} {
		if probe == nil {
			continue
		}
		if probe.TimeoutSeconds == 0 {
			probe.TimeoutSeconds = 1
		}
		if probe.PeriodSeconds == 0 {
			probe.PeriodSeconds = 10
		}
		if probe.SuccessThreshold == 0 {
			probe.SuccessThreshold = 1
		}
		if probe.FailureThreshold == 0 {
			probe.FailureThreshold = 3
		}
		if probe.HTTPGet != nil && probe.HTTPGet.Scheme == "" {
			probe.HTTPGet.Scheme = corev1.URISchemeHTTP
		}
		if probe.HTTPGet != nil && probe.HTTPGet.Path == "" {
			probe.HTTPGet.Path = "/"
		}
	}
}
