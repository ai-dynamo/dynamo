/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package validation

import (
	"context"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	admissionDCDKind = "DynamoComponentDeployment"
	admissionDGDKind = "DynamoGraphDeployment"
)

func admissionUnstructured(t *testing.T, deployment runtime.Object) map[string]any {
	t.Helper()
	request, err := runtime.DefaultUnstructuredConverter.ToUnstructured(deployment)
	if err != nil {
		t.Fatalf("convert %T to unstructured: %v", deployment, err)
	}

	// Several compact fixtures omit TypeMeta, but a real API request requires it.
	_, hasAPIVersion := request["apiVersion"]
	_, hasKind := request["kind"]
	if !hasAPIVersion || !hasKind {
		var apiVersion, kind string
		switch deployment.(type) {
		case *nvidiacomv1alpha1.DynamoGraphDeployment:
			apiVersion = nvidiacomv1alpha1.GroupVersion.String()
			kind = admissionDGDKind
		case *nvidiacomv1beta1.DynamoGraphDeployment:
			apiVersion = nvidiacomv1beta1.GroupVersion.String()
			kind = admissionDGDKind
		case *nvidiacomv1alpha1.DynamoComponentDeployment:
			apiVersion = nvidiacomv1alpha1.GroupVersion.String()
			kind = admissionDCDKind
		case *nvidiacomv1beta1.DynamoComponentDeployment:
			apiVersion = nvidiacomv1beta1.GroupVersion.String()
			kind = admissionDCDKind
		default:
			t.Fatalf("unsupported admission object type %T", deployment)
		}
		if !hasAPIVersion {
			request["apiVersion"] = apiVersion
		}
		if !hasKind {
			request["kind"] = kind
		}
	}
	delete(request, "status")
	return request
}

func admissionSourceVersion(t *testing.T, object runtime.Object) string {
	t.Helper()
	if version := object.GetObjectKind().GroupVersionKind().Version; version != "" {
		return version
	}
	switch object.(type) {
	case *nvidiacomv1alpha1.DynamoGraphDeployment, *nvidiacomv1alpha1.DynamoComponentDeployment:
		return nvidiacomv1alpha1.GroupVersion.Version
	case *nvidiacomv1beta1.DynamoGraphDeployment, *nvidiacomv1beta1.DynamoComponentDeployment:
		return nvidiacomv1beta1.GroupVersion.Version
	default:
		t.Fatalf("unsupported admission object type %T", object)
		return ""
	}
}

func TestValidateDynamoComponentDeploymentSharedSpecFieldPaths(t *testing.T) {
	minAvailable := int32(1)
	replicas := int32(2)
	frontendSidecar := "missing"
	sharedMemorySize := resource.MustParse("-1Gi")
	spec := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "epp",
		ComponentType: nvidiacomv1beta1.ComponentTypeEPP,
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid"},
			},
			Spec: corev1.PodSpec{
				Containers:     []corev1.Container{{Name: consts.MainContainerName}, {Name: "sidecar"}},
				InitContainers: []corev1.Container{{Name: "init"}},
			},
		},
		Replicas:         &replicas,
		MinAvailable:     &minAvailable,
		Multinode:        &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
		SharedMemorySize: &sharedMemorySize,
		EPPConfig: &nvidiacomv1beta1.EPPConfig{
			ConfigMapRef: &corev1.ConfigMapKeySelector{},
		},
		FrontendSidecar: &frontendSidecar,
	}
	validation := &sharedValidation{ctx: context.Background(), mgr: newGroveTopologyTestManager(t)}

	errs := validation.validateDynamoComponentDeploymentSharedSpec(spec, field.NewPath("spec", "components").Index(0), false, true)
	assertFieldPaths(t, errs, []string{
		"spec.components[0].minAvailable",
		"spec.components[0].sharedMemorySize",
		"spec.components[0].type",
		"spec.components[0].multinode",
		"spec.components[0].replicas",
		"spec.components[0].eppConfig.configMapRef.name",
		"spec.components[0].frontendSidecar",
	})
}

func TestValidateDynamoComponentDeploymentSharedSpecFrontendSidecar(t *testing.T) {
	validation := &sharedValidation{ctx: context.Background(), mgr: newGroveTopologyTestManager(t)}
	componentPath := field.NewPath("spec", "components").Index(0)

	t.Run("requires pod template", func(t *testing.T) {
		name := "frontend"
		spec := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{FrontendSidecar: &name}
		errs := validation.validateDynamoComponentDeploymentSharedSpec(spec, componentPath, true, true)
		assertFieldPaths(t, errs, []string{
			"spec.components[0].podTemplate.spec.containers",
		})
	})

	t.Run("rejects empty name", func(t *testing.T) {
		name := ""
		spec := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			PodTemplate:     &corev1.PodTemplateSpec{},
			FrontendSidecar: &name,
		}
		errs := validation.validateDynamoComponentDeploymentSharedSpec(spec, componentPath, true, true)
		assertFieldPaths(t, errs, []string{
			"spec.components[0].frontendSidecar",
		})
	})

	t.Run("accepts matching container", func(t *testing.T) {
		name := "frontend"
		spec := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			PodTemplate: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: name, Image: "frontend:latest"}}},
			},
			FrontendSidecar: &name,
		}
		errs := validation.validateDynamoComponentDeploymentSharedSpec(spec, componentPath, true, true)
		assertFieldPaths(t, errs, nil)
	})
}

func TestValidateComponentCheckpointJobConfigFieldPaths(t *testing.T) {
	validation := &sharedValidation{ctx: context.Background(), mgr: newGroveTopologyTestManager(t)}
	fldPath := field.NewPath("spec", "components").Index(0).Child("experimental", "checkpoint", "job")
	job := &nvidiacomv1beta1.ComponentCheckpointJobConfig{GMSClientContainers: []string{"saver"}}

	errs := validation.validateComponentCheckpointJobConfig(job, fldPath, nil)
	assertFieldPaths(t, errs, []string{
		"spec.components[0].experimental.checkpoint.job.gmsClientContainers",
	})
	errs = validation.validateComponentCheckpointJobConfig(
		job,
		fldPath,
		&nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeInterPod},
	)
	assertFieldPaths(t, errs, []string{"spec.components[0].experimental.checkpoint.job.gmsClientContainers"})
	errs = validation.validateComponentCheckpointJobConfig(
		job,
		fldPath,
		&nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
	)
	assertFieldPaths(t, errs, nil)
	errs = validation.validateComponentCheckpointJobConfig(
		&nvidiacomv1beta1.ComponentCheckpointJobConfig{},
		fldPath,
		nil,
	)
	assertFieldPaths(t, errs, nil)
}

func TestValidateDynamoComponentDeploymentSharedSpecV1alpha1FrontendSidecarFieldPaths(t *testing.T) {
	validation := &sharedValidation{ctx: context.Background(), mgr: newGroveTopologyTestManager(t)}
	fldPath := field.NewPath("spec", "services").Key("frontend")
	frontendSidecar := &nvidiacomv1alpha1.FrontendSidecarSpec{
		Image: "frontend:latest",
		Envs:  []corev1.EnvVar{{Name: "TOKEN", Value: "do-not-leak-this-value"}},
	}
	spec := &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{FrontendSidecar: frontendSidecar}
	errs := validation.validateDynamoComponentDeploymentSharedSpecV1alpha1(spec, fldPath, "dynamo")
	assertFieldPaths(t, errs, nil)
	spec.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{}}
	errs = validation.validateDynamoComponentDeploymentSharedSpecV1alpha1(spec, fldPath, "dynamo")
	assertFieldPaths(t, errs, nil)
	spec.ExtraPodSpec.PodSpec.Containers = []corev1.Container{{Name: consts.FrontendSidecarContainerName}}
	errs = validation.validateDynamoComponentDeploymentSharedSpecV1alpha1(spec, fldPath, "dynamo")
	assertFieldPaths(t, errs, []string{"spec.services[frontend].frontendSidecar"})
	if errs[0].BadValue != "" {
		t.Fatalf("error BadValue = %#v, want an empty non-sensitive scalar", errs[0].BadValue)
	}
	if strings.Contains(errs.ToAggregate().Error(), "do-not-leak-this-value") {
		t.Fatalf("error = %q, must not expose frontend sidecar environment values", errs.ToAggregate())
	}
}

func TestValidateExperimentalSpecDoesNotExposePodTemplate(t *testing.T) {
	validation := &sharedValidation{ctx: context.Background(), mgr: newGroveTopologyTestManager(t)}
	fldPath := field.NewPath("spec", "components").Index(0).Child("experimental")
	gms := &nvidiacomv1beta1.GPUMemoryServiceSpec{
		ExtraClientPods: []nvidiacomv1beta1.GMSClientPodSpec{{
			Name: "client",
			PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: consts.MainContainerName,
				Env:  []corev1.EnvVar{{Name: "TOKEN", Value: "do-not-leak-this-value"}},
			}}}},
		}},
	}

	errs := validation.validateExperimentalSpec(
		&nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: gms},
		fldPath,
		nvidiacomv1beta1.ComponentTypeWorker,
		corev1.ResourceRequirements{},
	)
	assertFieldPaths(t, errs, []string{"spec.components[0].experimental.gpuMemoryService"})
	if errs[0].BadValue != "" {
		t.Fatalf("error BadValue = %#v, want an empty non-sensitive scalar", errs[0].BadValue)
	}
	if strings.Contains(errs.ToAggregate().Error(), "do-not-leak-this-value") {
		t.Fatalf("error = %q, must not expose GPU memory service pod template values", errs.ToAggregate())
	}
}

func TestValidateDynamoComponentDeploymentSharedSpecV1alpha1WarningsAndErrors(t *testing.T) {
	legacyNamespace := "legacy"
	spec := &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		DynamoNamespace: &legacyNamespace,
		Annotations: map[string]string{
			consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
		},
	}

	//nolint:staticcheck // SA1019: Intentionally testing the deprecated compatibility warning.
	spec.Autoscaling = &nvidiacomv1alpha1.Autoscaling{Enabled: true}
	fldPath := field.NewPath("spec", "services").Key("worker")
	validation := &sharedValidation{ctx: context.Background(), mgr: newGroveTopologyTestManager(t)}

	errs := validation.validateDynamoComponentDeploymentSharedSpecV1alpha1(spec, fldPath, "replacement")
	if len(validation.warnings) != 2 {
		t.Fatalf("warnings = %v, want 2 compatibility warnings", validation.warnings)
	}
	if !strings.Contains(validation.warnings[0], "spec.services[worker].dynamoNamespace") ||
		!strings.Contains(validation.warnings[1], "spec.services[worker].autoscaling") {
		t.Fatalf("warnings = %v, want structural field paths", validation.warnings)
	}
	assertFieldPaths(t, errs, []string{
		"spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend]",
	})
}
