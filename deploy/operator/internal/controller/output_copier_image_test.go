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

package controller

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestCreateProfilingJobOutputCopierImage(t *testing.T) {
	tests := []struct {
		name          string
		operatorImage string
		requestImage  string
		want          string
	}{
		{name: "pinned default", want: configv1alpha1.DefaultDGDROutputCopierImage},
		{name: "operator digest", operatorImage: "registry.example/kubectl@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", want: "registry.example/kubectl@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
		{name: "request overrides operator", operatorImage: "registry.example/kubectl:operator", requestImage: "registry.example/kubectl:request", want: "registry.example/kubectl:request"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Configure an operator and a request with complete hardware")
			ctx := context.Background()
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, batchv1.AddToScheme(scheme))
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Name: "copier-image", Namespace: "default", UID: "copier-request",
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Model: "test-model", Backend: nvidiacomv1beta1.BackendTypeVllm, Image: "test-profiler:1.7.0",
					Hardware: &nvidiacomv1beta1.HardwareSpec{
						GPUSKU: nvidiacomv1beta1.GPUSKUTypeH100SXM, VRAMMB: new(81920.0),
						NumGPUsPerNode: new(int32(8)), TotalGPUs: new(int32(8)), Interconnect: "sxm", RDMA: new(false),
					},
				},
			}
			if tc.requestImage != "" {
				dgdr.Spec.Overrides = &nvidiacomv1beta1.OverridesSpec{
					ProfilingJob: &batchv1.JobSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: ContainerNameOutputCopier, Image: tc.requestImage}},
					}}},
				}
			}
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgdr).Build()
			reconciler := &DynamoGraphDeploymentRequestReconciler{
				Client: fakeClient, Recorder: &events.FakeRecorder{},
				Config: &configv1alpha1.OperatorConfiguration{
					Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"},
					DGDR:      configv1alpha1.DGDRConfiguration{OutputCopierImage: tc.operatorImage},
				},
				RuntimeConfig: &commonController.RuntimeConfig{Gate: features.Defaults()},
			}

			t.Log("Create the profiling Job and inspect the selected image and command")
			changed, err := reconciler.createProfilingJob(ctx, dgdr)
			require.NoError(t, err)
			require.True(t, changed)
			key := types.NamespacedName{Name: getProfilingJobName(dgdr), Namespace: dgdr.Namespace}
			job := &batchv1.Job{}
			require.NoError(t, fakeClient.Get(ctx, key, job))
			copier := job.Spec.Template.Spec.Containers[1]
			require.Equal(t, ContainerNameOutputCopier, copier.Name)
			require.Equal(t, tc.want, copier.Image)
			require.Equal(t, []string{"/bin/sh", "-c"}, copier.Command)
			require.Contains(t, copier.Args[0], "kubectl")

			t.Log("Change operator configuration without rewriting an existing Job")
			before := job.DeepCopy()
			reconciler.Config.DGDR.OutputCopierImage = "registry.example/kubectl:next"
			changed, err = reconciler.createProfilingJob(ctx, dgdr)
			require.NoError(t, err)
			require.False(t, changed)
			require.NoError(t, fakeClient.Get(ctx, key, job))
			require.Equal(t, before.Spec, job.Spec)
		})
	}
}
