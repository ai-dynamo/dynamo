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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	gpupkg "github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"k8s.io/utils/ptr"
)

func newFakeReconciler(objs ...client.Object) *DynamoGraphDeploymentRequestReconciler {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
	return &DynamoGraphDeploymentRequestReconciler{
		Client:    fakeClient,
		APIReader: fakeClient,
		Recorder:  &record.FakeRecorder{},
	}
}

func dcgmPod(name, ip string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "gpu-operator",
			Labels: map[string]string{gpupkg.LabelApp: gpupkg.LabelValueNvidiaDCGMExporter}},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: ip},
	}
}

func TestEnrichHardwareFromDiscovery(t *testing.T) {
	tests := []struct {
		name string

		// Input hardware spec (nil fields = not set by user).
		hardware *nvidiacomv1beta1.HardwareSpec

		// Discovery mock: what DCGM returns. Nil means no discovery available.
		discoveredGPU *gpupkg.GPUInfo

		// Expected outcome.
		wantErr       string // non-empty means error expected, substring match
		wantGPUSKU    string
		wantVRAM      float64
		wantGPUsNode  int32
		wantTotalGPUs int32
	}{
		{
			name: "all four fields set, discovery skipped",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "h100_sxm", VRAMMB: ptr.To(81920.0),
				NumGPUsPerNode: ptr.To(int32(8)), TotalGPUs: ptr.To(int32(16)),
			},
			wantGPUSKU: "h100_sxm", wantVRAM: 81920, wantGPUsNode: 8, wantTotalGPUs: 16,
		},
		{
			name:          "nothing set, full discovery",
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "H100-SXM5-80GB", VRAMPerGPU: 81920},
			wantGPUSKU:    "h100_sxm", wantVRAM: 81920, wantGPUsNode: 8, wantTotalGPUs: 8,
		},
		{
			name:          "nothing set, V100 discovered",
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "Tesla-V100-SXM2-16GB", VRAMPerGPU: 16384},
			wantGPUSKU:    "v100_sxm", wantVRAM: 16384, wantGPUsNode: 8, wantTotalGPUs: 8,
		},
		{
			name:          "nothing set, unknown GPU falls back to model name",
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 4, Model: "FutureGPU-X1000", VRAMPerGPU: 65536},
			wantGPUSKU:    "FutureGPU-X1000", wantVRAM: 65536, wantGPUsNode: 4, wantTotalGPUs: 4,
		},
		{
			name: "only totalGpus missing, discovery fills it",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "b200_sxm", VRAMMB: ptr.To(141312.0), NumGPUsPerNode: ptr.To(int32(8)),
			},
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "B200-SXM-180GB", VRAMPerGPU: 141312},
			wantGPUSKU:    "b200_sxm", wantVRAM: 141312, wantGPUsNode: 8, wantTotalGPUs: 8,
		},
		{
			name: "only gpuSku missing, discovery fills it",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				VRAMMB: ptr.To(81920.0), NumGPUsPerNode: ptr.To(int32(8)), TotalGPUs: ptr.To(int32(16)),
			},
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "H200-SXM5-141GB", VRAMPerGPU: 141312},
			wantGPUSKU:    "h200_sxm", wantVRAM: 81920, wantGPUsNode: 8, wantTotalGPUs: 16, // user overrides win
		},
		{
			name: "vramMb and numGpusPerNode override discovery",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "a100_sxm", VRAMMB: ptr.To(40960.0), NumGPUsPerNode: ptr.To(int32(4)),
			},
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "A100-SXM4-80GB", VRAMPerGPU: 81920},
			wantGPUSKU:    "a100_sxm", wantVRAM: 40960, wantGPUsNode: 4, wantTotalGPUs: 8,
		},
		{
			name:    "no fields set, discovery fails",
			wantErr: "auto-discovery failed",
		},
		{
			name: "three fields set, discovery fails",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "h100_sxm", VRAMMB: ptr.To(81920.0), NumGPUsPerNode: ptr.To(int32(8)),
			},
			wantErr: "auto-discovery failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r *DynamoGraphDeploymentRequestReconciler
			if tt.discoveredGPU != nil {
				// Set up a DCGM pod and mock scraper so discovery works.
				r = newFakeReconciler(dcgmPod("dcgm-exporter", "10.0.0.1"))
				r.GPUDiscovery = gpupkg.NewGPUDiscovery(func(ctx context.Context, endpoint string) (*gpupkg.GPUInfo, error) {
					return tt.discoveredGPU, nil
				})
				r.GPUDiscoveryCache = gpupkg.NewGPUDiscoveryCache()
			} else if tt.wantErr != "" {
				// No discovery — will fail if discovery is attempted.
				r = newFakeReconciler()
			} else {
				// All fields set — discovery not needed, no mock required.
				r = newFakeReconciler()
			}

			hw := tt.hardware
			if hw == nil {
				hw = &nvidiacomv1beta1.HardwareSpec{}
			}
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Hardware: hw,
				},
			}

			err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, dgdr.Spec.Hardware)
			assert.Equal(t, tt.wantGPUSKU, string(dgdr.Spec.Hardware.GPUSKU))
			assert.Equal(t, tt.wantVRAM, *dgdr.Spec.Hardware.VRAMMB)
			assert.Equal(t, tt.wantGPUsNode, *dgdr.Spec.Hardware.NumGPUsPerNode)
			assert.Equal(t, tt.wantTotalGPUs, *dgdr.Spec.Hardware.TotalGPUs)
		})
	}
}
