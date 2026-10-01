/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package gms

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOverlayCompatibleSnapshotClients(t *testing.T) {
	tests := []struct {
		name        string
		snapshotGMS *nvidiacomv1alpha1.GPUMemoryServiceSpec
		serviceGMS  *nvidiacomv1beta1.GPUMemoryServiceSpec
		wantErr     string
		wantDevice  string
		wantClients []string
	}{
		{
			name: "both disabled",
		},
		{
			name: "snapshot enabled but workload disabled",
			snapshotGMS: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled: true,
				Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
			},
			wantErr: "snapshot enabled=true, workload enabled=false",
		},
		{
			name: "snapshot disabled but workload enabled",
			serviceGMS: &nvidiacomv1beta1.GPUMemoryServiceSpec{
				Mode: nvidiacomv1beta1.GMSModeIntraPod,
			},
			wantErr: "snapshot enabled=false, workload enabled=true",
		},
		{
			name: "mode mismatch",
			snapshotGMS: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled: true,
				Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
			},
			serviceGMS: &nvidiacomv1beta1.GPUMemoryServiceSpec{
				Mode: nvidiacomv1beta1.GMSModeInterPod,
			},
			wantErr: `snapshot="intraPod", workload="interPod"`,
		},
		{
			name: "matching topology overlays destination clients",
			snapshotGMS: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled:               true,
				Mode:                  nvidiacomv1alpha1.GMSModeIntraPod,
				DeviceClassName:       "captured-device-class",
				ExtraClientContainers: []string{"captured-client"},
			},
			serviceGMS: &nvidiacomv1beta1.GPUMemoryServiceSpec{
				Mode:                  nvidiacomv1beta1.GMSModeIntraPod,
				DeviceClassName:       "destination-device-class",
				ExtraClientContainers: []string{"destination-client"},
			},
			wantDevice:  "destination-device-class",
			wantClients: []string{"destination-client"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Apply the destination workload GMS configuration to the resolved snapshot metadata")
			snapshotGMS := test.snapshotGMS.DeepCopy()
			err := OverlayCompatibleSnapshotClients(&snapshotGMS, "worker-snapshot", test.serviceGMS)

			if test.wantErr != "" {
				t.Log("Verify incompatible topology is rejected without changing the resolved snapshot")
				require.ErrorContains(t, err, test.wantErr)
				assert.Equal(t, test.snapshotGMS, snapshotGMS)
				return
			}

			t.Log("Verify compatible topology is preserved and only destination client fields are applied")
			require.NoError(t, err)
			if test.snapshotGMS == nil {
				assert.Nil(t, snapshotGMS)
				return
			}
			require.NotNil(t, snapshotGMS)
			assert.True(t, snapshotGMS.Enabled)
			assert.Equal(t, test.snapshotGMS.Mode, snapshotGMS.Mode)
			assert.Equal(t, test.wantDevice, snapshotGMS.DeviceClassName)
			assert.Equal(t, test.wantClients, snapshotGMS.ExtraClientContainers)
		})
	}
}
