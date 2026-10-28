//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"io"
	"net/http"
	"strings"
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestObserveWorkerStatusContract(t *testing.T) {
	// Keep wire fixtures independent of workerStatus so decoder changes cannot rewrite the producer contract.
	tests := []struct {
		name    string
		payload string
		members []domain.NativeMemberID
		wantErr bool
	}{
		{
			name: "single-member worker",
			payload: `{"pod_uid":"pod-0","member_ids":["dp-0"],"binding":null,"verified":false,"admitted":false,
				"listener_ready":true,"instance_id":"engine-0","retired":false,"withdrawn":true,"native_drain_enabled":true}`,
			members: []domain.NativeMemberID{"dp-0"},
		},
		{
			name:    "complete local member range",
			payload: `{"pod_uid":"pod-0","member_ids":["dp-0","dp-1"]}`,
			members: []domain.NativeMemberID{"dp-0", "dp-1"},
		},
		{
			name:    "singular field does not establish membership",
			payload: `{"pod_uid":"pod-0","member_id":"dp-0"}`,
			members: []domain.NativeMemberID{"dp-0"}, wantErr: true,
		},
		{
			name:    "missing member list",
			payload: `{"pod_uid":"pod-0"}`,
			members: []domain.NativeMemberID{"dp-0"}, wantErr: true,
		},
		{
			name:    "wrong member identity",
			payload: `{"pod_uid":"pod-0","member_ids":["dp-1"]}`,
			members: []domain.NativeMemberID{"dp-0"}, wantErr: true,
		},
		{
			name:    "incomplete local member range",
			payload: `{"pod_uid":"pod-0","member_ids":["dp-0"]}`,
			members: []domain.NativeMemberID{"dp-0", "dp-1"}, wantErr: true,
		},
		{
			name:    "unexpected additional member",
			payload: `{"pod_uid":"pod-0","member_ids":["dp-0","dp-1"]}`,
			members: []domain.NativeMemberID{"dp-0"}, wantErr: true,
		},
		{
			name:    "replacement Pod UID",
			payload: `{"pod_uid":"replacement","member_ids":["dp-0"]}`,
			members: []domain.NativeMemberID{"dp-0"}, wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("read the Python worker's list-shaped JSON contract without serializing the Go decoder type")
			control, err := NewWorkerClient("http://worker.test", &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				assert.Equal(t, http.MethodPost, request.Method)
				assert.Equal(t, "/engine/control/engine_group_status", request.URL.Path)
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(test.payload))}, nil
			})})
			require.NoError(t, err)
			process := Process{PodUID: "pod-0", Client: control}
			for _, id := range test.members {
				process.Membership.Members = append(process.Membership.Members, domain.NativeMemberIncarnation{ID: id})
			}

			t.Log("accept only the exact Pod UID and complete expected member list")
			_, err = observeWorker(t.Context(), process)
			if test.wantErr {
				require.ErrorContains(t, err, "belongs to another incarnation")
			} else {
				require.NoError(t, err)
			}
		})
	}
}
