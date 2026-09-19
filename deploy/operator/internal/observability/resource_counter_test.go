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

package observability

import (
	"context"
	"reflect"
	"testing"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

type listRecordingClient struct {
	client.Client
	listed client.ObjectList
}

func (c *listRecordingClient) List(_ context.Context, list client.ObjectList, _ ...client.ListOption) error {
	c.listed = list
	return nil
}

func TestResourceCounterDoesNotNeedLeaderElection(t *testing.T) {
	counter := NewResourceCounter(fake.NewClientBuilder().Build(), nil)

	if counter.NeedLeaderElection() {
		t.Fatal("ResourceCounter must run on every replica")
	}
}

func TestDeploymentResourceCountersListV1Beta1(t *testing.T) {
	tests := []struct {
		name     string
		update   func(context.Context, client.Client, ExcludedNamespaces, logr.Logger)
		wantList client.ObjectList
	}{
		{
			name:     "DynamoGraphDeployment",
			update:   updateDynamoGraphDeploymentCounts,
			wantList: &v1beta1.DynamoGraphDeploymentList{},
		},
		{
			name:     "DynamoComponentDeployment",
			update:   updateDynamoComponentDeploymentCounts,
			wantList: &v1beta1.DynamoComponentDeploymentList{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := &listRecordingClient{Client: fake.NewClientBuilder().Build()}
			test.update(context.Background(), c, nil, logr.Discard())

			if got, want := reflect.TypeOf(c.listed), reflect.TypeOf(test.wantList); got != want {
				t.Fatalf("listed %v, want %v", got, want)
			}
		})
	}
}
