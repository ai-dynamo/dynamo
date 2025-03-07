/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package fixtures

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/env"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
)

type MockNDSServer struct {
	server *httptest.Server
	throws *bool
}

func (s *MockNDSServer) Close() {
	s.server.Close()
}

func (s *MockNDSServer) Throws(throws bool) {
	s.throws = &throws
}

func CreateMockNDSServer(t *testing.T) *MockNDSServer {
	throws := false
	mockServer := MockNDSServer{}
	mockServer.throws = &throws
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if *mockServer.throws {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		urlParts := strings.Split(r.URL.String(), "/")
		n := len(urlParts)

		response := schemas.CompoundNimVersionSchema{
			ResourceSchema: schemas.ResourceSchema{
				BaseSchema: schemas.BaseSchema{
					Uid: "123456",
				},
				Name: "benquadinaros",
			},
			Version: urlParts[n-1],
		}

		jsonResponse, err := json.Marshal(response)
		if err != nil {
			t.Fatalf("Failed to marshal JSON %s", err.Error())
		}

		w.WriteHeader(http.StatusOK)
		w.Write(jsonResponse)
	}))

	idx := strings.LastIndex(server.URL, ":")
	os.Setenv("NDS_HOST", "localhost")
	os.Setenv("NDS_PORT", server.URL[idx+1:])
	env.NdsHostBase = fmt.Sprintf("localhost:%s", server.URL[idx+1:]) // This var is cached and must be set to the new value

	mockServer.server = server
	return &mockServer
}
