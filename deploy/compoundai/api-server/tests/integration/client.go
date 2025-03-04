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

package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"testing"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemasv2"
)

/**
	This file exposes a series of helper functions to utilize the CompoundAI API server
**/

type CompoundAIClient struct {
	Url     string
	Headers http.Header
}

func (c *CompoundAIClient) CreateCluster(t *testing.T, s schemas.CreateClusterSchema) (*http.Response, *schemas.ClusterFullSchema) {
	// Marshal the request body
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Create a new HTTP request
	req, err := http.NewRequest(http.MethodPost, c.Url+"/api/v1/clusters", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}

	// Set headers from the client
	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value) // Use Add to support multiple values for the same header key
		}
	}

	// Use http.Client to execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to create cluster: %v", err)
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	// Unmarshal the response into the schema
	var clusterFullSchema schemas.ClusterFullSchema
	if err = json.Unmarshal(respBody, &clusterFullSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &clusterFullSchema
}

func (c *CompoundAIClient) UpdateCluster(t *testing.T, name string, s schemas.UpdateClusterSchema) (*http.Response, *schemas.ClusterFullSchema) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Create the PATCH request with JSON data
	req, err := http.NewRequest(http.MethodPatch, c.Url+"/api/v1/clusters/"+name, bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed create update request %s", err.Error())
	}

	// Set the appropriate headers
	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value) // Use Add to support multiple values for the same header key
		}
	}

	// Create an HTTP client and send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to update cluster %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var clusterFullSchema schemas.ClusterFullSchema
	if err = json.Unmarshal(respBody, &clusterFullSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, &clusterFullSchema
}

func encodeListQuerySchema(query schemas.ListQuerySchema) string {
	params := url.Values{}
	params.Set("start", strconv.FormatUint(uint64(query.Start), 10))
	params.Set("count", strconv.FormatUint(uint64(query.Count), 10))
	if query.Search != nil {
		params.Set("search", *query.Search)
	}
	params.Set("q", query.Q)
	return params.Encode()
}

func (c *CompoundAIClient) GetCluster(t *testing.T, name string) (*http.Response, *schemas.ClusterFullSchema) {
	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v1/clusters/"+name, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}

	// Set headers from the client
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get cluster: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var clusterFullSchema schemas.ClusterFullSchema
	if err = json.Unmarshal(respBody, &clusterFullSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &clusterFullSchema
}

func (c *CompoundAIClient) GetClusterList(t *testing.T, s schemas.ListQuerySchema) (*http.Response, *schemas.ClusterListSchema) {
	form := encodeListQuerySchema(s)
	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v1/clusters?"+form, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}

	// Set headers from the client
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get cluster list: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var clusterListSchema schemas.ClusterListSchema
	if err = json.Unmarshal(respBody, &clusterListSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &clusterListSchema
}

func (c *CompoundAIClient) CreateDeployment(t *testing.T, clusterName string, s schemas.CreateDeploymentSchema) (*http.Response, *schemas.DeploymentSchema) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, c.Url+"/api/v1/clusters/"+clusterName+"/deployments", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema schemas.DeploymentSchema
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentSchema
}

func (c *CompoundAIClient) UpdateDeployment(t *testing.T, clusterName, namespace string, deploymentName string, s schemas.UpdateDeploymentSchema) (*http.Response, *schemas.DeploymentSchema) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Create the PATCH request with JSON data
	req, err := http.NewRequest(http.MethodPatch, c.Url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName, bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed create update request %s", err.Error())
	}

	// Set the appropriate headers
	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	// Create an HTTP client and send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to update deployment %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema schemas.DeploymentSchema
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, &deploymentSchema
}

func (c *CompoundAIClient) GetDeployment(t *testing.T, clusterName, namespace, deploymentName string) (*http.Response, *schemas.DeploymentSchema) {
	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get deployment: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema schemas.DeploymentSchema
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentSchema
}

func (c *CompoundAIClient) SyncDeploymentStatus(t *testing.T, clusterName, namespace, deploymentName string) (*http.Response, any) {
	req, err := http.NewRequest(http.MethodPost, c.Url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName+"/sync_status", nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to sync deployment status: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var statusResponse any
	if err = json.Unmarshal(respBody, &statusResponse); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, statusResponse
}

func (c *CompoundAIClient) TerminateDeployment(t *testing.T, clusterName, namespace, deploymentName string) (*http.Response, *schemas.DeploymentSchema) {
	req, err := http.NewRequest(http.MethodPost, c.Url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName+"/terminate", nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to terminate deployment: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var terminateResponse schemas.DeploymentSchema
	if err = json.Unmarshal(respBody, &terminateResponse); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &terminateResponse
}

func (c *CompoundAIClient) DeleteDeployment(t *testing.T, clusterName, namespace, deploymentName string) (*http.Response, any) {
	req, err := http.NewRequest(http.MethodDelete, c.Url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to delete deployment: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deleteResponse any
	if err = json.Unmarshal(respBody, &deleteResponse); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, deleteResponse
}

func (c *CompoundAIClient) GetDeploymentList(t *testing.T, queryParams schemas.ListQuerySchema) (*http.Response, *schemas.DeploymentListSchema) {
	query := url.Values{}
	query.Add("q", queryParams.Q)
	if queryParams.Start > 0 {
		query.Add("start", fmt.Sprintf("%d", queryParams.Start))
	}
	if queryParams.Count > 0 {
		query.Add("count", fmt.Sprintf("%d", queryParams.Count))
	}
	if queryParams.Search != nil {
		query.Add("search", *queryParams.Search)
	}

	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v1/deployments?"+query.Encode(), nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get deployment revisions list: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var deploymentListSchema schemas.DeploymentListSchema
	if err = json.Unmarshal(respBody, &deploymentListSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentListSchema
}

func (c *CompoundAIClient) GetClusterDeploymentList(t *testing.T, clusterName string, queryParams schemas.ListQuerySchema) (*http.Response, *schemas.DeploymentListSchema) {
	query := url.Values{}
	query.Add("q", queryParams.Q)
	if queryParams.Start > 0 {
		query.Add("start", fmt.Sprintf("%d", queryParams.Start))
	}
	if queryParams.Count > 0 {
		query.Add("count", fmt.Sprintf("%d", queryParams.Count))
	}
	if queryParams.Search != nil {
		query.Add("search", *queryParams.Search)
	}

	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v1/clusters/"+clusterName+"/deployments?"+query.Encode(), nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get deployment revisions list: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var deploymentListSchema schemas.DeploymentListSchema
	if err = json.Unmarshal(respBody, &deploymentListSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentListSchema
}

func (c *CompoundAIClient) GetDeploymentRevision(t *testing.T, clusterName, namespace, deploymentName, revisionUid string) (*http.Response, *schemas.DeploymentRevisionSchema) {
	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName+"/revisions/"+revisionUid, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get deployment revision: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var deploymentRevisionSchema schemas.DeploymentRevisionSchema
	if err = json.Unmarshal(respBody, &deploymentRevisionSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentRevisionSchema
}

func (c *CompoundAIClient) GetDeploymentRevisionList(t *testing.T, clusterName, namespace, deploymentName string, queryParams schemas.ListQuerySchema) (*http.Response, *schemas.DeploymentRevisionListSchema) {
	query := url.Values{}
	query.Add("q", queryParams.Q)
	if queryParams.Start > 0 {
		query.Add("start", fmt.Sprintf("%d", queryParams.Start))
	}
	if queryParams.Count > 0 {
		query.Add("count", fmt.Sprintf("%d", queryParams.Count))
	}
	if queryParams.Search != nil {
		query.Add("search", *queryParams.Search)
	}

	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName+"/revisions?"+query.Encode(), nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get deployment revisions list: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var deploymentRevisionListSchema schemas.DeploymentRevisionListSchema
	if err = json.Unmarshal(respBody, &deploymentRevisionListSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentRevisionListSchema
}

// V2 Client Functions

func (c *CompoundAIClient) CreateDeploymentV2(t *testing.T, clusterName string, s schemasv2.CreateDeploymentSchema) (*http.Response, *schemasv2.DeploymentSchema) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, c.Url+"/api/v2/deployments?cluster="+clusterName, bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema schemasv2.DeploymentSchema
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentSchema
}

func (c *CompoundAIClient) UpdateDeploymentV2(t *testing.T, clusterName, deploymentName string, s schemasv2.UpdateDeploymentSchema) (*http.Response, *schemasv2.DeploymentSchema) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Create the PATCH request with JSON data
	req, err := http.NewRequest(http.MethodPut, c.Url+"/api/v2/deployments/"+deploymentName+"?cluster="+clusterName, bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed create update request %s", err.Error())
	}

	// Set the appropriate headers
	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	// Create an HTTP client and send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to update deployment %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema schemasv2.DeploymentSchema
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, &deploymentSchema
}

func (c *CompoundAIClient) GetDeploymentV2(t *testing.T, clusterName, deploymentName string) (*http.Response, *schemasv2.DeploymentSchema) {
	req, err := http.NewRequest(http.MethodGet, c.Url+"/api/v2/deployments/"+deploymentName+"?cluster="+clusterName, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to get deployment: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema schemasv2.DeploymentSchema
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &deploymentSchema
}

func (c *CompoundAIClient) TerminateDeploymentV2(t *testing.T, clusterName, deploymentName string) (*http.Response, *schemasv2.DeploymentSchema) {
	req, err := http.NewRequest(http.MethodPost, c.Url+"/api/v2/deployments/"+deploymentName+"/terminate"+"?cluster="+clusterName, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to terminate deployment: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var terminateResponse schemasv2.DeploymentSchema
	if err = json.Unmarshal(respBody, &terminateResponse); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, &terminateResponse
}

func (c *CompoundAIClient) DeleteDeploymentV2(t *testing.T, clusterName, deploymentName string) (*http.Response, *schemasv2.DeploymentSchema) {
	req, err := http.NewRequest(http.MethodDelete, c.Url+"/api/v2/deployments/"+deploymentName+"?cluster="+clusterName, nil)
	if err != nil {
		t.Fatalf("Failed to create HTTP request: %v", err)
	}
	for key, values := range c.Headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to delete deployment: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deleteResponse *schemasv2.DeploymentSchema
	if err = json.Unmarshal(respBody, &deleteResponse); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}

	return resp, deleteResponse
}
