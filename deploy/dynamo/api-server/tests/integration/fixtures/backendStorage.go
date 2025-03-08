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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

type MockedBackendService struct {
	backendURL string
}

func (s *MockedBackendService) GetDynamoNimVersion(ctx context.Context, dynamoNim string, dynamoNimVersion string) (*schemas.DynamoNimVersionFullSchema, error) {
	return nil, nil
}

func (s *MockedBackendService) GetBackendURL() string {
	return s.backendURL
}

// MockBackendServer represents a mock backend server for testing
type MockBackendServer struct {
	Server      *httptest.Server
	router      *gin.Engine
	throwsError bool
	mutex       sync.Mutex
}

// CreateMockBackendServer creates a new mock backend server for testing
func CreateMockBackendServer(t *testing.T) *MockBackendServer {
	gin.SetMode(gin.TestMode)
	router := gin.New()

	// Add middleware to log all requests
	router.Use(func(c *gin.Context) {
		log.Info().Msgf("Mock server received request: %s %s", c.Request.Method, c.Request.URL.Path)
		c.Next()
	})

	mockServer := &MockBackendServer{
		router:      router,
		throwsError: false,
	}

	// DynamoNim routes
	router.GET("/api/v1/dynamo_nims", mockServer.handleListDynamoNims)
	router.POST("/api/v1/dynamo_nims", mockServer.handleCreateDynamoNim)
	router.GET("/api/v1/dynamo_nims/:dynamoNimName", mockServer.handleGetDynamoNim)
	router.PATCH("/api/v1/dynamo_nims/:dynamoNimName", mockServer.handleUpdateDynamoNim)

	// DynamoNim version routes
	router.GET("/api/v1/dynamo_nims/:dynamoNimName/versions", mockServer.handleListDynamoNimVersions)
	router.POST("/api/v1/dynamo_nims/:dynamoNimName/versions", mockServer.handleCreateDynamoNimVersion)
	router.GET("/api/v1/dynamo_nims/:dynamoNimName/versions/:version", mockServer.handleGetDynamoNimVersion)
	router.PATCH("/api/v1/dynamo_nims/:dynamoNimName/versions/:version/update_image_build_status", mockServer.handleUpdateImageBuildStatus)
	router.PATCH("/api/v1/dynamo_nims/:dynamoNimName/versions/:version/update_image_build_status_syncing_at", mockServer.handleUpdateImageBuildStatusSyncingAt)

	// Add a catch-all route with detailed logging
	router.NoRoute(func(c *gin.Context) {
		log.Warn().Msgf("⚠️ No route found for %s %s - returning mock success response", c.Request.Method, c.Request.URL.Path)

		// Return a generic success response based on the request method
		if c.Request.Method == "GET" {
			if strings.Contains(c.Request.URL.Path, "deployments") {
				// For deployment-related GET requests
				c.JSON(http.StatusOK, map[string]interface{}{
					"items": []map[string]interface{}{
						{
							"name":            "test-deployment",
							"namespace":       "default",
							"description":     "Test deployment",
							"status":          "ACTIVE",
							"cluster":         "test-cluster",
							"organization_id": c.GetHeader("X-NGC-Organization"),
							"latest_revision": map[string]interface{}{
								"uid":    "rev-123",
								"status": "ACTIVE",
							},
						},
					},
					"total": 1,
				})
			} else {
				// For other GET requests
				c.JSON(http.StatusOK, map[string]interface{}{
					"status": "success",
					"data": map[string]interface{}{
						"message": "Mock success response",
					},
				})
			}
		} else {
			// For non-GET requests
			c.JSON(http.StatusOK, map[string]interface{}{
				"status": "success",
				"data": map[string]interface{}{
					"message": "Mock success response",
				},
			})
		}
	})

	// Create test server
	server := httptest.NewServer(router)
	mockServer.Server = server
	os.Setenv("API_BACKEND_URL", server.URL)

	return mockServer
}

// Close shuts down the mock server
func (m *MockBackendServer) Close() {
	if m.Server != nil {
		m.Server.Close()
	}
}

// Throws sets whether the mock server should throw errors
func (m *MockBackendServer) Throws(throws bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.throwsError = throws
}

// handleGetDynamoNimVersion handles GET requests for Dynamo NIM versions
func (m *MockBackendServer) handleGetDynamoNimVersion(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	dynamoNim := c.Param("dynamoNim")
	version := c.Param("version")

	// Create a mock response
	response := schemas.DynamoNimVersionFullSchema{
		DynamoNimVersionSchema: schemas.DynamoNimVersionSchema{
			DynamoNimUid: dynamoNim,
			Version:      version,
			ResourceSchema: schemas.ResourceSchema{
				Name:         dynamoNim + ":" + version,
				ResourceType: "dynamo_nim_version",
			},
			ImageBuildStatus: "AVAILABLE",
		},
	}

	c.JSON(http.StatusOK, response)
}

// DefaultDynamoNimVersionResponse returns a default Dynamo NIM version response
func DefaultDynamoNimVersionResponse(dynamoNim, version string) []byte {
	response := schemas.DynamoNimVersionFullSchema{
		DynamoNimVersionSchema: schemas.DynamoNimVersionSchema{
			DynamoNimUid: dynamoNim,
			Version:      version,
			ResourceSchema: schemas.ResourceSchema{
				Name:         dynamoNim + ":" + version,
				ResourceType: "dynamo_nim_version",
			},
			ImageBuildStatus: "AVAILABLE",
		},
	}

	jsonBytes, _ := json.Marshal(response)
	return jsonBytes
}

// Add the new handler functions
func (m *MockBackendServer) handleListDynamoNims(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	// Get the organization ID from the header
	orgID := c.GetHeader("X-NGC-Organization")

	// Create a mock response
	response := map[string]interface{}{
		"items": []map[string]interface{}{
			{
				"name":            "test-nim",
				"description":     "Test Dynamo NIM",
				"status":          "ACTIVE",
				"organization_id": orgID,
			},
		},
		"total": 1,
	}

	c.JSON(http.StatusOK, response)
}

// Add the missing handler methods for DynamoNim operations

// handleCreateDynamoNim handles POST requests to create a Dynamo NIM
func (m *MockBackendServer) handleCreateDynamoNim(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	var requestBody map[string]interface{}
	if err := c.ShouldBindJSON(&requestBody); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	// Create a mock response
	response := map[string]interface{}{
		"name":        requestBody["name"],
		"description": requestBody["description"],
		"status":      "ACTIVE",
	}

	c.JSON(http.StatusOK, response)
}

// handleGetDynamoNim handles GET requests for a specific Dynamo NIM
func (m *MockBackendServer) handleGetDynamoNim(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	dynamoNimName := c.Param("dynamoNimName")

	// Create a mock response
	response := map[string]interface{}{
		"name":        dynamoNimName,
		"description": "Test DynamoNim",
		"status":      "ACTIVE",
	}

	c.JSON(http.StatusOK, response)
}

// handleUpdateDynamoNim handles PATCH requests to update a Dynamo NIM
func (m *MockBackendServer) handleUpdateDynamoNim(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	dynamoNimName := c.Param("dynamoNimName")

	var requestBody map[string]interface{}
	if err := c.ShouldBindJSON(&requestBody); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	// Create a mock response
	response := map[string]interface{}{
		"name":        dynamoNimName,
		"description": requestBody["description"],
		"status":      "ACTIVE",
	}

	c.JSON(http.StatusOK, response)
}

// handleListDynamoNimVersions handles GET requests for Dynamo NIM versions
func (m *MockBackendServer) handleListDynamoNimVersions(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	dynamoNimName := c.Param("dynamoNimName")

	// Create a mock response
	response := map[string]interface{}{
		"items": []map[string]interface{}{
			{
				"dynamo_nim_uid": dynamoNimName,
				"version":         "1.0.0",
				"status":          "ACTIVE",
			},
		},
		"total": 1,
	}

	c.JSON(http.StatusOK, response)
}

// handleCreateDynamoNimVersion handles POST requests to create a Dynamo NIM version
func (m *MockBackendServer) handleCreateDynamoNimVersion(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	dynamoNimName := c.Param("dynamoNimName")

	var requestBody map[string]interface{}
	if err := c.ShouldBindJSON(&requestBody); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	// Create a mock response
	response := map[string]interface{}{
		"dynamo_nim_uid": dynamoNimName,
		"version":         requestBody["version"],
		"status":          "ACTIVE",
	}

	c.JSON(http.StatusOK, response)
}

// handleUpdateImageBuildStatus handles PATCH requests to update image build status
func (m *MockBackendServer) handleUpdateImageBuildStatus(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	dynamoNimName := c.Param("dynamoNimName")
	version := c.Param("version")

	var requestBody map[string]interface{}
	if err := c.ShouldBindJSON(&requestBody); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	// Create a mock response
	response := map[string]interface{}{
		"dynamo_nim_uid":   dynamoNimName,
		"version":          version,
		"image_build_status": requestBody["image_build_status"],
	}

	c.JSON(http.StatusOK, response)
}

// handleUpdateImageBuildStatusSyncingAt handles PATCH requests to update image build status syncing time
func (m *MockBackendServer) handleUpdateImageBuildStatusSyncingAt(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	dynamoNimName := c.Param("dynamoNimName")
	version := c.Param("version")

	// Create a mock response
	response := map[string]interface{}{
		"dynamo_nim_uid":              dynamoNimName,
		"version":                       version,
		"image_build_status_syncing_at": "2025-03-07T10:39:40Z",
	}

	c.JSON(http.StatusOK, response)
}

// handleGetVersion handles GET requests for the API version
func (m *MockBackendServer) handleGetVersion(c *gin.Context) {
	m.mutex.Lock()
	throwsError := m.throwsError
	m.mutex.Unlock()

	if throwsError {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	// Create a mock response
	response := map[string]interface{}{
		"version": "1.0.0",
	}

	c.JSON(http.StatusOK, response)
}

// MockBackendClient provides a client for direct access to the mock backend server
type MockBackendClient struct {
	Server *MockBackendServer
}

func (m *MockBackendServer) Reset() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.throwsError = false
}
