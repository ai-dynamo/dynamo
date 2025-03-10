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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/consts"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/env"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/database"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/runtime"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemas"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/schemasv2"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/services"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/tests/integration/fixtures"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
)

const (
	port                = 9999
	expectedStatusOkMsg = "Expected status code 200"
)

var apiServerUrl = fmt.Sprintf("http://localhost:%d", port)
var client = DynamoClient{
	Url:     apiServerUrl,
	Headers: http.Header{},
}

type ApiServerSuite struct {
	suite.Suite
}

func TestApiServerSuite(t *testing.T) {
	suite.Run(t, new(ApiServerSuite))
}

// run once, before test suite methods
func (s *ApiServerSuite) SetupSuite() {
	log.Info().Msgf("Starting suite...")
	_, err := testContainers.CreatePostgresContainer()
	if err != nil {
		s.T().FailNow()
	}
	log.Info().Msgf("Created Postgres Container")

	// Setup server
	go func() {
		// Mute all logs for this goroutine
		gin.DefaultWriter = io.Discard
		runtime.Runtime.StartServer(port)
	}()

	s.waitUntilReady()
	log.Info().Msgf("API Server Ready")

	services.K8sService = &fixtures.MockedK8sService{}
	log.Info().Msgf("Mocked K8s Service")
}

func (s *ApiServerSuite) waitUntilReady() {
	url := fmt.Sprintf("%s/healthz", apiServerUrl)
	for {
		resp, err := http.Get(url)
		if err == nil && resp.StatusCode == http.StatusOK {
			log.Info().Msg("Dynamo API server is running")
			return // Server is ready
		}
		log.Info().Msgf("Waiting 500ms before checking /healthz again")
		time.Sleep(500 * time.Millisecond) // Wait before retrying
	}
}

// run once, after test suite methods
func (s *ApiServerSuite) TearDownSuite() {
	testContainers.TearDownPostgresContainer()
}

// run before every test
func (s *ApiServerSuite) SetupTest() {
	client.Headers.Set(consts.NgcOrganizationHeaderName, "test-org-nvidia")
	client.Headers.Set(consts.NgcUserHeaderName, "test-user-nvidia")

	env.ApplicationScope = env.UserScope
}

// run after each test
func (s *ApiServerSuite) TearDownTest() {
	ctx := context.Background()
	db := database.DatabaseUtil.GetDBSession(ctx)

	if err := db.Unscoped().Where("true").Delete(&models.Deployment{}).Error; err != nil {
		s.T().Fatalf("Failed to delete records from deployment table: %v", err)
	}

	if err := db.Unscoped().Where("true").Delete(&models.Cluster{}).Error; err != nil {
		s.T().Fatalf("Failed to delete records from cluster table: %v", err)
	}
}

func (s *ApiServerSuite) TestCreateCluster() {
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, clusterFullSchema := client.CreateCluster(s.T(), cluster)

	// Verify the response status code
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Additional checks on response content (optional)
	assert.Equal(s.T(), clusterFullSchema.Description, cluster.Description)
	assert.Equal(s.T(), *(clusterFullSchema.KubeConfig), cluster.KubeConfig)
	assert.Equal(s.T(), clusterFullSchema.Name, cluster.Name)
}

func (s *ApiServerSuite) TestGetCluster() {
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Get Cluster
	resp, clusterFullSchema := client.GetCluster(s.T(), cluster.Name)

	// Verify the response status code
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Additional checks on response content
	assert.Equal(s.T(), clusterFullSchema.Description, cluster.Description)
	assert.Equal(s.T(), *(clusterFullSchema.KubeConfig), cluster.KubeConfig)
	assert.Equal(s.T(), clusterFullSchema.Name, cluster.Name)
}

func (s *ApiServerSuite) TestGetUnknownClusterFails() {
	resp, _ := client.GetCluster(s.T(), "unknown")
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 404")
}

func (s *ApiServerSuite) TestGetMultipleClusters() {
	cluster1 := fixtures.DefaultCreateClusterSchema()
	cluster1.Name = "c1"
	resp, _ := client.CreateCluster(s.T(), cluster1)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	cluster2 := fixtures.DefaultCreateClusterSchema()
	cluster2.Name = "c2"
	resp, _ = client.CreateCluster(s.T(), cluster2)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	req := fixtures.DefaultListQuerySchema()
	resp, clusterListSchema := client.GetClusterList(s.T(), req)

	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	for _, item := range clusterListSchema.Items {
		assert.Contains(s.T(), []string{"c1", "c2"}, item.Name, expectedStatusOkMsg)
	}
}

func (s *ApiServerSuite) TestUpdateCluster() {
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	req := fixtures.DefaultUpdateClusterSchema()
	d := "Nemo"
	kc := "KcNemo"
	req.Description = &d
	req.KubeConfig = &kc

	resp, clusterFullSchema := client.UpdateCluster(s.T(), cluster.Name, req)

	// Verify the response status code
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Additional checks on response content (optional)
	assert.Equal(s.T(), clusterFullSchema.Description, *(req.Description))
	assert.Equal(s.T(), *(clusterFullSchema.KubeConfig), *(req.KubeConfig))
	assert.Equal(s.T(), clusterFullSchema.Name, cluster.Name)
}

func (s *ApiServerSuite) TestCreateDeployment() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, deploymentSchema := client.GetDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	assert.Equal(s.T(), deployment.Name, deploymentSchema.Name)
	assert.Equal(s.T(), schemas.DeploymentStatusNonDeployed, deploymentSchema.Status)
	assert.Equalf(s.T(), int(1), len(deploymentSchema.LatestRevision.Targets), "expected 1 target")
	assert.Equal(s.T(), deployment.Targets[0].Version, deploymentSchema.LatestRevision.Targets[0].DynamoNimVersion.Version)
}

func (s *ApiServerSuite) TestCreateDeploymentWithNDSErrorDoesNotChangeDB() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	d1, r1, t1, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}

	nds.Throws(true)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode)

	d2, r2, t2, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}

	assert.True(s.T(), compareDeployments(d1, d2))
	assert.True(s.T(), compareDeploymentRevisions(r1, r2))
	assert.True(s.T(), compareDeploymentTargets(t1, t2))
}

func (s *ApiServerSuite) TestCreateDeploymentUnknownClusterFails() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ := client.CreateDeployment(s.T(), "unknown", deployment)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestUpdateDeployment() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	// Create cluster
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Create deployment
	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Update deployment
	updateTarget := fixtures.DefaultCreateDeploymentTargetSchema()
	updateTarget.Version = "2025"
	updateTarget.DynamoNim = "dynamo"

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	updatedDescription := "new description"
	updateDeployment.Description = &updatedDescription

	updateDeployment.Targets = []*schemas.CreateDeploymentTargetSchema{
		updateTarget,
	}

	resp, deploymentSchema := client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Validate fields
	assert.Equal(s.T(), deployment.Name, deploymentSchema.Name)
	assert.Equal(s.T(), schemas.DeploymentStatusNonDeployed, deploymentSchema.Status)

	// Todo: once Deployment Schema is available make this test more rigorous
	ctx := context.Background()
	_, r1, _, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}

	assert.Equal(s.T(), 2, len(r1))

	status_ := schemas.DeploymentRevisionStatusActive
	activeRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}
	assert.Equal(s.T(), 1, len(activeRevisions))

	assert.Equal(s.T(), deployment.Name, deploymentSchema.Name)
	assert.Equal(s.T(), schemas.DeploymentStatusNonDeployed, deploymentSchema.Status)
	assert.Equalf(s.T(), int(1), len(deploymentSchema.LatestRevision.Targets), "expected 1 target")
	assert.Equal(s.T(), updateDeployment.Targets[0].Version, deploymentSchema.LatestRevision.Targets[0].DynamoNimVersion.Version)
}

func (s *ApiServerSuite) TestUpdateDeploymentWithNDSErrorDoesNotChangeDB() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	d1, r1, t1, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}

	nds.Throws(true)

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusInternalServerError, resp.StatusCode)

	d2, r2, t2, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}

	assert.True(s.T(), compareDeployments(d1, d2))
	assert.True(s.T(), compareDeploymentRevisions(r1, r2))
	assert.True(s.T(), compareDeploymentTargets(t1, t2))
}

func (s *ApiServerSuite) TestUpdateDeploymentWithoutDeployment() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	// Create cluster
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Create deployment
	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Update deployment
	updateTarget := fixtures.DefaultCreateDeploymentTargetSchema()
	updateTarget.Config.KubeResourceUid = "abc123"
	updateTarget.Config.KubeResourceVersion = "alphav1"

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	updateDeployment.Targets = []*schemas.CreateDeploymentTargetSchema{
		updateTarget,
	}
	updateDeployment.DoNotDeploy = true

	ctx := context.Background()
	_, r1, _, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}

	resp, deploymentSchema := client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Validate fields
	assert.Equal(s.T(), deployment.Name, deploymentSchema.Name)
	assert.Equal(s.T(), schemas.DeploymentStatusNonDeployed, deploymentSchema.Status)
	assert.Equalf(s.T(), 1, len(deploymentSchema.LatestRevision.Targets), "More deployment targets than expected")
	assert.Equal(s.T(), updateTarget.Config.KubeResourceUid, deploymentSchema.LatestRevision.Targets[0].Config.KubeResourceUid)
	assert.Equal(s.T(), updateTarget.Config.KubeResourceVersion, deploymentSchema.LatestRevision.Targets[0].Config.KubeResourceVersion)

	// Updating without deployment does not deactivate any deployment revision or create a new one
	_, r2, _, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}
	assert.True(s.T(), compareDeploymentRevisions(r1, r2))
}

func (s *ApiServerSuite) TestUpdateDeploymentUnknownClusterFails() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ := client.UpdateDeployment(s.T(), "unknown", "default", "unknown", updateDeployment)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 404")
}

func (s *ApiServerSuite) TestUpdateDeploymentUnknownDeploymentFails() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	// Create cluster
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, "default", "unknown", updateDeployment)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 404")
}

func (s *ApiServerSuite) TestTerminateDeployment() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	status_ := schemas.DeploymentRevisionStatusActive
	activeRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(activeRevisions))

	// Terminate deployment
	resp, deploymentSchema := client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	status_ = schemas.DeploymentRevisionStatusInactive
	inactiveRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(inactiveRevisions))

	var expectedRevision *schemas.DeploymentRevisionSchema = nil
	assert.Equal(s.T(), expectedRevision, deploymentSchema.LatestRevision)
}

func (s *ApiServerSuite) TestTerminateNonExistingDeployment() {
	resp, _ := client.TerminateDeployment(s.T(), "nonexistent-cluster", "nonexistent-namespace", "nonexistent-deployment")
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 404")
}

func (s *ApiServerSuite) TestTerminateNonIncorrectDeployment() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, _ = client.TerminateDeployment(s.T(), "nonexistent-cluster", deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 404")

	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, "nonexistent-namespace", deployment.Name)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 404")

	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, "nonexistent-deployment")
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, "Expected status code 404")
}

func (s *ApiServerSuite) TestDeleteDeactivatedDeployment() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Terminate the deployment
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Delete the deactivated deployment
	resp, _ = client.DeleteDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Check that there are no remaining deployment entities
	ctx := context.Background()
	d, r, t, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}
	assert.Equal(s.T(), 0, len(d))
	assert.Equal(s.T(), 0, len(r))
	assert.Equal(s.T(), 0, len(t))
}

func (s *ApiServerSuite) TestDeleteActiveDeploymentFails() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Attempt to delete the active deployment
	resp, _ = client.DeleteDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusInternalServerError, resp.StatusCode)
}

func (s *ApiServerSuite) TestUpdateDeploymentWithDMSErrorDoesNotChangeDB() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	d1, r1, t1, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}
	dms.Throws(true)

	// Attempt to update the deployment
	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusInternalServerError, resp.StatusCode)

	d2, r2, t2, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}

	assert.True(s.T(), compareDeployments(d1, d2))
	assert.True(s.T(), compareDeploymentRevisions(r1, r2))
	assert.True(s.T(), compareDeploymentTargets(t1, t2))
}

func (s *ApiServerSuite) TestTerminateDeploymentWithDMSErrorDoesNotChangeDB() {
	dms := fixtures.CreateMockDMSServer(s.T()) // Does not throw error initially
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	d1, r1, t1, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}
	dms.Throws(true)

	// Attempt to terminate the deployment
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusInternalServerError, resp.StatusCode)

	// Verify DB state remains unchanged
	d2, r2, t2, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}

	assert.True(s.T(), compareDeployments(d1, d2))
	assert.True(s.T(), compareDeploymentRevisions(r1, r2))
	assert.True(s.T(), compareDeploymentTargets(t1, t2))
}

func (s *ApiServerSuite) TestGetDeploymentRevisions() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, deploymentSchema := client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, deploymentRevisionSchema := client.GetDeploymentRevision(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, deploymentSchema.LatestRevision.Uid)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	assert.Equal(s.T(), deploymentSchema.LatestRevision.Uid, deploymentRevisionSchema.Uid)
	lr, err := json.Marshal(deploymentSchema.LatestRevision)
	if err != nil {
		s.T().Fatalf("%s", err.Error())
	}

	ar, err := json.Marshal(deploymentSchema.LatestRevision)
	if err != nil {
		s.T().Fatalf("%s", err.Error())
	}

	assert.Equal(s.T(), lr, ar)
}

func (s *ApiServerSuite) TestGetDeploymentRevisionsDifferentDeployments() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	d1 := fixtures.DefaultCreateDeploymentSchema()
	d1.Name = "dep1"
	resp, ds1 := client.CreateDeployment(s.T(), cluster.Name, d1)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	d2 := fixtures.DefaultCreateDeploymentSchema()
	d2.Name = "dep2"
	resp, ds2 := client.CreateDeployment(s.T(), cluster.Name, d2)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, drs1 := client.GetDeploymentRevision(s.T(), cluster.Name, d1.KubeNamespace, d1.Name, ds1.LatestRevision.Uid)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, drs2 := client.GetDeploymentRevision(s.T(), cluster.Name, d1.KubeNamespace, d1.Name, ds2.LatestRevision.Uid)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	assert.NotEqual(s.T(), drs2.Uid, drs1.Uid)
}

func (s *ApiServerSuite) TestGetDeploymentRevisionsList() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	query := schemas.ListQuerySchema{
		Count: 10,
	}
	resp, deploymentRevisionsListSchema := client.GetDeploymentRevisionList(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, query)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)
	assert.Equal(s.T(), 2, len(deploymentRevisionsListSchema.Items))
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForClusterWhenCreatingDeployment() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "some-other-org")
	deployment2 := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment2)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode)
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForClusterWhenUpdatingDeployment() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "some-other-org")
	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode)
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForClusterWhenTerminatingDeployment() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Terminate deployment
	client.Headers.Set(consts.NgcOrganizationHeaderName, "some-other-org")
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, expectedStatusOkMsg)
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForClusterWhenDeletingDeployment() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Terminate the deployment
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Delete the deactivated deployment
	client.Headers.Set(consts.NgcOrganizationHeaderName, "some-other-org")
	resp, _ = client.DeleteDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, expectedStatusOkMsg)
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForClusterWhenListingDeploymentRevisions() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "some-other-org")

	query := schemas.ListQuerySchema{}
	resp, _ = client.GetDeploymentRevisionList(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, query)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, expectedStatusOkMsg)
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForGetClusterDeployments() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	client.Headers.Set(consts.NgcOrganizationHeaderName, "org-1")
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "org-2")
	cluster = fixtures.DefaultCreateClusterSchema()
	resp, _ = client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment = fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "org-1")
	resp, deploymentSchemas := client.GetClusterDeploymentList(s.T(), cluster.Name, schemas.ListQuerySchema{Count: 10})
	assert.Equalf(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)
	assert.Equal(s.T(), 1, len(deploymentSchemas.Items))
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForGetDeployments() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	client.Headers.Set(consts.NgcOrganizationHeaderName, "org-1")
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "org-2")
	cluster = fixtures.DefaultCreateClusterSchema()
	resp, _ = client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment = fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "org-1")
	resp, deploymentSchemas := client.GetDeploymentList(s.T(), schemas.ListQuerySchema{Count: 10})
	assert.Equalf(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)
	assert.Equal(s.T(), 1, len(deploymentSchemas.Items))
}

func (s *ApiServerSuite) TestUserLevelScopeForClusterWhenCreatingDeployment() {
	env.ApplicationScope = env.UserScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	deployment.Name = "dep1"
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "some-other-user")
	deployment2 := fixtures.DefaultCreateDeploymentSchema()
	deployment2.Name = "dep2"
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment2)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode)

	resp, deploymentSchemas := client.GetClusterDeploymentList(s.T(), cluster.Name, schemas.ListQuerySchema{Count: 10})
	assert.Equalf(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)
	assert.Equal(s.T(), 1, len(deploymentSchemas.Items))
}

func (s *ApiServerSuite) TestUserLevelScopeForClusterWhenUpdatingDeployment() {
	env.ApplicationScope = env.UserScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "some-other-user")
	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode)
}

func (s *ApiServerSuite) TestUserLevelScopeForClusterWhenTerminatingDeployment() {
	env.ApplicationScope = env.UserScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "some-other-user")
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, expectedStatusOkMsg)
}

func (s *ApiServerSuite) TestUserLevelScopeForClusterWhenDeletingDeployment() {
	env.ApplicationScope = env.UserScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "some-other-user")
	resp, _ = client.DeleteDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, expectedStatusOkMsg)
}

func (s *ApiServerSuite) TestUserLevelScopeForClusterWhenListingDeploymentRevisions() {
	env.ApplicationScope = env.UserScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "some-other-user")

	query := schemas.ListQuerySchema{}
	resp, _ = client.GetDeploymentRevisionList(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, query)
	assert.Equal(s.T(), http.StatusNotFound, resp.StatusCode, expectedStatusOkMsg)
}

func (s *ApiServerSuite) TestUserLevelScopeForGetClusterDeployments() {
	env.ApplicationScope = env.UserScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	client.Headers.Set(consts.NgcUserHeaderName, "user-1")
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	deployment.Name = "dep1"
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "user-2")
	deployment = fixtures.DefaultCreateDeploymentSchema()
	deployment.Name = "dep2"
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "user-1")
	resp, deploymentSchemas := client.GetClusterDeploymentList(s.T(), cluster.Name, schemas.ListQuerySchema{Count: 10})
	assert.Equalf(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)
	assert.Equal(s.T(), 1, len(deploymentSchemas.Items))
}

func (s *ApiServerSuite) TestUserLevelScopeForGetDeployments() {
	env.ApplicationScope = env.UserScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	client.Headers.Set(consts.NgcUserHeaderName, "user-1")
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	deployment.Name = "dep1"
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "user-2")
	deployment = fixtures.DefaultCreateDeploymentSchema()
	deployment.Name = "dep2"
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcUserHeaderName, "user-1")
	resp, deploymentSchemas := client.GetDeploymentList(s.T(), schemas.ListQuerySchema{Count: 10})
	assert.Equalf(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)
	assert.Equal(s.T(), 1, len(deploymentSchemas.Items))
}

func (s *ApiServerSuite) TestOrganizationLevelScopeForListClusters() {
	env.ApplicationScope = env.OrganizationScope
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	client.Headers.Set(consts.NgcOrganizationHeaderName, "some-other-org")
	resp, _ = client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	req := fixtures.DefaultListQuerySchema()
	resp, clusterListSchema := client.GetClusterList(s.T(), req)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)
	assert.Equal(s.T(), 1, len(clusterListSchema.Items))
}

// V2 API Tests
func (s *ApiServerSuite) TestCreateDeploymentV2() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchemaV2()
	resp, _ = client.CreateDeploymentV2(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, deploymentSchema := client.GetDeploymentV2(s.T(), cluster.Name, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	assert.Equal(s.T(), deployment.Name, deploymentSchema.Name)
	assert.Equal(s.T(), schemas.DeploymentStatusNonDeployed, deploymentSchema.Status)
	assert.Equalf(s.T(), int(1), len(deploymentSchema.LatestRevision.Targets), "expected 1 target")
	assert.Equal(s.T(), deployment.Services["default-service"].ConfigOverrides.Resources, *deploymentSchema.LatestRevision.Targets[0].Config.Resources)
}

func (s *ApiServerSuite) TestUpdateDeploymentV2() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	// Create cluster
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Create deployment
	deployment := fixtures.DefaultCreateDeploymentSchemaV2()
	resp, _ = client.CreateDeploymentV2(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Update deployment
	resourceItem := &schemas.ResourceItem{
		CPU:    "123m",
		GPU:    "2",
		Memory: "5Gi",
	}

	updateService := fixtures.DefaultServiceSpec()
	updateService.ConfigOverrides.Resources.Limits = resourceItem
	updateService.ConfigOverrides.Resources.Requests = resourceItem

	updateDeployment := fixtures.DefaultUpdateDeploymentSchemaV2()
	updateDeployment.DynamoNim = "new:654321"
	updateDeployment.DeploymentConfigSchema.Services = map[string]schemasv2.ServiceSpec{
		"new-service": updateService,
	}

	resp, deploymentSchema := client.UpdateDeploymentV2(s.T(), cluster.Name, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	assert.Equal(s.T(), deployment.Name, deploymentSchema.Name)
	assert.Equal(s.T(), schemas.DeploymentStatusNonDeployed, deploymentSchema.Status)
	assert.Equalf(s.T(), int(1), len(deploymentSchema.LatestRevision.Targets), "expected 1 target")
	assert.Equal(s.T(), updateDeployment.Services["new-service"].ConfigOverrides.Resources, *deploymentSchema.LatestRevision.Targets[0].Config.Resources)
}

func (s *ApiServerSuite) TestTerminateDeploymentV2() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchemaV2()
	resp, _ = client.CreateDeploymentV2(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	status_ := schemas.DeploymentRevisionStatusActive
	activeRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(activeRevisions))

	// Terminate deployment
	resp, deploymentSchema := client.TerminateDeploymentV2(s.T(), cluster.Name, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	status_ = schemas.DeploymentRevisionStatusInactive
	inactiveRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(inactiveRevisions))

	var expectedRevision *schemas.DeploymentRevisionSchema = nil
	assert.Equal(s.T(), expectedRevision, deploymentSchema.LatestRevision)
}

func (s *ApiServerSuite) TestDeleteDeactivatedDeploymentV2() {
	dms := fixtures.CreateMockDMSServer(s.T())
	defer dms.Close()
	nds := fixtures.CreateMockNDSServer(s.T())
	defer nds.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchemaV2()
	resp, _ = client.CreateDeploymentV2(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Terminate the deployment
	resp, _ = client.TerminateDeploymentV2(s.T(), cluster.Name, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Delete the deactivated deployment
	resp, _ = client.DeleteDeploymentV2(s.T(), cluster.Name, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Check that there are no remaining deployment entities
	ctx := context.Background()
	d, r, t, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}
	assert.Equal(s.T(), 0, len(d))
	assert.Equal(s.T(), 0, len(r))
	assert.Equal(s.T(), 0, len(t))
}

func compareDeployments(slice1, slice2 []*models.Deployment) bool {
	// Check if lengths are equal
	if len(slice1) != len(slice2) {
		return false
	}

	// Compare each element using reflect.DeepEqual
	for i := range slice1 {
		if !reflect.DeepEqual(slice1[i], slice2[i]) {
			log.Info().Msgf("Expected deployment: %+v", slice1[i])
			log.Info().Msgf("Actual deployment: %+v", slice2[i])
			return false
		}
	}

	return true
}

func compareDeploymentRevisions(slice1, slice2 []*models.DeploymentRevision) bool {
	// Check if lengths are equal
	if len(slice1) != len(slice2) {
		return false
	}

	// Compare each element using reflect.DeepEqual
	for i := range slice1 {
		if !reflect.DeepEqual(slice1[i], slice2[i]) {
			log.Info().Msgf("Expected revision: %+v", slice1[i])
			log.Info().Msgf("Actual revision: %+v", slice2[i])
			return false
		}
	}

	return true
}

func compareDeploymentTargets(slice1, slice2 []*models.DeploymentTarget) bool {
	// Check if lengths are equal
	if len(slice1) != len(slice2) {
		return false
	}

	// Compare each element using reflect.DeepEqual
	for i := range slice1 {
		if !reflect.DeepEqual(slice1[i], slice2[i]) {
			log.Info().Msgf("Expected target: %+v", slice1[i])
			log.Info().Msgf("Actual target: %+v", slice2[i])
			return false
		}
	}

	return true
}

func getDeploymentEntitiesSnapshot(ctx context.Context) ([]*models.Deployment, []*models.DeploymentRevision, []*models.DeploymentTarget, error) {
	deployments, _, err := services.DeploymentService.List(ctx, services.ListDeploymentOption{})
	if err != nil {
		return nil, nil, nil, err
	}

	revisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{})
	if err != nil {
		return nil, nil, nil, err
	}

	targets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{})
	if err != nil {
		return nil, nil, nil, err
	}

	return deployments, revisions, targets, nil
}
