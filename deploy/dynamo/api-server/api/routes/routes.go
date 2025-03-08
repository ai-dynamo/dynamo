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

package routes

import (
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/controllers"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/mocks"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
)

func SetupRouter() *gin.Engine {
	router := gin.Default()
	router.Use(injectCurrentOrganization)

	baseGroup := router.Group("")
	createK8sRoutes(baseGroup)

	api := router.Group("/api")
	api.Use(getAuthInfo)
	v1 := api.Group("/v1")
	v2 := api.Group("/v2")

	/* Start V1 APIs */
	createClusterRoutes(v1)

	// These routes are proxied to NDS
	createDynamoNimRoutes(v1)
	createBentoRepositoriesRoutes(v1)

	createMiscellaneousRoutes(v1)
	createMockedRoutes(v1)
	createOrganizationRoutes(v1)
	createPublicRoutes(v1)
	/* End V1 APIs */

	/* Start V2 APIs */
	deploymentRoutesV2(v2)
	/* End V2 APIs */

	return router
}

func createK8sRoutes(grp *gin.RouterGroup) {
	healthGroup := grp.Group("/healthz")
	healthGroup.GET("", controllers.HealthController.Get)

	readyGroup := grp.Group("/readyz")
	readyGroup.GET("", controllers.HealthController.Get)
}

func createClusterRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/clusters")

	resourceGrp := grp.Group("/:clusterName")

	resourceGrp.GET("", controllers.ClusterController.Get)

	resourceGrp.PATCH("", controllers.ClusterController.Update)

	grp.GET("", controllers.ClusterController.List)

	grp.POST("", controllers.ClusterController.Create)

	dynamoComponentRoutes(resourceGrp)
	deploymentRoutes(resourceGrp)
}

func deploymentRoutes(grp *gin.RouterGroup) {
	namespacedGrp := grp.Group("/namespaces/:kubeNamespace/deployments")
	grp = grp.Group("/deployments")

	resourceGrp := namespacedGrp.Group("/:deploymentName")

	resourceGrp.GET("", controllers.DeploymentController.Get)

	resourceGrp.PATCH("", controllers.DeploymentController.Update)

	resourceGrp.POST("/sync_status", controllers.DeploymentController.SyncStatus)

	resourceGrp.POST("/terminate", controllers.DeploymentController.Terminate)

	resourceGrp.DELETE("", controllers.DeploymentController.Delete)

	// resourceGrp.GET("/terminal_records", controllers.DeploymentController.ListTerminalRecords)

	grp.GET("", controllers.DeploymentController.ListClusterDeployments)

	grp.POST("", controllers.DeploymentController.Create)

	deploymentRevisionRoutes(resourceGrp)
}

func deploymentRevisionRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/revisions")

	resourceGrp := grp.Group("/:revisionUid")

	resourceGrp.GET("", controllers.DeploymentRevisionController.Get)

	grp.GET("", controllers.DeploymentRevisionController.List)
}

func deploymentRoutesV2(grp *gin.RouterGroup) {
	grp = grp.Group("/deployments")
	grp.POST("", controllers.DeploymentController.CreateV2)

	resourceGrp := grp.Group("/:deploymentName")
	resourceGrp.GET("", controllers.DeploymentController.GetV2)
	resourceGrp.PUT("", controllers.DeploymentController.UpdateV2)
	resourceGrp.POST("/terminate", controllers.DeploymentController.TerminateV2)
	resourceGrp.DELETE("", controllers.DeploymentController.DeleteV2)
}

func createBentoRepositoriesRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/bento_repositories")

	resourceGrp := grp.Group("/:bentoRepositoryName")

	resourceGrp.GET("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/deployments", controllers.ProxyController.ReverseProxy)

	grp.GET("", controllers.ProxyController.ReverseProxy)

	grp.POST("", controllers.ProxyController.ReverseProxy)

	bentoRoutes(resourceGrp)
}

func bentoRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/bentos")

	resourceGrp := grp.Group("/:version")

	resourceGrp.GET("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/update_image_build_status_syncing_at", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/update_image_build_status", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/models", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/deployments", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/start_multipart_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_multipart_upload_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/complete_multipart_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_upload_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_download_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/start_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/finish_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PUT("/upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/download", controllers.ProxyController.ReverseProxy)

	grp.GET("", controllers.ProxyController.ReverseProxy)

	grp.POST("", controllers.ProxyController.ReverseProxy)
}

func createDynamoNimRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/dynamo_nims")

	resourceGrp := grp.Group("/:dynamoNimName")

	resourceGrp.GET("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/deployments", controllers.DeploymentController.ListDynamoNimDeployments)

	grp.GET("", controllers.ProxyController.ReverseProxy)

	grp.POST("", controllers.ProxyController.ReverseProxy)

	dynamoNimVersionRoutes(resourceGrp)
}

func dynamoNimVersionRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/versions")

	resourceGrp := grp.Group("/:version")

	resourceGrp.GET("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/update_image_build_status_syncing_at", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/update_image_build_status", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/models", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/deployments", controllers.DeploymentController.ListDynamoNimVersionDeployments)

	resourceGrp.PATCH("/start_multipart_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_multipart_upload_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/complete_multipart_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_upload_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_download_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/start_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/finish_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PUT("/upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/download", controllers.ProxyController.ReverseProxy)

	grp.GET("", controllers.ProxyController.ReverseProxy)

	grp.POST("", controllers.ProxyController.ReverseProxy)
}

func createMiscellaneousRoutes(grp *gin.RouterGroup) {
	versionGrp := grp.Group("/version")

	versionGrp.GET("", controllers.VersionController.Get)
}

// Legacy APIs used by the CLI
func createMockedRoutes(grp *gin.RouterGroup) {
	grp.GET("auth/current", controllers.UserController.GetDefaultUser)
}

func createOrganizationRoutes(grp *gin.RouterGroup) {
	resourceGrp := grp.Group("/current_org")

	resourceGrp.GET("", controllers.OrganizationController.Get)

	resourceGrp.GET("/major_cluster", controllers.OrganizationController.GetMajorCluster)

	resourceGrp.PATCH("", controllers.OrganizationController.Update)

	resourceGrp.GET("/events", controllers.OrganizationController.ListEvents)

	resourceGrp.GET("/event_operation_names", controllers.OrganizationController.ListEventOperationNames)

	grp.GET("/members", controllers.OrganizationMemberController.List)

	grp.POST("/members", controllers.OrganizationMemberController.Create)

	grp.DELETE("/members", controllers.OrganizationMemberController.Delete)

	grp.GET("/deployments", controllers.DeploymentController.ListDeployments)

	grp.GET("/deployment_creation_json_schema", controllers.DeploymentController.CreationJSONSchema)

	grp.GET("/yatai_components", controllers.DynamoComponentController.ListAll)

	grp.GET("/orgs", controllers.OrganizationController.List)

	grp.POST("/orgs", controllers.OrganizationController.Create)
}

func createPublicRoutes(grp *gin.RouterGroup) {
	grp.GET("/info", controllers.InfoController.GetInfo)
}

func dynamoComponentRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/yatai_components")

	grp.GET("", controllers.DynamoComponentController.List)

	grp.POST("", controllers.DynamoComponentController.Register)
}

func injectCurrentOrganization(c *gin.Context) {
	orgName := strings.TrimSpace(c.GetHeader(consts.YataiOrganizationHeaderName))
	if orgName == "" {
		orgName = strings.TrimSpace(c.Query("organization_name"))
	}
	org := mocks.DefaultOrg()

	if orgName != "" {
		org.Name = orgName
	}

	orgId := c.GetHeader(consts.NgcOrganizationHeaderName)
	if orgId == "" {
		orgId = "default"
	}
	org.Uid = orgId

	c.Set(controllers.CurrentOrganizationKey, org)
	c.Next()
}

func getAuthInfo(c *gin.Context) {
	orgId := c.GetHeader(consts.NgcOrganizationHeaderName)
	if orgId == "" {
		orgId = "default"
	}

	userId := c.GetHeader(consts.NgcUserHeaderName)
	if userId == "" {
		userId = "default"
	}

	ownership := &schemas.OwnershipSchema{
		UserId:         userId,
		OrganizationId: orgId,
	}
	log.Info().Msgf("Setting ownership info %+v", ownership)
	c.Set(controllers.OwnershipInfoKey, ownership)
	c.Next()
}
