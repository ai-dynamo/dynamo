package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/mocks"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

type organizationMemberController struct {
	organizationController
}

var OrganizationMemberController = organizationMemberController{}

func (c *organizationMemberController) Create(ctx *gin.Context) {
	ctx.JSON(501, gin.H{"error": "not supported."})
}

func (c *organizationMemberController) List(ctx *gin.Context) {
	organizationMemberSchemaList := []*schemas.OrganizationMemberSchema{mocks.DefaultOrgMember()}
	ctx.JSON(200, organizationMemberSchemaList)
}

func (c *organizationMemberController) Delete(ctx *gin.Context) {
	ctx.JSON(501, gin.H{"error": "not supported."})
}
