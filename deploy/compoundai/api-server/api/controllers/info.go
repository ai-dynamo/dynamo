package controllers

import (
	"github.com/gin-gonic/gin"
)

type infoController struct{}

var InfoController = infoController{}

type InfoSchema struct {
	IsSaas           bool   `json:"is_saas"`
	SaasDomainSuffix string `json:"saas_domain_suffix"`
}

func (c *infoController) GetInfo(ctx *gin.Context) {
	schema := InfoSchema{
		IsSaas:           true,
		SaasDomainSuffix: "",
	}

	ctx.JSON(200, schema)
}
