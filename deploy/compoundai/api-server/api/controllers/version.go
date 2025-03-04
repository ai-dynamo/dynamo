package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

var (
	Version   = "0.0.1"
	GitCommit = "HEAD"
	BuildDate = "1970-01-01T00:00:00Z"
)

type versionController struct{}

var VersionController = versionController{}

func (c *versionController) Get(ctx *gin.Context) {
	versionSchema := &schemas.VersionSchema{
		Version:   Version,
		GitCommit: GitCommit,
		BuildDate: BuildDate,
	}

	ctx.JSON(200, versionSchema)
}
