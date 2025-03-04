package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/mocks"
)

type userController struct{}

const CurrentUserIdKey = "currentUserId"

var UserController = userController{}

func (c *userController) GetDefaultUser(ctx *gin.Context) {
	user := mocks.DefaultUser()
	ctx.JSON(200, user)
}
