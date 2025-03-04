package controllers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type healthController struct{}

var HealthController = healthController{}

func (h *healthController) Get(gin *gin.Context) {
	gin.JSON(http.StatusOK, "ok")
}
