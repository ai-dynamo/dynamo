package controllers

import (
	"errors"

	"github.com/gin-gonic/gin"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/consts"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

const OwnershipInfoKey = "_ownershipInfoKey"

func GetOwnershipInfo(ctx *gin.Context) (*schemas.OwnershipSchema, error) {
	ownership_ := ctx.Value(OwnershipInfoKey)
	if ownership_ == nil {
		return nil, consts.ErrNotFound
	}

	ownership, ok := ownership_.(*schemas.OwnershipSchema)
	if !ok {
		return nil, errors.New("current ownership is not an ownership struct")
	}

	return ownership, nil
}
