package consts

import (
	"errors"

	"gorm.io/gorm"
)

var (
	ErrNotFound      = gorm.ErrRecordNotFound
	ErrNoPermission  = errors.New("no permission")
	ErrEmptyData     = errors.New("data is nil")
	ErrNoImplemented = errors.New("no implemented")
	ErrTimeout       = errors.New("timeout")

	YataiOrganizationHeaderName = "X-Yatai-Organization"
	NgcOrganizationHeaderName   = "Nv-Ngc-Org"
	NgcUserHeaderName           = "Nv-Actor-Id"

	CompoundNimContainerPortName = "http"
)
