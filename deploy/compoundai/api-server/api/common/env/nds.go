package env

import (
	"fmt"
	"sync"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/utils"
)

var (
	NdsHostBase string
	once        sync.Once
)

func GetNdsUrl() string {
	baseUrl := GetNdsHost()
	return fmt.Sprintf("http://%s", baseUrl)
}

func GetNdsHost() string {
	return NdsHostBase
}

func SetNdsHost() (string, error) {
	var err error
	once.Do(func() { // We cache and reuse the same NDS host
		NDS_HOST, syncErr := utils.MustGetEnv("NDS_HOST")
		if syncErr != nil {
			err = syncErr
			return
		}

		NDS_PORT, syncErr := utils.MustGetEnv("NDS_PORT")
		if syncErr != nil {
			err = syncErr
			return
		}

		NdsHostBase = fmt.Sprintf("%s:%s", NDS_HOST, NDS_PORT)
	})

	if err != nil {
		return "", err
	}

	return NdsHostBase, nil
}
