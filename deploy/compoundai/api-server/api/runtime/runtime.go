package runtime

import (
	"fmt"

	"github.com/rs/zerolog/log"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/env"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/database"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/routes"
)

type runtime struct{}

var Runtime = runtime{}

func (r *runtime) StartServer(port int) {
	env.SetupEnv()

	database.SetupDB()
	router := routes.SetupRouter()

	log.Info().Msgf("Starting CompoundAI API server on port %d", port)

	router.Run(fmt.Sprintf("0.0.0.0:%d", port))
}
