package env

import (
	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
)

func SetupEnv() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal().Msgf("Failed to load env during setup %s", err.Error())
	}

	_, err = SetResourceScope()
	if err != nil {
		log.Fatal().Msgf("Failed to set resource scope during env setup %s", err.Error())
	}

	_, err = SetNdsHost()
	if err != nil {
		log.Fatal().Msgf("Failed to set nds urls during env setup %s", err.Error())
	}
}
