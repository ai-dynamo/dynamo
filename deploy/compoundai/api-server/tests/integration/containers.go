package integration

import (
	"context"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/database"
)

type TestContainers struct {
	postgres *postgres.PostgresContainer
}

var testContainers = TestContainers{}

const (
	postgresImage = "postgres:16.2"
)

func (c *TestContainers) CreatePostgresContainer() (*postgres.PostgresContainer, error) {
	err := godotenv.Load()
	if err != nil {
		log.Error().Msgf("Failed to load env vars for during integration test setup: %s", err.Error())
	}

	ctx := context.Background()
	postgres, err := postgres.Run(ctx,
		postgresImage,
		postgres.WithDatabase(os.Getenv(database.DB_NAME)),
		postgres.WithUsername(os.Getenv(database.DB_USER)),
		postgres.WithPassword(os.Getenv(database.DB_PASSWORD)),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).WithStartupTimeout(10*time.Second)),
	)
	if err != nil {
		log.Error().Msgf("Could not create Postgres container: %s", err.Error())
		return nil, err
	}

	containerPort, err := postgres.MappedPort(ctx, "5432")
	if err != nil {
		log.Error().Msgf("Could not get mapped port: %s", err.Error())
		return nil, err
	}
	os.Setenv(database.DB_PORT, containerPort.Port())

	log.Info().Msgf("Started postgres container %+v on port %s", postgres, containerPort.Port())
	c.postgres = postgres
	return postgres, nil
}

func (c *TestContainers) TearDownPostgresContainer() error {
	log.Info().Msgf("terminating postgres container")
	ctx := context.Background()
	err := c.postgres.Terminate(ctx)
	if err != nil {
		log.Error().Msgf("Failed to terminate test Postgres container: %s", err.Error())
		return err
	}

	return nil
}
