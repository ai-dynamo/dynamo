/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package integration

import (
	"context"
	"os"
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/database"
	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
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
