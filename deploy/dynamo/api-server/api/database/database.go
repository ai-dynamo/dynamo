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

package database

import (
	"context"
	"fmt"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/utils"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/models"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/schema"
)

var db *gorm.DB

type databaseUtil struct{}

var DatabaseUtil = databaseUtil{}

type DbCtxKeyType string

const DbSessionKey DbCtxKeyType = "session"

var openDbOnce = sync.Once{}

const (
	DB_USER     = "DB_USER"
	DB_PASSWORD = "DB_PASSWORD"
	DB_HOST     = "DB_HOST"
	DB_NAME     = "DB_NAME"
	DB_PORT     = "DB_PORT"
)

func SetupDB() {
	openDbOnce.Do(func() {
		var err error
		db, err = openDBConnection()
		if err != nil {
			log.Fatal().Msgf("Could not connect to Postgres database! %s", err.Error())
		}
		db.AutoMigrate(&models.Cluster{})
		db.AutoMigrate(&models.Deployment{})
		db.AutoMigrate(&models.DeploymentRevision{})
		db.AutoMigrate(&models.DeploymentTarget{})
		db.AutoMigrate(&models.DynamoComponent{})

		db.Exec("CREATE UNIQUE INDEX uk_cluster_orgId_name ON cluster (organization_id, name);")
		db.Exec("CREATE UNIQUE INDEX uk_deployment_clusterId_name ON deployment (cluster_id, name);")
	})
}

func openDBConnection() (*gorm.DB, error) {
	dbUser, err := utils.MustGetEnv(DB_USER)
	if err != nil {
		log.Error().Msgf("Failed to get %s from env: %s", DB_USER, err.Error())
		return nil, err
	}

	dbPass, err := utils.MustGetEnv(DB_PASSWORD)
	if err != nil {
		log.Error().Msgf("Failed to get %s from env: %s", DB_PASSWORD, err.Error())
		return nil, err
	}

	dbHost, err := utils.MustGetEnv(DB_HOST)
	if err != nil {
		log.Error().Msgf("Failed to get %s from env: %s", DB_HOST, err.Error())
		return nil, err
	}

	dbPort, err := utils.MustGetEnv(DB_PORT)
	if err != nil {
		log.Error().Msgf("Failed to get %s from env: %s", DB_PORT, err.Error())
		return nil, err
	}

	dbName, err := utils.MustGetEnv(DB_NAME)
	if err != nil {
		log.Error().Msgf("Failed to get %s from env: %s", DB_NAME, err.Error())
		return nil, err
	}

	uri := fmt.Sprintf("postgres://%s:%s@%s:%s/%s",
		dbUser,
		dbPass,
		dbHost,
		dbPort,
		dbName,
	)

	log.Info().Msgf("Connecting to Postgres")
	db, err := gorm.Open(postgres.Open(uri), &gorm.Config{
		NamingStrategy: schema.NamingStrategy{SingularTable: true},
		PrepareStmt:    false,
	})
	if err != nil {
		return nil, err
	}
	log.Info().Msgf("Successfully connected to Postgres")

	return db, nil
}

func (d *databaseUtil) GetDB(ctx context.Context) *gorm.DB {
	return db.WithContext(ctx)
}

func (d *databaseUtil) GetDBSession(ctx context.Context) *gorm.DB {
	session := ctx.Value(DbSessionKey)
	if session != nil {
		db := session.(*gorm.DB)
		return db
	}
	return d.GetDB(ctx)
}

func (d *databaseUtil) StartTransaction(ctx context.Context) (*gorm.DB, context.Context, func(error), error) {
	session := ctx.Value(DbSessionKey)

	if session != nil {
		db_ := session.(*gorm.DB)
		return db_, ctx, func(err error) {}, nil
	}

	db := d.GetDB(ctx)
	tx := db.Begin()
	if tx.Error != nil {
		return nil, ctx, func(err error) {}, tx.Error
	}

	ctx = context.WithValue(ctx, DbSessionKey, tx)
	return tx, ctx, func(err error) {
		select {
		case <-ctx.Done():
			return
		default:
		}
		// nolint: gocritic
		if p := recover(); p != nil {
			tx.Rollback()
			panic(p)
		} else if err != nil {
			tx.Rollback()
		} else {
			tx.Commit()
		}
	}, nil
}
