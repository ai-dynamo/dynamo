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

package main

import (
	"log"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/utils"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/runtime"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/services"
)

const (
	port = 8181
)

// getDatabaseURL gets the database URL from the Python db.py script
func getDatabaseURL() string {
	// First check environment variable
	dbURL, err := utils.MustGetEnv("API_DATABASE_URL")
	if err != nil {
		log.Fatalf("Failed to get database URL: %v", err)
	}
	return dbURL
}

func startDatabase() {
	// Check if the database is already running
	cmd := exec.Command("python3", "../db/db.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		log.Printf("Failed to start database: %v", err)
		return
	}

	// Give the database time to initialize
	time.Sleep(2 * time.Second)

	log.Println("Database started successfully")

	// Set up graceful shutdown
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		log.Println("Shutting down database...")
		if err := cmd.Process.Kill(); err != nil {
			log.Printf("Failed to kill database process: %v", err)
		}
		os.Exit(0)
	}()
}

func main() {
	// Start the database first
	startDatabase()

	// Get the database URL
	dbURL := getDatabaseURL()
	log.Printf("Using database URL: %s", dbURL)

	// Set the database URL for services
	services.InitDatabaseService(dbURL)

	// Start the API server
	runtime.Runtime.StartServer(port)
}
