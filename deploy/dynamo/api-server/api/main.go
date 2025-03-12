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

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/env"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/runtime"
)

const (
	port = 8181
)

func startDatabaseServer() {
	/*
		Starts the Python FastAPI server for the database.
	*/

	// Check if the database is already running
	cmd := exec.Command("python3", "db/start_db.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		log.Printf("Failed to start database: %v", err)
		return
	}

	// Give the database time to initialize
	time.Sleep(5 * time.Second)
	// Check the backend URL is reachable
	_ = env.GetBackendUrl()

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
	// Start the database server first
	startDatabaseServer()

	// Start the API server
	runtime.Runtime.StartServer(port)
}
