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
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/runtime"
)

const (
	port = 8181
)

func waitForDatabase(port string) error {
	hostname, _ := os.Hostname()
	url := fmt.Sprintf("http://%s:%s/api/v1/healthz", hostname, port)
	client := http.Client{Timeout: 1 * time.Second}

	log.Printf("Waiting for database to be ready at %s", url)
	for i := 0; i < 30; i++ { // try for 30 seconds
		resp, err := client.Get(url)
		if err != nil {
			log.Printf("Error connecting to database: %v", err)
		} else {
			log.Printf("Got response with status: %d", resp.StatusCode)
			return nil
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("database server failed to start on port %s", port)
}

func startDatabaseServer() error {
	dbPort := os.Getenv("API_DATABASE_PORT")
	if dbPort == "" {
		dbPort = "8001"
	}

	// Set the backend URL based on the database port
	hostname, _ := os.Hostname()
	backendUrl := fmt.Sprintf("http://%s:%s", hostname, dbPort)
	os.Setenv("API_BACKEND_URL", backendUrl)

	cmd := exec.Command("python3", "db/start_db.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start database: %v", err)
	}

	// Wait for database to be ready
	if err := waitForDatabase(dbPort); err != nil {
		cmd.Process.Kill()
		return err
	}

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

	return nil
}

func main() {
	// Start the database server first
	if err := startDatabaseServer(); err != nil {
		log.Fatalf("Failed to start database server: %v", err)
	}

	// Start the API server
	runtime.Runtime.StartServer(port)
}
