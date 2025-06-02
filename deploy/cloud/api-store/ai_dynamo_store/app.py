#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os
import sys

import uvicorn
from fastapi import FastAPI

from .api.deployments import router as deployments_router
from .api.dynamo import router as dynamo_router  # type: ignore
from .api.health_check import router as health_check_router
from .api.storage import create_db_and_tables_async

# Configure logging to write to stdout


def setup_logging():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    logger = logging.getLogger("ai_dynamo_store")
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


async def initialize_database():
    try:
        await create_db_and_tables_async()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def run_app():
    """Create and configure the FastAPI application.
    Returns:
        FastAPI: The configured application instance
    """
    app = FastAPI(
        title="AI Dynamo Store",
        description="AI Dynamo Store for managing Dynamo artifacts",
        version="0.1.0",
    )

    app.include_router(health_check_router)
    app.include_router(dynamo_router)
    app.include_router(deployments_router)
    port = int(os.getenv("SERVICE_PORT", "8000"))

    await initialize_database()
    # Start the FastAPI server
    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
