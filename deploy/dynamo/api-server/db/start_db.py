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

import asyncio
import logging
import os
import sys

import uvicorn
from db.api import router
from db.storage import create_db_and_tables_async
from fastapi import FastAPI
from fastapi.routing import APIRoute

# Configure logging to write to stdout
logging.basicConfig(
    level=logging.INFO, format="PYTHON_APP: %(message)s", stream=sys.stdout
)

logger = logging.getLogger(__name__)


async def init_db():
    try:
        await create_db_and_tables_async()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def main():
    # Add debug logging for import verification
    logger.info("Imported app from api.py")
    logger.info("Router routes:")

    app = FastAPI(title="Dynamo API Database Server")
    app.include_router(router)
    for route in router.routes:
        if isinstance(
            route, APIRoute
        ):  # Ensure it's an APIRoute before accessing attributes
            route: APIRoute  # Explicitly tell MyPy that route is an APIRoute
            logger.info(f"  {route.path} {route.methods} {route.name}")

    await init_db()

    # Log all registered routes more verbosely
    logger.info("=== REGISTERED ROUTES ===")
    for route in app.routes:
        logger.info(f"PATH: {route.path}")
        logger.info(f"METHODS: {route.methods}")
        logger.info(f"NAME: {route.name}")
        logger.info("=====================")

    # Get port from environment or default to 8001
    port = int(os.getenv("API_DATABASE_PORT", "8001"))

    # Log the backend URL for debugging
    backend_url = os.getenv("API_BACKEND_URL")
    logger.info(f"Starting FastAPI server with backend URL: {backend_url}")

    # Start the FastAPI server
    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
