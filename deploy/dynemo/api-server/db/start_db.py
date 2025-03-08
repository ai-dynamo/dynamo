import asyncio
import logging
import os
import uvicorn
from api import app
from db import create_db_and_tables_async

logger = logging.getLogger(__name__)

async def init_db():
    try:
        await create_db_and_tables_async()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

async def main():
    # Initialize the database
    await init_db()
    
    # Get port from environment or default to 8001
    port = int(os.getenv("API_DATABASE_PORT", "8001"))
    os.environ["API_BACKEND_URL"] = f"http://0.0.0.0:{port}"
    
    # Start the FastAPI server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())