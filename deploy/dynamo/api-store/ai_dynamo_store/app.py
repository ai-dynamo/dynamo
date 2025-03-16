from fastapi import FastAPI

from .api import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured application instance
    """
    app = FastAPI(
        title="AI Dynamo Store",
        description="AI Dynamo Store for managing Dynamo artifacts",
        version="0.1.0",
    )

    app.include_router(router)

    return app
