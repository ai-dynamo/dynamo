import uvicorn

from .app import create_app


def main():
    """Run the application server."""
    uvicorn.run(
        "ai_dynamo_store:create_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=True,
    )


if __name__ == "__main__":
    main()
