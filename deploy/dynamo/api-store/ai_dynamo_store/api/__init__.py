from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
@router.get("/readyz")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Status information
    """
    return {"status": "healthy"}
