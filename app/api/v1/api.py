"""API v1 router configuration.

Exposes the educational-video pipeline plus the template's auth + chatbot
routers.
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.chatbot import router as chatbot_router
from app.api.v1.videos import router as videos_router
from app.core.logging import logger

api_router = APIRouter()

# Educational-video pipeline.
api_router.include_router(videos_router, prefix="/videos", tags=["videos"])

# Template auth + chatbot.
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])


@api_router.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Health status information.
    """
    logger.info("health_check_called")
    return {"status": "healthy", "version": "1.0.0"}
