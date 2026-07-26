"""Entry point: FastAPI server."""

import uvicorn

from config.settings import get_settings
from api.main import app

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
    )
