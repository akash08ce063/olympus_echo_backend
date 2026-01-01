#!/usr/bin/env python3
"""
Main entry point for the Voice Testing Platform.

This module starts the FastAPI application with all routes loaded.
"""

import os
import uvicorn
from telemetrics.logger import logger

# Import the FastAPI app
from api.app import app


def main():
    """Main entry point."""
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting Voice Testing Platform on {host}:{port}")

    # Start the server
    uvicorn.run(
        "main:app",  # module:app
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
