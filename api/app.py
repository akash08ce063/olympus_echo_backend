"""FastAPI application for Agent Bridge service."""

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from api.v1.routes import (
    test_suite_routes,
    test_suite_crud_routes, target_agents_routes,
    user_agents_routes, test_cases_routes, test_history_routes,
    test_execution_routes, test_runs_routes
)
from telemetrics.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cleanup on shutdown."""
    logger.info("Voice Testing Platform startup")
    yield
    logger.info("Voice Testing Platform shutdown")


app = FastAPI(
    title="Agent Bridge Service",
    description="Enables two voice agents to have real-time conversations",
    version="1.0.0",
    lifespan=lifespan,
)

# Create v1 API router
v1_router = APIRouter(prefix="/v1")

# Include routers under v1
v1_router.include_router(test_suite_routes.router)  # Scaled testing (twilio-test, web-test)
v1_router.include_router(test_suite_crud_routes.router)  # Test suites CRUD
v1_router.include_router(target_agents_routes.router)  # Target agents CRUD
v1_router.include_router(user_agents_routes.router)  # User agents CRUD
v1_router.include_router(test_cases_routes.router)  # Test cases CRUD
v1_router.include_router(test_history_routes.router)  # Test history read-only
v1_router.include_router(test_execution_routes.router)  # Test execution
v1_router.include_router(test_runs_routes.router)  # Test runs, recordings, transcripts

# Include v1 router in main app
app.include_router(v1_router)


@app.get("/health", tags=["Health"])
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "voice_testing_platform",
        "version": "0.1.0"
    }
