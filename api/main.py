"""FastAPI main application entrypoint for Route.it.

Configures CORS for frontend dashboard development servers, registers
REST and WebSocket routers (both at root and under /api for unified hosting),
mounts the prebuilt React dashboard if present, and initializes telemetry hooks.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import __version__
from api.routes.events import router as events_router
from api.routes.health import router as health_router
from api.routes.problems import router as problems_router
from api.routes.ws import router as ws_router
from api.websocket import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("route_it_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for event loop capture and graceful shutdown."""
    loop = asyncio.get_running_loop()
    manager.set_event_loop(loop)
    logger.info(f"[API] Route.it Service v{__version__} initialized.")
    yield
    logger.info("[API] Route.it Service shutting down.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    app = FastAPI(
        title="Route.it Fleet Routing API",
        description=(
            "Adaptive Quantum-Inspired Fleet Routing Engine (CVRPTW) "
            "with Dynamic Incident Re-optimization for Smart India Hackathon 2026 (PS SIH26137)."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Enable CORS for React + MapLibre GL Dashboard development servers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "*",  # Permissive for development and cloud previews
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API and WebSocket routers at root (for direct API clients & tests)
    app.include_router(health_router)
    app.include_router(problems_router)
    app.include_router(events_router)
    app.include_router(ws_router)

    # Also expose under /api/* so the React dashboard works without a separate reverse proxy
    app.include_router(health_router, prefix="/api", include_in_schema=False)
    app.include_router(problems_router, prefix="/api", include_in_schema=False)
    app.include_router(events_router, prefix="/api", include_in_schema=False)
    app.include_router(ws_router, prefix="/api", include_in_schema=False)

    # Serve prebuilt React production bundle (dashboard/dist) when available
    dist_dir = Path(__file__).resolve().parent.parent / "dashboard" / "dist"
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="dashboard")

    return app


app = create_app()
