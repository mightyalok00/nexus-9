"""FastAPI application gateway for NEXUS-9 research system."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import datasets, experiments, models, simulations
from nexus9.config import settings

app = FastAPI(
    title="NEXUS-9 AI/ML System API",
    description="Research-grade API for Model Arena, FutureBench Simulator, Causal Engine, and Autonomous Agents",
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(simulations.router)
app.include_router(experiments.router)

@app.get("/")
def root():
    return {
        "system": "NEXUS-9",
        "status": "OPERATIONAL",
        "version": settings.version,
        "docs_url": "/docs",
    }
