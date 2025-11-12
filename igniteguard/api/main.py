"""FastAPI entry point (optional Phase 4 implementation)."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="IgniteGuard API", version="0.0.1")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok"}
