"""FastAPI application entry point."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.database.connection import get_db_manager
from src.database.operations import create_model, get_all_models, get_model_by_name
from src.api.routes import router

app = FastAPI(title="Bitcoin Trading Model API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

DEFAULT_MODEL_NAME = "nhits_best_local"
DEFAULT_MODEL_DIR = Path("models_nhits_best")


def _bootstrap_default_model() -> None:
    metadata_path = DEFAULT_MODEL_DIR / "metadata.json"
    model_path = DEFAULT_MODEL_DIR / "model.pt"
    if not metadata_path.exists() or not model_path.exists():
        return

    metadata = json.loads(metadata_path.read_text())
    config = metadata.get("config", {})
    metrics = metadata.get("metrics")
    model_type = str(config.get("model_type", "nhits"))
    scaler_path = metadata.get("scaler_path")

    db = get_db_manager()
    with db.session() as session:
        if get_model_by_name(session, DEFAULT_MODEL_NAME):
            return
        if get_all_models(session):
            return
        create_model(
            session,
            name=DEFAULT_MODEL_NAME,
            model_type=model_type,
            version="1.0",
            file_path=str(model_path),
            scaler_path=str(scaler_path) if scaler_path else None,
            config=config,
            metrics=metrics,
        )


@app.on_event("startup")
async def _startup_seed_default_model() -> None:
    _bootstrap_default_model()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    code = detail.get("code", "HTTP_ERROR")
    message = detail.get("message", "Request failed")
    details = detail.get("details", {})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": message, "details": details}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Internal server error",
                "details": {},
            }
        },
    )
