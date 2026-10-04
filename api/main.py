import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.routers import router as investigate_router
from api.routes.knowledge import router as knowledge_router

# ---------------------------------------------------------------------------
# LOGGING & APP INITIALIZATION
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraud_investigation_api")

app = FastAPI(
    title="AI-Assisted Financial Transaction Risk Investigation API",
    description=(
        "Production REST backend combining Machine Learning anomaly detection "
        "with Regulatory RAG (RBI, FATF, FinCEN) to generate human-readable "
        "investigation briefs for financial compliance analysts."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS MIDDLEWARE
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# STATIC FILES MOUNTING & UI SERVING
# ---------------------------------------------------------------------------

STATIC_DIR = os.getenv("STATIC_DIR", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------------------------

app.include_router(investigate_router)
app.include_router(knowledge_router)


# ---------------------------------------------------------------------------
# SYSTEM HEALTH & ROOT UI
# ---------------------------------------------------------------------------

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System Health"])
def health_check():
    """System health check and model status verification."""
    models_dir = os.getenv("MODELS_DIR", "models")
    ml_model_exists = os.path.exists(os.path.join(models_dir, "isolation_forest.joblib"))
    rag_index_exists = os.path.exists(os.path.join(models_dir, "rag_matrix.joblib"))

    status = "READY" if (ml_model_exists and rag_index_exists) else "DEGRADED"

    return {
        "status": status,
        "services": {
            "ml_isolation_forest": "LOADED" if ml_model_exists else "MISSING",
            "rag_knowledge_index": "LOADED" if rag_index_exists else "MISSING",
        },
        "version": "1.0.0",
    }


@app.api_route("/", methods=["GET", "HEAD"], tags=["UI Portal"])
def serve_dashboard():
    """Serve the interactive web frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "AI-Assisted Financial Transaction Risk Investigation API",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
