"""
api/routes/knowledge.py
=======================
FastAPI route handlers for exploring and searching the Regulatory &
Historical Case Knowledge Base.

Author: AI-Assisted Financial Transaction Risk Investigation System
"""

import json
import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from rag.retriever import retrieve_evidence

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base & Regulations"])

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")


@router.get("/regulations", summary="List Regulatory Frameworks")
def list_regulations():
    """Retrieve all indexed RBI, FATF, and FinCEN directives."""
    try:
        regulations = []
        for filename in ["rbi_regulations.json", "fatf_recommendations.json", "fincen_typologies.json"]:
            path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    regulations.extend(data)
        return {"total": len(regulations), "regulations": regulations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load regulations: {str(e)}")


@router.get("/cases", summary="List Historical Case Precedents")
def list_historical_cases():
    """Retrieve all historical fraud investigation precedents."""
    try:
        path = os.path.join(KNOWLEDGE_BASE_DIR, "historical_cases.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            return {"total": len(cases), "cases": cases}
        return {"total": 0, "cases": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load cases: {str(e)}")


@router.get("/search", summary="Search Evidence Knowledge Base")
def search_knowledge_base(q: str = Query(..., description="Search query string"), top_k: int = 3):
    """Perform semantic search across regulations and historical precedents."""
    try:
        results = retrieve_evidence(query=q, top_k=top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
