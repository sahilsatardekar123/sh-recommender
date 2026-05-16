"""
main.py — FastAPI application for the SHL Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {reply, recommendations, end_of_conversation}
"""

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="Full conversation history, oldest first",
        min_length=1,
    )


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False


# ─────────────────────────────────────────────
# Application state (loaded once at startup)
# ─────────────────────────────────────────────

class AppState:
    catalog: list[dict] = []


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog and warm up FAISS index on startup."""
    catalog_path = os.path.join(os.path.dirname(__file__), "catalog.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            app_state.catalog = json.load(f)
        print(f"✓ Loaded catalog: {len(app_state.catalog)} assessments")
    else:
        print("WARNING: catalog.json not found — run scraper.py first")
        app_state.catalog = []

    # Warm up FAISS index (loads model + index into memory)
    try:
        from retriever import _get_index_and_meta, _get_model
        _get_model()
        _get_index_and_meta()
        print("✓ FAISS index and sentence-transformer model loaded")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    except Exception as e:
        print(f"WARNING: Could not pre-load FAISS index: {e}")

    yield
    # Cleanup (none needed for this service)


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "AI-powered service that helps hiring managers select the right "
        "SHL Individual Test Solutions based on their hiring context."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health_check():
    """Returns 200 OK when the service is running."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="SHL Assessment Recommender chat",
)
def chat(request: ChatRequest):
    """
    Stateless chat endpoint.
    The full conversation history must be sent on every call.
    Returns a reply, optional assessment recommendations, and a conversation-end flag.
    """
    from agent import run_agent

    # Validate message roles
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'.",
            )

    # Ensure conversation starts with a user message
    if request.messages[0].role != "user":
        raise HTTPException(
            status_code=422,
            detail="Conversation must start with a user message.",
        )

    if not app_state.catalog:
        raise HTTPException(
            status_code=503,
            detail="Catalog not loaded. Run `python scraper.py` and `python build_index.py` first.",
        )

    # Convert Pydantic models to plain dicts for agent
    messages_dicts = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]

    result = run_agent(messages_dicts, app_state.catalog)

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r.get("test_type", "N/A"),
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )

@app.get("/evaluate", summary="Run evaluation metrics")
def evaluate(with_agent: bool = False):
    from evaluation import run_full_evaluation
    if not app_state.catalog:
        raise HTTPException(status_code=503, detail="Catalog not loaded.")
    if with_agent:
        from agent import run_agent
        report = run_full_evaluation(run_agent_func=run_agent, catalog=app_state.catalog)
    else:
        report = run_full_evaluation(catalog=app_state.catalog)
    return report
