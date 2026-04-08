"""
app.py
------
FastAPI HTTP server that exposes the PM-OS environment as REST endpoints.

This is required for:
  - HuggingFace Space deployment
  - OpenEnv validation (it calls /reset to check the env works)
  - Remote agents that talk to the env over HTTP

Endpoints:
  POST /reset         → Start a new episode, return Observation
  POST /step          → Take one action, return StepResult
  GET  /state         → Get current observation without advancing
  GET  /health        → Health check (returns 200 if alive)
  GET  /tasks         → List available task names
"""

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from my_env import PMEnv, Action, Observation, StepResult

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="PM-OS OpenEnv",
    description="AI Product Manager Simulator — OpenEnv compatible environment",
    version="1.0.0",
)

# Allow cross-origin requests (needed for HF Spaces + browser clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global environment instance per server
# For production you'd use sessions — this is fine for evaluation
_env: Optional[PMEnv] = None

# ─────────────────────────────────────────────
# REQUEST/RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "bug_triage_easy"

class StepRequest(BaseModel):
    action: Action

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — OpenEnv validation calls this first."""
    return {"status": "ok", "service": "pm_os_env"}


@app.get("/tasks")
async def list_tasks():
    """List available task names."""
    return {
        "tasks": [
            "bug_triage_easy",
            "sprint_planning_medium",
            "product_crisis_hard",
        ]
    }


@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest):
    """
    Start a new episode.

    Body: {"task_name": "bug_triage_easy"}
    Returns: Observation (the initial state)

    This is the first call an agent/validator makes.
    """
    global _env
    _env = PMEnv(task_name=request.task_name)
    obs = await _env.reset(request.task_name)
    return obs


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest):
    """
    Take one action and advance the environment.

    Body: {"action": {"action_type": "fix_bug", "target_id": "bug_001", "reasoning": "..."}}
    Returns: StepResult (new observation + reward + done + info)

    Must call /reset first.
    """
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    result = await _env.step(request.action)
    return result


@app.get("/state", response_model=Observation)
async def state():
    """
    Get the current observation without advancing the episode.

    Useful for agents that want to re-examine the state
    without spending a step.
    """
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    obs = await _env.state()
    return obs


# ─────────────────────────────────────────────
# LOCAL DEV RUNNER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
