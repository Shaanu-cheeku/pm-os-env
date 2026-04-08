"""
models.py
---------
All Pydantic data models for the PM-OS environment.

Think of these like "blueprints" for data objects — they ensure
every piece of data has the right shape and types at all times.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# BACKLOG ITEM MODELS
# ─────────────────────────────────────────────

class Feature(BaseModel):
    """
    A product feature waiting to be built.
    - impact: how much it will boost user_growth (0.0–1.0)
    - effort: sprint points it consumes (1–10)
    - deadline: which step it expires on (if missed, small penalty)
    """
    id: str
    impact: float = Field(ge=0.0, le=1.0)
    effort: int = Field(ge=1, le=10)
    deadline: int = Field(ge=1)


class Bug(BaseModel):
    """
    A bug hurting users right now.
    - severity: 1 (minor) to 5 (critical — product is breaking)
    - users_affected: number of users experiencing the bug
    """
    id: str
    severity: int = Field(ge=1, le=5)
    users_affected: int = Field(ge=0)


class Request(BaseModel):
    """
    A stakeholder request (CEO, investor, customer, etc.)
    - urgency: 1 (nice-to-have) to 5 (drop everything now)
    """
    id: str
    stakeholder: str
    urgency: int = Field(ge=1, le=5)


# ─────────────────────────────────────────────
# WHAT THE AGENT SEES (Observation)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """
    Everything the AI agent can see at each step.
    Hidden variables are NOT included here — the agent
    must infer them from changes in these metrics.
    """
    backlog: List[Feature]
    bugs: List[Bug]
    sprint_capacity: int                  # Points left this sprint
    stakeholder_requests: List[Request]
    user_growth: float                    # 0.0–1.0 (current growth rate)
    stability: float                      # 0.0–1.0 (product health)
    stakeholder_satisfaction: float       # 0.0–1.0
    step_count: int                       # How many steps taken so far


# ─────────────────────────────────────────────
# WHAT THE AGENT DOES (Action)
# ─────────────────────────────────────────────

class Action(BaseModel):
    """
    The agent's decision each step.
    - action_type: one of four valid actions
    - target_id: which Feature/Bug/Request to act on
    - reasoning: free-text explanation (for logging/debugging)
    """
    action_type: Literal[
        "prioritize_feature",
        "fix_bug",
        "defer_task",
        "respond_to_stakeholder"
    ]
    target_id: str
    reasoning: str = ""


# ─────────────────────────────────────────────
# WHAT THE ENV RETURNS AFTER EACH STEP
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """
    The full result of one agent step.
    - observation: the new world state
    - reward: float in [0.0, 1.0]
    - done: True if episode is over
    - info: extra debug info (errors, hidden metrics, etc.)
    """
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any]
