"""
tasks.py
--------
Defines the 3 tasks the environment can run.

Each task is a function that returns a fully-specified initial state dict.
ALL tasks are 100% deterministic — same seed = same result every time.

Tasks go from easy → medium → hard:
  1. bug_triage_easy       — fix bugs, nothing else
  2. sprint_planning_medium — balance bugs vs features
  3. product_crisis_hard   — everything on fire at once
"""

from typing import Any, Dict


def get_task(task_name: str) -> Dict[str, Any]:
    """
    Factory function — call this with a task name, get back
    the full initial state dict for that task.

    Raises ValueError for unknown task names (prevents silent mistakes).
    """
    tasks = {
        "bug_triage_easy": _bug_triage_easy,
        "sprint_planning_medium": _sprint_planning_medium,
        "product_crisis_hard": _product_crisis_hard,
    }
    if task_name not in tasks:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Valid tasks: {list(tasks.keys())}"
        )
    return tasks[task_name]()


# ─────────────────────────────────────────────
# TASK 1 — BUG TRIAGE (EASY)
# ─────────────────────────────────────────────

def _bug_triage_easy() -> Dict[str, Any]:
    """
    Scenario: Production just broke. Three bugs, no features.
    The agent's only job: fix bugs by severity order.

    What a good agent does:
      - Fix severity-5 bugs first (most damage)
      - Fix severity-3 next
      - Fix severity-1 last (or defer if out of capacity)

    Hidden traps:
      - Ignoring severity-5 bug causes stability to collapse
      - Deferring too many times triggers stakeholder decay
    """
    return {
        # ── Visible State ──────────────────────────────────────
        "backlog": [],          # No features — pure bug triage
        "bugs": [
            {"id": "bug_001", "severity": 5, "users_affected": 5000},
            {"id": "bug_002", "severity": 3, "users_affected": 800},
            {"id": "bug_003", "severity": 1, "users_affected": 50},
        ],
        "sprint_capacity": 10,  # Enough to fix all 3 if efficient
        "stakeholder_requests": [
            {
                "id": "req_001",
                "stakeholder": "CTO",
                "urgency": 5,
            }
        ],
        "user_growth": 0.5,
        "stability": 0.4,       # Already degraded — bugs have been ignored
        "stakeholder_satisfaction": 0.6,
        "step_count": 0,

        # ── Hidden State (internal only) ─────────────────────
        "true_bug_impact": {    # Actual damage per step if ignored
            "bug_001": 0.08,    # Critical — loses 8% stability/step
            "bug_002": 0.03,
            "bug_003": 0.005,
        },
        "stakeholder_patience": {
            "req_001": 3,       # Will escalate after 3 steps
        },
        "delayed_feature_queue": [],
        "technical_debt": 0.3,  # Already has 30% debt from past ignored bugs

        # ── Task Config ───────────────────────────────────────
        "max_steps": 10,
        "task_name": "bug_triage_easy",
        "revenue": None,        # Not tracked in this task
    }


# ─────────────────────────────────────────────
# TASK 2 — SPRINT PLANNING (MEDIUM)
# ─────────────────────────────────────────────

def _sprint_planning_medium() -> Dict[str, Any]:
    """
    Scenario: End of quarter sprint. You have bugs AND features.
    Capacity is limited — you CANNOT do everything.

    What a good agent does:
      - Fix the severity-4 bug (real stability risk)
      - Pick the highest impact/effort ratio features
      - Respond to the high-urgency CEO request
      - Defer low-value items strategically

    Hidden traps:
      - Feature B (high impact, high effort) might not fit in one sprint
      - Feature C has a deadline at step 5 — miss it and it's gone
      - Deferring the CEO request twice = satisfaction collapses
    """
    return {
        # ── Visible State ──────────────────────────────────────
        "backlog": [
            {"id": "feat_001", "impact": 0.3, "effort": 3, "deadline": 15},
            {"id": "feat_002", "impact": 0.5, "effort": 7, "deadline": 20},
            {"id": "feat_003", "impact": 0.2, "effort": 2, "deadline": 5},  # ← Urgent deadline!
        ],
        "bugs": [
            {"id": "bug_004", "severity": 4, "users_affected": 2000},
            {"id": "bug_005", "severity": 2, "users_affected": 300},
        ],
        "sprint_capacity": 12,  # Tight! All items cost 14 total
        "stakeholder_requests": [
            {"id": "req_002", "stakeholder": "CEO", "urgency": 4},
            {"id": "req_003", "stakeholder": "Investor", "urgency": 2},
        ],
        "user_growth": 0.45,
        "stability": 0.65,
        "stakeholder_satisfaction": 0.7,
        "step_count": 0,

        # ── Hidden State ─────────────────────────────────────
        "true_bug_impact": {
            "bug_004": 0.06,
            "bug_005": 0.01,
        },
        "stakeholder_patience": {
            "req_002": 2,       # CEO has little patience
            "req_003": 5,
        },
        "delayed_feature_queue": [],
        "technical_debt": 0.15,

        # ── Task Config ───────────────────────────────────────
        "max_steps": 15,
        "task_name": "sprint_planning_medium",
        "revenue": None,
    }


# ─────────────────────────────────────────────
# TASK 3 — PRODUCT CRISIS (HARD)
# ─────────────────────────────────────────────

def _product_crisis_hard() -> Dict[str, Any]:
    """
    Scenario: Full product crisis.
      - Critical bug causing data loss (severity 5)
      - Revenue dropping 5% per step
      - Board calling for an emergency response
      - Two features competitors just shipped
      - Rapid stakeholder patience decay

    What a good agent does:
      1. IMMEDIATELY fix the data-loss bug (severity 5)
      2. Respond to the Board (urgency 5) — else satisfaction implodes
      3. Fix the churn bug next
      4. Then tackle features strategically

    What a bad agent does:
      - Works on features while the crisis bug bleeds stability
      - Ignores the Board request
      - Defers without communicating

    This task ends early if the crisis is resolved:
      - stability > 0.8 AND stakeholder_satisfaction > 0.7
    """
    return {
        # ── Visible State ──────────────────────────────────────
        "backlog": [
            {"id": "feat_004", "impact": 0.4, "effort": 4, "deadline": 25},
            {"id": "feat_005", "impact": 0.35, "effort": 5, "deadline": 30},
        ],
        "bugs": [
            {"id": "bug_006", "severity": 5, "users_affected": 10000},  # Data loss!
            {"id": "bug_007", "severity": 4, "users_affected": 3000},   # Causing churn
            {"id": "bug_008", "severity": 2, "users_affected": 500},
        ],
        "sprint_capacity": 15,
        "stakeholder_requests": [
            {"id": "req_004", "stakeholder": "Board",    "urgency": 5},
            {"id": "req_005", "stakeholder": "CEO",      "urgency": 4},
            {"id": "req_006", "stakeholder": "Customer", "urgency": 3},
        ],
        "user_growth": 0.2,           # Already declining
        "stability": 0.25,            # Nearly broken
        "stakeholder_satisfaction": 0.3,  # Everyone is furious
        "step_count": 0,

        # ── Hidden State ─────────────────────────────────────
        "true_bug_impact": {
            "bug_006": 0.12,    # Catastrophic — 12% stability loss/step
            "bug_007": 0.07,
            "bug_008": 0.02,
        },
        "stakeholder_patience": {
            "req_004": 1,       # Board: 1 step before they escalate
            "req_005": 2,
            "req_006": 4,
        },
        "delayed_feature_queue": [],
        "technical_debt": 0.6,      # High debt from years of shortcuts
        "revenue": 1.0,             # Normalized; drops 0.05/step if crisis unresolved

        # ── Task Config ───────────────────────────────────────
        "max_steps": 20,
        "task_name": "product_crisis_hard",

        # Crisis ends early if both conditions met:
        "crisis_resolved": False,
        "crisis_resolve_stability_threshold": 0.8,
        "crisis_resolve_satisfaction_threshold": 0.7,
    }
