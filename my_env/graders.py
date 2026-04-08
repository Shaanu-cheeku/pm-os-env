"""
graders.py
----------
Deterministic graders — one per task.

A grader looks at the FINAL state after an episode ends and returns
a score from 0.0 (total failure) to 1.0 (perfect run).

Graders are separate from the per-step reward function.
They answer: "How well did the agent do overall?"

Used by the inference script and OpenEnv validation.
"""

from typing import Any, Dict


def grade(task_name: str, final_state: Dict[str, Any]) -> float:
    """
    Route to the correct grader based on task name.
    Returns score in [0.0, 1.0].
    """
    graders = {
        "bug_triage_easy": grade_bug_triage,
        "sprint_planning_medium": grade_sprint,
        "product_crisis_hard": grade_crisis,
    }
    if task_name not in graders:
        raise ValueError(f"No grader found for task: '{task_name}'")
    return graders[task_name](final_state)


# ─────────────────────────────────────────────
# GRADER 1 — Bug Triage (Easy)
# ─────────────────────────────────────────────

def grade_bug_triage(state: Dict[str, Any]) -> float:
    """
    Score = weighted average of:
      - 60%: critical bugs fixed (severity >= 4)
      - 25%: final stability level
      - 15%: stakeholder satisfaction

    Why this weighting?
      The whole point of this task is to fix critical bugs.
      Stability is a secondary signal of whether bugs were
      actually impactful. Stakeholder matters but is minor here.
    """
    # ── Bug fix score ──────────────────────────────────────────
    remaining_bugs = state.get("bugs", [])

    # Count bugs that were initially critical (severity >= 4)
    # We track this via initial counts stored at reset time
    initial_critical = state.get("initial_critical_bugs", 1)  # avoid div-by-0
    remaining_critical = sum(
        1 for b in remaining_bugs if b.get("severity", 0) >= 4
    )
    bugs_fixed = max(0, initial_critical - remaining_critical)
    bug_score = min(1.0, bugs_fixed / initial_critical)

    # ── Stability score ───────────────────────────────────────
    stability_score = float(state.get("stability", 0.0))

    # ── Stakeholder score ─────────────────────────────────────
    sat_score = float(state.get("stakeholder_satisfaction", 0.0))

    # ── Weighted final score ──────────────────────────────────
    score = (
        0.60 * bug_score +
        0.25 * stability_score +
        0.15 * sat_score
    )
    return round(min(1.0, max(0.0, score)), 4)


# ─────────────────────────────────────────────
# GRADER 2 — Sprint Planning (Medium)
# ─────────────────────────────────────────────

def grade_sprint(state: Dict[str, Any]) -> float:
    """
    Score = weighted average of:
      - 30%: bugs resolved (weighted by severity)
      - 30%: features delivered (weighted by impact)
      - 20%: stakeholder satisfaction
      - 20%: capacity efficiency (didn't waste or over-spend)

    This rewards BALANCE — the best agents do a bit of everything
    rather than hyper-focusing on one dimension.
    """
    # ── Bug score ─────────────────────────────────────────────
    remaining_bugs = state.get("bugs", [])
    initial_bugs = state.get("initial_bugs", [])

    # Weight fixes by severity (fixing a sev-4 is worth more than sev-1)
    initial_sev_sum = sum(b.get("severity", 1) for b in initial_bugs) or 1
    remaining_sev_sum = sum(b.get("severity", 1) for b in remaining_bugs)
    fixed_sev_sum = max(0, initial_sev_sum - remaining_sev_sum)
    bug_score = min(1.0, fixed_sev_sum / initial_sev_sum)

    # ── Feature score ─────────────────────────────────────────
    initial_backlog = state.get("initial_backlog", [])
    remaining_backlog = state.get("backlog", [])

    initial_impact_sum = sum(f.get("impact", 0) for f in initial_backlog) or 1
    remaining_impact_sum = sum(f.get("impact", 0) for f in remaining_backlog)
    delivered_impact = max(0.0, initial_impact_sum - remaining_impact_sum)
    feature_score = min(1.0, delivered_impact / initial_impact_sum)

    # ── Stakeholder score ─────────────────────────────────────
    sat_score = float(state.get("stakeholder_satisfaction", 0.0))

    # ── Capacity efficiency ───────────────────────────────────
    # Full score if 70-100% of capacity used; penalize waste or over-spend
    initial_capacity = state.get("initial_capacity", 12)
    remaining_capacity = state.get("sprint_capacity", 0)
    used = initial_capacity - remaining_capacity
    utilization = used / initial_capacity if initial_capacity > 0 else 0
    # Reward range [0.7, 1.0] utilization as full marks
    if utilization >= 0.7:
        capacity_score = 1.0
    elif utilization > 0:
        capacity_score = utilization / 0.7
    else:
        capacity_score = 0.0

    # ── Weighted final score ──────────────────────────────────
    score = (
        0.30 * bug_score +
        0.30 * feature_score +
        0.20 * sat_score +
        0.20 * capacity_score
    )
    return round(min(1.0, max(0.0, score)), 4)


# ─────────────────────────────────────────────
# GRADER 3 — Product Crisis (Hard)
# ─────────────────────────────────────────────

def grade_crisis(state: Dict[str, Any]) -> float:
    """
    Score = weighted average of:
      - 30%: revenue recovery (how much revenue was saved)
      - 30%: final stability (did the product stabilize?)
      - 20%: stakeholder satisfaction
      - 20%: efficiency (didn't burn all capacity on non-crisis items)

    The hard task requires juggling ALL dimensions simultaneously.
    A score > 0.7 means the agent genuinely handled the crisis well.
    """
    # ── Revenue recovery ──────────────────────────────────────
    # Revenue starts at 1.0, drops each step if crisis unresolved
    revenue = float(state.get("revenue", 0.0))
    revenue_recovery = min(1.0, revenue)  # Already normalized

    # ── Stability ─────────────────────────────────────────────
    stability = float(state.get("stability", 0.0))

    # ── Stakeholder satisfaction ──────────────────────────────
    sat = float(state.get("stakeholder_satisfaction", 0.0))

    # ── Efficiency ────────────────────────────────────────────
    # Penalize if technical debt is very high (shortcuts taken)
    debt = float(state.get("technical_debt", 0.0))
    efficiency = max(0.0, 1.0 - debt)

    # ── Bonus: early crisis resolution ───────────────────────
    crisis_resolved = state.get("crisis_resolved", False)
    early_bonus = 0.05 if crisis_resolved else 0.0

    # ── Weighted final score ──────────────────────────────────
    score = (
        0.30 * revenue_recovery +
        0.30 * stability +
        0.20 * sat +
        0.20 * efficiency +
        early_bonus
    )
    return round(min(1.0, max(0.0, score)), 4)
