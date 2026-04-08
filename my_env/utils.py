"""
utils.py
--------
Pure helper functions — no side effects, fully testable.

These are the "math engine" of the environment.
Keeping them here (separate from env.py) makes the code:
  - Easier to test in isolation
  - Easier to reason about
  - Easier to tune reward weights
"""

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# REWARD CALCULATION
# ─────────────────────────────────────────────

def compute_reward(
    action_type: str,
    target_id: str,
    prev_state: Dict[str, Any],
    next_state: Dict[str, Any],
    action_valid: bool,
    error: Optional[str] = None,
) -> float:
    """
    Compute the per-step reward for one agent action.

    Design philosophy:
      - Dense rewards: agent gets feedback every step (not just at end)
      - Multi-factor: captures all dimensions of PM work
      - Normalized: always returns a value in [0.0, 1.0]
      - Punishes invalidity: wrong target_id = penalty

    Returns: float in [0.0, 1.0]
    """
    raw = 0.0

    # ── Penalty for invalid actions ───────────────────────────
    if not action_valid:
        # Invalid action = wasted step, mild penalty
        return max(0.0, 0.1)

    # ── Per-step existence penalty (encourages urgency) ───────
    # Doing nothing optimal still bleeds a small cost each step
    raw -= 0.05

    # ── Bug fixing reward ─────────────────────────────────────
    if action_type == "fix_bug":
        fixed_bug = _find_in_list(prev_state.get("bugs", []), target_id)
        if fixed_bug:
            severity = fixed_bug.get("severity", 1)
            # Critical bugs (4-5) get disproportionately large reward
            if severity >= 4:
                raw += 0.30    # Critical fix: big reward
            elif severity == 3:
                raw += 0.15
            else:
                raw += 0.05    # Minor bug: small reward

            # Stability improvement bonus
            stability_delta = (
                next_state.get("stability", 0) -
                prev_state.get("stability", 0)
            )
            if stability_delta > 0:
                raw += 0.20 * stability_delta

    # ── Feature prioritization reward ─────────────────────────
    elif action_type == "prioritize_feature":
        feature = _find_in_list(prev_state.get("backlog", []), target_id)
        if feature:
            # Immediate partial reward for queuing a feature
            raw += 0.10 * feature.get("impact", 0)
            # Note: full impact reward comes when delayed effect fires

    # ── Stakeholder response reward ───────────────────────────
    elif action_type == "respond_to_stakeholder":
        request = _find_in_list(
            prev_state.get("stakeholder_requests", []), target_id
        )
        if request:
            urgency = request.get("urgency", 1)
            sat_delta = (
                next_state.get("stakeholder_satisfaction", 0) -
                prev_state.get("stakeholder_satisfaction", 0)
            )
            # Higher urgency responses matter more
            raw += 0.20 * (urgency / 5.0) * max(0, sat_delta + 0.1)

    # ── Defer action — small penalty ─────────────────────────
    elif action_type == "defer_task":
        # Deferring is sometimes necessary but never free
        raw -= 0.05

    # ── Penalty: ignored critical bugs are bleeding stability ──
    remaining_critical = sum(
        1 for b in next_state.get("bugs", [])
        if b.get("severity", 0) >= 4
    )
    if remaining_critical > 0:
        raw -= 0.40 * (remaining_critical * 0.3)  # Scales with unresolved criticals

    # ── Penalty: over-capacity ─────────────────────────────────
    if next_state.get("sprint_capacity", 0) < 0:
        raw -= 0.30

    # ── Reward: delayed feature effects ───────────────────────
    growth_delta = (
        next_state.get("user_growth", 0) -
        prev_state.get("user_growth", 0)
    )
    if growth_delta > 0:
        raw += 0.20 * growth_delta

    # ── Crisis task: revenue tracking ─────────────────────────
    if "revenue" in next_state and next_state["revenue"] is not None:
        rev_delta = (
            next_state.get("revenue", 0) -
            prev_state.get("revenue", 0)
        )
        if rev_delta > 0:
            raw += 0.15 * rev_delta
        elif rev_delta < 0:
            raw += 0.15 * rev_delta  # Penalty if revenue dropped

    # ── Normalize to [0.0, 1.0] ───────────────────────────────
    return float(max(0.0, min(1.0, raw + 0.5)))  # Shift up so 0.5 = neutral


# ─────────────────────────────────────────────
# METRIC UPDATE HELPERS
# ─────────────────────────────────────────────

def apply_bug_decay(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Each step that passes with unfixed bugs, stability degrades.
    Uses the hidden true_bug_impact values (not visible to agent).
    """
    remaining_bug_ids = {b["id"] for b in state.get("bugs", [])}
    true_impact = state.get("true_bug_impact", {})

    total_damage = sum(
        impact
        for bug_id, impact in true_impact.items()
        if bug_id in remaining_bug_ids
    )

    new_stability = max(0.0, state["stability"] - total_damage)
    state["stability"] = round(new_stability, 4)
    return state


def apply_delayed_features(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Features queued via prioritize_feature take 2-3 steps to ship.
    This function fires them when their delay expires.

    Each item in delayed_feature_queue looks like:
      {"feature_id": "feat_001", "impact": 0.3, "fires_at_step": 5}
    """
    current_step = state.get("step_count", 0)
    queue = state.get("delayed_feature_queue", [])
    still_pending = []

    for item in queue:
        if item["fires_at_step"] <= current_step:
            # Feature shipped! Apply growth boost
            growth_boost = item["impact"] * 0.15  # Each feature = 15% of its impact
            state["user_growth"] = min(1.0, state["user_growth"] + growth_boost)
        else:
            still_pending.append(item)

    state["delayed_feature_queue"] = still_pending
    return state


def apply_stakeholder_decay(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stakeholders lose patience each step their request goes unaddressed.
    When patience hits 0, satisfaction drops and they escalate.
    """
    patience = state.get("stakeholder_patience", {})
    active_request_ids = {r["id"] for r in state.get("stakeholder_requests", [])}

    total_escalations = 0
    new_patience = {}

    for req_id, p in patience.items():
        if req_id in active_request_ids:
            # Request still unresolved — patience ticks down
            new_p = p - 1
            new_patience[req_id] = new_p
            if new_p <= 0:
                total_escalations += 1
        else:
            # Request was resolved — remove from patience tracker
            pass  # Don't carry it forward

    # Each escalation hits satisfaction hard
    if total_escalations > 0:
        hit = 0.08 * total_escalations
        state["stakeholder_satisfaction"] = max(
            0.0, state["stakeholder_satisfaction"] - hit
        )

    state["stakeholder_patience"] = new_patience
    return state


def apply_technical_debt(state: Dict[str, Any], action_type: str) -> Dict[str, Any]:
    """
    Technical debt accumulates when:
      - Bugs are deferred
      - Features are rushed (prioritized with bugs still present)

    Debt reduces the efficiency of future actions (tracked in graders).
    """
    debt = state.get("technical_debt", 0.0)

    if action_type == "defer_task":
        debt = min(1.0, debt + 0.03)
    elif action_type == "fix_bug":
        debt = max(0.0, debt - 0.05)  # Fixing bugs reduces debt

    state["technical_debt"] = round(debt, 4)
    return state


def apply_revenue_decay(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only for product_crisis_hard task.
    Revenue bleeds 5% per step while the critical bug (severity 5) exists.
    """
    if state.get("revenue") is None:
        return state  # Not tracked for this task

    has_critical = any(
        b.get("severity", 0) >= 5 for b in state.get("bugs", [])
    )
    if has_critical:
        state["revenue"] = max(0.0, round(state["revenue"] - 0.05, 4))
    else:
        # Slight recovery when crisis bug is fixed
        state["revenue"] = min(1.0, round(state["revenue"] + 0.02, 4))

    return state


def check_crisis_resolved(state: Dict[str, Any]) -> bool:
    """
    For product_crisis_hard: episode ends early if both conditions met.
    """
    if state.get("task_name") != "product_crisis_hard":
        return False

    stability_ok = state.get("stability", 0) >= state.get(
        "crisis_resolve_stability_threshold", 0.8
    )
    sat_ok = state.get("stakeholder_satisfaction", 0) >= state.get(
        "crisis_resolve_satisfaction_threshold", 0.7
    )
    return stability_ok and sat_ok


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _find_in_list(items: List[Dict], target_id: str) -> Optional[Dict]:
    """Find an item by its 'id' field in a list of dicts. Returns None if missing."""
    for item in items:
        if item.get("id") == target_id:
            return item
    return None


def build_info(state: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
    """
    Build the info dict returned with every StepResult.
    Exposes enough debug data for agents and evaluators to understand
    what's happening without revealing all hidden state.
    """
    return {
        "stability": round(state.get("stability", 0.0), 4),
        "growth": round(state.get("user_growth", 0.0), 4),
        "remaining_bugs": len(state.get("bugs", [])),
        "technical_debt": round(state.get("technical_debt", 0.0), 4),
        "step_count": state.get("step_count", 0),
        "revenue": state.get("revenue"),  # None for non-crisis tasks
        "crisis_resolved": state.get("crisis_resolved", False),
        "error": error,
    }
