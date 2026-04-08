"""
env.py
------
The main PM-OS environment class. This is the heart of the project.

It implements the OpenEnv interface:
  - reset()  → Observation
  - step()   → StepResult
  - state()  → Observation

Think of this like a video game engine:
  - reset() starts a new game
  - step() processes one player action and returns what happens
  - state() peeks at the current game state

All game logic flows through this class.
"""

import copy
import random
from typing import Any, Dict, Optional

from .graders import grade
from .models import Action, Bug, Feature, Observation, Request, StepResult
from .tasks import get_task
from .utils import (
    apply_bug_decay,
    apply_delayed_features,
    apply_revenue_decay,
    apply_stakeholder_decay,
    apply_technical_debt,
    build_info,
    check_crisis_resolved,
    compute_reward,
)


class PMEnv:
    """
    PM-OS: Product Manager Operating System

    An OpenEnv-compatible environment that simulates the real-world
    work of a Product Manager: balancing bugs, features, stakeholders,
    and business metrics under resource constraints.

    Usage:
        env = PMEnv()
        obs = await env.reset("bug_triage_easy")
        result = await env.step(Action(
            action_type="fix_bug",
            target_id="bug_001",
            reasoning="Critical severity-5 bug"
        ))
    """

    # Fixed seed for determinism — same result every time
    RANDOM_SEED = 42

    # Effort cost for each action type (consumes sprint_capacity)
    ACTION_COSTS = {
        "fix_bug": 2,
        "prioritize_feature": None,  # Uses feature.effort value
        "defer_task": 0,             # Free — but has side effects
        "respond_to_stakeholder": 1,
    }

    # Feature delay: 2–3 steps before it ships (simulates dev time)
    FEATURE_DELAY_MIN = 2
    FEATURE_DELAY_MAX = 3

    def __init__(self, task_name: str = "bug_triage_easy"):
        """
        Create the environment. Does NOT start an episode.
        Call reset() to begin.

        Args:
            task_name: which scenario to load. Options:
                "bug_triage_easy"
                "sprint_planning_medium"
                "product_crisis_hard"
        """
        random.seed(self.RANDOM_SEED)
        self._task_name = task_name
        self._state: Dict[str, Any] = {}
        self._episode_started = False

    # ─────────────────────────────────────────────────────────────────
    # OpenEnv API — the 3 required methods
    # ─────────────────────────────────────────────────────────────────

    async def reset(self, task_name: Optional[str] = None) -> Observation:
        """
        Start a fresh episode.

        1. (Re)seeds the random number generator for determinism
        2. Loads the task's initial state
        3. Snapshots initial values for graders
        4. Returns the first Observation (what the agent sees)

        Args:
            task_name: Override the task for this episode (optional).
                       If not given, uses the task set in __init__.

        Returns:
            Observation: Everything the agent can see at step 0.
        """
        # Determinism: always re-seed at episode start
        random.seed(self.RANDOM_SEED)

        # Load fresh task state
        if task_name:
            self._task_name = task_name
        raw = get_task(self._task_name)
        self._state = copy.deepcopy(raw)

        # Snapshot initial values so graders can measure improvement
        self._state["initial_critical_bugs"] = sum(
            1 for b in self._state.get("bugs", []) if b.get("severity", 0) >= 4
        )
        self._state["initial_bugs"] = copy.deepcopy(self._state.get("bugs", []))
        self._state["initial_backlog"] = copy.deepcopy(self._state.get("backlog", []))
        self._state["initial_capacity"] = self._state.get("sprint_capacity", 0)

        self._episode_started = True
        return self._build_observation()

    async def step(self, action: Action) -> StepResult:
        """
        Process one agent action and advance the world by one step.

        The full pipeline:
          1. Validate the action (does target exist? enough capacity?)
          2. Apply the action's direct effects
          3. Update hidden state (bug decay, debt, etc.)
          4. Fire any delayed feature effects
          5. Compute reward
          6. Increment step counter
          7. Check episode termination
          8. Return StepResult

        Args:
            action: The agent's chosen action this step.

        Returns:
            StepResult: new observation + reward + done flag + debug info.
        """
        if not self._episode_started:
            raise RuntimeError("Call reset() before step().")

        # Keep a copy of state before action (needed for reward computation)
        prev_state = copy.deepcopy(self._state)

        # ── STEP 1: Validate action ───────────────────────────
        valid, error = self._validate_action(action)

        # ── STEP 2: Apply action effects ──────────────────────
        if valid:
            self._apply_action(action)
        else:
            # Invalid action still costs a step (no free retries)
            pass

        # ── STEP 3: Update hidden state ───────────────────────
        self._state = apply_bug_decay(self._state)
        self._state = apply_stakeholder_decay(self._state)
        self._state = apply_technical_debt(self._state, action.action_type)
        self._state = apply_revenue_decay(self._state)

        # ── STEP 4: Fire delayed feature effects ──────────────
        self._state = apply_delayed_features(self._state)

        # ── STEP 5: Compute reward ────────────────────────────
        reward = compute_reward(
            action_type=action.action_type,
            target_id=action.target_id,
            prev_state=prev_state,
            next_state=self._state,
            action_valid=valid,
            error=error,
        )

        # ── STEP 6: Increment step counter ───────────────────
        self._state["step_count"] += 1

        # ── STEP 7: Check if crisis resolved (hard task only) ──
        if check_crisis_resolved(self._state):
            self._state["crisis_resolved"] = True

        # ── STEP 8: Determine done ────────────────────────────
        done = self._is_done()

        # ── STEP 9: Build return value ────────────────────────
        obs = self._build_observation()
        info = build_info(self._state, error=error)

        # Add final grade to info when episode ends
        if done:
            info["final_score"] = grade(self._task_name, self._state)

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    async def state(self) -> Observation:
        """
        Peek at the current state without advancing the episode.
        Useful for agents that want to re-check the observation.

        Returns:
            Observation: current state (same shape as reset() output).
        """
        if not self._episode_started:
            raise RuntimeError("Call reset() before state().")
        return self._build_observation()

    # ─────────────────────────────────────────────────────────────────
    # Action Validation
    # ─────────────────────────────────────────────────────────────────

    def _validate_action(self, action: Action):
        """
        Check if an action is valid.

        Rules:
          - target_id must exist in the relevant list
          - Agent must have enough sprint_capacity
          - Can't fix a bug that doesn't exist
          - Can't prioritize a feature not in the backlog
          - Can't respond to a request not in the queue

        Returns:
            (bool, Optional[str]): (is_valid, error_message_or_None)
        """
        t = action.action_type
        tid = action.target_id

        # ── fix_bug: must exist in bugs list ─────────────────
        if t == "fix_bug":
            if not any(b["id"] == tid for b in self._state.get("bugs", [])):
                return False, f"Bug '{tid}' not found in bug list."
            if self._state.get("sprint_capacity", 0) < self.ACTION_COSTS["fix_bug"]:
                return False, "Not enough sprint capacity to fix bug."

        # ── prioritize_feature: must exist in backlog ─────────
        elif t == "prioritize_feature":
            feat = next(
                (f for f in self._state.get("backlog", []) if f["id"] == tid), None
            )
            if not feat:
                return False, f"Feature '{tid}' not found in backlog."
            if self._state.get("sprint_capacity", 0) < feat["effort"]:
                return False, (
                    f"Feature '{tid}' needs {feat['effort']} capacity, "
                    f"only {self._state['sprint_capacity']} remaining."
                )

        # ── defer_task: must be a bug, feature, or request ───
        elif t == "defer_task":
            all_ids = (
                [b["id"] for b in self._state.get("bugs", [])] +
                [f["id"] for f in self._state.get("backlog", [])] +
                [r["id"] for r in self._state.get("stakeholder_requests", [])]
            )
            if tid not in all_ids:
                return False, f"Target '{tid}' not found anywhere in the backlog/bugs/requests."

        # ── respond_to_stakeholder: must be in requests ───────
        elif t == "respond_to_stakeholder":
            if not any(
                r["id"] == tid for r in self._state.get("stakeholder_requests", [])
            ):
                return False, f"Stakeholder request '{tid}' not found."
            if self._state.get("sprint_capacity", 0) < self.ACTION_COSTS["respond_to_stakeholder"]:
                return False, "Not enough sprint capacity to respond to stakeholder."

        else:
            return False, f"Unknown action_type: '{t}'."

        return True, None

    # ─────────────────────────────────────────────────────────────────
    # Action Application
    # ─────────────────────────────────────────────────────────────────

    def _apply_action(self, action: Action):
        """
        Apply the direct effects of a valid action.
        Side effects update self._state in-place.
        """
        t = action.action_type
        tid = action.target_id

        if t == "fix_bug":
            self._do_fix_bug(tid)

        elif t == "prioritize_feature":
            self._do_prioritize_feature(tid)

        elif t == "defer_task":
            self._do_defer_task(tid)

        elif t == "respond_to_stakeholder":
            self._do_respond_stakeholder(tid)

    def _do_fix_bug(self, bug_id: str):
        """
        Fix a bug:
          - Remove it from the bugs list
          - Improve stability based on severity
          - Consume sprint capacity (cost = 2)
        """
        bugs = self._state.get("bugs", [])
        bug = next((b for b in bugs if b["id"] == bug_id), None)
        if not bug:
            return

        # Remove from active bugs
        self._state["bugs"] = [b for b in bugs if b["id"] != bug_id]

        # Stability boost proportional to severity
        severity = bug.get("severity", 1)
        stability_gain = 0.05 * severity   # sev-5 = +0.25 stability
        self._state["stability"] = min(1.0, self._state["stability"] + stability_gain)

        # Consume capacity
        self._state["sprint_capacity"] = max(
            0, self._state["sprint_capacity"] - self.ACTION_COSTS["fix_bug"]
        )

    def _do_prioritize_feature(self, feature_id: str):
        """
        Queue a feature for development:
          - Remove from backlog
          - Consume sprint capacity = feature.effort
          - Add to delayed_feature_queue (fires after 2-3 steps)
        """
        backlog = self._state.get("backlog", [])
        feature = next((f for f in backlog if f["id"] == feature_id), None)
        if not feature:
            return

        # Remove from backlog
        self._state["backlog"] = [f for f in backlog if f["id"] != feature_id]

        # Consume capacity
        effort = feature.get("effort", 1)
        self._state["sprint_capacity"] = max(
            0, self._state["sprint_capacity"] - effort
        )

        # Queue delayed effect — will fire in 2 or 3 steps
        delay = random.randint(self.FEATURE_DELAY_MIN, self.FEATURE_DELAY_MAX)
        fires_at = self._state["step_count"] + delay

        self._state["delayed_feature_queue"].append({
            "feature_id": feature_id,
            "impact": feature.get("impact", 0.1),
            "fires_at_step": fires_at,
        })

    def _do_defer_task(self, target_id: str):
        """
        Defer a task (mark it as low priority for now):
          - No immediate effect on bugs/features lists
          - Small stakeholder satisfaction hit if it's a request
          - Technical debt increases (handled in utils)
          - No capacity consumed
        """
        # Check if it's a stakeholder request being deferred
        requests = self._state.get("stakeholder_requests", [])
        for req in requests:
            if req["id"] == target_id:
                urgency = req.get("urgency", 1)
                # Higher urgency request deferred = bigger satisfaction hit
                hit = 0.03 * urgency
                self._state["stakeholder_satisfaction"] = max(
                    0.0, self._state["stakeholder_satisfaction"] - hit
                )
                break

    def _do_respond_stakeholder(self, request_id: str):
        """
        Respond to a stakeholder request:
          - Remove from active requests
          - Satisfaction boost based on urgency
          - Consume 1 capacity point
        """
        requests = self._state.get("stakeholder_requests", [])
        req = next((r for r in requests if r["id"] == request_id), None)
        if not req:
            return

        # Remove from active queue
        self._state["stakeholder_requests"] = [
            r for r in requests if r["id"] != request_id
        ]

        # Also remove from patience tracker (no longer decaying)
        patience = self._state.get("stakeholder_patience", {})
        patience.pop(request_id, None)
        self._state["stakeholder_patience"] = patience

        # Satisfaction boost — higher urgency request handled = bigger boost
        urgency = req.get("urgency", 1)
        boost = 0.06 * urgency   # urgency-5 = +0.30 satisfaction
        self._state["stakeholder_satisfaction"] = min(
            1.0, self._state["stakeholder_satisfaction"] + boost
        )

        # Consume 1 capacity point
        self._state["sprint_capacity"] = max(
            0, self._state["sprint_capacity"] - self.ACTION_COSTS["respond_to_stakeholder"]
        )

    # ─────────────────────────────────────────────────────────────────
    # Episode Termination
    # ─────────────────────────────────────────────────────────────────

    def _is_done(self) -> bool:
        """
        Check if the episode should end.

        Ends when ANY of:
          1. Reached max_steps (time limit)
          2. sprint_capacity is 0 (out of resources)
          3. Crisis resolved (hard task only: stability+satisfaction both high)
        """
        if self._state["step_count"] >= self._state.get("max_steps", 10):
            return True

        if self._state.get("sprint_capacity", 1) <= 0:
            return True

        if self._state.get("crisis_resolved", False):
            return True

        return False

    # ─────────────────────────────────────────────────────────────────
    # Observation Builder
    # ─────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        """
        Convert internal state dict → typed Observation model.

        IMPORTANT: Hidden state fields (true_bug_impact, stakeholder_patience,
        delayed_feature_queue, technical_debt) are NOT included here.
        The agent must infer them from observable metric changes.
        """
        s = self._state

        # Convert raw dicts → typed Pydantic models
        backlog = [Feature(**f) for f in s.get("backlog", [])]
        bugs = [Bug(**b) for b in s.get("bugs", [])]
        requests = [Request(**r) for r in s.get("stakeholder_requests", [])]

        return Observation(
            backlog=backlog,
            bugs=bugs,
            sprint_capacity=s.get("sprint_capacity", 0),
            stakeholder_requests=requests,
            user_growth=round(s.get("user_growth", 0.0), 4),
            stability=round(s.get("stability", 0.0), 4),
            stakeholder_satisfaction=round(s.get("stakeholder_satisfaction", 0.0), 4),
            step_count=s.get("step_count", 0),
        )
