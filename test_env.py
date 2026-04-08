"""
test_env.py
-----------
Tests for PM-OS environment.

Run with: python test_env.py
(No pytest needed — pure asyncio + assert)

Tests cover:
  - All 3 tasks reset correctly
  - All 4 action types work
  - Invalid actions are handled gracefully
  - Reward is always in [0.0, 1.0]
  - Done flag triggers correctly
  - Graders return valid scores
  - Determinism: same seed = same result
"""

import asyncio
import sys

from my_env import PMEnv, Action
from my_env.graders import grade_bug_triage, grade_crisis, grade_sprint


# ─────────────────────────────────────────────
# TEST HELPERS
# ─────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
results = []

def check(name: str, condition: bool, detail: str = ""):
    symbol = PASS if condition else FAIL
    msg = f"  {symbol} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, condition))
    return condition


# ─────────────────────────────────────────────
# TEST: Reset
# ─────────────────────────────────────────────

async def test_reset():
    print("\n📋 TEST: Reset")

    for task in ["bug_triage_easy", "sprint_planning_medium", "product_crisis_hard"]:
        env = PMEnv(task_name=task)
        obs = await env.reset()
        check(f"reset/{task} returns Observation", obs is not None)
        check(f"reset/{task} step_count=0", obs.step_count == 0)
        check(
            f"reset/{task} sprint_capacity > 0",
            obs.sprint_capacity > 0,
            f"got {obs.sprint_capacity}"
        )
        check(
            f"reset/{task} stability in [0,1]",
            0.0 <= obs.stability <= 1.0,
            f"got {obs.stability}"
        )


# ─────────────────────────────────────────────
# TEST: Fix Bug
# ─────────────────────────────────────────────

async def test_fix_bug():
    print("\n🐛 TEST: Fix Bug")

    env = PMEnv("bug_triage_easy")
    obs = await env.reset()

    initial_bugs = len(obs.bugs)
    initial_stability = obs.stability
    bug_id = obs.bugs[0].id  # First bug (should be severity-5)

    result = await env.step(Action(
        action_type="fix_bug",
        target_id=bug_id,
        reasoning="Testing fix_bug action"
    ))

    check("fix_bug returns StepResult", result is not None)
    check(
        "fix_bug removes bug",
        len(result.observation.bugs) == initial_bugs - 1,
        f"{initial_bugs} → {len(result.observation.bugs)}"
    )
    check(
        "fix_bug reward in [0,1]",
        0.0 <= result.reward <= 1.0,
        f"reward={result.reward}"
    )
    check(
        "fix_bug step_count increments",
        result.observation.step_count == 1
    )
    check(
        "fix_bug done=False (still steps left)",
        result.done == False
    )


# ─────────────────────────────────────────────
# TEST: Prioritize Feature
# ─────────────────────────────────────────────

async def test_prioritize_feature():
    print("\n🚀 TEST: Prioritize Feature")

    env = PMEnv("sprint_planning_medium")
    obs = await env.reset()

    initial_backlog = len(obs.backlog)
    initial_capacity = obs.sprint_capacity
    feat = obs.backlog[0]  # First feature

    result = await env.step(Action(
        action_type="prioritize_feature",
        target_id=feat.id,
        reasoning="Testing feature prioritization"
    ))

    check(
        "prioritize_feature removes from backlog",
        len(result.observation.backlog) == initial_backlog - 1,
        f"{initial_backlog} → {len(result.observation.backlog)}"
    )
    check(
        "prioritize_feature consumes capacity",
        result.observation.sprint_capacity == initial_capacity - feat.effort,
        f"{initial_capacity} → {result.observation.sprint_capacity} (effort={feat.effort})"
    )
    check(
        "prioritize_feature reward in [0,1]",
        0.0 <= result.reward <= 1.0
    )


# ─────────────────────────────────────────────
# TEST: Respond to Stakeholder
# ─────────────────────────────────────────────

async def test_respond_stakeholder():
    print("\n🤝 TEST: Respond to Stakeholder")

    env = PMEnv("bug_triage_easy")
    obs = await env.reset()

    initial_sat = obs.stakeholder_satisfaction
    req = obs.stakeholder_requests[0]

    result = await env.step(Action(
        action_type="respond_to_stakeholder",
        target_id=req.id,
        reasoning="Testing stakeholder response"
    ))

    check("respond_to_stakeholder returns StepResult", result is not None)
    check(
        "respond_to_stakeholder removes request",
        len(result.observation.stakeholder_requests) == len(obs.stakeholder_requests) - 1
    )
    check(
        "respond_to_stakeholder reward in [0,1]",
        0.0 <= result.reward <= 1.0
    )


# ─────────────────────────────────────────────
# TEST: Defer Task
# ─────────────────────────────────────────────

async def test_defer_task():
    print("\n⏸️  TEST: Defer Task")

    env = PMEnv("sprint_planning_medium")
    obs = await env.reset()

    bug_id = obs.bugs[0].id

    result = await env.step(Action(
        action_type="defer_task",
        target_id=bug_id,
        reasoning="Testing defer"
    ))

    check("defer_task returns StepResult", result is not None)
    check(
        "defer_task reward in [0,1]",
        0.0 <= result.reward <= 1.0,
        f"reward={result.reward}"
    )
    # Deferred bug stays in the list
    check(
        "defer_task bug remains in bugs list",
        any(b.id == bug_id for b in result.observation.bugs)
    )


# ─────────────────────────────────────────────
# TEST: Invalid Actions
# ─────────────────────────────────────────────

async def test_invalid_actions():
    print("\n🚫 TEST: Invalid Actions")

    env = PMEnv("bug_triage_easy")
    obs = await env.reset()

    # Invalid: nonexistent bug ID
    result = await env.step(Action(
        action_type="fix_bug",
        target_id="nonexistent_id_xyz",
        reasoning="This should fail"
    ))

    check(
        "invalid target_id returns error in info",
        result.info.get("error") is not None,
        f"error='{result.info.get('error')}'"
    )
    check(
        "invalid action still gives reward in [0,1]",
        0.0 <= result.reward <= 1.0,
        f"reward={result.reward}"
    )
    check(
        "invalid action still increments step_count",
        result.observation.step_count == 1
    )


# ─────────────────────────────────────────────
# TEST: Episode Termination
# ─────────────────────────────────────────────

async def test_episode_termination():
    print("\n🏁 TEST: Episode Termination")

    env = PMEnv("bug_triage_easy")
    obs = await env.reset()

    # Run until done, deferring every step (worst strategy)
    done = False
    steps = 0
    last_result = None

    while not done and steps < 50:  # Hard cap prevents infinite loop
        result = await env.step(Action(
            action_type="defer_task",
            target_id="bug_001",
            reasoning="Deferring everything to test termination"
        ))
        done = result.done
        steps += 1
        last_result = result

    check(
        "episode eventually terminates",
        done,
        f"terminated after {steps} steps"
    )
    check(
        "final StepResult has final_score",
        last_result is not None and "final_score" in last_result.info,
        f"score={last_result.info.get('final_score')}"
    )
    check(
        "final_score in [0,1]",
        0.0 <= last_result.info.get("final_score", -1) <= 1.0
    )


# ─────────────────────────────────────────────
# TEST: Determinism
# ─────────────────────────────────────────────

async def test_determinism():
    print("\n🔁 TEST: Determinism")

    # Run the same sequence twice and compare results
    async def run_sequence():
        env = PMEnv("bug_triage_easy")
        obs = await env.reset()
        result = await env.step(Action(
            action_type="fix_bug",
            target_id="bug_001",
            reasoning="Determinism test"
        ))
        return result.reward, result.observation.stability

    r1 = await run_sequence()
    r2 = await run_sequence()

    check(
        "same seed produces same reward",
        r1[0] == r2[0],
        f"{r1[0]} == {r2[0]}"
    )
    check(
        "same seed produces same stability",
        r1[1] == r2[1],
        f"{r1[1]} == {r2[1]}"
    )


# ─────────────────────────────────────────────
# TEST: Graders
# ─────────────────────────────────────────────

async def test_graders():
    print("\n📊 TEST: Graders")

    # Build minimal state dicts for grader testing
    bug_triage_state = {
        "bugs": [],                  # All bugs fixed
        "stability": 0.9,
        "stakeholder_satisfaction": 0.8,
        "initial_critical_bugs": 1,
    }
    score = grade_bug_triage(bug_triage_state)
    check("grade_bug_triage in [0,1]", 0.0 <= score <= 1.0, f"score={score}")
    check("grade_bug_triage perfect state ≈ high score", score > 0.7, f"score={score}")

    sprint_state = {
        "bugs": [],
        "backlog": [],
        "sprint_capacity": 0,
        "stakeholder_satisfaction": 0.9,
        "initial_bugs": [{"id": "b1", "severity": 4}],
        "initial_backlog": [{"id": "f1", "impact": 0.5}],
        "initial_capacity": 12,
    }
    score2 = grade_sprint(sprint_state)
    check("grade_sprint in [0,1]", 0.0 <= score2 <= 1.0, f"score={score2}")

    crisis_state = {
        "revenue": 0.9,
        "stability": 0.85,
        "stakeholder_satisfaction": 0.75,
        "technical_debt": 0.1,
        "crisis_resolved": True,
    }
    score3 = grade_crisis(crisis_state)
    check("grade_crisis in [0,1]", 0.0 <= score3 <= 1.0, f"score={score3}")
    check("grade_crisis with crisis resolved ≈ high score", score3 > 0.7, f"score={score3}")


# ─────────────────────────────────────────────
# TEST: State endpoint
# ─────────────────────────────────────────────

async def test_state_endpoint():
    print("\n👁️  TEST: State endpoint")

    env = PMEnv("bug_triage_easy")
    obs1 = await env.reset()
    obs2 = await env.state()

    check(
        "state() matches reset() observation",
        obs1.step_count == obs2.step_count and obs1.stability == obs2.stability
    )

    # After a step, state() should reflect the new state
    await env.step(Action(
        action_type="fix_bug",
        target_id="bug_001",
        reasoning="State test"
    ))
    obs3 = await env.state()
    check(
        "state() updates after step()",
        obs3.step_count == 1
    )


# ─────────────────────────────────────────────
# MAIN TEST RUNNER
# ─────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("  PM-OS Environment Test Suite")
    print("=" * 55)

    await test_reset()
    await test_fix_bug()
    await test_prioritize_feature()
    await test_respond_stakeholder()
    await test_defer_task()
    await test_invalid_actions()
    await test_episode_termination()
    await test_determinism()
    await test_graders()
    await test_state_endpoint()

    # ── Summary ──────────────────────────────────────────────
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    failed = [(name, ok) for name, ok in results if not ok]

    print(f"\n{'='*55}")
    print(f"  Results: {passed}/{total} passed")
    if failed:
        print(f"\n  Failed tests:")
        for name, _ in failed:
            print(f"    ❌ {name}")
    else:
        print("  🎉 All tests passed!")
    print("=" * 55)

    # Return exit code 1 if any test failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
