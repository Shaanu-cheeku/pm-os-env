"""
inference.py
------------
Runs an AI agent (via OpenAI-compatible API) through the PM-OS environment.

This script:
  1. Connects to a language model (Claude, GPT-4, Llama, etc.)
  2. Shows it the current observation
  3. Asks it to choose an action
  4. Sends that action to the environment
  5. Logs everything in the required [START]/[STEP]/[END] format
  6. Repeats until done

Environment variables required:
  - API_BASE_URL  : The base URL of your LLM API
  - MODEL_NAME    : Model identifier (e.g. "claude-3-5-sonnet-20241022")
  - HF_TOKEN      : HuggingFace token (used as API key)

Usage:
  export API_BASE_URL="https://api.anthropic.com"
  export MODEL_NAME="claude-3-5-sonnet-20241022"
  export HF_TOKEN="your_token_here"
  python inference.py
"""

import asyncio
import json
import os
import sys

from openai import OpenAI

from my_env import PMEnv, Action

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# Which tasks to run. Remove any you want to skip.
TASKS_TO_RUN = [
    "bug_triage_easy",
    "sprint_planning_medium",
    "product_crisis_hard",
]

# ─────────────────────────────────────────────
# LLM CLIENT SETUP
# ─────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key-needed",
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an experienced AI Product Manager.

Each turn you receive a JSON observation describing:
- backlog: features waiting to be built
- bugs: active bugs hurting users
- sprint_capacity: points you have left this sprint
- stakeholder_requests: demands from CEO/Board/customers
- user_growth, stability, stakeholder_satisfaction: your KPIs
- step_count: how many turns have passed

You must respond with ONLY a valid JSON object — no prose, no markdown:
{
    "action_type": "fix_bug" | "prioritize_feature" | "defer_task" | "respond_to_stakeholder",
    "target_id": "the exact id of the item you are acting on",
    "reasoning": "one sentence explaining your decision"
}

Decision guidelines:
- ALWAYS fix severity-4 or 5 bugs immediately — they bleed stability every step you wait
- Respond to urgency-4+ stakeholder requests within 2 steps or satisfaction collapses
- Balance capacity — don't overspend, but don't hoard capacity either
- Features take 2-3 steps to ship after prioritization — plan ahead
- Deferring is a valid tactic but overusing it increases technical debt
- In a crisis (stability < 0.3): all actions should be crisis-focused

Think step by step but return ONLY the JSON object."""

# ─────────────────────────────────────────────
# AGENT ACTION PARSER
# ─────────────────────────────────────────────

def parse_action(llm_response: str) -> Action:
    """
    Parse the LLM's JSON response into an Action object.
    Handles markdown code fences and whitespace.

    Falls back to a 'defer_task' with a dummy ID if parsing fails
    (ensures the episode continues even if the model returns garbage).
    """
    # Strip markdown code fences if present
    text = llm_response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        data = json.loads(text)
        return Action(
            action_type=data["action_type"],
            target_id=data["target_id"],
            reasoning=data.get("reasoning", ""),
        )
    except Exception as e:
        print(f"  ⚠️  Failed to parse LLM response: {e}")
        print(f"  ⚠️  Raw response: {llm_response[:200]}")
        # Fallback: return a safe no-op
        return Action(
            action_type="defer_task",
            target_id="__invalid__",
            reasoning="Parse error fallback",
        )

# ─────────────────────────────────────────────
# SINGLE-TASK RUNNER
# ─────────────────────────────────────────────

async def run_task(task_name: str) -> float:
    """
    Run one full episode of the given task.
    Returns the final grade score (0.0–1.0).
    """
    print(f"\n{'='*60}")
    print(f"[START] Task: {task_name}")
    print(f"{'='*60}")

    env = PMEnv(task_name=task_name)
    obs = await env.reset(task_name)

    # Conversation history — we send the full history each time
    # so the LLM understands context (what actions it already took)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    final_score = 0.0
    step_num = 0

    while True:
        step_num += 1

        # Format current observation as JSON for the LLM
        obs_dict = obs.model_dump()
        obs_json = json.dumps(obs_dict, indent=2)

        # Build the user message
        user_message = f"Step {step_num} — Current State:\n{obs_json}\n\nWhat is your action?"
        messages.append({"role": "user", "content": user_message})

        # ── Ask the LLM for an action ──────────────────────────
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=300,
                temperature=0.2,   # Low temp = more consistent PM decisions
            )
            llm_text = response.choices[0].message.content
        except Exception as e:
            print(f"  ❌ LLM API error: {e}")
            llm_text = '{"action_type": "defer_task", "target_id": "__error__", "reasoning": "API error"}'

        # Add assistant response to history
        messages.append({"role": "assistant", "content": llm_text})

        # ── Parse the action ───────────────────────────────────
        action = parse_action(llm_text)

        # ── Send action to environment ─────────────────────────
        result = await env.step(action)

        # ── Log the step ───────────────────────────────────────
        print(f"\n[STEP] {step_num}")
        print(f"  Action   : {action.action_type} → {action.target_id}")
        print(f"  Reasoning: {action.reasoning}")
        print(f"  Reward   : {result.reward:.2f}")
        print(f"  Stability: {result.info.get('stability', 0):.2f} | "
              f"Growth: {result.info.get('growth', 0):.2f} | "
              f"Bugs left: {result.info.get('remaining_bugs', 0)}")
        if result.info.get("error"):
            print(f"  ⚠️  Error: {result.info['error']}")
        if result.info.get("revenue") is not None:
            print(f"  Revenue  : {result.info['revenue']:.2f}")
        if result.info.get("crisis_resolved"):
            print(f"  ✅ CRISIS RESOLVED EARLY!")

        # Update observation for next step
        obs = result.observation

        # ── Check if episode is done ───────────────────────────
        if result.done:
            final_score = result.info.get("final_score", 0.0)
            print(f"\n[END] Task: {task_name}")
            print(f"  Steps taken : {step_num}")
            print(f"  Final score : {final_score:.2f}")
            print(f"  Stability   : {result.observation.stability:.2f}")
            print(f"  Growth      : {result.observation.user_growth:.2f}")
            print(f"  Satisfaction: {result.observation.stakeholder_satisfaction:.2f}")
            break

    return final_score

# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

async def main():
    """Run all tasks and print a summary scorecard."""
    print("\n🚀 PM-OS OpenEnv Inference Runner")
    print(f"   Model : {MODEL_NAME}")
    print(f"   API   : {API_BASE_URL}")

    scores = {}
    for task_name in TASKS_TO_RUN:
        score = await run_task(task_name)
        scores[task_name] = score

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 FINAL SCORECARD")
    print(f"{'='*60}")
    for task, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task:<30} {bar} {score:.2f}")

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  {'AVERAGE':<30} {avg:.2f}")
    print(f"{'='*60}\n")

    # Exit with error code if average score is terrible
    if avg < 0.2:
        print("⚠️  WARNING: Average score < 0.2 — check your model/API config.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
