import asyncio
from my_env import PMEnv, Action

async def test():
    env = PMEnv("bug_triage_easy")
    obs = await env.reset()
    print("=== Starting Episode ===")
    print(f"Bugs: {len(obs.bugs)}, Capacity: {obs.sprint_capacity}, Stability: {obs.stability}")

    actions = [
        Action(action_type="fix_bug", target_id="bug_001", reasoning="Critical bug first"),
        Action(action_type="fix_bug", target_id="bug_002", reasoning="Next priority"),
        Action(action_type="respond_to_stakeholder", target_id="req_001", reasoning="CTO needs answer"),
        Action(action_type="fix_bug", target_id="bug_003", reasoning="Last bug"),
    ]

    for action in actions:
        result = await env.step(action)
        print(f"Action: {action.action_type} -> {action.target_id} | Reward: {result.reward:.2f} | Stability: {result.observation.stability:.2f} | Done: {result.done}")
        if result.done:
            print(f"Episode finished! Final score: {result.info['final_score']}")
            break

asyncio.run(test())