# 🧠 PM-OS — AI Product Manager Simulator

**An OpenEnv-compatible environment that simulates the real-world workflow of a Product Manager.**

Agents must prioritize features, fix bugs, manage sprint capacity, respond to stakeholders, and maintain product metrics — all under realistic constraints and delayed consequences.

---

## 📁 Project Structure

```
pm_os_env/
│
├── my_env/                    ← The environment package
│   ├── __init__.py            ← Exports PMEnv, Action, Observation, StepResult
│   ├── env.py                 ← Main PMEnv class (reset/step/state logic)
│   ├── models.py              ← All Pydantic data models
│   ├── tasks.py               ← 3 task definitions (easy/medium/hard)
│   ├── graders.py             ← Deterministic graders for each task
│   └── utils.py               ← Reward math + hidden state helpers
│
├── app.py                     ← FastAPI server (HTTP endpoints)
├── inference.py               ← LLM agent runner script
├── test_env.py                ← Test suite (run before submitting!)
├── openenv.yaml               ← OpenEnv config
├── Dockerfile                 ← Container config
├── requirements.txt           ← Python dependencies
└── README.md                  ← This file
```

---

## 🚀 Quick Start (Step-by-Step for Beginners)

### Step 1: Make sure you have Python 3.10+

```bash
python --version
# Should print: Python 3.10.x or higher
```

### Step 2: Create a virtual environment

A virtual environment keeps your project's dependencies separate from your system Python.

```bash
# Navigate to the project folder
cd pm_os_env

# Create a virtual environment named "venv"
python -m venv venv

# Activate it:
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Your terminal prompt should now show (venv) at the start
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs: pydantic, fastapi, uvicorn, openai, pyyaml.

### Step 4: Run the tests

Always run tests first to confirm everything is working:

```bash
python test_env.py
```

Expected output:
```
=======================================================
  PM-OS Environment Test Suite
=======================================================

📋 TEST: Reset
  ✅ reset/bug_triage_easy returns Observation
  ✅ reset/bug_triage_easy step_count=0
  ...

=======================================================
  Results: 28/28 passed
  🎉 All tests passed!
=======================================================
```

### Step 5: Run the FastAPI server locally

```bash
python app.py
```

Then open: http://localhost:7860/docs

You'll see the interactive API docs. Try:
- `GET /health` → should return `{"status": "ok"}`
- `POST /reset` with body `{"task_name": "bug_triage_easy"}`
- `POST /step` with an action

### Step 6: Run the inference script (with a real LLM)

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_openai_api_key_here"

python inference.py
```

---

## 🎮 How the Environment Works

### The Agent's View (Observation)

Every step, the agent sees:

```python
Observation(
    backlog=[Feature(id="feat_001", impact=0.3, effort=3, deadline=15)],
    bugs=[Bug(id="bug_001", severity=5, users_affected=5000)],
    sprint_capacity=10,
    stakeholder_requests=[Request(id="req_001", stakeholder="CTO", urgency=5)],
    user_growth=0.5,
    stability=0.4,
    stakeholder_satisfaction=0.6,
    step_count=0
)
```

### The Agent's Choices (Actions)

```python
# Fix a bug
Action(action_type="fix_bug", target_id="bug_001", reasoning="Critical severity-5")

# Queue a feature
Action(action_type="prioritize_feature", target_id="feat_001", reasoning="High ROI")

# Skip something
Action(action_type="defer_task", target_id="bug_003", reasoning="Low severity, no time")

# Reply to stakeholder
Action(action_type="respond_to_stakeholder", target_id="req_001", reasoning="CTO needs answer")
```

### What Happens Each Step

1. Agent submits action
2. Environment validates it (wrong ID? not enough capacity? → penalty)
3. Direct effects applied (bug removed, capacity reduced, satisfaction updated)
4. Hidden decay runs (unfixed bugs hurt stability, impatient stakeholders hurt satisfaction)
5. Delayed features ship if their timer fires
6. Reward computed (see below)
7. Step counter increments
8. Check if episode is done

### Reward Function

| Event | Reward |
|-------|--------|
| Fix critical bug (sev 4-5) | +0.30 |
| Fix medium bug (sev 3) | +0.15 |
| Fix minor bug (sev 1-2) | +0.05 |
| Stability improvement | +0.20 × delta |
| Feature impact realized | +0.10 × impact |
| Stakeholder response (weighted by urgency) | +0.20 × (urgency/5) |
| Ignored critical bug each step | -0.40 × count |
| Over capacity | -0.30 |
| Defer action | -0.05 |
| Per-step existence penalty | -0.05 |

All rewards normalized to [0.0, 1.0].

### Hidden State (Agent Cannot See)

| Variable | Effect |
|----------|--------|
| `true_bug_impact` | Actual stability loss per step per unfixed bug |
| `stakeholder_patience` | How many steps before stakeholders escalate |
| `delayed_feature_queue` | Features in flight (ship in 2-3 steps) |
| `technical_debt` | Accumulates from defers, reduces via bug fixes |

---

## 📋 Tasks

### Task 1: `bug_triage_easy`
- **Scenario**: Production is broken. 3 bugs, no features.
- **Goal**: Fix high-severity bugs before they collapse stability
- **Key challenge**: Severity-5 bug loses 8% stability/step if ignored
- **Max steps**: 10

### Task 2: `sprint_planning_medium`
- **Scenario**: End-of-quarter sprint with mixed priorities
- **Goal**: Balance bugs + features within limited capacity (12 points for 14 points of work)
- **Key challenges**: Feature with urgent deadline at step 5; CEO patience = 2 steps
- **Max steps**: 15

### Task 3: `product_crisis_hard`
- **Scenario**: Full product crisis with data-loss bug, revenue bleeding, furious board
- **Goal**: Resolve crisis (stability > 0.8, satisfaction > 0.7)
- **Key challenges**: Revenue drops 5%/step, Board patience = 1 step, high technical debt
- **Max steps**: 20 (or early exit if crisis resolved)

---

## 🐳 Docker

### Build the image:

```bash
docker build -t pm-os-env .
```

### Run it:

```bash
docker run -p 7860:7860 pm-os-env
```

Then test: `curl http://localhost:7860/health`

### Memory usage:
- Base image: ~150MB
- App + dependencies: ~300MB total
- Well under the 8GB limit

---

## 🤗 Deploying to HuggingFace Spaces

1. Create a new Space on huggingface.co
2. Set Space type to "Docker"
3. Upload all project files
4. Set environment variables in the Space settings:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
5. Space will build and deploy automatically
6. Validate with: `curl https://your-space.hf.space/health`

---

## ✅ OpenEnv Validation Checklist

Before submitting, verify:

- [ ] `python test_env.py` → all tests pass
- [ ] `docker build -t pm-os-env .` → builds successfully
- [ ] `docker run -p 7860:7860 pm-os-env` → starts without errors
- [ ] `curl localhost:7860/reset` with POST body → returns 200 + Observation JSON
- [ ] `openenv validate` → passes (if openenv CLI installed)
- [ ] All rewards are in [0.0, 1.0]
- [ ] All grader scores are in [0.0, 1.0]
- [ ] Episode always terminates (max_steps is finite)
- [ ] `random.seed(42)` used (deterministic)

---

## 🧪 Testing Individual Components

```python
import asyncio
from my_env import PMEnv, Action

async def quick_test():
    env = PMEnv("product_crisis_hard")
    obs = await env.reset()
    print(f"Bugs: {len(obs.bugs)}, Stability: {obs.stability}")

    result = await env.step(Action(
        action_type="fix_bug",
        target_id="bug_006",
        reasoning="Fix data-loss crisis bug immediately"
    ))
    print(f"Reward: {result.reward}, Done: {result.done}")

asyncio.run(quick_test())
```

---

## 📝 License

MIT — free to use, modify, and submit to competitions.
