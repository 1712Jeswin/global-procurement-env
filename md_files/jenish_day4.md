# Jenish (P1) — Day 4 Instructions
**Role:** Env Core | **Theme:** Train remaining models, harden, final verification

> ⚠️ **Today's Critical Gates (both must pass):**
> 1. `baseline.py` runs cleanly against the live HF Space URL and prints scores for all 3 tasks
> 2. `inference.py` runs cleanly against the live HF Space URL and prints LLM agent scores for all 3 tasks

> 🚫 **No new features today.** Fix, train, and verify only.

---

## 🔗 Cross-Team Dependency — What You Need Today

**From Jamal (morning):** The live Hugging Face Space URL from Day 3's deployment. Update `baseline.py` and `trained_agent.py` to point at that URL instead of `localhost`. Confirm Jamal's `/metrics` route is live so your episode counts appear there.

**From Jeswin (ongoing):** Jeswin is running 500 random episodes today to stress-test the system. If he finds crashes or unexpected behaviour in the environment, coordinate immediately — some bugs may be in your code.

---

## Phase 1 — Train PPO on Task 2

**What you're building:** The second trained model. Task 2 includes India and EU suppliers (6 total), runs for 50 steps, and includes one disruption at step 8 (port strike India). It requires 100,000 timesteps to learn because it has a larger action space and more complex constraints.

Open `train.py` and add Task 2 training:

```python
if __name__ == "__main__":
    # Day 3 already trained Task 1 — don't re-run that unless needed
    train_task(task_id=2, total_timesteps=100000, save_path="models/task2_ppo")
```

Run this and let it complete. Expect 15–30 minutes on CPU.

### ✅ Pitstop Audit — Phase 1

After training completes, verify the model file exists and is under 150MB:

```bash
ls -lh models/task2_ppo.zip
```

Also do a quick sanity evaluation:
```python
# Quick check — not the full trained_agent.py run
from stable_baselines3 import PPO
model = PPO.load("models/task2_ppo")
print("Model loaded successfully")
print(model.policy)  # Should show MlpPolicy architecture
```
**File exists, loads without error, and is under 150MB → Phase 1 complete.**

---

## Phase 2 — Train PPO on Task 3

**What you're building:** The hardest model. Task 3 includes all 10 suppliers across 3 countries, runs 100 steps, has all 3 disruptions, and the agent has access to all 7 actions including `escalate`. Despite the added complexity, 50,000 timesteps is sufficient because the graders reward meaningful behaviour, not perfect play.

```python
train_task(task_id=3, total_timesteps=50000, save_path="models/task3_ppo")
```

**Important:** Task 3's action space is size 7 (actions 0–6). Double-check that your `ProcurementGymWrapper` initialised with `task=3` uses `spaces.Discrete(7)` — not 4 or 6. A mismatch here will cause SB3 to crash silently during training with a shape error.

### ✅ Pitstop Audit — Phase 2

```bash
ls -lh models/task3_ppo.zip  # exists, < 150MB
```

```python
from stable_baselines3 import PPO
from env.gym_wrapper import ProcurementGymWrapper

model = PPO.load("models/task3_ppo")
env = ProcurementGymWrapper(task=3, seed=42)
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print(0 <= int(action) <= 6)  # True — valid action range for Task 3
```
**Model predicts a valid action in [0, 6] → Phase 2 complete.**

---

## Phase 3 — Full Evaluation Against Live URL

**What you're building:** The final proof that everything works end-to-end. Update `trained_agent.py` to point at the live Hugging Face URL and run all 3 tasks. The trained agent must score higher than the random baseline on every task.

Update the URL at the top of both scripts:

```python
# In baseline.py and trained_agent.py
BASE_URL = "https://YOUR_USERNAME-global-procurement-env.hf.space"
```

Then add Tasks 2 and 3 to `trained_agent.py`:

```python
if __name__ == "__main__":
    baseline_scores = {}
    trained_scores = {}

    for task_id in [1, 2, 3]:
        # You can also run baseline here for direct comparison
        trained_avg = evaluate_model(task_id, f"models/task{task_id}_ppo")
        trained_scores[task_id] = trained_avg

    print("\n--- Final Comparison ---")
    for t in [1, 2, 3]:
        print(f"Task {t}: trained={trained_scores[t]:.4f}")
```

### ✅ Pitstop Audit — Phase 3

Expected output (your numbers will differ, but the pattern should hold):
```
Task 1: avg_reward over 5 episodes = 0.6123
Task 2: avg_reward over 5 episodes = 0.4871
Task 3: avg_reward over 5 episodes = 0.3509

--- Final Comparison ---
Task 1: trained=0.6123
Task 2: trained=0.4871
Task 3: trained=0.3509
```

Task 3 scoring lower than Task 1 is completely expected — it's a harder problem. What matters is that each trained score is higher than the random baseline from Day 3. **If any trained score equals or is lower than random → re-train that task with 2× timesteps.**

---

## Phase 4 — Verify `inference.py` Against Live URL

**What you're doing:** Running the mandatory LLM agent script against the live HF Space and confirming it completes within the 20-minute budget. This is a separate submission requirement from `baseline.py` — judges check both files. `inference.py` missing or crashing = disqualification.

Update the `ENV_URL` in `inference.py` to point at the live HF Space:

```python
ENV_URL = os.getenv("ENV_URL", "https://YOUR_USERNAME-global-procurement-env.hf.space")
```

Then run it with real credentials:

```bash
ENV_URL=https://YOUR_USERNAME-global-procurement-env.hf.space \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
HF_TOKEN=your_hf_token \
python inference.py
```

**Expected output:**
```
=== GlobalProcurementEnv — LLM Inference ===
Model: mistralai/Mistral-7B-Instruct-v0.3
Environment: https://YOUR_USERNAME-global-procurement-env.hf.space

Task 1: LLM agent score = 0.XXXX
Task 2: LLM agent score = 0.XXXX
Task 3: LLM agent score = 0.XXXX
```

**If the LLM calls time out:** Reduce `MAX_STEPS` to 10 — it's better to get a lower score than to timeout. The 20-minute limit is hard.

**If you get HTTP 401 from the LLM API:** The `HF_TOKEN` is wrong or not set. Double-check the token has inference access on Hugging Face.

### ✅ Pitstop Audit — Phase 4

```bash
# Time the full run to verify <20 min
time python inference.py
# Must print 3 lines of scores AND exit in under 20 minutes
```

Pass conditions: 3 score lines printed, no Python exceptions, no HTTP 500s from the environment API, total runtime under 20 minutes. **Both `baseline.py` and `inference.py` must pass before proceeding to the hardening patches below.**

---

## Phase 5 — Hardening Patches (Scoring Improvements)

> ℹ️ **Why these patches exist:** After the core submission gates are cleared (Phases 1–4), these five targeted fixes address real weaknesses flagged in the scoring rubric — without touching any file that would break the running HF Space. Each patch is self-contained. Apply them in the order listed.

> 🚫 **Do not apply these patches before Phase 4 is confirmed passing.** The critical gates come first.

---

### Patch A — Fix disruption duration bug (`env/disruption_engine.py`)

**What's wrong:** The current `check()` method only fires a disruption on the exact `trigger_step`. A port strike that should last 5 steps vanishes after one step — which means agents (and judges) see an unrealistic simulation.

**Why this is safe:** You are only changing the condition inside `check()`. The return type (a list of dicts) and the method signature are unchanged. Nothing in `api/main.py`, `train.py`, `baseline.py`, or `inference.py` needs to change.

**File:** `env/disruption_engine.py`

Find the `check()` method and replace the trigger condition:

```python
# BEFORE — fires only at the exact step number
def check(self, step_count: int) -> list:
    active = []
    for scenario in self.scenarios:
        if scenario["trigger_step"] == step_count:
            if self.task >= scenario.get("min_task", 1):
                active.append(scenario)
    return active
```

```python
# AFTER — respects duration_steps from scenarios.json
def check(self, step_count: int) -> list:
    active = []
    for scenario in self.scenarios:
        start = scenario["trigger_step"]
        end = start + scenario.get("duration_steps", 1)
        if start <= step_count < end:
            if self.task >= scenario.get("min_task", 1):
                active.append(scenario)
    return active
```

**Verify `scenarios.json` has `duration_steps` on every scenario.** If any are missing, add the field using the documented values:

| Scenario | `trigger_step` | `duration_steps` |
|---|---|---|
| `port_strike_india` | 8 | 5 |
| `war_reroute_eu` | 15 | 10 |
| `protest_france` | 22 | 3 |

### ✅ Pitstop Audit — Patch A

```python
from env.disruption_engine import DisruptionEngine

engine = DisruptionEngine(task=2, seed=42)

# Step 7: strike has not started yet
print(engine.check(7))   # []

# Step 8: strike starts
print(engine.check(8))   # [{'name': 'port_strike_india', ...}]

# Step 12: still within 5-step window (8 + 5 = 13, so 8–12 inclusive)
print(engine.check(12))  # [{'name': 'port_strike_india', ...}]

# Step 13: outside the window
print(engine.check(13))  # []
```

**Steps 8–12 return the disruption, step 13 does not → Patch A complete.**

---

### Patch B — Surface `lead_days` and `carbon` in the public Observation schema

**What's wrong:** The gym wrapper reads `lead_days` and `carbon` via `env._state` — a private attribute bypass. This means the public `Observation` Pydantic model is missing two fields that judges will check with `openenv validate`, and the leaky access breaks if the internal class ever changes.

**Two files to edit. Coordinate with Jamal on `api/schemas.py` before touching it.**

#### Step B-1 — Add fields to `api/schemas.py` (Jamal's file — tell him first)

Find the `Observation` class and add two new optional fields at the bottom of the field list:

```python
class Observation(BaseModel):
    step: int
    budget_remaining: float
    inventory: Dict[str, float]
    suppliers: List[SupplierObservation]
    active_disruptions: List[str]
    policy_violations_this_episode: int
    current_task: int
    grader_score: Optional[float] = None
    # --- ADD THESE TWO LINES ---
    cumulative_lead_days: int = 0
    cumulative_carbon: float = 0.0
```

Both fields default to `0` / `0.0`, so any existing client code that doesn't read them is unaffected.

#### Step B-2 — Populate the fields in `_build_observation()` (`env/global_procurement_env.py`)

Find `_build_observation()` and add the two new fields to the `Observation(...)` call:

```python
def _build_observation(self) -> Observation:
    supplier_obs = [
        SupplierObservation(
            id=s["id"], country=s["country"],
            price_usd=s["price_usd"], lead_days=s["lead_days"],
            carbon_tons=s["carbon_tons"], available=s["available"],
            applied_duty_rate=s.get("applied_duty_rate", 0.0)
        )
        for s in self.state.suppliers
    ]
    return Observation(
        step=self.state.step_count,
        budget_remaining=self.state.budget,
        inventory=self.state.inventory,
        suppliers=supplier_obs,
        active_disruptions=[],
        policy_violations_this_episode=self.state.violations,
        current_task=self.task,
        grader_score=getattr(self, "_grader_score", None),
        # --- ADD THESE TWO LINES ---
        cumulative_lead_days=self.state.lead_days,
        cumulative_carbon=self.state.carbon,
    )
```

#### Step B-3 — Remove private `_state` access from `env/gym_wrapper.py`

Find `_dict_to_array()` and replace the two private accesses with public dict reads:

```python
# BEFORE — reads internal private attribute, bypasses public contract
def _dict_to_array(self, obs: dict) -> np.ndarray:
    MAX_BUDGET = 200000.0
    return np.array([
        obs.get("budget_remaining", 0.0) / MAX_BUDGET,
        obs.get("inventory", {}).get("steel", 0.0) / 1000.0,
        obs.get("inventory", {}).get("chips", 0.0) / 1000.0,
        obs.get("inventory", {}).get("fabric", 0.0) / 1000.0,
        self.env._state.lead_days / 30.0,        # ← private access
        self.env._state.carbon / 50.0,           # ← private access
        obs.get("violations", 0) / 10.0,
        obs.get("step", 0) / 100.0,
    ], dtype=np.float32)
```

```python
# AFTER — reads from the public Observation dict
def _dict_to_array(self, obs: dict) -> np.ndarray:
    MAX_BUDGET = 200000.0
    return np.array([
        obs.get("budget_remaining", 0.0) / MAX_BUDGET,
        obs.get("inventory", {}).get("steel", 0.0) / 1000.0,
        obs.get("inventory", {}).get("chips", 0.0) / 1000.0,
        obs.get("inventory", {}).get("fabric", 0.0) / 1000.0,
        obs.get("cumulative_lead_days", 0) / 30.0,    # ← public schema
        obs.get("cumulative_carbon", 0.0) / 50.0,     # ← public schema
        obs.get("policy_violations_this_episode", 0) / 10.0,
        obs.get("step", 0) / 100.0,
    ], dtype=np.float32)
```

> ⚠️ **Note:** The observation array stays 8-dimensional. Shape is unchanged. The SB3 models you already trained are still compatible — no retraining needed.

### ✅ Pitstop Audit — Patch B

```python
from env.global_procurement_env import GlobalProcurementEnv
from api.schemas import Observation

env = GlobalProcurementEnv()
obs = env.reset(task=2, seed=42)

# Fields now exist on the public Observation object
assert hasattr(obs, "cumulative_lead_days"), "cumulative_lead_days missing from Observation"
assert hasattr(obs, "cumulative_carbon"), "cumulative_carbon missing from Observation"
print(obs.cumulative_lead_days)   # 0 at reset
print(obs.cumulative_carbon)      # 0.0 at reset

# Take a step and confirm they update
result_obs, _, _, _ = env.step(0)
print(result_obs.cumulative_lead_days)   # > 0 after a purchase
print(result_obs.cumulative_carbon)      # > 0.0 after a purchase

# Gym wrapper still returns correct shape
from env.gym_wrapper import ProcurementGymWrapper
import numpy as np
wrapper = ProcurementGymWrapper(task=2, seed=42)
arr, _ = wrapper.reset()
print(arr.shape)   # (8,) — unchanged
print(arr.dtype)   # float32 — unchanged
print(all(0.0 <= v <= 1.0 for v in arr))  # True
```

**All assertions pass and array shape is still (8,) → Patch B complete.**

---

### Patch C — Add stochastic disruption to Task 3 hard mode

**What's wrong:** All three disruptions fire at fixed, predictable steps. Any agent that runs a few episodes memorises the schedule. The hackathon rubric asks that the hard task *genuinely challenges frontier models* — a fully deterministic disruption pattern doesn't meet that bar.

**Why this is safe:** The stochastic path is gated behind `self.task == 3` and `step_count > 5`. Tasks 1 and 2 are completely unaffected. The change uses `random.random()` which is already seeded by `reset()` via `random.seed(seed)`, so runs with the same seed remain reproducible.

**Two files to edit.**

#### Step C-1 — Mark one scenario as stochastic in `disruptions/scenarios.json`

Add `"stochastic": true` to the `war_reroute_eu` scenario (the one at step 15). This flag is only read by the new disruption engine code — existing logic ignores unknown fields:

```json
{
  "name": "war_reroute_eu",
  "trigger_step": 15,
  "min_task": 3,
  "duration_steps": 10,
  "stochastic": true,
  "affected_suppliers": ["EU_01", "EU_02", "FR_01"],
  "effect": "lead_days_multiplier",
  "multiplier": 2.5
}
```

Leave `port_strike_india` and `protest_france` without the flag — they stay fully deterministic across all tasks.

#### Step C-2 — Add the stochastic check to `env/disruption_engine.py`

Add this block at the end of the `check()` method, **after** the existing loop (Patch A must already be applied):

```python
def check(self, step_count: int) -> list:
    active = []
    for scenario in self.scenarios:
        start = scenario["trigger_step"]
        end = start + scenario.get("duration_steps", 1)
        if start <= step_count < end:
            if self.task >= scenario.get("min_task", 1):
                active.append(scenario)

    # --- ADD THIS BLOCK (Task 3 only) ---
    # For Task 3, stochastic-flagged disruptions can also fire randomly
    # outside their fixed window. random.random() is seeded in reset() so
    # runs with the same seed are still reproducible.
    if self.task == 3 and step_count > 5:
        for scenario in self.scenarios:
            if scenario.get("stochastic") and scenario not in active:
                if random.random() < 0.30:   # 30% chance per step
                    active.append(scenario)
    # --- END ADDED BLOCK ---

    return active
```

Make sure `import random` is at the top of `disruption_engine.py` — it should already be there from Day 1, but verify.

### ✅ Pitstop Audit — Patch C

```python
import random
from env.disruption_engine import DisruptionEngine

# Tasks 1 and 2 must be unaffected
for task in [1, 2]:
    engine = DisruptionEngine(task=task, seed=42)
    at_step_20 = engine.check(20)
    stochastic_fired = any(s.get("stochastic") for s in at_step_20)
    assert not stochastic_fired, f"Task {task} should never see stochastic disruption"
    print(f"Task {task} at step 20: {[s['name'] for s in at_step_20]}")

# Task 3 should occasionally fire the stochastic disruption at step 20
# (outside the fixed window of step 15–24)
random.seed(99)  # use a seed where the 30% roll fires
engine3 = DisruptionEngine(task=3, seed=99)
hits = sum(1 for _ in range(20) if any(
    s.get("stochastic") for s in engine3.check(20)
))
print(f"Task 3 step 20: stochastic fired {hits}/20 times (expect ~6)")
assert hits > 0, "Stochastic disruption never fired in 20 rolls — check the probability logic"
```

**Tasks 1 and 2 unaffected, Task 3 fires stochastic disruption roughly 30% of the time → Patch C complete.**

---

### Patch D — Fix MAX_STEPS consistency for Task 2

**What's wrong:** The system design doc and task config are inconsistent. Phase 1 of this document correctly states Task 2 runs for 50 steps — but `env/tasks/task2_medium.py` may still have `MAX_STEPS = 15`. If that mismatch exists, the port strike at step 8 fires inside the episode but the episode terminates at step 15, meaning the agent only ever sees 7 steps of disruption at most. More critically, `openenv validate` reads the live API's episode length and will flag a mismatch against whatever `openenv.yaml` declares.

**Why this is safe:** This is a single constant change in one task config file. No method signatures, no data structures, no API contracts change. The only downstream effect is that episodes run longer — which is what the training and disruption schedule were already designed around.

**File:** `env/tasks/task2_medium.py`

Open the file and verify the current value, then correct it:

```python
# BEFORE — incorrect, causes premature episode termination
MAX_STEPS = 15
```

```python
# AFTER — matches Phase 1 description and disruption schedule
MAX_STEPS = 50
```

While you have the file open, also confirm these other constants are correct — they should already be set from Day 2, but verify:

```python
TASK_ID = 2
MAX_STEPS = 50                              # ← the fix
SUPPLIER_COUNTRIES = ["India", "EU"]        # 7 suppliers total
VALID_ACTIONS = [0, 1, 2, 3, 4, 5]         # actions 0–5, no escalate
DISRUPTIONS_ENABLED = True
```

Then open `openenv.yaml` (Jamal's file — tell him) and confirm the `max_steps` field for task 2 matches:

```yaml
tasks:
  - id: 1
    name: "Easy — India Only"
    max_steps: 20
  - id: 2
    name: "Medium — India + EU"
    max_steps: 50        # ← must match task2_medium.py
  - id: 3
    name: "Hard — All Countries"
    max_steps: 100
```

### ✅ Pitstop Audit — Patch D

```python
from env.global_procurement_env import GlobalProcurementEnv

env = GlobalProcurementEnv()
obs = env.reset(task=2, seed=42)

# Run until done and count steps
steps = 0
done = False
while not done:
    _, _, done, _ = env.step(0)
    steps += 1

print(f"Task 2 episode length: {steps}")
assert steps == 50, f"Expected 50 steps, got {steps} — check task2_medium.py MAX_STEPS"

# Also confirm the disruption fires AND persists inside the episode
env.reset(task=2, seed=42)
from env.disruption_engine import DisruptionEngine
engine = DisruptionEngine(task=2, seed=42)
assert len(engine.check(8))  > 0, "Disruption must fire at step 8"
assert len(engine.check(12)) > 0, "Disruption must still be active at step 12"
assert len(engine.check(50)) == 0, "No disruption at final step"
print("Task 2 MAX_STEPS = 50 confirmed, disruption window verified → Patch D complete.")
```

**Episode runs exactly 50 steps and disruption window fits inside it → Patch D complete.**

---

### Patch E — Fix inventory logic and surface available supplier count

**What's wrong — Part 1 (inventory):** Every purchase currently increments all three inventory categories (`steel += 1`, `chips += 1`, `fabric += 1`) regardless of what supplier was selected or what material they supply. This means buying from a chip supplier increases your steel count, which is visible in every `/step` response. Judges running the agentic evaluation will notice inventory grows uniformly.

**What's wrong — Part 2 (available supplier count):** When all suppliers are knocked out by a disruption, actions 0, 1, 2, and 4 silently do nothing — the agent burns a step with no feedback about why. The LLM agent in `inference.py` has no way to detect this because the observation only shows disruption *names*, not *how many suppliers are actually reachable right now*. Adding a single integer field fixes this without changing the observation array shape.

**Why this is safe:** Part 1 is a change inside `apply_action()` in `supply_chain_sim.py` only — the return dict shape and state mutation pattern are unchanged. Part 2 adds one field to the Observation schema with a default value, so all existing callers continue working. The gym wrapper observation array stays 8-dimensional (the new field is only for the HTTP/LLM path, not the neural network input).

---

#### Step E-1 — Add `material` field to `data/suppliers.json` (Jeswin's file — tell him)

Each supplier entry needs a `"material"` key specifying what it supplies. This is a data change only — no Python code reads it yet until Step E-2.

```json
[
  { "id": "IN_01", "name": "...", "country": "India", "price_usd": 40000,
    "lead_days": 5, "carbon_tons": 2.0, "available": true,
    "sanctioned_category": null, "material": "steel" },

  { "id": "IN_02", "name": "...", "country": "India", "price_usd": 35000,
    "lead_days": 7, "carbon_tons": 4.0, "available": true,
    "sanctioned_category": null, "material": "chips" },

  { "id": "IN_03", "name": "...", "country": "India", "price_usd": 38000,
    "lead_days": 6, "carbon_tons": 3.5, "available": true,
    "sanctioned_category": null, "material": "fabric" },

  { "id": "IN_04", "name": "...", "country": "India", "price_usd": 50000,
    "lead_days": 4, "carbon_tons": 1.8, "available": true,
    "sanctioned_category": null, "material": "steel" },

  { "id": "EU_01", "name": "...", "country": "EU", "price_usd": 42000,
    "lead_days": 8, "carbon_tons": 1.5, "available": true,
    "sanctioned_category": null, "material": "chips" },

  { "id": "EU_02", "name": "...", "country": "EU", "price_usd": 48000,
    "lead_days": 6, "carbon_tons": 1.2, "available": true,
    "sanctioned_category": null, "material": "fabric" },

  { "id": "FR_01", "name": "...", "country": "EU", "price_usd": 55000,
    "lead_days": 5, "carbon_tons": 0.9, "available": true,
    "sanctioned_category": null, "material": "steel" },

  { "id": "US_01", "name": "...", "country": "USA", "price_usd": 60000,
    "lead_days": 10, "carbon_tons": 2.2, "available": true,
    "sanctioned_category": null, "material": "chips" },

  { "id": "US_02", "name": "...", "country": "USA", "price_usd": 58000,
    "lead_days": 9,  "carbon_tons": 2.5, "available": true,
    "sanctioned_category": null, "material": "fabric" },

  { "id": "US_03", "name": "...", "country": "USA", "price_usd": 65000,
    "lead_days": 8,  "carbon_tons": 2.0, "available": true,
    "sanctioned_category": null, "material": "steel" }
]
```

> ⚠️ Keep all existing fields exactly as they are — only add the `"material"` key to each entry. If your actual supplier names and values differ from the above, preserve your values and only add the missing `"material"` field.

---

#### Step E-2 — Fix `apply_action()` in `env/supply_chain_sim.py`

Find every action branch that increments inventory and replace the three-line uniform increment with a single targeted increment using the supplier's `material` field.

```python
# BEFORE — uniform increment regardless of supplier type
# (appears inside action == 0, 1, 2, 4, 5 branches)
state.inventory["steel"] += 1
state.inventory["chips"] += 1
state.inventory["fabric"] += 1
```

```python
# AFTER — targeted increment using the supplier's declared material
# Replace all three lines above with this single line wherever they appear:
material = selected_supplier.get("material", "steel")  # default steel if missing
state.inventory[material] = state.inventory.get(material, 0.0) + 1
```

**For action 5 (split_order)** — which splits between two suppliers — apply one increment per supplier using each supplier's own material:

```python
# AFTER for split_order specifically
for supplier in [cheapest_1, cheapest_2]:
    mat = supplier.get("material", "steel")
    state.inventory[mat] = state.inventory.get(mat, 0.0) + 0.5  # half order each
```

**For actions 3 (reject_all) and 6 (escalate)** — no supplier is selected, so no inventory change. These branches should already not touch inventory — verify and leave them as-is.

---

#### Step E-3 — Add `available_supplier_count` to `api/schemas.py` (coordinate with Jamal)

Add one new field to the `Observation` class, after the fields added in Patch B:

```python
class Observation(BaseModel):
    step: int
    budget_remaining: float
    inventory: Dict[str, float]
    suppliers: List[SupplierObservation]
    active_disruptions: List[str]
    policy_violations_this_episode: int
    current_task: int
    grader_score: Optional[float] = None
    cumulative_lead_days: int = 0       # added in Patch B
    cumulative_carbon: float = 0.0      # added in Patch B
    # --- ADD THIS LINE ---
    available_supplier_count: int = 0
```

---

#### Step E-4 — Populate `available_supplier_count` in `_build_observation()` (`env/global_procurement_env.py`)

Add the count calculation and pass it into the `Observation(...)` call:

```python
def _build_observation(self) -> Observation:
    supplier_obs = [
        SupplierObservation(
            id=s["id"], country=s["country"],
            price_usd=s["price_usd"], lead_days=s["lead_days"],
            carbon_tons=s["carbon_tons"], available=s["available"],
            applied_duty_rate=s.get("applied_duty_rate", 0.0)
        )
        for s in self.state.suppliers
    ]
    # --- ADD THIS LINE ---
    available_count = sum(1 for s in self.state.suppliers if s.get("available", True))

    return Observation(
        step=self.state.step_count,
        budget_remaining=self.state.budget,
        inventory=self.state.inventory,
        suppliers=supplier_obs,
        active_disruptions=[],
        policy_violations_this_episode=self.state.violations,
        current_task=self.task,
        grader_score=getattr(self, "_grader_score", None),
        cumulative_lead_days=self.state.lead_days,
        cumulative_carbon=self.state.carbon,
        # --- ADD THIS LINE ---
        available_supplier_count=available_count,
    )
```

---

#### Step E-5 — Update `inference.py` prompt to use the new field

The LLM can now make a smarter decision when suppliers are knocked out. Find the `user_msg` string in `choose_action()` and add one line:

```python
# BEFORE
user_msg = f"""Current procurement state:
- Step: {observation.get('step', 0)}
- Budget remaining: ${observation.get('budget_remaining', 0):,.0f}
- Violations so far: {observation.get('policy_violations_this_episode', 0)}
- Active disruptions: {observation.get('active_disruptions', [])}
- Available suppliers: {len([s for s in observation.get('suppliers', []) if s.get('available')])}

Choose action 0-{max_action}. Reply with ONE integer only."""
```

```python
# AFTER — use the pre-computed field, add no-supplier guidance
available = observation.get('available_supplier_count', 0)
user_msg = f"""Current procurement state:
- Step: {observation.get('step', 0)}
- Budget remaining: ${observation.get('budget_remaining', 0):,.0f}
- Violations so far: {observation.get('policy_violations_this_episode', 0)}
- Active disruptions: {observation.get('active_disruptions', [])}
- Available suppliers: {available}

{"WARNING: No suppliers available this step. Use action 3 (reject_all) or action 6 (escalate) to avoid wasting the step." if available == 0 else ""}
Choose action 0-{max_action}. Reply with ONE integer only."""
```

> ⚠️ The gym wrapper `_dict_to_array()` is **not changed** in this patch. `available_supplier_count` is intentionally excluded from the 8-dimensional neural network input — the PPO agent already learns from disruption effects through the inventory and budget signals. Adding it to the array would require retraining all three models. The field is only surfaced to the HTTP API and the LLM agent.

### ✅ Pitstop Audit — Patch E

```python
from env.global_procurement_env import GlobalProcurementEnv
from env.disruption_engine import DisruptionEngine

env = GlobalProcurementEnv()

# --- Part 1: Inventory is now material-specific ---
obs = env.reset(task=1, seed=42)
steel_before = obs.inventory.get("steel", 0)
chips_before  = obs.inventory.get("chips", 0)
fabric_before = obs.inventory.get("fabric", 0)

# Action 0 = approve_cheapest — IN_03 (cheapest India supplier, material="fabric")
result_obs, _, _, _ = env.step(0)

# Only the material matching the selected supplier should increase
# (exact material depends on which supplier is cheapest — check suppliers.json)
total_before = steel_before + chips_before + fabric_before
total_after  = sum(result_obs.inventory.values())
assert total_after == total_before + 1, "Exactly one unit of one material should be added per purchase"

# Confirm it's NOT uniformly distributed
changed = sum(
    1 for k in ["steel", "chips", "fabric"]
    if result_obs.inventory.get(k, 0) != obs.inventory.get(k, 0)
)
assert changed == 1, f"Only 1 material should change per purchase, got {changed} changes"
print("Inventory correctly increments one material only ✅")

# --- Part 2: available_supplier_count is correct ---
obs2 = env.reset(task=2, seed=42)
assert hasattr(obs2, "available_supplier_count"), "available_supplier_count missing from Observation"

# At reset all suppliers should be available (no disruption yet)
assert obs2.available_supplier_count == 7, \
    f"Task 2 has 7 suppliers, got {obs2.available_supplier_count}"

# Advance to step 8 (port strike — IN_02, IN_03 knocked out → 5 of 7 remain)
for _ in range(8):
    obs2, _, _, _ = env.step(0)

assert obs2.available_supplier_count == 5, \
    f"During port strike, 5 of 7 suppliers should be available, got {obs2.available_supplier_count}"
print(f"available_supplier_count at step 8 (disruption active): {obs2.available_supplier_count} ✅")

# --- Part 3: gym wrapper shape still 8-dimensional ---
from env.gym_wrapper import ProcurementGymWrapper
import numpy as np
wrapper = ProcurementGymWrapper(task=2, seed=42)
arr, _ = wrapper.reset()
assert arr.shape == (8,), f"Gym wrapper shape changed — expected (8,), got {arr.shape}"
print(f"Gym wrapper observation shape: {arr.shape} ✅")
```

**Inventory increments one material only, available_supplier_count is correct, gym wrapper shape unchanged → Patch E complete.**

---

## Phase 6 — Add Model Scores to README & Push Everything

Tell Jeswin the three score columns to put in the README — he owns that file:

```markdown
## Model Performance

| Task | Random Baseline | Trained PPO | LLM Agent (inference.py) |
|------|----------------|-------------|--------------------------|
| Task 1 (Easy)   | 0.XX | 0.XX | 0.XX |
| Task 2 (Medium) | 0.XX | 0.XX | 0.XX |
| Task 3 (Hard)   | 0.XX | 0.XX | 0.XX |
```

Then push all model files, the patched env files, and `inference.py` to the HF Space:

```bash
git add models/task1_ppo.zip models/task2_ppo.zip models/task3_ppo.zip
git add env/disruption_engine.py env/global_procurement_env.py env/gym_wrapper.py
git add env/supply_chain_sim.py env/tasks/task2_medium.py
git add api/schemas.py disruptions/scenarios.json data/suppliers.json inference.py
git commit -m "Day 4 — trained models all 3 tasks + hardening patches A/B/C/D/E + verified inference.py"
git push origin main
git push space main
```

Watch the HF Space build logs. Confirm it goes green after the push.

### ✅ Pitstop Audit — Phase 6 (Day 4 Final Gate)

```bash
# 1. Space still responds after rebuild
curl -s https://YOUR_USERNAME-global-procurement-env.hf.space/health
# → {"status":"ok"}

# 2. New schema fields are visible in the live API
curl -s -X POST https://YOUR_USERNAME-global-procurement-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": 2, "seed": 42}' | python -m json.tool | grep -E "cumulative|available_supplier"
# → "cumulative_lead_days": 0,
# → "cumulative_carbon": 0.0,
# → "available_supplier_count": 7,

# 3. Random baseline agent — reproducible scores
python baseline.py
# → Task 1: 0.XXXX, Task 2: 0.XXXX, Task 3: 0.XXXX (no errors)

# 4. LLM inference agent — under 20 minutes
time python inference.py
# → Task 1: 0.XXXX, Task 2: 0.XXXX, Task 3: 0.XXXX (no errors, <20 min)
```

**All four commands succeed → Day 4 complete. Tell Jeswin and Jamal you're ready to submit.**

---

## Summary of Files Changed in Day 4

| File | Changed by | What changed |
|---|---|---|
| `models/task2_ppo.zip` | Jenish | New — trained Task 2 PPO |
| `models/task3_ppo.zip` | Jenish | New — trained Task 3 PPO |
| `baseline.py` | Jenish | Updated `BASE_URL` to live HF Space |
| `trained_agent.py` | Jenish | Added Task 2 and Task 3 evaluation |
| `inference.py` | Jenish | Updated `ENV_URL`; smarter no-supplier prompt (Patch E) |
| `env/disruption_engine.py` | Jenish | Patch A — duration fix; Patch C — stochastic Task 3 |
| `env/global_procurement_env.py` | Jenish | Patch B — cumulative fields; Patch E — available_supplier_count |
| `env/gym_wrapper.py` | Jenish | Patch B — replace private `_state` reads with public dict |
| `env/supply_chain_sim.py` | Jenish | Patch E — material-specific inventory increment |
| `env/tasks/task2_medium.py` | Jenish | Patch D — MAX_STEPS corrected to 50 |
| `api/schemas.py` | Jamal (coordinate) | Patch B — cumulative fields; Patch E — available_supplier_count |
| `disruptions/scenarios.json` | Jenish | Patch C — `"stochastic": true` on `war_reroute_eu` |
| `data/suppliers.json` | Jeswin (coordinate) | Patch E — `"material"` field added to all 10 suppliers |