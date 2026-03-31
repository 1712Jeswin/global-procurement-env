# test_day2_all.py -- All 4 Phase Audits for Jenish Day 2
import random
import sys

print("=" * 60)
print("JENISH DAY 2 -- FULL AUDIT")
print("=" * 60)

# =====================================================================
# PHASE 1 -- Main Environment Class
# =====================================================================
print("\n--- PHASE 1: Environment Class ---")
from env.global_procurement_env import GlobalProcurementEnv
from api.schemas import Observation, Reward

env = GlobalProcurementEnv()
obs = env.reset(task=1, seed=42)

assert isinstance(obs, Observation), "reset() must return Observation, got %s" % type(obs)
assert obs.step == 0
assert obs.budget_remaining == 200000.0
assert obs.current_task == 1
print("  reset() returns Observation: step=%d, budget=%.0f, task=%d" % (obs.step, obs.budget_remaining, obs.current_task))

result = env.step(0)
assert isinstance(result, tuple) and len(result) == 4, "step() must return a 4-tuple"
obs2, reward, done, info = result
assert isinstance(obs2, Observation), "First element must be Observation, got %s" % type(obs2)
assert isinstance(reward, Reward), "Second element must be Reward, got %s" % type(reward)
assert isinstance(done, bool), "Third element must be bool, got %s" % type(done)
assert isinstance(info, dict), "Fourth element must be dict, got %s" % type(info)
print("  step() returns 4-tuple: (%s, %s, %s, %s)" % (type(obs2).__name__, type(reward).__name__, done, info))

obs_a = env.reset(task=1, seed=42)
obs_b = env.reset(task=1, seed=42)
assert obs_a.budget_remaining == obs_b.budget_remaining
print("  Deterministic seeding: seed=42 -> budget=%.0f both times" % obs_a.budget_remaining)
print("  PHASE 1 PASSED")

# =====================================================================
# PHASE 2 -- Reward Formula
# =====================================================================
print("\n--- PHASE 2: Reward Formula ---")
env = GlobalProcurementEnv()
env.reset(task=1, seed=42)

obs1, reward1, done1, info1 = env.step(0)
print("  Step 1 (approve_cheapest): reward=%.4f" % reward1.value)
print("    compliance=%.4f, cost=%.4f, speed=%.4f, carbon=%.4f" % (
    reward1.compliance, reward1.cost_efficiency, reward1.delivery_speed, reward1.carbon_score))
assert -1.0 <= reward1.value <= 1.0, "Reward out of range: %f" % reward1.value
assert isinstance(reward1, Reward)
assert not done1, "Episode should not end on step 1"

obs2, reward2, done2, info2 = env.step(3)
print("  Step 2 (reject_all): reward=%.4f" % reward2.value)
assert reward1.value != reward2.value, "Rewards must vary between actions"
print("  Rewards differ: %.4f vs %.4f" % (reward1.value, reward2.value))

rewards_seen = [reward1.value, reward2.value]
for i in range(3):
    _, r, _, _ = env.step(0)
    rewards_seen.append(r.value)
print("  5 rewards: %s" % rewards_seen)
assert len(set(rewards_seen)) > 1, "Rewards must not all be constant"
print("  PHASE 2 PASSED")

# =====================================================================
# PHASE 3 -- Disruption Engine
# =====================================================================
print("\n--- PHASE 3: Disruption Engine ---")
from env.disruption_engine import DisruptionEngine

engine = DisruptionEngine(task=3, seed=42)

result7 = engine.check(7)
assert len(result7) == 0, "Expected no disruptions at step 7"
print("  Step 7: [] -- OK")

disruptions8 = engine.check(8)
assert len(disruptions8) == 1, "Expected 1 disruption at step 8, got %d" % len(disruptions8)
assert disruptions8[0]["name"] == "port_strike_india"
print("  Step 8: %s -- OK" % disruptions8[0]["name"])

disruptions15 = engine.check(15)
assert len(disruptions15) == 1
assert disruptions15[0]["name"] == "war_reroute_eu"
print("  Step 15: %s -- OK" % disruptions15[0]["name"])

disruptions22 = engine.check(22)
assert len(disruptions22) == 1
assert disruptions22[0]["name"] == "protest_france"
print("  Step 22: %s -- OK" % disruptions22[0]["name"])

engine1 = DisruptionEngine(task=1, seed=42)
assert len(engine1.check(8)) == 0, "Task 1 should have no disruptions"
print("  Task 1 at step 8: no disruptions -- OK")
print("  PHASE 3 PASSED")

# =====================================================================
# PHASE 4 -- 100-Step Stress Test (Day 2 Critical Gate)
# =====================================================================
print("\n--- PHASE 4: 100-Step Stress Test (Critical Gate) ---")
env = GlobalProcurementEnv()
rewards = []
all_done = True

for episode in range(10):
    obs = env.reset(task=1, seed=episode)
    done = False
    ep_reward = 0.0
    steps = 0

    while not done and steps < 20:
        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        ep_reward += reward.value
        steps += 1

    if not done:
        all_done = False
    rewards.append(ep_reward)
    print("  Episode %d: steps=%d, total_reward=%.3f, done=%s" % (episode, steps, ep_reward, done))

print("")
print("  Reward range: %.3f to %.3f" % (min(rewards), max(rewards)))
print("  All episodes reached done=True: %s" % all_done)
print("  Rewards vary: %s" % (min(rewards) != max(rewards)))

gate_passed = min(rewards) != max(rewards) and all_done

# =====================================================================
# FINAL RESULT
# =====================================================================
print("\n" + "=" * 60)
if gate_passed:
    print("DAY 2 CRITICAL GATE: PASSED")
else:
    print("DAY 2 CRITICAL GATE: FAILED")
print("=" * 60)

if not gate_passed:
    sys.exit(1)
