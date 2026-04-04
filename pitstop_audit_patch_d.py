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
