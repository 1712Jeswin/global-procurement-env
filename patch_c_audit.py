import random
from env.disruption_engine import DisruptionEngine

# Tasks 1 and 2 must be unaffected
for task in [1, 2]:
    engine = DisruptionEngine(task=task, seed=42)
    at_step_20 = engine.check(20)
    stochastic_fired = any(s.get("stochastic") for s in at_step_20)
    assert not stochastic_fired, f"Task {task} should never see stochastic disruption"
    print(f"Task {task} at step 20: {[s['name'] for s in at_step_20]}")

# Task 3 should occasionally fire the stochastic disruption at step 30
# (outside the fixed window of step 15–24)
random.seed(99)  # use a seed where the 30% roll fires
engine3 = DisruptionEngine(task=3, seed=99)
hits = sum(1 for _ in range(20) if any(
    s.get("stochastic") for s in engine3.check(30)
))
print(f"Task 3 step 30: stochastic fired {hits}/20 times (expect ~6)")
assert hits > 0, "Stochastic disruption never fired in 20 rolls — check the probability logic"
