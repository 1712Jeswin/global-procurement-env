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
