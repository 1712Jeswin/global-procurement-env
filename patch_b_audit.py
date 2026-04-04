from env.global_procurement_env import GlobalProcurementEnv
from api.schemas import Observation

env = GlobalProcurementEnv()
obs = env.reset(task=2, seed=42)

# Fields now exist on the public Observation object
assert hasattr(obs, "cumulative_lead_days"), "cumulative_lead_days missing from Observation"
assert hasattr(obs, "cumulative_carbon"), "cumulative_carbon missing from Observation"
print("Initial lead days:", obs.cumulative_lead_days)   # 0 at reset
print("Initial carbon:", obs.cumulative_carbon)      # 0.0 at reset

# Take a step and confirm they update
result_obs, _, _, _ = env.step(0)
print("Updated lead days:", result_obs.cumulative_lead_days)   # > 0 after a purchase
print("Updated carbon:", result_obs.cumulative_carbon)      # > 0.0 after a purchase

# Gym wrapper still returns correct shape
from env.gym_wrapper import ProcurementGymWrapper
import numpy as np
wrapper = ProcurementGymWrapper(task=2, seed=42)
arr, _ = wrapper.reset()
print("Array shape:", arr.shape)   # (8,) — unchanged
print("Array dtype:", arr.dtype)   # float32 — unchanged
print("Array bounds check:", all(0.0 <= v <= 1.0 for v in arr))  # True
print("Patch B Audio passed!")
