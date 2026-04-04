"""
Phase 5 — Patch E Pitstop Audit
Tests:
  Part 1: inventory increments exactly one material per purchase
  Part 2: available_supplier_count is correct at rest and during disruption
  Part 3: gym wrapper shape remains (8,)
"""
import sys, traceback

def run():
    from env.global_procurement_env import GlobalProcurementEnv
    from env.gym_wrapper import ProcurementGymWrapper
    import numpy as np

    env = GlobalProcurementEnv()

    # ── Part 1: material-specific inventory ────────────────────────────────────
    print("Part 1: material-specific inventory")
    obs = env.reset(task=1, seed=42)
    # Task 1, India only: IN_01(steel,38k), IN_02(chips,42k), IN_03(fabric,35k), IN_04(steel,50k)
    # approve_cheapest → IN_03 (35k, material=fabric)
    before = dict(obs.inventory)
    print(f"  before : {before}")

    step_obs, _, _, _ = env.step(0)   # approve_cheapest
    after = dict(step_obs.inventory)
    print(f"  after  : {after}")

    changed = [k for k in ("steel","chips","fabric") if after.get(k,0) != before.get(k,0)]
    assert len(changed) == 1, f"Expected 1 material to change, got {len(changed)}: {changed}"
    assert changed[0] == "fabric", f"Expected 'fabric', got '{changed[0]}'"
    total_added = sum(after.values()) - sum(before.values())
    assert total_added == 1.0, f"Expected +1 unit total, got +{total_added}"
    print(f"  ✅ Only '{changed[0]}' changed (+1 unit)")

    # ── Part 2: available_supplier_count ──────────────────────────────────────
    print("Part 2: available_supplier_count")
    obs2 = env.reset(task=2, seed=42)
    total_suppliers = len(obs2.suppliers)   # India+EU = 7
    print(f"  Task 2 total suppliers: {total_suppliers}")
    assert hasattr(obs2, "available_supplier_count"), "Field missing from Observation"
    assert obs2.available_supplier_count == total_suppliers, \
        f"At reset expected {total_suppliers}, got {obs2.available_supplier_count}"
    print(f"  ✅ At reset: available_supplier_count = {obs2.available_supplier_count}")

    # Step 7 times (action 3=reject_all, no purchase) then take one step each
    # After 8 steps, step_count=8 → disruption fires
    for i in range(8):
        obs2, _, _, _ = env.step(3)   # reject_all → no supplier selected

    print(f"  Step 8 disruptions: {obs2.active_disruptions}")
    # port_strike_india knocks out IN_02 and IN_03 (2 suppliers)
    expected_available = total_suppliers - 2
    assert obs2.available_supplier_count == expected_available, \
        f"During port strike expected {expected_available}, got {obs2.available_supplier_count}"
    print(f"  ✅ During disruption: available_supplier_count = {obs2.available_supplier_count}")

    # ── Part 3: gym wrapper shape ──────────────────────────────────────────────
    print("Part 3: gym wrapper shape")
    wrapper = ProcurementGymWrapper(task=2, seed=42)
    arr, _ = wrapper.reset()
    assert arr.shape == (8,), f"Expected (8,), got {arr.shape}"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    print(f"  ✅ shape={arr.shape}, dtype={arr.dtype}")

    print("\n✅ Patch E Audit PASSED")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\n❌ Audit FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
