from env.global_procurement_env import GlobalProcurementEnv
from env.disruption_engine import DisruptionEngine

def test_patch_d():
    print("--- Patch D Local Audit ---")
    
    # 1. Check Task 2 Episode Length
    env = GlobalProcurementEnv()
    obs = env.reset(task=2, seed=42)
    
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
    
    print(f"Task 2 episode length: {steps}")
    assert steps == 50, f"Expected 50 steps, but got {steps}"
    
    # 2. Check Disruption Window for Task 2
    # port_strike_india: trigger_step=8, duration_steps=5
    # Should be active for step 8, 9, 10, 11, 12.
    engine = DisruptionEngine(task=2, seed=42)
    
    assert len(engine.check(8))  > 0, "Disruption SHOULD be active at step 8"
    assert len(engine.check(12)) > 0, "Disruption SHOULD still be active at step 12"
    assert len(engine.check(13)) == 0, "Disruption SHOULD NOT be active at step 13"
    assert len(engine.check(50)) == 0, "No disruption at final step"
    
    print("Task 2 MAX_STEPS = 50 confirmed, disruption window verified ✅")
    print("Patch D Local Audit PASSED! ✅")

if __name__ == "__main__":
    try:
        test_patch_d()
    except Exception as e:
        print(f"Audit FAILED: {e}")
        import traceback
        traceback.print_exc()
