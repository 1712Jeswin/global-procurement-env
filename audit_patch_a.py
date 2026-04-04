"""
Patch A Audit — disruption_engine.py duration fix
Run from project root: python audit_patch_a.py
"""
from env.disruption_engine import DisruptionEngine

print("=== Patch A Audit — DisruptionEngine.check() duration fix ===\n")

engine = DisruptionEngine(task=2, seed=42)

# Step 7: strike has not started yet
result_7 = engine.check(7)
print(f"Step  7 (before strike):  {result_7}")
assert result_7 == [], f"FAIL — expected [] at step 7, got {result_7}"

# Step 8: strike starts
result_8 = engine.check(8)
names_8 = [s["name"] for s in result_8]
print(f"Step  8 (strike starts):  {names_8}")
assert any(s["name"] == "port_strike_india" for s in result_8), \
    f"FAIL — expected port_strike_india at step 8, got {result_8}"

# Step 10: mid-window
result_10 = engine.check(10)
names_10 = [s["name"] for s in result_10]
print(f"Step 10 (mid window):     {names_10}")
assert any(s["name"] == "port_strike_india" for s in result_10), \
    f"FAIL — expected port_strike_india at step 10, got {result_10}"

# Step 12: last step in window (8 + 5 = 13, so 8–12 inclusive)
result_12 = engine.check(12)
names_12 = [s["name"] for s in result_12]
print(f"Step 12 (last in window): {names_12}")
assert any(s["name"] == "port_strike_india" for s in result_12), \
    f"FAIL — expected port_strike_india at step 12, got {result_12}"

# Step 13: outside the window
result_13 = engine.check(13)
print(f"Step 13 (outside window): {result_13}")
assert result_13 == [], f"FAIL — expected [] at step 13, got {result_13}"

print("\n✅ All Patch A assertions passed.")
print("   Steps 8–12 return port_strike_india ✓")
print("   Steps 7 and 13 return [] ✓")
print("\n=== Patch A Audit COMPLETE ===")
