import requests
import time

BASE_URL = "http://localhost:7860"

def audit_patch_d():
    print("--- Patch D Audit: Task 2 Episode Length ---")
    
    # 1. Reset for Task 2
    try:
        response = requests.post(f"{BASE_URL}/reset", json={"task": 2, "seed": 42})
        if response.status_code != 200:
            print(f"Error: /reset failed with {response.status_code}: {response.text}")
            return
        obs = response.json()
        print(f"Reset Task 2: step={obs['step']}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    # 2. Step through until done
    steps = 0
    done = False
    while not done:
        # Action 0 = approve_cheapest
        resp = requests.post(f"{BASE_URL}/step", json={"action": 0})
        if resp.status_code != 200:
            print(f"Error: /step failed at step {steps} with {resp.status_code}: {resp.text}")
            break
        
        data = resp.json()
        obs = data["observation"]
        done = data["done"]
        steps += 1
        
        # 3. Verify disruption window (from Patch D)
        # Port strike starts at 8, duration 5 -> steps 8, 9, 10, 11, 12 should have disruptions
        if steps == 8:
            disruptions = obs.get("active_disruptions", [])
            print(f"Step 8 disruptions: {disruptions}")
            assert len(disruptions) > 0, "Disruption SHOULD be active at step 8"
        
        if steps == 12:
            disruptions = obs.get("active_disruptions", [])
            print(f"Step 12 disruptions: {disruptions}")
            assert len(disruptions) > 0, "Disruption SHOULD still be active at step 12"
            
        if steps == 13:
            disruptions = obs.get("active_disruptions", [])
            print(f"Step 13 disruptions: {disruptions}")
            assert len(disruptions) == 0, "Disruption SHOULD NOT be active at step 13"

    print(f"Total steps in Task 2: {steps}")
    assert steps == 50, f"Expected 50 steps, but got {steps}"
    
    print("\nPatch D Audit PASSED! ✅")

if __name__ == "__main__":
    audit_patch_d()
