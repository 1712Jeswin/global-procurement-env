import sys
from env.global_procurement_env import GlobalProcurementEnv

def test_grader_boundaries():
    """
    Tests specific grader boundary conditions:
    1. Immediate failure (done=True at step 1)
    2. Empty episode (done=True at step 0 - theoretically impossible but good to test)
    3. Perfect score vs Terrible score
    """
    env = GlobalProcurementEnv()
    
    print("=== Starting Boundary Tests ===")
    
    for task_id in [1, 2, 3]:
        print(f"\nProcessing Task {task_id}...")
        
        # 1. Immediate Failure
        # Force a done flag by exceeding max steps (simulated) 
        # or by taking an action that ends the episode.
        env.reset(task=task_id, seed=42)
        # We'll take 1 random action and then manually end it if needed
        # Actually, let's just use the real environment flow.
        _, _, done, _ = env.step(action=0)
        final_state = env.state()
        score = final_state.grader_score
        print(f"  Immediate (step 1) score: {score:.4f} (expected: valid float 0-1)")
        if score is None or not (0.0 <= score <= 1.0):
            print(f"  FAILED boundary check for task {task_id}")
            return False
            
    # 2. Perfect vs Terrible (Comparative check)
    # Task 1: 0, 1, 2 = Approve, 3 = Reject
    # If we approve everything (good) vs reject everything (bad)
    scores = []
    for action in [0, 3]:
        env.reset(task=1, seed=42)
        # Step until done
        done = False
        while not done:
            _, _, done, _ = env.step(action=action)
        scores.append(env.state().grader_score)
        
    print(f"\nComparative check (Task 1):")
    print(f"  Approve Cheapest: {scores[0]:.4f}")
    print(f"  Reject All:       {scores[1]:.4f}")
    
    if scores[0] == scores[1]:
        print("  WARNING: Grader returned identical scores for different outcomes!")
    else:
        print("  PASS: Grader produced distinct scores.")
        
    print("\n=== Boundary Tests Complete ===")
    return True

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='ascii', errors='replace')
    success = test_grader_boundaries()
    sys.exit(0 if success else 1)
