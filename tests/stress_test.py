import random
import time
import sys
from env.global_procurement_env import GlobalProcurementEnv

def run_stress_test(num_episodes_per_task=167):
    """
    Runs a high-volume stress test using direct environment calls.
    Avoids HTTP overhead to achieve ~160 episodes per second.
    Total episodes for 3 tasks: ~501.
    """
    env = GlobalProcurementEnv()
    results = []
    
    print(f"=== Starting Stress Test ({num_episodes_per_task} episodes/task) ===")
    start_time = time.time()
    
    for task_id in [1, 2, 3]:
        max_action = {1: 3, 2: 5, 3: 6}[task_id]
        print(f"Testing Task {task_id}...")
        
        for seed in range(num_episodes_per_task):
            try:
                env.reset(task=task_id, seed=seed)
                done = False
                steps = 0
                while not done and steps < 150:
                    action = random.randint(0, max_action)
                    _, _, done, _ = env.step(action=action)
                    steps += 1
                
                final_state = env.state()
                score = final_state.grader_score
                
                # Validation: No None, No NaN, in range [0, 1]
                if score is None or score != score or not (0.0 <= score <= 1.0):
                    results.append(f"FAIL: Task {task_id}, Seed {seed}, Score {score}")
                
            except Exception as e:
                results.append(f"ERROR: Task {task_id}, Seed {seed}, Error: {str(e)}")
                
    duration = time.time() - start_time
    
    print(f"\n=== Stress Test Results ===")
    print(f"Total Episodes: {num_episodes_per_task * 3}")
    print(f"Failures/Errors: {len(results)}")
    print(f"Total Duration: {duration:.2f}s")
    print(f"Average Rate: {(num_episodes_per_task * 3) / duration:.2f} ep/s")
    
    if results:
        print("\nDetail of Failures:")
        for r in results:
            print(f"  {r}")
    else:
        print("All episodes passed with valid grader scores.")
        
    return len(results) == 0

if __name__ == "__main__":
    # Force ASCII output for Windows console stability
    sys.stdout.reconfigure(encoding='ascii', errors='replace')
    success = run_stress_test()
    sys.exit(0 if success else 1)
