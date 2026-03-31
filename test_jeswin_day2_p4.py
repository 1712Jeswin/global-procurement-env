# test_jeswin_day2_p4.py
from graders.grader1 import grade as grade1
from graders.grader2 import grade as grade2
from graders.grader3 import grade as grade3

# Test 1: Good episode - should score well (near 1.0)
good_ep = {
    "task_id": 1, "seed": 42,
    "steps": [{"step": i, "action": 0, "reward": 0.7, "done": i==19}
              for i in range(1, 21)],
    "total_violations": 0,
    "final_budget": 140000.0,
    "total_lead_days": 50,
    "total_carbon": 10.0
}

# Test 2: Bad episode - immediate hard violation at step 1
bad_ep = {
    "task_id": 1, "seed": 42,
    "steps": [{"step": 1, "action": 0, "reward": -1.0, "done": True}],
    "total_violations": 1,
    "final_budget": 200000.0,
    "total_lead_days": 0,
    "total_carbon": 0.0
}

# Test 3: Empty episode (edge case)
empty_ep = {"task_id": 1, "seed": 42, "steps": [], "total_violations": 0}

for grader_fn in [grade1, grade2, grade3]:
    good_score = grader_fn(good_ep)
    bad_score = grader_fn(bad_ep)
    empty_score = grader_fn(empty_ep)

    assert isinstance(good_score, float), "Score must be float"
    assert 0.0 <= good_score <= 1.0, f"Good score out of range: {good_score}"
    assert 0.0 <= bad_score <= 1.0, f"Bad score out of range: {bad_score}"
    assert empty_score == 0.0, f"Empty episode should return 0.0, got {empty_score}"
    assert good_score != bad_score, "Grader returns same score for different episodes!"
    print(f"  Good: {good_score:.4f}, Bad: {bad_score:.4f}, Empty: {empty_score:.4f}")

print("All graders pass critical gate checks")
