# test_jeswin_day2_p1_p2.py
import importlib

# Phase 1
task1 = importlib.import_module("env.tasks.task1_easy")
assert task1.MAX_STEPS == 20
assert task1.DISRUPTIONS_ENABLED == False
assert set(task1.VALID_ACTIONS) == {0, 1, 2, 3}
assert "India" in task1.SUPPLIER_COUNTRIES
print("Task 1 config valid")

# Phase 2
for task_num, expected_max_steps, expected_actions in [(2, 50, 6), (3, 100, 7)]:
    mod = importlib.import_module(f"env.tasks.task{task_num}_{'medium' if task_num==2 else 'hard'}")
    assert mod.MAX_STEPS == expected_max_steps
    assert len(mod.VALID_ACTIONS) == expected_actions
    print(f"Task {task_num} config valid")
