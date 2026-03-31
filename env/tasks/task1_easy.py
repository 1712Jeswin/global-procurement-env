"""
Task 1 — Easy Mode
India-only suppliers, 3 active suppliers, 20 steps maximum.
Actions 0-3 only (no negotiate, no split, no escalate).
No disruptions.
Teaches the agent basic cost/speed/carbon trade-offs.
"""

TASK_ID = 1
TASK_NAME = "easy_india_only"

MAX_STEPS = 20
VALID_ACTIONS = [0, 1, 2, 3]  # approve_cheapest, approve_fastest, approve_greenest, reject_all
SUPPLIER_COUNTRIES = ["India"]
NUM_SUPPLIERS = 3  # Use only 3 of the 4 India suppliers (randomly selected at reset)
DISRUPTIONS_ENABLED = False
STARTING_BUDGET = 200000.0

# Grader weights for Task 1 — simpler scoring, compliance matters most
GRADER_WEIGHTS = {
    "compliance_rate": 0.50,
    "delivery_rate": 0.30,
    "cost_rate": 0.20
}
