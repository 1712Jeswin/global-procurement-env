# disruption_engine.py — Tracks seeded crisis events that fire at specific step numbers.
# Because the same seed is used, disruptions are deterministic and reproducible.

import json
import os


class DisruptionEngine:
    def __init__(self, task: int, seed: int):
        self.task = task
        self.seed = seed
        self._load_scenarios()

    def _load_scenarios(self):
        path = os.path.join(os.path.dirname(__file__), "..", "disruptions", "scenarios.json")
        with open(path, "r") as f:
            self.scenarios = json.load(f)  # list of scenario dicts

    def check(self, step_count: int) -> list:
        """
        Returns a list of active disruption dicts for the current step.
        Task 1: no disruptions
        Task 2: disruption at step 8 only
        Task 3: all 3 disruptions
        """
        active = []
        for scenario in self.scenarios:
            if scenario["trigger_step"] == step_count:
                if self.task >= scenario.get("min_task", 1):
                    active.append(scenario)
        return active
