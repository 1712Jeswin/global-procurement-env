# disruption_engine.py — Tracks seeded crisis events that fire at specific step numbers.
# Because the same seed is used, disruptions are deterministic and reproducible.

import json
import os
import random


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
        Respects duration_steps — disruptions persist across the full window
        [trigger_step, trigger_step + duration_steps).
        Task 1: no disruptions
        Task 2: port_strike_india (step 8–12, duration 5)
        Task 3: all 3 disruptions
        """
        active = []
        for scenario in self.scenarios:
            start = scenario["trigger_step"]
            end = start + scenario.get("duration_steps", 1)
            if start <= step_count < end:
                if self.task >= scenario.get("min_task", 1):
                    active.append(scenario)

        # For Task 3, stochastic-flagged disruptions can also fire randomly
        # outside their fixed window. random.random() is seeded in reset() so
        # runs with the same seed are still reproducible.
        if self.task == 3 and step_count > 5:
            for scenario in self.scenarios:
                if scenario.get("stochastic") and scenario not in active:
                    if random.random() < 0.30:   # 30% chance per step
                        active.append(scenario)

        return active
