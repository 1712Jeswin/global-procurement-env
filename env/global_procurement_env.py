"""
STUB — global_procurement_env.py
=================================
This is a temporary mock so that api/main.py can be developed and tested
independently. Replace this entire file with Jenish's real implementation
once it is delivered.

Contract that Jenish's class MUST honour:
  • reset(task: int, seed: int) -> Observation
  • step(action: int) -> tuple[Observation, Reward, bool, dict]
  • state() -> Observation          (property OR zero-arg method)
  • env.state is None               before first reset()
  • env.is_done                     bool, True after episode ends

Jenish imports Observation, Action, Reward from api.schemas — do NOT
redefine those classes here.
"""

from api.schemas import Observation, Reward, SupplierObservation


class GlobalProcurementEnv:
    """Minimal stub that satisfies the api/main.py interface."""

    def __init__(self):
        # state == None signals "not yet reset"; main.py checks this
        self._state: Observation | None = None
        self.is_done: bool = False
        self.current_task: int = 1
        self._step_count: int = 0

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, task: int = 1, seed: int = 42) -> Observation:
        """
        Initialise the environment.
        Returns the starting Observation for the chosen task.
        """
        self.current_task = task
        self.is_done = False
        self._step_count = 0

        # Example suppliers — Jenish will replace with real procurement data
        suppliers = [
            SupplierObservation(
                id="SUP-001", country="IN", price_usd=12000.0,
                lead_days=7, carbon_tons=2.3, available=True,
                applied_duty_rate=0.05,
            ),
            SupplierObservation(
                id="SUP-002", country="CN", price_usd=9500.0,
                lead_days=14, carbon_tons=4.1, available=True,
                applied_duty_rate=0.12,
            ),
            SupplierObservation(
                id="SUP-003", country="DE", price_usd=18000.0,
                lead_days=5, carbon_tons=1.1, available=True,
                applied_duty_rate=0.0,
            ),
        ]

        self._state = Observation(
            step=0,
            budget_remaining=200_000.0,
            inventory={"steel": 0.0, "chips": 0.0, "fabric": 0.0},
            suppliers=suppliers,
            active_disruptions=[],
            policy_violations_this_episode=0,
            current_task=task,
            grader_score=None,
        )
        return self._state

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: int = 0) -> tuple[Observation, Reward, bool, dict]:
        """
        Advance the episode by one action.
        Returns: (Observation, Reward, done: bool, info: dict)
        """
        self._step_count += 1

        # --- placeholder logic (Jenish replaces this) ---
        budget_used = 12_000.0 + action * 500.0
        new_budget = (self._state.budget_remaining - budget_used
                      if self._state else 200_000.0)

        suppliers = [
            SupplierObservation(
                id="SUP-001", country="IN", price_usd=12000.0,
                lead_days=7, carbon_tons=2.3, available=True,
                applied_duty_rate=0.05,
            ),
        ]

        obs = Observation(
            step=self._step_count,
            budget_remaining=max(new_budget, 0.0),
            inventory={"steel": float(self._step_count * 10),
                       "chips": 0.0, "fabric": 0.0},
            suppliers=suppliers,
            active_disruptions=[],
            policy_violations_this_episode=0,
            current_task=self.current_task,
            grader_score=None,
        )

        reward = Reward(
            value=0.65,
            compliance=1.0,
            cost_efficiency=0.7,
            delivery_speed=0.6,
            carbon_score=0.8,
        )

        # Mark done after 10 steps (placeholder; Jenish defines real horizon)
        done = self._step_count >= 10 or new_budget <= 0
        self.is_done = done
        self._state = obs

        info: dict = {"stub": True, "step_taken": self._step_count}
        return obs, reward, done, info

    # ── state ─────────────────────────────────────────────────────────────────
    def state(self) -> Observation:
        """Returns the current observation without advancing the episode."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._state