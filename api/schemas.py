from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

# ─── OpenEnv Interface Models ─────────────────────────────────────────────────
# These are the typed models the OpenEnv spec requires.
# Jenish's GlobalProcurementEnv imports Observation, Action, Reward from here.
# reset()  → returns Observation
# step()   → returns (Observation, Reward, bool, dict)
# state()  → returns Observation

class SupplierObservation(BaseModel):
    id: str
    country: str
    price_usd: float
    lead_days: int
    carbon_tons: float
    available: bool
    applied_duty_rate: float


class Observation(BaseModel):
    step: int
    budget_remaining: float
    inventory: Dict[str, float]
    suppliers: List[SupplierObservation]
    active_disruptions: List[str]
    policy_violations_this_episode: int
    current_task: int
    grader_score: Optional[float] = None
    cumulative_lead_days: int = 0
    cumulative_carbon: float = 0.0
    available_supplier_count: int = 0


class Action(BaseModel):
    action: int = Field(ge=0, le=6, description="Action integer 0–6")


class Reward(BaseModel):
    value: float
    compliance: float
    cost_efficiency: float
    delivery_speed: float
    carbon_score: float


# ─── FastAPI HTTP Request / Response Models ───────────────────────────────────
# These validate JSON bodies coming in from HTTP callers
# (judges' automated system, baseline.py, inference.py).

class ResetRequest(BaseModel):
    task: int = Field(
        default=1, ge=1, le=3,
        description="Task ID: 1=easy, 2=medium, 3=hard"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    action: int = Field(
        ge=0, le=6,
        description=(
            "Action integer: "
            "0=approve_cheapest, 1=approve_fastest, 2=approve_greenest, "
            "3=reject_all, 4=partial_order, 5=negotiate, 6=escalate"
        )
    )


class HealthResponse(BaseModel):
    status: str = "ok"