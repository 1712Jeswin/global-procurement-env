import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    Observation, Action, Reward,
    ResetRequest, StepRequest, HealthResponse,
)
from env.global_procurement_env import GlobalProcurementEnv

app = FastAPI(
    title="GlobalProcurementEnv",
    description="AI procurement simulation — OpenEnv Hackathon",
    version="1.0.0",
)

# ─── Single global env instance + thread lock ─────────────────────────────────
# One env object is shared across all requests; the lock prevents concurrent
# mutations from racing each other.
env = GlobalProcurementEnv()
lock = threading.Lock()
# ─────────────────────────────────────────────────────────────────────────────


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """
    Always returns HTTP 200 {"status": "ok"}.
    No try/except — if this fails the whole server is broken.
    Judges ping this first to confirm the server is alive.
    """
    return HealthResponse(status="ok")


# ── /reset ────────────────────────────────────────────────────────────────────
@app.post("/reset")
def reset(request: ResetRequest):
    """
    Initialise (or re-initialise) the environment for a given task.

    Body:
        task  (int, 1–3): difficulty level
        seed  (int):      RNG seed for reproducibility

    Returns:
        Observation JSON — the starting state of the episode.
    """
    try:
        with lock:
            observation: Observation = env.reset(
                task=request.task,
                seed=request.seed,
            )
        return observation.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── /step ─────────────────────────────────────────────────────────────────────
@app.post("/step")
def step(request: StepRequest):
    """
    Advance the environment by one action.

    env.step() returns a 4-tuple: (Observation, Reward, bool, dict).
    We unpack it and return a flat JSON that merges all four parts.

    Body:
        action (int, 0–6): procurement decision

    Returns:
        Flat JSON with:
            • all Observation fields (step, budget_remaining, suppliers, …)
            • reward           (float)  — scalar score for this step
            • reward_breakdown (dict)   — per-component breakdown
            • done             (bool)   — True when the episode has ended
            • info             (dict)   — auxiliary diagnostic info

    Errors:
        400 if /reset has not been called yet, or the episode is already done.
        Task 1 accepts actions 0–3; Task 2: 0–5; Task 3: 0–6.
        Pydantic rejects action > 6 with a 422 before this function runs.
    """
    try:
        with lock:
            if env._state is None:
                raise ValueError("Call /reset before /step.")
            if env.is_done:
                raise ValueError("Episode has ended. Call /reset to start a new one.")

            # env.step() contract: returns (Observation, Reward, bool, dict)
            observation, reward, done, info = env.step(action=request.action)

        # Flatten observation + reward metadata into a single response dict
        response = observation.model_dump()
        response["reward"] = reward.value                  # scalar float
        response["reward_breakdown"] = reward.model_dump() # full breakdown
        response["done"] = done                            # bool
        response["info"] = info                            # aux dict
        return response

    except HTTPException:
        raise  # re-raise 400/422 errors unchanged
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── /state ────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    """
    Returns the current Observation without advancing the episode.
    Useful for polling / logging without consuming a step.
    """
    try:
        with lock:
            observation: Observation = env.state()
        return observation.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))