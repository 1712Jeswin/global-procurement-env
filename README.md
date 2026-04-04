# GlobalProcurementEnv

> **OpenEnv Hackathon — Round 1 Submission**

An AI procurement simulation where an agent acts as a procurement officer sourcing materials from **India**, **EU**, and **USA** suppliers under real government trade policies and seeded supply-chain disruptions.

Built with [`openenv-core`](https://pypi.org/project/openenv-core/), deployed on Hugging Face Spaces as a Docker-based FastAPI application.

---

## Live Environment

**Base URL:** `https://JEN-chad-global-procurement-env.hf.space`

| Endpoint       | Method | Description                                |
|----------------|--------|--------------------------------------------|
| `/health`      | GET    | Server health check — returns `{"status": "ok"}` |
| `/reset`       | POST   | Start a new episode for a given task & seed |
| `/step`        | POST   | Advance the episode by one action          |
| `/state`       | GET    | Read current observation without stepping  |

---

## Quick Start

### Health check

```bash
curl https://JEN-chad-global-procurement-env.hf.space/health
```

Expected response:
```json
{"status": "ok"}
```

### Reset (start a new episode)

```bash
curl -X POST https://JEN-chad-global-procurement-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": 1, "seed": 42}'
```

Expected response (truncated):
```json
{
  "step": 0,
  "budget_remaining": 200000.0,
  "inventory": {"steel": 0.0, "copper": 0.0, "silicon": 0.0},
  "suppliers": [...],
  "active_disruptions": [],
  "policy_violations_this_episode": 0,
  "current_task": 1,
  "grader_score": null
}
```

### Step (take an action)

```bash
curl -X POST https://JEN-chad-global-procurement-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": 0}'
```

Expected response (truncated):
```json
{
  "step": 1,
  "budget_remaining": 195000.0,
  "suppliers": [...],
  "reward": 0.65,
  "reward_breakdown": {
    "value": 0.65,
    "compliance": 1.0,
    "cost_efficiency": 0.5,
    "delivery_speed": 0.7,
    "carbon_score": 0.4
  },
  "done": false,
  "info": {}
}
```

### State (read current observation)

```bash
curl https://JEN-chad-global-procurement-env.hf.space/state
```

---

## Tasks

| # | Name | Countries | Actions | Max Steps | Disruptions |
|---|------|-----------|---------|-----------|-------------|
| 1 | Easy — India Only | India | 0–3 | 20 | None |
| 2 | Medium — India + EU | India, EU | 0–5 | 50 | `port_strike_india` |
| 3 | Hard — All 3 Countries | India, EU, USA | 0–6 | 100 | `port_strike_india`, `war_reroute_eu`, `protest_france` |

## Action Space

| Action | Name | Description |
|--------|------|-------------|
| 0 | `approve_cheapest` | Buy from the lowest-cost available supplier |
| 1 | `approve_fastest` | Buy from the fastest-delivering supplier |
| 2 | `approve_greenest` | Buy from the lowest-carbon supplier |
| 3 | `reject_all` | Skip this round — buy nothing |
| 4 | `partial_order` | Split order across multiple suppliers (Task 2+) |
| 5 | `negotiate` | Attempt to lower price with a supplier (Task 2+) |
| 6 | `escalate` | Override policy restrictions (Task 3 only) |

---

## Model Performance

| Task | Random Baseline | Trained PPO | Improvement |
|------|----------------|-------------|-------------|
| Task 1 (Easy — India only) | 0.38 | TBD | TBD |
| Task 2 (Medium — India + EU) | 0.21 | TBD | TBD |
| Task 3 (Hard — All 3 Countries) | 0.26 | TBD | TBD |

> **Note:** Random baseline scores are averaged over 20 seeds. Trained PPO and LLM inference scores will be updated once Jenish's training runs complete.

---

## Reward Formula

Each step returns a composite reward in `[-1.0, 1.0]`:

```
reward = 0.30 * compliance + 0.30 * cost_efficiency + 0.20 * delivery_speed + 0.20 * carbon_score
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `compliance` | 0.30 | Policy violation penalty — hard violations end the episode |
| `cost_efficiency` | 0.30 | Budget preservation — buying cheap is better |
| `delivery_speed` | 0.20 | Lower lead times score higher |
| `carbon_score` | 0.20 | Lower carbon emissions score higher |

---

## Grader Scoring

Each task has its own grader (in `graders/`) that scores the full episode on completion:

| Grader | Weights | Key Factor |
|--------|---------|------------|
| `grader1` (Task 1) | 50% compliance, 30% delivery, 20% cost | Compliance-heavy |
| `grader2` (Task 2) | 40% compliance, 25% delivery, 20% cost, 15% carbon | EU carbon rules |
| `grader3` (Task 3) | 45% compliance, 25% delivery, 15% cost, 15% carbon | OFAC risk weighting |

All graders:
- Return `float` in `[0.0, 1.0]`
- Return `0.0` for empty episodes (`{"steps": []}`)
- Never return `None` or `NaN`
- Produce different scores for different episode outcomes

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current step number (0-indexed) |
| `budget_remaining` | float | USD remaining from starting budget |
| `inventory` | object | Quantities of each material procured |
| `suppliers` | array | List of available suppliers with prices, lead times, carbon |
| `active_disruptions` | array | Active supply-chain disruptions |
| `policy_violations_this_episode` | int | Cumulative hard violations |
| `current_task` | int | Task ID (1, 2, or 3) |
| `grader_score` | float \| null | Final grader score (set when episode ends) |
| `cumulative_lead_days` | int | Total lead days accumulated |
| `cumulative_carbon` | float | Total carbon tons accumulated |
| `available_supplier_count` | int | Number of currently available suppliers |

---

## Project Structure

```
global-procurement-env/
├── api/
│   ├── main.py               # FastAPI server with /health, /reset, /step, /state
│   └── schemas.py             # Pydantic models (Observation, Action, Reward)
├── env/
│   ├── global_procurement_env.py  # Main environment class
│   ├── supply_chain_sim.py        # Supply chain state & action logic
│   ├── constraint_engine.py       # Policy validation & compliance
│   ├── disruption_engine.py       # Supply-chain disruption simulation
│   ├── gym_wrapper.py             # Gymnasium wrapper for SB3 training
│   └── tasks/
│       ├── task1_easy.py          # India-only, 4 actions, 20 steps
│       ├── task2_medium.py        # India+EU, 6 actions, 50 steps
│       └── task3_hard.py          # All countries, 7 actions, 100 steps
├── graders/
│   ├── grader1.py             # Task 1 scoring
│   ├── grader2.py             # Task 2 scoring (includes carbon)
│   └── grader3.py             # Task 3 scoring (OFAC-weighted)
├── data/
│   └── suppliers.json         # Supplier catalog (India, EU, USA)
├── policies/
│   ├── india.json             # India trade policy
│   ├── eu.json                # EU trade policy (carbon levy)
│   └── usa.json               # USA trade policy (OFAC sanctions)
├── disruptions/
│   └── disruptions.json       # Supply-chain disruption scenarios
├── baseline.py                # Random baseline agent
├── inference.py               # LLM-driven agent (OpenAI client)
├── train.py                   # PPO training script (SB3)
├── trained_agent.py           # Trained PPO evaluation
├── openenv.yaml               # OpenEnv specification file
├── requirements.txt           # Pinned dependencies
└── Dockerfile                 # Docker deployment config
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- Docker (optional, for container testing)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 7860
```

### Run the baseline agent

```bash
python baseline.py
```

### Run the inference agent (requires API key)

```bash
API_BASE_URL=https://api-inference.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
HF_TOKEN=your_token_here \
python inference.py
```

---

## Docker

### Build

```bash
docker build -t gpe .
```

### Run

```bash
docker run -p 7860:7860 gpe
```

### Verify

```bash
curl http://localhost:7860/health
# → {"status": "ok"}
```

---

## Stress Test Results

**500-episode stress test** — 167 episodes per task, random actions, all seeds:

| Metric | Result |
|--------|--------|
| Total episodes | 501 |
| Failures | **0** |
| Rate | ~160 ep/s |
| Duration | 3.1s |
| None scores | 0 |
| NaN scores | 0 |
| Out-of-range scores | 0 |

**Grader boundary tests** — all 9/9 passed:
- Immediate failure (done=True at step 1): all graders return valid `[0, 1]` float
- Empty episode: all graders return `0.0`
- Perfect vs terrible: all graders produce distinct scores

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11 |
| Framework | FastAPI | 0.111.0 |
| Server | Uvicorn | 0.35.0 |
| Models | Pydantic | 2.11.7 |
| Training | Stable-Baselines3 | 2.3.2 |
| Gym | Gymnasium | 0.29.1 |
| OpenEnv | openenv-core | 0.2.2 |
| LLM Client | OpenAI | ≥1.0.0 |

---

## Team

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Jenish | Env Core | `env/`, `baseline.py`, `train.py`, `inference.py` |
| Jeswin | Data & Tasks | `graders/`, `policies/`, `data/`, `disruptions/`, `README.md` |
| Jamal | Infra & Deploy | `api/`, `Dockerfile`, `requirements.txt`, `openenv.yaml` |

---

## License

MIT
