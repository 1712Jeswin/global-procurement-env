def grade(episode_log: dict) -> float:
    """
    Returns a float in [0.0, 1.0] representing Task 2 performance.
    Adds carbon_rate to the calculation.
    """
    steps = episode_log.get("steps", [])
    if not steps:
        return 0.0

    total_steps = len(steps)
    violations = episode_log.get("total_violations", total_steps)

    compliance_rate = max(0.0, 1.0 - (violations / max(total_steps, 1)))

    total_lead = episode_log.get("total_lead_days", 30 * total_steps)
    avg_lead = total_lead / max(total_steps, 1)
    delivery_rate = max(0.0, 1.0 - (avg_lead / 30.0))

    starting_budget = 200000.0
    final_budget = episode_log.get("final_budget", 0.0)
    cost_rate = max(0.0, final_budget / starting_budget)

    # Carbon rate (new in Task 2 -- EU carbon rules apply)
    total_carbon = episode_log.get("total_carbon", 50.0)
    carbon_rate = max(0.0, 1.0 - (total_carbon / 50.0))

    score = 0.40 * compliance_rate + 0.25 * delivery_rate + 0.20 * cost_rate + 0.15 * carbon_rate
    return round(max(0.0, min(1.0, score)), 4)
