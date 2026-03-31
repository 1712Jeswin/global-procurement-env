def grade(episode_log: dict) -> float:
    """
    Returns a float in [0.0, 1.0] representing Task 1 performance.
    Disqualification checks:
    - If episode_log is empty or has 0 steps -> return 0.0 (don't crash)
    - Never return NaN or None
    """
    steps = episode_log.get("steps", [])

    # Handle edge case: done=True at step 1 (immediate hard violation)
    if not steps:
        return 0.0

    total_steps = len(steps)
    violations = episode_log.get("total_violations", total_steps)  # default to worst case

    # compliance_rate: fraction of steps with no violation
    compliance_rate = max(0.0, 1.0 - (violations / max(total_steps, 1)))

    # delivery_rate: inverse of average lead days normalised to max (30 days)
    total_lead = episode_log.get("total_lead_days", 30 * total_steps)
    avg_lead = total_lead / max(total_steps, 1)
    delivery_rate = max(0.0, 1.0 - (avg_lead / 30.0))

    # cost_rate: how much budget remains (higher = better)
    starting_budget = 200000.0
    final_budget = episode_log.get("final_budget", 0.0)
    cost_rate = max(0.0, final_budget / starting_budget)

    score = 0.50 * compliance_rate + 0.30 * delivery_rate + 0.20 * cost_rate

    # Final safety clamp -- never return outside [0, 1]
    return round(max(0.0, min(1.0, score)), 4)
