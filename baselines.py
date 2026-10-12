def fixed_action(env, target):
    # Count pending capacity so the controller does not request it twice.
    total = env.active + int(env.timer > 0)
    if total < target:
        return 2
    if total > target:
        return 0
    return 1


def threshold_action(env, upper, lower):
    # Wait for the current addition before making another scaling decision.
    if env.timer > 0:
        return 1
    arrivals = env.trace["arrivals"][env.interval]
    # Compare arriving and queued work with capacity available in this interval.
    pressure = (arrivals + env.queue) / (env.active * env.config["capacity_per_replica"])
    if pressure > upper:
        return 2
    if pressure < lower:
        return 0
    return 1
