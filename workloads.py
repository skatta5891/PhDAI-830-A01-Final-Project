import json
from pathlib import Path

import numpy as np


def load_config():
    return json.loads(Path(__file__).with_name("config.json").read_text())


def generate_trace(seed, config, switch_probability=None):
    rng = np.random.default_rng(seed)
    levels = config["arrival_levels"]
    probability = config["switch_probability"] if switch_probability is None else switch_probability
    mean = int(rng.choice(levels))
    means, arrivals = [], []
    for interval in range(config["episode_intervals"]):
        # A switch must choose a different demand level.
        if interval > 0 and rng.random() < probability:
            mean = int(rng.choice([value for value in levels if value != mean]))
        means.append(mean)
        arrivals.append(int(rng.poisson(mean)))
    return {"seed": seed, "means": means, "arrivals": arrivals}


def make_traces(config):
    # Generate traffic separately from controller actions for fair comparisons.
    sets = {}
    for name in ["validation", "nominal_test", "robustness_test"]:
        start, stop = config[name + "_seeds"]
        probability = config["robustness_switch_probability"] if name == "robustness_test" else config["switch_probability"]
        sets[name] = [generate_trace(seed, config, probability) for seed in range(start, stop)]
    return sets
